#include <torch/extension.h>
#include <ATen/core/TensorAccessor.h>   // <-- 추가 (packed_accessor*, RestrictPtrTraits)

#include <c10/util/Half.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <vector>
#include <limits>

#define BACKWARD_W_BATCH_THREADS 32

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)

// adapted from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans)                                                                                                 \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(const cudaError_t code, const char *const file, const int line, const bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }


/**********************************************************************************************************************/


template <typename T> struct AtomicFPOp;

template <> struct AtomicFPOp<at::Half> {
    template <typename func_t> inline __device__ at::Half operator()(at::Half *address, at::Half val, const func_t &func) {
        unsigned int *address_as_ui = (unsigned int *)((char *)address - ((size_t)address & 2));
        unsigned int old = *address_as_ui;
        unsigned int assumed;

        at::Half hsum;
        do {
            assumed = old;
            hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
            hsum = func(hsum, val);
            old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
            old = atomicCAS(address_as_ui, assumed, old);
        } while (assumed != old);
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        return hsum;
    }
};

static inline __device__ at::Half gpuAtomicAdd(at::Half *address, at::Half val) {
#if defined(USE_ROCM) || ((defined(CUDA_VERSION) && CUDA_VERSION < 10000) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))

    unsigned int *aligned = (unsigned int *)((size_t)address - ((size_t)address & 2));
    unsigned int old = *aligned;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned short old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
        __half sum = c10::Half(__ushort_as_half(old_as_us)) + c10::Half(__float2half((float)val));
        unsigned short sum_as_us = __half_as_ushort(sum);
        unsigned int sum_as_ui = (size_t)address & 2 ? (sum_as_us << 16) | (old & 0xffff) : (old & 0xffff0000) | sum_as_us;
        old = atomicCAS(aligned, assumed, sum_as_ui);
    } while (assumed != old);
    unsigned short old_as_us = (unsigned short)((size_t)address & 2 ? old >> 16 : old & 0xffff);
    return c10::Half((__half_raw)__ushort_as_half(old_as_us));
#else
    return atomicAdd(reinterpret_cast<__half *>(address), val);
#endif
}

static inline __device__ float gpuAtomicAdd(float *address, float val) { return atomicAdd(address, val); }

static inline __device__ double gpuAtomicAdd(double *address, double val) { return atomicAdd(address, val); }




/**********************************************************************************************************************/
/** TRAINING MODE  ****************************************************************************************************/
/**********************************************************************************************************************/


template <typename scalar_t>
__global__ void logic_layer_cuda_forward_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> y
) {

    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < y.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < y.size(0);
            col += blockDim.y * gridDim.y
        ) {

            const auto idx_a = a[col];
            const auto idx_b = b[col];
            const auto a_ = x[idx_a][row];
            const auto b_ = x[idx_b][row];

            const auto w_ = w[col];

            y[col][row] = (
                  ((w_[1] * (a_ * b_)
                  + w_[2] * (a_ - a_ * b_))
                 + (w_[3] * a_
                  + w_[4] * (b_ - a_ * b_)))
                + ((w_[5] * b_
                  + w_[6] * (a_ + b_ - static_cast<scalar_t>(2) * a_ * b_))
                 + (w_[7] * (a_ + b_ - a_ * b_)
                  + w_[8] * (static_cast<scalar_t>(1) - (a_ + b_ - a_ * b_)))))
              + (((w_[9] * (static_cast<scalar_t>(1) - (a_ + b_ - static_cast<scalar_t>(2) * a_ * b_))
                 + w_[10] * (static_cast<scalar_t>(1) - b_)) +
                   (w_[11] * (static_cast<scalar_t>(1) - b_ + a_ * b_)
                  + w_[12] * (static_cast<scalar_t>(1) - a_))) +
                   (w_[13] * (static_cast<scalar_t>(1) - a_ + a_ * b_)
                  + w_[14] * (static_cast<scalar_t>(1) - a_ * b_)
                  + w_[15])
            );
    }}
}


template <typename scalar_t>
__global__ void
logic_layer_cuda_backward_w_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 3, at::RestrictPtrTraits> grad_w_
) {

    const auto row_ = blockIdx.x * blockDim.x + threadIdx.x;

    for (  // neuron dim
        auto col = blockIdx.y * blockDim.y + threadIdx.y;
        col < grad_y.size(0);
        col += blockDim.y * gridDim.y
    ) {
        const auto idx_a = a[col];
        const auto idx_b = b[col];
        scalar_t grad_w_local_1 = 0;
        scalar_t grad_w_local_3 = 0;
        scalar_t grad_w_local_5 = 0;
        scalar_t grad_w_local_15 = 0;
        for (int row = row_; row < grad_y.size(1); row += BACKWARD_W_BATCH_THREADS) {  // batch dim
            const auto a_ = x[idx_a][row];
            const auto b_ = x[idx_b][row];
            const auto grad_y_ = grad_y[col][row];

            // compute grad_w
            grad_w_local_1 += (a_ * b_) * grad_y_;
            grad_w_local_3 += a_ * grad_y_;
            grad_w_local_5 += b_ * grad_y_;
            grad_w_local_15 += grad_y_;
        }

        grad_w_[col][row_][0] = grad_w_local_1;
        grad_w_[col][row_][1] = grad_w_local_3;
        grad_w_[col][row_][2] = grad_w_local_5;
        grad_w_[col][row_][3] = grad_w_local_15;
    }
}


template <typename scalar_t>
__global__ void
logic_layer_cuda_backward_x_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> grad_x,
    torch::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> given_x_indices_of_y_start,
    torch::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> given_x_indices_of_y
) {

    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < grad_x.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < grad_x.size(0);
            col += blockDim.y * gridDim.y
        ) {

            scalar_t grad_x_ = 0;

            const auto start = given_x_indices_of_y_start[col];
            const auto end = given_x_indices_of_y_start[col + 1];

            for (int cur = start; cur < end; ++cur) {
                const auto idx_y = given_x_indices_of_y[cur];
                const auto idx_a = a[idx_y];
                const auto idx_b = b[idx_y];
                const auto grad_y_ = grad_y[idx_y][row];
                const auto idx_is_a = idx_a == col;

                // compute grad_x
                if (idx_is_a) {
                    const auto b_ = x[idx_b][row];
                    const auto dy_dx = (
                         (w[idx_y][1] * b_
                        + w[idx_y][2] * (static_cast<scalar_t>(1) - b_)
                        + w[idx_y][3]) +
                         (w[idx_y][4] * -b_
                        + w[idx_y][6] * (static_cast<scalar_t>(1) - static_cast<scalar_t>(2) * b_)
                        + w[idx_y][7] * (static_cast<scalar_t>(1) - b_)))
                       + ((w[idx_y][8] * (b_ - static_cast<scalar_t>(1))
                        + w[idx_y][9] * (static_cast<scalar_t>(2) * b_ - static_cast<scalar_t>(1))
                        + w[idx_y][11] * b_)
                        + (-w[idx_y][12]
                        + w[idx_y][13] * (b_ - static_cast<scalar_t>(1))
                        + w[idx_y][14] * -b_)
                      );
                    grad_x_ += dy_dx * grad_y_;
                } else {
                    const auto a_ = x[idx_a][row];
                    const auto dy_dx = (
                         (w[idx_y][1] * a_
                        + w[idx_y][2] * -a_
                        + w[idx_y][4] * (static_cast<scalar_t>(1) - a_))
                       + (w[idx_y][5]
                        + w[idx_y][6] * (static_cast<scalar_t>(1) - static_cast<scalar_t>(2) * a_)
                        + w[idx_y][7] * (static_cast<scalar_t>(1) - a_)))
                     + ((w[idx_y][8] * (a_ - static_cast<scalar_t>(1))
                        + w[idx_y][9] * (static_cast<scalar_t>(2) * a_ - static_cast<scalar_t>(1))
                        - w[idx_y][10])
                       + (w[idx_y][11] * (a_ - static_cast<scalar_t>(1))
                        + w[idx_y][13] * a_
                        + w[idx_y][14] * -a_)
                      );
                    grad_x_ += dy_dx * grad_y_;
                }
            }
            grad_x[col][row] = grad_x_;
    }}
}


torch::Tensor logic_layer_cuda_forward(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);

    const auto batch_size = x.size(1);
    const auto in_size = x.size(0);
    const auto out_size = w.size(0);

    auto y = torch::empty({out_size, batch_size}, torch::dtype(x.dtype()).device(x.device()));

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "logic_layer_cuda_forward", ([&] {
                                              logic_layer_cuda_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                                                  x.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(),
                                                  a.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                                                  b.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                                                  w.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(),
                                                  y.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>()
                                              );
                                          }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return y;
}


torch::Tensor logic_layer_cuda_backward_w(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_y
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(grad_y);


    const auto batch_size = x.size(1);
    const auto in_size = x.size(0);
    const auto out_size = grad_y.size(0);

    auto grad_w_4 = torch::empty({out_size, BACKWARD_W_BATCH_THREADS, 4}, torch::dtype(x.dtype()).device(x.device()));

    dim3 threads_per_block(BACKWARD_W_BATCH_THREADS, 1024 / BACKWARD_W_BATCH_THREADS);

    const dim3 blocks_per_grid(
        1,
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "logic_layer_cuda_backward_w", ([&] {
                                              logic_layer_cuda_backward_w_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                                                  x.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(),
                                                  a.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                                                  b.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                                                  grad_y.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(),
                                                  grad_w_4.packed_accessor64<scalar_t, 3, at::RestrictPtrTraits>());
                                          }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    const auto grad_w_components = grad_w_4.sum(1);
    const auto grad_w_ab = grad_w_components.index({torch::indexing::Slice(), 0});
    const auto grad_w_a = grad_w_components.index({torch::indexing::Slice(), 1});
    const auto grad_w_b = grad_w_components.index({torch::indexing::Slice(), 2});
    const auto grad_w_ = grad_w_components.index({torch::indexing::Slice(), 3});

    const auto grad_w = torch::stack({
        torch::zeros({out_size}, torch::dtype(x.dtype()).device(x.device())),
        grad_w_ab,
        grad_w_a - grad_w_ab,
        grad_w_a,
        grad_w_b - grad_w_ab,
        grad_w_b,
        grad_w_a + grad_w_b - grad_w_ab - grad_w_ab,
        grad_w_a + grad_w_b - grad_w_ab,
        grad_w_ - grad_w_a - grad_w_b + grad_w_ab,
        grad_w_ - grad_w_a - grad_w_b + grad_w_ab + grad_w_ab,
        grad_w_ - grad_w_b,
        grad_w_ - grad_w_b + grad_w_ab,
        grad_w_ - grad_w_a,
        grad_w_ - grad_w_a + grad_w_ab,
        grad_w_ - grad_w_ab,
        grad_w_,
    }, 1);


    return grad_w;
}


torch::Tensor logic_layer_cuda_backward_x(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor grad_y,
    torch::Tensor given_x_indices_of_y_start,
    torch::Tensor given_x_indices_of_y
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);
    CHECK_INPUT(grad_y);
    CHECK_INPUT(given_x_indices_of_y_start);
    CHECK_INPUT(given_x_indices_of_y);

    auto grad_x = torch::empty_like(x);

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(x.size(1), static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(x.size(0), static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "logic_layer_cuda_backward_x", ([&] {
                                              logic_layer_cuda_backward_x_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                                                  x.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(),
                                                  a.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                                                  b.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                                                  w.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(),
                                                  grad_y.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(),
                                                  grad_x.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(),
                                                  given_x_indices_of_y_start.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                                                  given_x_indices_of_y.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>()
                                              );
                                          }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return grad_x;
}


/**********************************************************************************************************************/
/** INFERENCE MODE  ***************************************************************************************************/
/**********************************************************************************************************************/


// | id | Operator           | AB=00 | AB=01 | AB=10 | AB=11 |
// |----|----------------------|-------|-------|-------|-------|
// | 0  | 0                    | 0     | 0     | 0     | 0     |
// | 1  | A and B              | 0     | 0     | 0     | 1     |
// | 2  | not(A implies B)     | 0     | 0     | 1     | 0     |
// | 3  | A                    | 0     | 0     | 1     | 1     |
// | 4  | not(B implies A)     | 0     | 1     | 0     | 0     |
// | 5  | B                    | 0     | 1     | 0     | 1     |
// | 6  | A xor B              | 0     | 1     | 1     | 0     |
// | 7  | A or B               | 0     | 1     | 1     | 1     |
// | 8  | not(A or B)          | 1     | 0     | 0     | 0     |
// | 9  | not(A xor B)         | 1     | 0     | 0     | 1     |
// | 10 | not(B)               | 1     | 0     | 1     | 0     |
// | 11 | B implies A          | 1     | 0     | 1     | 1     |
// | 12 | not(A)               | 1     | 1     | 0     | 0     |
// | 13 | A implies B          | 1     | 1     | 0     | 1     |
// | 14 | not(A and B)         | 1     | 1     | 1     | 0     |
// | 15 | 1                    | 1     | 1     | 1     | 1     |

template <typename T> __device__ __forceinline__ T bin_op_eval(const T a_, const T b_, const int op_idx) {
    switch (op_idx) {
    case 0:
        return static_cast<T>(0);
    case 1:
        return a_ & b_;
    case 2:
        return a_ & ~b_;
    case 3:
        return a_;
    case 4:
        return b_ & ~a_;
    case 5:
        return b_;
    case 6:
        return a_ ^ b_;
    case 7:
        return a_ | b_;
    case 8:
        return ~(a_ | b_);
    case 9:
        return ~(a_ ^ b_);
    case 10:
        return ~b_;
    case 11:
        return ~b_ | a_;
    case 12:
        return ~a_;
    case 13:
        return ~a_ | b_;
    case 14:
        return ~(a_ & b_);
    default:
        return ~static_cast<T>(0);
    }
}

template <typename scalar_t>
__global__ void logic_layer_cuda_eval_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> a,
    torch::PackedTensorAccessor64<int64_t, 1, at::RestrictPtrTraits> b,
    torch::PackedTensorAccessor64<uint8_t, 1, at::RestrictPtrTraits> w,
    torch::PackedTensorAccessor64<scalar_t, 2, at::RestrictPtrTraits> y
) {
    for (  // batch dim
        auto row = blockIdx.x * blockDim.x + threadIdx.x;
        row < y.size(1);
        row += blockDim.x * gridDim.x
    ) {
        for (  // neuron dim
            auto col = blockIdx.y * blockDim.y + threadIdx.y;
            col < y.size(0);
            col += blockDim.y * gridDim.y
        ) {

            const auto idx_a = a[col];
            const auto idx_b = b[col];
            const auto a_ = x[idx_a][row];
            const auto b_ = x[idx_b][row];
            const auto w_ = w[col];
            y[col][row] = bin_op_eval(a_, b_, w_);
        }
    }
}

torch::Tensor logic_layer_cuda_eval(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);

    const auto batch_size = x.size(1);
    const auto in_size = x.size(0);
    const auto out_size = w.size(0);

    auto y = torch::zeros({out_size, batch_size}, torch::dtype(x.dtype()).device(x.device()));

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(x.size(1), static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(x.size(0), static_cast<int64_t>(threads_per_block.y)))
    );

    AT_DISPATCH_INTEGRAL_TYPES(x.type(), "logic_layer_cuda_eval_kernel", ([&] {
                                          logic_layer_cuda_eval_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                                              x.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>(),
                                              a.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                                              b.packed_accessor64<int64_t, 1, at::RestrictPtrTraits>(),
                                              w.packed_accessor64<uint8_t, 1, at::RestrictPtrTraits>(),
                                              y.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>()
                                          );
                                      }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return y;
}


/**********************************************************************************************************************/


template <typename scalar_t>
__global__ void tensor_packbits_cuda_kernel(
    torch::PackedTensorAccessor32<bool, 2, at::RestrictPtrTraits> t,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> b
) {

    for (  // neuron in b and t
        auto row = blockIdx.y * blockDim.y + threadIdx.y;
        row < t.size(0);
        row += blockDim.y * gridDim.y
    ) {
        for (  // batch in b
            auto col = blockIdx.x * blockDim.x + threadIdx.x;
            col < b.size(1);
            col += blockDim.x * gridDim.x
        ) {

            typedef typename std::make_unsigned<scalar_t>::type unsigned_scalar_t;
            union {
                unsigned_scalar_t unsigned_scalar;
                scalar_t signed_scalar;
            } val;
            constexpr int bit_count = std::numeric_limits<unsigned_scalar_t>::digits;
            val.signed_scalar = b[row][col];
            for (unsigned int i = 0; i < bit_count; ++i) {
                const auto t_col = bit_count * col + i;
                if (t_col < t.size(1)) {    
                    const unsigned_scalar_t bit_mask = static_cast<unsigned_scalar_t>(t[row][t_col]) << i;
                    val.unsigned_scalar = val.unsigned_scalar | bit_mask;
                }
            }
            b[row][col] = val.signed_scalar;
        }
    }
}

std::tuple<torch::Tensor, int> tensor_packbits_cuda(
    torch::Tensor t,
    const int bit_count
) {
    CHECK_INPUT(t);

    const auto batch_in_size = t.size(1);
    const auto batch_out_size = ceil_div(batch_in_size, static_cast<int64_t>(bit_count));
    const auto out_size = t.size(0);
    const auto pad_len = (bit_count - batch_in_size % bit_count) % bit_count;

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_out_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    auto dispatch_type = [bit_count]() {
        switch (bit_count) {
        case 8:
            return torch::kInt8;
        case 16:
            return torch::kInt16;
        case 32:
            return torch::kInt32;
        case 64:
            return torch::kInt64;
        default:
            throw std::invalid_argument("`bit_count` has to be in { 8, 16, 32, 64 }");
        }
    }();
    auto b = torch::zeros({out_size, batch_out_size}, torch::dtype(dispatch_type).device(t.device()));

    AT_DISPATCH_INTEGRAL_TYPES(b.type(), "tensor_packbits_cuda_kernel", ([&] {
                                          tensor_packbits_cuda_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(t.packed_accessor32<bool, 2, at::RestrictPtrTraits>(),
                                                                                                                        b.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>());
                                      }));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return {b, pad_len};
}


/**********************************************************************************************************************/


template <typename scalar_t>
__global__ void groupbitsum_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> t
) {

    for (  // class in t
        auto row = blockIdx.y * blockDim.y + threadIdx.y;
        row < t.size(0);
        row += blockDim.y * gridDim.y
    ) {
        for (  // batch in t
            auto col = blockIdx.x * blockDim.x + threadIdx.x;
            col < t.size(1);
            col += blockDim.x * gridDim.x
        ) {

            typedef typename std::make_unsigned<scalar_t>::type unsigned_scalar_t;
            union scalar_t_ {
                unsigned_scalar_t unsigned_scalar;
                scalar_t signed_scalar;
            };
            constexpr int bit_count = std::numeric_limits<unsigned_scalar_t>::digits;
            int res = 0;
            const auto class_size = b.size(0) / t.size(0);
            for (int i = 0; i < class_size; ++i) {
                const scalar_t_ val = {.signed_scalar = b[row * class_size + i][col / bit_count]};
                const unsigned_scalar_t bit_mask = static_cast<unsigned_scalar_t>(1) << static_cast<uint32_t>(col % bit_count);
                res += !!(val.unsigned_scalar & bit_mask);
            }
            t[row][col] = res;
        }
    }
}

torch::Tensor groupbitsum(
    torch::Tensor b,
    const int pad_len,
    const int k
) {
    CHECK_INPUT(b);

    const int bit_count = 8 * b.element_size();

    const auto batch_in_size = b.size(1);
    const auto in_size = b.size(0);
    const auto batch_out_size = batch_in_size * bit_count - pad_len;
    const auto out_size = static_cast<int64_t>(k);
    assert(in_size % k == 0);

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_out_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    auto t = torch::zeros({out_size, batch_out_size}, torch::dtype(torch::kInt32).device(b.device()));

    AT_DISPATCH_INTEGRAL_TYPES(b.type(), "groupbitsum_kernel", ([&] {
                                          groupbitsum_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                                              b.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                                              t.packed_accessor32<int, 2, at::RestrictPtrTraits>()
                                              );
                                      }));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return t.transpose(0, 1).contiguous();
}


/**********************************************************************************************************************/
// ▼▼▼ weighted_groupbitsum CUDA kernel and C++ function ▼▼▼
/**********************************************************************************************************************/

template <typename scalar_t_in, typename scalar_t_out>
__global__ void weighted_groupbitsum_kernel(
    torch::PackedTensorAccessor32<scalar_t_in, 2, at::RestrictPtrTraits> b,      // Packed bits
    torch::PackedTensorAccessor32<scalar_t_out, 2, at::RestrictPtrTraits> weights, // Weights
    torch::PackedTensorAccessor32<scalar_t_out, 2, at::RestrictPtrTraits> t        // Output tensor
) {
    // Each thread calculates one element of the output tensor t (one class, one batch sample).
    for (  // class in t
        auto row = blockIdx.y * blockDim.y + threadIdx.y;
        row < t.size(0);
        row += blockDim.y * gridDim.y
    ) {
        for (  // batch in t
            auto col = blockIdx.x * blockDim.x + threadIdx.x;
            col < t.size(1);
            col += blockDim.x * gridDim.x
        ) {

            // Define the unsigned integer type corresponding to the packed input type
            typedef typename std::make_unsigned<scalar_t_in>::type unsigned_scalar_t;
            union scalar_t_in_union {
                unsigned_scalar_t unsigned_scalar;
                scalar_t_in signed_scalar;
            };
            constexpr int bit_count = std::numeric_limits<unsigned_scalar_t>::digits;
            
            scalar_t_out res = 0; // The result is a weighted sum, so its type is scalar_t_out
            const auto class_size = b.size(0) / t.size(0); // Number of features (neurons) per group

            // Iterate over all features (i) belonging to the current class (row).
            for (int i = 0; i < class_size; ++i) {
                // Get the value of the current feature from the PackBitsTensor.
                const scalar_t_in_union val = {.signed_scalar = b[row * class_size + i][col / bit_count]};
                
                // Mask to check if the corresponding bit is 1.
                const unsigned_scalar_t bit_mask = static_cast<unsigned_scalar_t>(1) << static_cast<uint32_t>(col % bit_count);

                // If the bit is 1, add the corresponding weight to the result.
                if (val.unsigned_scalar & bit_mask) {
                    res += weights[row][i];
                }
            }
            t[row][col] = res; // Store the weighted sum
        }
    }
}

torch::Tensor weighted_groupbitsum(
    torch::Tensor b,
    const int pad_len,
    const int k,
    torch::Tensor weights
) {
    CHECK_INPUT(b);
    CHECK_INPUT(weights);

    const int bit_count = 8 * b.element_size();

    const auto batch_in_size = b.size(1);
    const auto in_size = b.size(0);
    const auto batch_out_size = batch_in_size * bit_count - pad_len;
    const auto out_size = static_cast<int64_t>(k);
    assert(in_size % k == 0);

    dim3 threads_per_block(32, 32);

    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_out_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    // The output tensor t must be of the same type as the weights (float, half, etc.).
    auto t = torch::zeros({out_size, batch_out_size}, torch::dtype(weights.dtype()).device(b.device()));

    // Dispatch the kernel based on the types of the input PackBitsTensor (b), the output weighted sum (t), and the weights.
    AT_DISPATCH_INTEGRAL_TYPES(b.type(), "weighted_groupbitsum_integral", ([&] {
        // Store the integral type of b as integral_scalar_t
        using integral_scalar_t = scalar_t; 
        
        // Dispatch on the float type of t and weights (this macro internally uses the name 'scalar_t')
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(t.type(), "weighted_groupbitsum_float", ([&] {
            // Clearly distinguish and pass the types of b and t/weights when calling the kernel
            weighted_groupbitsum_kernel<integral_scalar_t, scalar_t><<<blocks_per_grid, threads_per_block>>>(
                b.packed_accessor32<integral_scalar_t, 2, at::RestrictPtrTraits>(),
                weights.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                t.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>()
            );
        }));
    }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return t.transpose(0, 1).contiguous();
}



/**********************************************************************************************************************/
// ▼▼▼ pruned_groupbitsum CUDA kernel and C++ function ▼▼▼
/**********************************************************************************************************************/
template <typename scalar_t>
__global__ void pruned_groupbitsum_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> t,
    torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> group_sizes,
    torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> group_offsets
) {
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= t.size(0)) return;

    // Get the size and starting position of the current group (row).
    const int current_group_size = group_sizes[row];
    const int current_group_offset = group_offsets[row];

    for (
        auto col = blockIdx.x * blockDim.x + threadIdx.x;
        col < t.size(1);
        col += blockDim.x * gridDim.x
    ) {
        typedef typename std::make_unsigned<scalar_t>::type unsigned_scalar_t;
        union scalar_t_ {
            unsigned_scalar_t unsigned_scalar;
            scalar_t signed_scalar;
        };
        constexpr int bit_count = std::numeric_limits<unsigned_scalar_t>::digits;
        int res = 0;

        // Use the size and offset of the current group instead of a fixed class_size.
        for (int i = 0; i < current_group_size; ++i) {
            const scalar_t_ val = {.signed_scalar = b[current_group_offset + i][col / bit_count]};
            const unsigned_scalar_t bit_mask = static_cast<unsigned_scalar_t>(1) << static_cast<uint32_t>(col % bit_count);
            res += !!(val.unsigned_scalar & bit_mask);
        }
        t[row][col] = res;
    }
}

torch::Tensor pruned_groupbitsum(
    torch::Tensor b,
    const int pad_len,
    const int k,
    torch::Tensor group_sizes // Receive a Tensor instead of const int class_size
) {
    CHECK_INPUT(b);
    TORCH_CHECK(group_sizes.dim() == 1 && group_sizes.size(0) == k, "group_sizes must be a 1D tensor of size k");
    TORCH_CHECK(group_sizes.scalar_type() == torch::kInt, "group_sizes must be an Int tensor");
    group_sizes = group_sizes.to(b.device()); // Move the tensor to the same device.

    // Calculate the group_offsets tensor (cumulative sum).
    // Example: group_sizes = [5, 3, 4] -> group_offsets = [0, 5, 8]
    auto group_offsets = torch::zeros_like(group_sizes);
    if (k > 1) {
        group_offsets.slice(0, 1, k) = torch::cumsum(group_sizes.slice(0, 0, k - 1), 0);
    }

    // --- The rest of the logic is similar to before ---
    const int bit_count = 8 * b.element_size();
    const auto batch_in_size = b.size(1);
    const auto batch_out_size = batch_in_size * bit_count - pad_len;
    const auto out_size = static_cast<int64_t>(k);

    dim3 threads_per_block(32, 32);
    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_out_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    auto t = torch::zeros({out_size, batch_out_size}, torch::dtype(torch::kInt32).device(b.device()));

    AT_DISPATCH_INTEGRAL_TYPES(b.type(), "pruned_groupbitsum_kernel", ([&] {
        pruned_groupbitsum_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            b.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            t.packed_accessor32<int, 2, at::RestrictPtrTraits>(),
            group_sizes.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
            group_offsets.packed_accessor32<int, 1, at::RestrictPtrTraits>()
        );
    }));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return t.transpose(0, 1).contiguous();
}

/**********************************************************************************************************************/
// ▼▼▼ pruned_weighted_groupbitsum CUDA kernel and C++ function ▼▼▼
/**********************************************************************************************************************/
template <typename scalar_t_in, typename scalar_t_out>
__global__ void pruned_weighted_groupbitsum_kernel(
    torch::PackedTensorAccessor32<scalar_t_in, 2, at::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t_out, 1, at::RestrictPtrTraits> weights, // Weights (1D)
    torch::PackedTensorAccessor32<scalar_t_out, 2, at::RestrictPtrTraits> t,
    torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> group_sizes,
    torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> group_offsets
) {
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= t.size(0)) return;

    // Get the size and starting offset of the current group (row).
    const int current_group_size = group_sizes[row];
    const int current_group_offset = group_offsets[row];

    for (
        auto col = blockIdx.x * blockDim.x + threadIdx.x;
        col < t.size(1);
        col += blockDim.x * gridDim.x
    ) {
        typedef typename std::make_unsigned<scalar_t_in>::type unsigned_scalar_t;
        union scalar_t_in_union {
            unsigned_scalar_t unsigned_scalar;
            scalar_t_in signed_scalar;
        };
        constexpr int bit_count = std::numeric_limits<unsigned_scalar_t>::digits;
        
        scalar_t_out res = 0; // The result is a weighted sum.

        // Iterate over all features (i) in the current group.
        for (int i = 0; i < current_group_size; ++i) {
            const int feature_index = current_group_offset + i;
            const scalar_t_in_union val = {.signed_scalar = b[feature_index][col / bit_count]};
            const unsigned_scalar_t bit_mask = static_cast<unsigned_scalar_t>(1) << static_cast<uint32_t>(col % bit_count);

            // If the bit is 1, add the corresponding weight.
            // The weight is accessed from the 1D tensor using the absolute feature index.
            if (val.unsigned_scalar & bit_mask) {
                res += weights[feature_index];
            }
        }
        t[row][col] = res;
    }
}


torch::Tensor pruned_weighted_groupbitsum(
    torch::Tensor b,
    const int pad_len,
    const int k,
    torch::Tensor group_sizes,
    torch::Tensor weights
) {
    CHECK_INPUT(b);
    CHECK_INPUT(weights);
    TORCH_CHECK(group_sizes.dim() == 1 && group_sizes.size(0) == k, "group_sizes must be a 1D tensor of size k");
    TORCH_CHECK(group_sizes.scalar_type() == torch::kInt, "group_sizes must be an Int tensor");
    group_sizes = group_sizes.to(b.device());

    auto group_offsets = torch::zeros_like(group_sizes);
    if (k > 1) {
        group_offsets.slice(0, 1, k) = torch::cumsum(group_sizes.slice(0, 0, k - 1), 0);
    }

    const int bit_count = 8 * b.element_size();
    const auto batch_in_size = b.size(1);
    const auto batch_out_size = batch_in_size * bit_count - pad_len;
    const auto out_size = static_cast<int64_t>(k);

    dim3 threads_per_block(32, 32);
    const dim3 blocks_per_grid(
        min(static_cast<int64_t>(65535), ceil_div(batch_out_size, static_cast<int64_t>(threads_per_block.x))),
        min(static_cast<int64_t>(65535), ceil_div(out_size, static_cast<int64_t>(threads_per_block.y)))
    );

    auto t = torch::zeros({out_size, batch_out_size}, torch::dtype(weights.dtype()).device(b.device()));

    AT_DISPATCH_INTEGRAL_TYPES(b.type(), "pruned_weighted_groupbitsum_integral", ([&] {
        using integral_scalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(t.type(), "pruned_weighted_groupbitsum_float", ([&] {
            pruned_weighted_groupbitsum_kernel<integral_scalar_t, scalar_t><<<blocks_per_grid, threads_per_block>>>(
                b.packed_accessor32<integral_scalar_t, 2, at::RestrictPtrTraits>(),
                weights.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                t.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                group_sizes.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
                group_offsets.packed_accessor32<int, 1, at::RestrictPtrTraits>()
            );
        }));
    }));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return t.transpose(0, 1).contiguous();
}













// === ADD: 게이트 16개 조합의 전방/미분 헬퍼 (float/half 공통) ==========================
template <typename T>
__device__ __forceinline__ T gate_forward_16(const T* __restrict__ w, T a, T b) {
    const T ab = a * b;
    // 기존 logic_layer_cuda_forward_kernel 수식과 동일 (인덱스 1..15 사용)
    return  (w[1]*ab + w[2]*(a - ab) + w[3]*a + w[4]*(b - ab))
          + (w[5]*b  + w[6]*(a + b - T(2)*ab) + w[7]*(a + b - ab) + w[8]*(T(1) - (a + b - ab)))
          + (w[9]*(T(1) - (a + b - T(2)*ab)) + w[10]*(T(1) - b) + w[11]*(T(1) - b + ab) + w[12]*(T(1) - a))
          + (w[13]*(T(1) - a + ab) + w[14]*(T(1) - ab) + w[15]);
}

template <typename T>
__device__ __forceinline__ void gate_backward_partials_16(
    const T* __restrict__ w, T a, T b, T& dza, T& dzb
){
    // 기존 logic_layer_cuda_backward_x_kernel의 dy/dx 식을 그대로 사용
    // (정확히 같은 다항식, 변수명만 맞춤)
    dza =
      ( w[1]*b + w[2]*(T(1)-b) + w[3]
      + w[4]*(-b) + w[6]*(T(1)-T(2)*b) + w[7]*(T(1)-b) )
    + ( w[8]*(b - T(1)) + w[9]*(T(2)*b - T(1)) + w[11]*b
      - w[12] + w[13]*(b - T(1)) + w[14]*(-b) );

    dzb =
      ( w[1]*a + w[2]*(-a) + w[4]*(T(1)-a)
      + w[5] + w[6]*(T(1)-T(2)*a) + w[7]*(T(1)-a) )
    + ( w[8]*(a - T(1)) + w[9]*(T(2)*a - T(1)) - w[10]
      + w[11]*(a - T(1)) + w[13]*a + w[14]*(-a) );
}



// Forward kernel declaration (per-level version)
template <typename scalar_t>
__global__ void fused_tree_orpool_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4> xp,      // [B, C, Hp, Wp]
    torch::PackedTensorAccessor32<scalar_t,4> wL,      // [L, Co, maxN, 16]
    torch::PackedTensorAccessor32<int64_t,3> aL,       // [L, Co, maxN]
    torch::PackedTensorAccessor32<int64_t,3> bL,       // [L, Co, maxN]
    torch::PackedTensorAccessor32<int64_t,2> leaf_ici, // [Co, N0]
    torch::PackedTensorAccessor32<int64_t,2> leaf_ipx, // [Co, N0]
    torch::PackedTensorAccessor32<int64_t,2> leaf_ipy, // [Co, N0]
    torch::PackedTensorAccessor32<int,1> nodesL,        // [L]
    torch::PackedTensorAccessor32<scalar_t,4> y,       // [B, Co, Ho2, Wo2]
    int kernel_size, int stride, int groups, int icpg,
    int Ho2, int Wo2, int L);

// Backward kernel declaration (per-level version)
template <typename scalar_t>
__global__ void fused_tree_orpool_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4> xp,      // [B, C, Hp, Wp]
    torch::PackedTensorAccessor32<scalar_t,4> gy,      // [B, Co, Ho2, Wo2]
    torch::PackedTensorAccessor32<scalar_t,4> wL,       // [L, Co, maxN, 16]
    torch::PackedTensorAccessor32<int64_t,3> aL,        // [L, Co, maxN]
    torch::PackedTensorAccessor32<int64_t,3> bL,        // [L, Co, maxN]
    torch::PackedTensorAccessor32<int64_t,2> leaf_ici,  // [Co, N0]
    torch::PackedTensorAccessor32<int64_t,2> leaf_ipx, // [Co, N0]
    torch::PackedTensorAccessor32<int64_t,2> leaf_ipy, // [Co, N0]
    torch::PackedTensorAccessor32<int,1> nodesL,        // [L]
    torch::PackedTensorAccessor32<scalar_t,4> gx,      // [B, C, Hp, Wp]
    torch::PackedTensorAccessor32<scalar_t,4> gwL,      // [L, Co, maxN, 16]
    int kernel_size, int stride, int groups, int icpg,
    int Ho2, int Wo2, int L);

// Host wrapper for forward pass
torch::Tensor fused_forward_ablevels_cuda(
    const torch::Tensor& x_padded,
    const torch::Tensor& weights_L,
    const torch::Tensor& a_idx_L,
    const torch::Tensor& b_idx_L,
    const torch::Tensor& leaf_ici,
    const torch::Tensor& leaf_ipx,
    const torch::Tensor& leaf_ipy,
    const torch::Tensor& nodes_per_level,
    int out_h, int out_w, int kernel_size, int stride, int groups, int in_channels_per_group)
{
    auto y = torch::empty({x_padded.size(0), weights_L.size(1), out_h, out_w}, x_padded.options());
    dim3 block(32);
    dim3 grid((out_w + block.x - 1)/block.x, out_h, x_padded.size(0)*weights_L.size(1));

    const int N0 = leaf_ici.size(1);
    size_t shmem = static_cast<size_t>(2 * N0 * block.x) * x_padded.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_padded.scalar_type(), "fused_forward_ablevels", [&]{
        fused_tree_orpool_forward_kernel<scalar_t><<<grid, block, shmem>>>(
            x_padded.packed_accessor32<scalar_t,4>(),
            weights_L.packed_accessor32<scalar_t,4>(),
            a_idx_L.packed_accessor32<int64_t,3>(),
            b_idx_L.packed_accessor32<int64_t,3>(),
            leaf_ici.packed_accessor32<int64_t,2>(),
            leaf_ipx.packed_accessor32<int64_t,2>(),
            leaf_ipy.packed_accessor32<int64_t,2>(),
            nodes_per_level.packed_accessor32<int,1>(),
            y.packed_accessor32<scalar_t,4>(),
            kernel_size, stride, groups, in_channels_per_group,
            out_h, out_w, weights_L.size(0)
        );
    });
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fused_forward_ablevels launch failed: ", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "fused_forward_ablevels sync failed: ", cudaGetErrorString(err));
    return y;
}

std::vector<torch::Tensor> fused_backward_ablevels_cuda(
    const torch::Tensor& x_padded, const torch::Tensor& grad_out,
    const torch::Tensor& weights_L, const torch::Tensor& a_idx_L, const torch::Tensor& b_idx_L,
    const torch::Tensor& leaf_ici, const torch::Tensor& leaf_ipx, const torch::Tensor& leaf_ipy,
    const torch::Tensor& nodes_per_level,
    int out_h, int out_w, int kernel_size, int stride, int groups, int in_channels_per_group)
{
    auto gx  = torch::zeros_like(x_padded);
    auto gwL = torch::zeros_like(weights_L);

    dim3 block(32);
    dim3 grid((out_w + block.x - 1)/block.x, out_h, x_padded.size(0)*weights_L.size(1));
    const int N0 = leaf_ici.size(1);
    size_t shmem = static_cast<size_t>(2 * N0 * block.x) * x_padded.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_padded.scalar_type(), "fused_backward_ablevels", [&]{
        fused_tree_orpool_backward_kernel<scalar_t><<<grid, block, shmem>>>(
            x_padded.packed_accessor32<scalar_t,4>(),
            grad_out.packed_accessor32<scalar_t,4>(),
            weights_L.packed_accessor32<scalar_t,4>(),
            a_idx_L.packed_accessor32<int64_t,3>(),
            b_idx_L.packed_accessor32<int64_t,3>(),
            leaf_ici.packed_accessor32<int64_t,2>(),
            leaf_ipx.packed_accessor32<int64_t,2>(),
            leaf_ipy.packed_accessor32<int64_t,2>(),
            nodes_per_level.packed_accessor32<int,1>(),
            gx.packed_accessor32<scalar_t,4>(),
            gwL.packed_accessor32<scalar_t,4>(),
            kernel_size, stride, groups, in_channels_per_group,
            out_h, out_w, weights_L.size(0)
        );
    });
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fused_backward_ablevels launch failed: ", cudaGetErrorString(err));
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "fused_backward_ablevels sync failed: ", cudaGetErrorString(err));
    return {gx, gwL};
}



// Forward kernel implementation
template <typename scalar_t>
__global__ void fused_tree_orpool_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4> xp,      // [B, C, Hp, Wp]
    torch::PackedTensorAccessor32<scalar_t, 4> wL,     // [L, Co, maxN, 16]
    torch::PackedTensorAccessor32<int64_t, 3> aL,       // [L, Co, maxN]
    torch::PackedTensorAccessor32<int64_t, 3> bL,       // [L, Co, maxN]
    torch::PackedTensorAccessor32<int64_t, 2> leaf_ici, // [Co, N0]
    torch::PackedTensorAccessor32<int64_t, 2> leaf_ipx, // [Co, N0]
    torch::PackedTensorAccessor32<int64_t, 2> leaf_ipy, // [Co, N0]
    torch::PackedTensorAccessor32<int, 1> nodesL,       // [L]
    torch::PackedTensorAccessor32<scalar_t, 4> y,       // [B, Co, Ho2, Wo2]
    int kernel_size, int stride, int groups, int icpg,
    int Ho2, int Wo2, int L
){
    // Shared memory for ping-pong buffers (per-thread region)
    extern __shared__ char smem_raw[];
    const int ow2 = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh2 = blockIdx.y;
    const int bk  = blockIdx.z;
    const int B   = xp.size(0);
    const int Co  = y.size(1);
    const int b   = bk / Co;
    const int k   = bk % Co;
    if (b >= B || ow2 >= Wo2) return;

    // Conv output coordinates (before pooling)
    const int oh_base = oh2 * 2;
    const int ow_base = ow2 * 2;

    // Group calculation
    const int gidx = (k * groups) / Co;
    const int cstart = gidx * icpg;

    scalar_t best = -std::numeric_limits<float>::infinity();

    // Iterate over 2x2 ORPool window (4 positions), compare root values only
    for (int h_off = 0; h_off < 2; ++h_off){
        for (int w_off = 0; w_off < 2; ++w_off){
            const int oh = oh_base + h_off;
            const int ow = ow_base + w_off;

            // Patch top-left corner
            const int ph0 = oh * stride;
            const int pw0 = ow * stride;

            // --- Level 0 input (leaf) gather ---
            const int N0 = leaf_ici.size(1);
            scalar_t* sh = reinterpret_cast<scalar_t*>(smem_raw);
            scalar_t* curr = sh + threadIdx.x * N0;  // thread-local region
            scalar_t* next = curr;  // Level 0 fills curr (ping-pong starts from next level)

            for (int i=0; i<N0; ++i){
                const int64_t c_rel = leaf_ici[k][i];
                const int64_t py    = leaf_ipy[k][i];
                const int64_t px    = leaf_ipx[k][i];
                const int cabs = cstart + (int)c_rel;
                const int yabs = ph0 + (int)py;
                const int xabs = pw0 + (int)px;
                curr[i] = xp[b][cabs][yabs][xabs];
            }

            // --- Level traversal (0..L-1), compute nodesL[ℓ] nodes ---
            int prev_count = N0;
            for (int li=0; li<L; ++li){
                const int nL = nodesL[li];
                scalar_t* outbuf = (li % 2 == 0) ? (curr + N0) : curr;  // simple ping-pong
                scalar_t* inbuf  = (li % 2 == 0) ? curr         : (curr + N0);

                for (int n=0; n<nL; ++n){
                    const int64_t ai = aL[li][k][n];  // prev-level space index
                    const int64_t bi = bL[li][k][n];
                    const scalar_t a = inbuf[ai];
                    const scalar_t b = inbuf[bi];
                    const scalar_t* w = &wL[li][k][n][0];
                    outbuf[n] = gate_forward_16(w, a, b);
                }
                prev_count = nL;
            }

            // Root of the last level is outbuf[0]
            scalar_t root = ((L % 2 == 0) ? curr[0] : (curr + N0)[0]);
            if (root > best) best = root;
        }
    }

    y[b][k][oh2][ow2] = best;
}




// Backward kernel implementation
// Strategy: Recompute ORPool to find winner, then backpropagate only through the selected path
template <typename scalar_t>
__global__ void fused_tree_orpool_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4> xp,      // [B, C, Hp, Wp]
    torch::PackedTensorAccessor32<scalar_t, 4> gy,      // [B, Co, Ho2, Wo2]
    torch::PackedTensorAccessor32<scalar_t, 4> wL,       // [L, Co, maxN, 16]
    torch::PackedTensorAccessor32<int64_t, 3> aL,         // [L, Co, maxN]
    torch::PackedTensorAccessor32<int64_t, 3> bL,         // [L, Co, maxN]
    torch::PackedTensorAccessor32<int64_t, 2> leaf_ici,  // [Co, N0]
    torch::PackedTensorAccessor32<int64_t, 2> leaf_ipx,   // [Co, N0]
    torch::PackedTensorAccessor32<int64_t, 2> leaf_ipy,   // [Co, N0]
    torch::PackedTensorAccessor32<int, 1> nodesL,         // [L]
    torch::PackedTensorAccessor32<scalar_t, 4> gx,        // [B, C, Hp, Wp]
    torch::PackedTensorAccessor32<scalar_t, 4> gwL,       // [L, Co, maxN, 16]
    int kernel_size, int stride, int groups, int icpg,
    int Ho2, int Wo2, int L
){
    extern __shared__ char smem_raw[];
    const int ow2 = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh2 = blockIdx.y;
    const int bk  = blockIdx.z;
    const int B   = xp.size(0);
    const int Co  = gy.size(1);
    const int b   = bk / Co;
    const int k   = bk % Co;
    if (b >= B || ow2 >= Wo2) return;

    const int gidx   = (k * groups) / Co;
    const int cstart = gidx * icpg;

    const int oh_base = oh2 * 2;
    const int ow_base = ow2 * 2;

    // Step 1: Recompute root for 4 positions and select argmax
    scalar_t best = -std::numeric_limits<float>::infinity();
    int best_idx = 0;

    // Shared ping-pong buffer layout
    const int N0 = leaf_ici.size(1);
    scalar_t* sh = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* buf0 = sh + threadIdx.x * (2 * N0);
    scalar_t* buf1 = buf0 + N0;

    for (int idx=0; idx<4; ++idx){
        const int h_off = idx >> 1;
        const int w_off = idx & 1;
        const int oh = oh_base + h_off;
        const int ow = ow_base + w_off;

        const int ph0 = oh * stride;
        const int pw0 = ow * stride;

        // Level 0 gather → buf0
        for (int i=0; i<N0; ++i){
            const int cabs = cstart + (int)leaf_ici[k][i];
            const int yabs = ph0 + (int)leaf_ipy[k][i];
            const int xabs = pw0 + (int)leaf_ipx[k][i];
            buf0[i] = xp[b][cabs][yabs][xabs];
        }

        // Forward recomputation through levels
        int prev_count = N0;
        scalar_t* curr = buf0;
        scalar_t* next = buf1;
        for (int li=0; li<L; ++li){
            const int nL = nodesL[li];
            scalar_t* outbuf = (li % 2 == 0) ? next : curr;
            scalar_t* inbuf  = (li % 2 == 0) ? curr : next;
            for (int n=0; n<nL; ++n){
                const int64_t ai = aL[li][k][n];
                const int64_t bi = bL[li][k][n];
                const scalar_t a = inbuf[ai];
                const scalar_t b = inbuf[bi];
                const scalar_t* w = &wL[li][k][n][0];
                outbuf[n] = gate_forward_16(w, a, b);
            }
            prev_count = nL;
            scalar_t* tmp = curr; curr = outbuf; outbuf = tmp;
        }
        const scalar_t root = curr[0];  // Root is in the last curr
        if (root > best){ best = root; best_idx = idx; }
    }

    const scalar_t upstream = gy[b][k][oh2][ow2];
    if (upstream == scalar_t(0)) return;

    // Step 2: Forward recompute + backward pass for selected position only
    const int h_off = best_idx >> 1;
    const int w_off = best_idx & 1;
    const int oh = oh_base + h_off;
    const int ow = ow_base + w_off;
    const int ph0 = oh * stride;
    const int pw0 = ow * stride;

    // Level 0 gather
    for (int i=0; i<N0; ++i){
        const int cabs = cstart + (int)leaf_ici[k][i];
        const int yabs = ph0 + (int)leaf_ipy[k][i];
        const int xabs = pw0 + (int)leaf_ipx[k][i];
        buf0[i] = xp[b][cabs][yabs][xabs];
    }

    // Forward recomputation through all levels
    // Note: For simplicity, we recompute activations during backward.
    // For optimization, intermediate activations could be cached.
    int prev_count = N0;
    scalar_t* curr = buf0;
    scalar_t* next = buf1;
    for (int li=0; li<L; ++li){
        const int nL = nodesL[li];
        scalar_t* outbuf = (li % 2 == 0) ? next : curr;
        scalar_t* inbuf  = (li % 2 == 0) ? curr : next;
        for (int n=0; n<nL; ++n){
            const int64_t ai = aL[li][k][n];
            const int64_t bi = bL[li][k][n];
            const scalar_t a = inbuf[ai];
            const scalar_t b = inbuf[bi];
            const scalar_t* w = &wL[li][k][n][0];
            outbuf[n] = gate_forward_16(w, a, b);
        }
        prev_count = nL;
        scalar_t* tmp = curr; curr = outbuf; outbuf = tmp;
    }

    // Backward pass: top to bottom (L-1 → 0)
    // Reuse curr/next buffers: curr for activations, next for gradients
    scalar_t* grad_curr = buf1;  // Reuse
    scalar_t* grad_next = buf0;  // Reuse
    for (int i=0; i<N0; ++i) grad_next[i] = scalar_t(0);  // Initialize

    // Set root gradient
    grad_curr[0] = upstream;

    // Level reverse loop
    for (int li=L-1; li>=0; --li){
        const int nL = nodesL[li];
        // Reconstruct inbuf using the same rules as forward
        // Since curr was the last level output in forward,
        // we need to recompute the input (lower level output) for this level in backward
        
        // Recompute inbuf for level li
        // For simplicity, we recompute from level 0 to li-1
        // Optimization: could cache level activations
        {
            // Local evaluation from leaf to li-1
            // Put leaf in buf0
            for (int i=0; i<N0; ++i){
                const int cabs = cstart + (int)leaf_ici[k][i];
                const int yabs = ph0 + (int)leaf_ipy[k][i];
                const int xabs = pw0 + (int)leaf_ipx[k][i];
                buf0[i] = xp[b][cabs][yabs][xabs];
            }
            scalar_t* c = buf0;
            scalar_t* n = buf1;
            for (int j=0; j<li; ++j){
                const int nj = nodesL[j];
                scalar_t* outb = (j % 2 == 0) ? n : c;
                scalar_t* inb  = (j % 2 == 0) ? c : n;
                for (int m=0; m<nj; ++m){
                    const int64_t ai = aL[j][k][m];
                    const int64_t bi = bL[j][k][m];
                    const scalar_t a = inb[ai];
                    const scalar_t b = inb[bi];
                    const scalar_t* w = &wL[j][k][m][0];
                    outb[m] = gate_forward_16(w, a, b);
                }
                scalar_t* tmp2 = c; c = outb; outb = tmp2;
            }
            // c is now the input vector for level li
            // Use it as in_vec
            // grad_curr: gradients of level li outputs (size nL)
            scalar_t* in_vec = c;
            scalar_t* out_grad = grad_curr;  // size nL

            // Initialize grad_next to zero for propagation to lower level
            // Size is the input dimension of li (= previous level node count or leaf count)
            // Safely initialize N0 elements
            for (int z=0; z<N0; ++z) grad_next[z] = scalar_t(0);

            for (int n=0; n<nL; ++n){
                const int64_t ai = aL[li][k][n];
                const int64_t bi = bL[li][k][n];
                const scalar_t a = in_vec[ai];
                const scalar_t b = in_vec[bi];
                const scalar_t* w = &wL[li][k][n][0];
                const scalar_t go = out_grad[n];

                // Compute gradients: ∂L/∂w += basis(a,b) * go
                // Compute partial derivatives w.r.t. a and b
                scalar_t dza, dzb;
                gate_backward_partials_16(w, a, b, dza, dzb);
                // Accumulate gradients to lower level
                gpuAtomicAdd(&grad_next[ai], dza * go);
                gpuAtomicAdd(&grad_next[bi], dzb * go);

                // Weight gradients (16 basis terms)
                // Compute all 16 basis terms from forward formula
                const scalar_t ab  = a*b;
                const scalar_t a_n = a - ab;
                const scalar_t b_n = b - ab;
                const scalar_t a_pb = a + b;
                const scalar_t a_xor_b = a + b - scalar_t(2)*ab;   // (A xor B) 유사 항
                const scalar_t a_or_b  = a + b - ab;
                const scalar_t n_or    = scalar_t(1) - a_or_b;
                const scalar_t n_xor   = scalar_t(1) - a_xor_b;
                // 1..15 항만 누적
                gpuAtomicAdd(&gwL[li][k][n][ 1], ab      * go);
                gpuAtomicAdd(&gwL[li][k][n][ 2], a_n     * go);
                gpuAtomicAdd(&gwL[li][k][n][ 3], a       * go);
                gpuAtomicAdd(&gwL[li][k][n][ 4], b_n     * go);
                gpuAtomicAdd(&gwL[li][k][n][ 5], b       * go);
                gpuAtomicAdd(&gwL[li][k][n][ 6], a_xor_b * go);
                gpuAtomicAdd(&gwL[li][k][n][ 7], a_or_b  * go);
                gpuAtomicAdd(&gwL[li][k][n][ 8], n_or    * go);
                gpuAtomicAdd(&gwL[li][k][n][ 9], n_xor   * go);
                gpuAtomicAdd(&gwL[li][k][n][10], (scalar_t(1)-b)        * go);
                gpuAtomicAdd(&gwL[li][k][n][11], (scalar_t(1)-b + ab)   * go);
                gpuAtomicAdd(&gwL[li][k][n][12], (scalar_t(1)-a)        * go);
                gpuAtomicAdd(&gwL[li][k][n][13], (scalar_t(1)-a + ab)   * go);
                gpuAtomicAdd(&gwL[li][k][n][14], (scalar_t(1)-ab)       * go);
                gpuAtomicAdd(&gwL[li][k][n][15], go);
            }

            // Swap grad_curr and grad_next for next level
            scalar_t* tmp3 = grad_curr; grad_curr = grad_next; grad_next = tmp3;
        }
    }

    // Step 3: Scatter-add level 0 gradients to input x_padded
    for (int i=0; i<N0; ++i){
        const scalar_t gvi = grad_curr[i];
        if (gvi != scalar_t(0)){
            const int cabs = cstart + (int)leaf_ici[k][i];
            const int yabs = ph0 + (int)leaf_ipy[k][i];
            const int xabs = pw0 + (int)leaf_ipx[k][i];
            gpuAtomicAdd(&gx[b][cabs][yabs][xabs], gvi);
        }
    }
}

/**********************************************************************************************************************/
/** BLOCK EFFICIENT CROSSBAR LAYER  **********************************************************************************/
/**********************************************************************************************************************/

torch::Tensor block_efficient_crossbar_forward(
    torch::Tensor x,          // [B, in_dim]
    torch::Tensor w_sparse,   // [out_dim, block_size]
    int num_blocks,
    int block_size,
    int out_per_block
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w_sparse);

    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, in_dim]");
    TORCH_CHECK(w_sparse.dim() == 2, "w_sparse must be 2D [out_dim, block_size]");

    const auto B       = x.size(0);
    const auto in_dim  = x.size(1);
    const auto out_dim = w_sparse.size(0);

    TORCH_CHECK(in_dim == num_blocks * block_size,
                "in_dim must be num_blocks * block_size (",
                num_blocks, " * ", block_size, "), got ", in_dim);
    TORCH_CHECK(out_dim == num_blocks * out_per_block,
                "out_dim must be num_blocks * out_per_block (",
                num_blocks, " * ", out_per_block, "), got ", out_dim);

    // x: [B, in_dim] -> [B, num_blocks, block_size] -> [num_blocks, B, block_size]
    auto x_3d = x.view({B, num_blocks, block_size})
                 .permute({1, 0, 2})       // [N, B, Cb]
                 .contiguous();

    // w_sparse: [out_dim, block_size] = [N * Ob, Cb]
    // -> [N, Ob, Cb] -> [N, Cb, Ob]
    auto w_3d = w_sparse.view({num_blocks, out_per_block, block_size})
                 .permute({0, 2, 1})       // [N, Cb, Ob]
                 .contiguous();

    // batched GEMM: [N, B, Cb] x [N, Cb, Ob] = [N, B, Ob]
    auto y_3d = torch::matmul(x_3d, w_3d);  // [N, B, Ob]

    // [N, B, Ob] -> [B, N, Ob] -> [B, out_dim]
    auto y = y_3d.permute({1, 0, 2})        // [B, N, Ob]
                 .contiguous()
                 .view({B, out_dim});       // [B, out_dim]

    return y;
}


torch::Tensor block_efficient_crossbar_backward_w(
    torch::Tensor x,          // [B, in_dim]
    torch::Tensor grad_y,     // [B, out_dim]
    int num_blocks,
    int block_size,
    int out_per_block
) {
    CHECK_INPUT(x);
    CHECK_INPUT(grad_y);

    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, in_dim]");
    TORCH_CHECK(grad_y.dim() == 2, "grad_y must be 2D [B, out_dim]");

    const auto B       = x.size(0);
    const auto in_dim  = x.size(1);
    const auto out_dim = grad_y.size(1);

    TORCH_CHECK(in_dim == num_blocks * block_size,
                "in_dim must be num_blocks * block_size");
    TORCH_CHECK(out_dim == num_blocks * out_per_block,
                "out_dim must be num_blocks * out_per_block");

    // x: [B, in_dim] -> [B, N, Cb] -> [N, B, Cb]
    auto x_3d = x.view({B, num_blocks, block_size})
                 .permute({1, 0, 2})            // [N, B, Cb]
                 .contiguous();

    // grad_y: [B, out_dim] -> [B, N, Ob] -> [N, B, Ob]
    auto gy_3d = grad_y.view({B, num_blocks, out_per_block})
                   .permute({1, 0, 2})          // [N, B, Ob]
                   .contiguous();

    // grad_w_block = X_block^T @ grad_y_block
    //   X_block:   [B, Cb]
    //   grad_y_block: [B, Ob]
    // => [Cb, B] @ [B, Ob] = [Cb, Ob]
    // batched 형태:
    // gy_3d^T: [N, Ob, B], x_3d: [N, B, Cb] -> [N, Ob, Cb]
    auto gy_T = gy_3d.transpose(1, 2);          // [N, Ob, B]
    auto grad_w_3d = torch::matmul(gy_T, x_3d); // [N, Ob, Cb]

    // [N, Ob, Cb] -> [N * Ob, Cb] = [out_dim, block_size]
    auto grad_w = grad_w_3d.contiguous()
                            .view({num_blocks * out_per_block, block_size});

    return grad_w;
}


torch::Tensor block_efficient_crossbar_backward_x(
    torch::Tensor w_sparse,   // [out_dim, block_size]
    torch::Tensor grad_y,     // [B, out_dim]
    int num_blocks,
    int block_size,
    int out_per_block
) {
    CHECK_INPUT(w_sparse);
    CHECK_INPUT(grad_y);

    TORCH_CHECK(w_sparse.dim() == 2, "w_sparse must be 2D [out_dim, block_size]");
    TORCH_CHECK(grad_y.dim() == 2, "grad_y must be 2D [B, out_dim]");

    const auto B       = grad_y.size(0);
    const auto out_dim = grad_y.size(1);

    TORCH_CHECK(out_dim == num_blocks * out_per_block,
                "out_dim must be num_blocks * out_per_block");

    const auto in_dim = num_blocks * block_size;

    // w_sparse: [out_dim, block_size] = [N*Ob, Cb]
    // -> [N, Ob, Cb] -> [N, Cb, Ob]
    auto w_3d = w_sparse.view({num_blocks, out_per_block, block_size})
                 .permute({0, 2, 1})            // [N, Cb, Ob]
                 .contiguous();

    // grad_y: [B, out_dim] -> [B, N, Ob] -> [N, B, Ob]
    auto gy_3d = grad_y.view({B, num_blocks, out_per_block})
                   .permute({1, 0, 2})          // [N, B, Ob]
                   .contiguous();

    // grad_x_block = grad_y_block @ W_block^T
    //   grad_y_block: [B, Ob]
    //   W_block:      [Cb, Ob]
    // => [B, Ob] @ [Ob, Cb] = [B, Cb]
    // batched 형태: gy_3d [N,B,Ob], w_T [N,Ob,Cb] -> [N,B,Cb]
    auto w_T = w_3d.transpose(1, 2);            // [N, Ob, Cb]
    auto grad_x_3d = torch::matmul(gy_3d, w_T); // [N, B, Cb]

    // [N, B, Cb] -> [B, N, Cb] -> [B, in_dim]
    auto grad_x = grad_x_3d.permute({1, 0, 2})  // [B, N, Cb]
                             .contiguous()
                             .view({B, in_dim});

    return grad_x;
}



/**********************************************************************************************************************/
/** FUSED TREE CONV LAYER (NO ORPOOL, PER-PATCH) **********************************************************************/
/**********************************************************************************************************************/

// Forward kernel: each thread = one (patch n, channel c)
template <typename scalar_t>
__global__ void fused_tree_conv_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> x,   // [N, Din]
    torch::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> wL,  // [L, Co, maxN, 16]
    torch::PackedTensorAccessor32<int64_t, 3, at::RestrictPtrTraits> aL,   // [L, Co, maxN]
    torch::PackedTensorAccessor32<int64_t, 3, at::RestrictPtrTraits> bL,   // [L, Co, maxN]
    torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> nodesL,   // [L]
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> y,   // [N, Co]
    int L
){
    extern __shared__ char smem_raw[];

    const int n = blockIdx.x * blockDim.x + threadIdx.x;   // patch index
    const int c = blockIdx.y * blockDim.y + threadIdx.y;   // channel index

    const int N  = x.size(0);
    const int Co = wL.size(1);
    const int max_nodes = wL.size(2);

    if (n >= N || c >= Co) return;

    // per-thread shared region: 2 * max_nodes
    scalar_t* sh   = reinterpret_cast<scalar_t*>(smem_raw);
    const int tid  = threadIdx.y * blockDim.x + threadIdx.x;
    scalar_t* curr = sh + tid * (2 * max_nodes);
    scalar_t* next = curr + max_nodes;

    const int Din = x.size(1);

    // ----- Level 0: copy input features into curr[0..Din) -----
    for (int i = 0; i < Din && i < max_nodes; ++i) {
        curr[i] = x[n][i];
    }
    for (int i = Din; i < max_nodes; ++i) {
        curr[i] = scalar_t(0);
    }

    // ----- Level traversal -----
    for (int li = 0; li < L; ++li) {
        const int nL = nodesL[li];
        scalar_t* outbuf = (li % 2 == 0) ? next : curr;
        scalar_t* inbuf  = (li % 2 == 0) ? curr : next;

        for (int node = 0; node < nL; ++node) {
            const int64_t ai = aL[li][c][node];
            const int64_t bi = bL[li][c][node];
            const scalar_t a = inbuf[ai];
            const scalar_t b = inbuf[bi];
            const scalar_t* w = &wL[li][c][node][0];
            outbuf[node] = gate_forward_16(w, a, b);
        }

        // ping-pong swap
        scalar_t* tmp = curr; curr = outbuf; outbuf = tmp;
    }

    // 마지막 레벨 출력에서 이 채널 c에 해당하는 노드를 그대로 사용한다고 가정
    const int last_nL = (L > 0) ? nodesL[L-1] : Din;
    if (c < last_nL) {
        y[n][c] = curr[c];
    } else {
        // c가 last_nL보다 크면 마지막 노드 값 재사용 (안전장치)
        y[n][c] = curr[last_nL - 1];
    }
}


// Backward kernel: each thread = one (patch n, channel c)
template <typename scalar_t>
__global__ void fused_tree_conv_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> x,    // [N, Din]
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> gy,   // [N, Co]
    torch::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> wL,   // [L, Co, maxN, 16]
    torch::PackedTensorAccessor32<int64_t, 3, at::RestrictPtrTraits> aL,    // [L, Co, maxN]
    torch::PackedTensorAccessor32<int64_t, 3, at::RestrictPtrTraits> bL,    // [L, Co, maxN]
    torch::PackedTensorAccessor32<int, 1, at::RestrictPtrTraits> nodesL,    // [L]
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> gx,   // [N, Din]
    torch::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> gwL,  // [L, Co, maxN, 16]
    int L
){
    extern __shared__ char smem_raw[];

    const int n  = blockIdx.x * blockDim.x + threadIdx.x;  // patch index
    const int c  = blockIdx.y * blockDim.y + threadIdx.y;  // channel index
    const int N  = x.size(0);
    const int Co = wL.size(1);
    const int max_nodes = wL.size(2);

    if (n >= N || c >= Co) return;

    const scalar_t upstream = gy[n][c];
    if (upstream == scalar_t(0)) return;

    scalar_t* sh   = reinterpret_cast<scalar_t*>(smem_raw);
    const int tid  = threadIdx.y * blockDim.x + threadIdx.x;
    scalar_t* buf0 = sh + tid * (2 * max_nodes);
    scalar_t* buf1 = buf0 + max_nodes;

    const int Din = x.size(1);

    // ----- Forward recomputation (level 0) -----
    for (int i = 0; i < Din && i < max_nodes; ++i) {
        buf0[i] = x[n][i];
    }
    for (int i = Din; i < max_nodes; ++i) {
        buf0[i] = scalar_t(0);
    }

    scalar_t* curr = buf0;
    scalar_t* next = buf1;

    for (int li = 0; li < L; ++li) {
        const int nL = nodesL[li];
        scalar_t* outbuf = (li % 2 == 0) ? next : curr;
        scalar_t* inbuf  = (li % 2 == 0) ? curr : next;

        for (int node = 0; node < nL; ++node) {
            const int64_t ai = aL[li][c][node];
            const int64_t bi = bL[li][c][node];
            const scalar_t a = inbuf[ai];
            const scalar_t b = inbuf[bi];
            const scalar_t* w = &wL[li][c][node][0];
            outbuf[node] = gate_forward_16(w, a, b);
        }
        scalar_t* tmp = curr; curr = outbuf; outbuf = tmp;
    }

    // ----- Backward: top (L-1) → 0 -----
    scalar_t* grad_curr = buf1;
    scalar_t* grad_next = buf0;

    // grad_curr 초기화
    for (int i = 0; i < max_nodes; ++i) grad_curr[i] = scalar_t(0);
    for (int i = 0; i < max_nodes; ++i) grad_next[i] = scalar_t(0);

    const int last_nL = (L > 0) ? nodesL[L-1] : Din;
    const int root_idx = (c < last_nL) ? c : (last_nL - 1);
    grad_curr[root_idx] = upstream;

    for (int li = L-1; li >= 0; --li) {
        const int nL = nodesL[li];

        // in_vec 재구성: 레벨 0~li-1 forward 다시
        for (int i = 0; i < Din && i < max_nodes; ++i) {
            buf0[i] = x[n][i];
        }
        for (int i = Din; i < max_nodes; ++i) {
            buf0[i] = scalar_t(0);
        }
        scalar_t* cbuf = buf0;
        scalar_t* nbuf = buf1;
        for (int j = 0; j < li; ++j) {
            const int nj = nodesL[j];
            scalar_t* outb = (j % 2 == 0) ? nbuf : cbuf;
            scalar_t* inb  = (j % 2 == 0) ? cbuf : nbuf;
            for (int m = 0; m < nj; ++m) {
                const int64_t ai = aL[j][c][m];
                const int64_t bi = bL[j][c][m];
                const scalar_t a = inb[ai];
                const scalar_t b = inb[bi];
                const scalar_t* w = &wL[j][c][m][0];
                outb[m] = gate_forward_16(w, a, b);
            }
            scalar_t* tmp2 = cbuf; cbuf = outb; outb = tmp2;
        }
        scalar_t* in_vec   = cbuf;
        scalar_t* out_grad = grad_curr;

        // grad_next 초기화
        for (int z = 0; z < max_nodes; ++z) grad_next[z] = scalar_t(0);

        for (int node = 0; node < nL; ++node) {
            const scalar_t go = out_grad[node];
            if (go == scalar_t(0)) continue;

            const int64_t ai = aL[li][c][node];
            const int64_t bi = bL[li][c][node];
            const scalar_t a = in_vec[ai];
            const scalar_t b = in_vec[bi];
            const scalar_t* w = &wL[li][c][node][0];

            // a,b 편미분
            scalar_t dza, dzb;
            gate_backward_partials_16(w, a, b, dza, dzb);
            gpuAtomicAdd(&grad_next[ai], dza * go);
            gpuAtomicAdd(&grad_next[bi], dzb * go);

            // weight grad 16개 basis
            const scalar_t ab      = a * b;
            const scalar_t a_n     = a - ab;
            const scalar_t b_n     = b - ab;
            const scalar_t a_xor_b = a + b - scalar_t(2) * ab;
            const scalar_t a_or_b  = a + b - ab;
            const scalar_t n_or    = scalar_t(1) - a_or_b;
            const scalar_t n_xor   = scalar_t(1) - a_xor_b;

            gpuAtomicAdd(&gwL[li][c][node][ 1], ab      * go);
            gpuAtomicAdd(&gwL[li][c][node][ 2], a_n     * go);
            gpuAtomicAdd(&gwL[li][c][node][ 3], a       * go);
            gpuAtomicAdd(&gwL[li][c][node][ 4], b_n     * go);
            gpuAtomicAdd(&gwL[li][c][node][ 5], b       * go);
            gpuAtomicAdd(&gwL[li][c][node][ 6], a_xor_b * go);
            gpuAtomicAdd(&gwL[li][c][node][ 7], a_or_b  * go);
            gpuAtomicAdd(&gwL[li][c][node][ 8], n_or    * go);
            gpuAtomicAdd(&gwL[li][c][node][ 9], n_xor   * go);
            gpuAtomicAdd(&gwL[li][c][node][10], (scalar_t(1) - b)      * go);
            gpuAtomicAdd(&gwL[li][c][node][11], (scalar_t(1) - b + ab) * go);
            gpuAtomicAdd(&gwL[li][c][node][12], (scalar_t(1) - a)      * go);
            gpuAtomicAdd(&gwL[li][c][node][13], (scalar_t(1) - a + ab) * go);
            gpuAtomicAdd(&gwL[li][c][node][14], (scalar_t(1) - ab)     * go);
            gpuAtomicAdd(&gwL[li][c][node][15], go);
        }

        // swap grad_curr / grad_next
        scalar_t* tmp3 = grad_curr; grad_curr = grad_next; grad_next = tmp3;
    }

    // ----- level 0 (입력) grad를 gx에 scatter-add -----
    for (int i = 0; i < Din && i < max_nodes; ++i) {
        const scalar_t gvi = grad_curr[i];
        if (gvi != scalar_t(0)) {
            gpuAtomicAdd(&gx[n][i], gvi);
        }
    }
}


// Host wrapper: forward
torch::Tensor fused_tree_conv_forward_cuda(
    const torch::Tensor& x,              // [N, Din]
    const torch::Tensor& weights_L,      // [L, Co, maxN, 16]
    const torch::Tensor& a_idx_L,        // [L, Co, maxN]
    const torch::Tensor& b_idx_L,        // [L, Co, maxN]
    const torch::Tensor& nodes_per_level // [L]
){
    CHECK_INPUT(x);
    CHECK_INPUT(weights_L);
    CHECK_INPUT(a_idx_L);
    CHECK_INPUT(b_idx_L);
    CHECK_INPUT(nodes_per_level);

    const int N   = x.size(0);
    const int Co  = weights_L.size(1);
    const int L   = weights_L.size(0);
    const int max_nodes = weights_L.size(2);

    auto y = torch::empty({N, Co}, x.options());

    dim3 block(32, 8);
    dim3 grid(
        (N  + block.x - 1) / block.x,
        (Co + block.y - 1) / block.y
    );

    size_t shmem = static_cast<size_t>(2 * max_nodes * block.x * block.y) * x.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_tree_conv_forward", [&]{
        fused_tree_conv_forward_kernel<scalar_t><<<grid, block, shmem>>>(
            x.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            weights_L.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
            a_idx_L.packed_accessor32<int64_t, 3, at::RestrictPtrTraits>(),
            b_idx_L.packed_accessor32<int64_t, 3, at::RestrictPtrTraits>(),
            nodes_per_level.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
            y.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            L
        );
    });
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return y;
}


// Host wrapper: backward
std::vector<torch::Tensor> fused_tree_conv_backward_cuda(
    const torch::Tensor& x,              // [N, Din]
    const torch::Tensor& grad_out,       // [N, Co]
    const torch::Tensor& weights_L,      // [L, Co, maxN, 16]
    const torch::Tensor& a_idx_L,        // [L, Co, maxN]
    const torch::Tensor& b_idx_L,        // [L, Co, maxN]
    const torch::Tensor& nodes_per_level // [L]
){
    CHECK_INPUT(x);
    CHECK_INPUT(grad_out);
    CHECK_INPUT(weights_L);
    CHECK_INPUT(a_idx_L);
    CHECK_INPUT(b_idx_L);
    CHECK_INPUT(nodes_per_level);

    const int N   = x.size(0);
    const int Co  = weights_L.size(1);
    const int L   = weights_L.size(0);
    const int max_nodes = weights_L.size(2);

    auto gx  = torch::zeros_like(x);
    auto gwL = torch::zeros_like(weights_L);

    dim3 block(32, 8);
    dim3 grid(
        (N  + block.x - 1) / block.x,
        (Co + block.y - 1) / block.y
    );

    size_t shmem = static_cast<size_t>(2 * max_nodes * block.x * block.y) * x.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_tree_conv_backward", [&]{
        fused_tree_conv_backward_kernel<scalar_t><<<grid, block, shmem>>>(
            x.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            grad_out.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            weights_L.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
            a_idx_L.packed_accessor32<int64_t, 3, at::RestrictPtrTraits>(),
            b_idx_L.packed_accessor32<int64_t, 3, at::RestrictPtrTraits>(),
            nodes_per_level.packed_accessor32<int, 1, at::RestrictPtrTraits>(),
            gx.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            gwL.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>(),
            L
        );
    });
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return {gx, gwL};
}







/*************************************************************************************************/
/** TRIPLE LOGIC LAYER FUSED FORWARD (x -> h1 -> h2 -> y) ***************************************/
/*************************************************************************************************/

// Forward declaration of the kernel
template <typename scalar_t>
__global__ void logic_triple_backward_kernel_grouped(
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> x,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> gy,
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> a1,
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> b1,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> w1,
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> a2,
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> b2,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> w2,
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> a3,
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> b3,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> w3,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> gx,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> gw1_,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> gw2_,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> gw3_,
    int D0, int D1, int D2, int D3,
    int groups
);

std::vector<torch::Tensor> logic_triple_backward_cuda(
    torch::Tensor x,       // [N, D0]
    torch::Tensor grad_y,  // [N, D3]
    torch::Tensor a1, torch::Tensor b1, torch::Tensor w1,
    torch::Tensor a2, torch::Tensor b2, torch::Tensor w2,
    torch::Tensor a3, torch::Tensor b3, torch::Tensor w3,
    int groups
){
    CHECK_INPUT(x);
    CHECK_INPUT(grad_y);
    CHECK_INPUT(w1); CHECK_INPUT(w2); CHECK_INPUT(w3);
    CHECK_INPUT(a1); CHECK_INPUT(b1);
    CHECK_INPUT(a2); CHECK_INPUT(b2);
    CHECK_INPUT(a3); CHECK_INPUT(b3);

    const int64_t N  = x.size(0);
    const int64_t D0 = x.size(1);
    const int64_t D1 = a1.size(0);
    const int64_t D2 = a2.size(0);
    const int64_t D3 = a3.size(0);

    TORCH_CHECK(D1 % groups == 0 && D2 % groups == 0 && D3 % groups == 0,
                "logic_triple_backward_cuda: D1,D2,D3 must be divisible by groups.");

    auto gx  = torch::zeros_like(x);
    auto gw1 = torch::zeros_like(w1);
    auto gw2 = torch::zeros_like(w2);
    auto gw3 = torch::zeros_like(w3);

    const int threads = 128;
    const dim3 block(threads);
    const dim3 grid(N, groups);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "logic_triple_backward_cuda", [&]{
        const int D1_g = (int)(D1 / groups);
        const int D2_g = (int)(D2 / groups);
        const size_t shmem = (D1_g + D2_g + D1_g + D2_g) * sizeof(scalar_t); // h1,h2,gh1,gh2

        int dev;
        cudaGetDevice(&dev);
        int max_shmem_bytes = 0;
        cudaDeviceGetAttribute(&max_shmem_bytes,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if (max_shmem_bytes == 0) {
            cudaDeviceGetAttribute(&max_shmem_bytes,
                                   cudaDevAttrMaxSharedMemoryPerBlock, dev);
        }
        TORCH_CHECK(shmem <= (size_t)max_shmem_bytes,
                    "logic_triple_backward_cuda: required shared memory (", shmem,
                    ") > device limit (", max_shmem_bytes,
                    "). Reduce out_channels or groups, or fallback to unfused.");

        logic_triple_backward_kernel_grouped<scalar_t><<<grid, block, shmem>>>(
            x.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            grad_y.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            a1.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            b1.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            w1.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            a2.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            b2.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            w2.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            a3.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            b3.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
            w3.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            gx.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            gw1.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            gw2.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            gw3.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
            (int)D0, (int)D1, (int)D2, (int)D3,
            groups
        );
    });
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return {gx, gw1, gw2, gw3};
}

template <typename scalar_t>
__global__ void logic_triple_backward_kernel_grouped(
    // x: [N, D0]
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> x,
    // grad_y: [N, D3]
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> gy,

    // Layer1
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> a1,
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> b1,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> w1,
    // Layer2
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> a2,
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> b2,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> w2,
    // Layer3
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> a3,
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> b3,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> w3,

    // gradients
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> gx,   // [N, D0]
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> gw1_, // [D1,16]
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> gw2_, // [D2,16]
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> gw3_, // [D3,16]

    int D0, int D1, int D2, int D3,
    int groups
){
    extern __shared__ char smem_raw[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);

    const int n   = blockIdx.x;  // patch index
    const int g   = blockIdx.y;  // group index
    const int tid = threadIdx.x;
    const int N   = x.size(0);
    if (n >= N || g >= groups) return;

    const int D1_g = D1 / groups;
    const int D2_g = D2 / groups;
    const int D3_g = D3 / groups;

    const int o1_start = g * D1_g;
    const int o2_start = g * D2_g;
    const int o3_start = g * D3_g;

    // shared layout: h1_g[D1_g], h2_g[D2_g], gh1_g[D1_g], gh2_g[D2_g]
    scalar_t* h1  = smem;
    scalar_t* h2  = h1  + D1_g;
    scalar_t* gh1 = h2  + D2_g;
    scalar_t* gh2 = gh1 + D1_g;

    // -----------------------------
    // 0) forward 재계산: h1_g, h2_g
    // -----------------------------
    // Layer1: x -> h1_g
    for (int i_local = tid; i_local < D1_g; i_local += blockDim.x) {
        const int i = o1_start + i_local;
        const int64_t ia = a1[i];
        const int64_t ib = b1[i];
        if (ia < 0 || ia >= D0 || ib < 0 || ib >= D0) {
            h1[i_local] = scalar_t(0);
            continue;
        }
        const scalar_t va = x[n][ia];
        const scalar_t vb = x[n][ib];
        const scalar_t* w = &w1[i][0];
        h1[i_local] = gate_forward_16<scalar_t>(w, va, vb);
    }
    __syncthreads();

    // Layer2: h1_g -> h2_g
    // Pairwise connection: i_local -> (2*i_local) % D1_g, (2*i_local+1) % D1_g
    for (int i_local = tid; i_local < D2_g; i_local += blockDim.x) {
        const int i = o2_start + i_local;
        
        // Pairwise: 직접 계산 (인덱스 배열 불필요)
        const int ia_local = (2 * i_local) % D1_g;
        const int ib_local = (2 * i_local + 1) % D1_g;
        
        const scalar_t va = h1[ia_local];
        const scalar_t vb = h1[ib_local];
        const scalar_t* w = &w2[i][0];
        h2[i_local] = gate_forward_16<scalar_t>(w, va, vb);
    }
    __syncthreads();

    // -----------------------------
    // 1) Layer3 backward: grad_y -> gh2_g, gw3
    // -----------------------------
    for (int i_local = tid; i_local < D2_g; i_local += blockDim.x) {
        gh2[i_local] = scalar_t(0);
    }

    for (int i_local = tid; i_local < D3_g; i_local += blockDim.x) {
        const int i = o3_start + i_local;
        const scalar_t go = gy[n][i];
        if (go == scalar_t(0)) continue;

        // Pairwise: 직접 계산 (인덱스 배열 불필요)
        const int ia_local = (2 * i_local) % D2_g;
        const int ib_local = (2 * i_local + 1) % D2_g;
        
        const scalar_t va = h2[ia_local];
        const scalar_t vb = h2[ib_local];
        const scalar_t* w = &w3[i][0];

        scalar_t dza, dzb;
        gate_backward_partials_16<scalar_t>(w, va, vb, dza, dzb);

        gpuAtomicAdd(&gh2[ia_local], dza * go);
        gpuAtomicAdd(&gh2[ib_local], dzb * go);

        const scalar_t ab      = va * vb;
        const scalar_t a_n     = va - ab;
        const scalar_t b_n     = vb - ab;
        const scalar_t a_xor_b = va + vb - scalar_t(2) * ab;
        const scalar_t a_or_b  = va + vb - ab;
        const scalar_t n_or    = scalar_t(1) - a_or_b;
        const scalar_t n_xor   = scalar_t(1) - a_xor_b;

        gpuAtomicAdd(&gw3_[i][ 1], ab        * go);
        gpuAtomicAdd(&gw3_[i][ 2], a_n       * go);
        gpuAtomicAdd(&gw3_[i][ 3], va        * go);
        gpuAtomicAdd(&gw3_[i][ 4], b_n       * go);
        gpuAtomicAdd(&gw3_[i][ 5], vb        * go);
        gpuAtomicAdd(&gw3_[i][ 6], a_xor_b   * go);
        gpuAtomicAdd(&gw3_[i][ 7], a_or_b    * go);
        gpuAtomicAdd(&gw3_[i][ 8], n_or      * go);
        gpuAtomicAdd(&gw3_[i][ 9], n_xor     * go);
        gpuAtomicAdd(&gw3_[i][10], (scalar_t(1) - vb)      * go);
        gpuAtomicAdd(&gw3_[i][11], (scalar_t(1) - vb + ab) * go);
        gpuAtomicAdd(&gw3_[i][12], (scalar_t(1) - va)      * go);
        gpuAtomicAdd(&gw3_[i][13], (scalar_t(1) - va + ab) * go);
        gpuAtomicAdd(&gw3_[i][14], (scalar_t(1) - ab)      * go);
        gpuAtomicAdd(&gw3_[i][15], go);
    }
    __syncthreads();

    // -----------------------------
    // 2) Layer2 backward: gh2_g -> gh1_g, gw2
    // -----------------------------
    for (int i_local = tid; i_local < D1_g; i_local += blockDim.x) {
        gh1[i_local] = scalar_t(0);
    }

    for (int i_local = tid; i_local < D2_g; i_local += blockDim.x) {
        const scalar_t go = gh2[i_local];
        if (go == scalar_t(0)) continue;

        const int i = o2_start + i_local;
        
        // Pairwise: 직접 계산 (인덱스 배열 불필요)
        const int ia_local = (2 * i_local) % D1_g;
        const int ib_local = (2 * i_local + 1) % D1_g;
        
        const scalar_t va = h1[ia_local];
        const scalar_t vb = h1[ib_local];
        const scalar_t* w = &w2[i][0];

        scalar_t dza, dzb;
        gate_backward_partials_16<scalar_t>(w, va, vb, dza, dzb);

        gpuAtomicAdd(&gh1[ia_local], dza * go);
        gpuAtomicAdd(&gh1[ib_local], dzb * go);

        const scalar_t ab      = va * vb;
        const scalar_t a_n     = va - ab;
        const scalar_t b_n     = vb - ab;
        const scalar_t a_xor_b = va + vb - scalar_t(2) * ab;
        const scalar_t a_or_b  = va + vb - ab;
        const scalar_t n_or    = scalar_t(1) - a_or_b;
        const scalar_t n_xor   = scalar_t(1) - a_xor_b;

        gpuAtomicAdd(&gw2_[i][ 1], ab        * go);
        gpuAtomicAdd(&gw2_[i][ 2], a_n       * go);
        gpuAtomicAdd(&gw2_[i][ 3], va        * go);
        gpuAtomicAdd(&gw2_[i][ 4], b_n       * go);
        gpuAtomicAdd(&gw2_[i][ 5], vb        * go);
        gpuAtomicAdd(&gw2_[i][ 6], a_xor_b   * go);
        gpuAtomicAdd(&gw2_[i][ 7], a_or_b    * go);
        gpuAtomicAdd(&gw2_[i][ 8], n_or      * go);
        gpuAtomicAdd(&gw2_[i][ 9], n_xor     * go);
        gpuAtomicAdd(&gw2_[i][10], (scalar_t(1) - vb)      * go);
        gpuAtomicAdd(&gw2_[i][11], (scalar_t(1) - vb + ab) * go);
        gpuAtomicAdd(&gw2_[i][12], (scalar_t(1) - va)      * go);
        gpuAtomicAdd(&gw2_[i][13], (scalar_t(1) - va + ab) * go);
        gpuAtomicAdd(&gw2_[i][14], (scalar_t(1) - ab)      * go);
        gpuAtomicAdd(&gw2_[i][15], go);
    }
    __syncthreads();

    // -----------------------------
    // 3) Layer1 backward: gh1_g -> gx, gw1
    // -----------------------------
    for (int i_local = tid; i_local < D1_g; i_local += blockDim.x) {
        const scalar_t go = gh1[i_local];
        if (go == scalar_t(0)) continue;

        const int i = o1_start + i_local;
        const int64_t ia = a1[i];
        const int64_t ib = b1[i];
        if (ia < 0 || ia >= D0 || ib < 0 || ib >= D0) continue;

        const scalar_t va = x[n][ia];
        const scalar_t vb = x[n][ib];
        const scalar_t* w = &w1[i][0];

        scalar_t dza, dzb;
        gate_backward_partials_16<scalar_t>(w, va, vb, dza, dzb);

        // gx는 patch n 전용이므로 atomic 불필요
        gx[n][ia] += dza * go;
        gx[n][ib] += dzb * go;

        const scalar_t ab      = va * vb;
        const scalar_t a_n     = va - ab;
        const scalar_t b_n     = vb - ab;
        const scalar_t a_xor_b = va + vb - scalar_t(2) * ab;
        const scalar_t a_or_b  = va + vb - ab;
        const scalar_t n_or    = scalar_t(1) - a_or_b;
        const scalar_t n_xor   = scalar_t(1) - a_xor_b;

        gpuAtomicAdd(&gw1_[i][ 1], ab        * go);
        gpuAtomicAdd(&gw1_[i][ 2], a_n       * go);
        gpuAtomicAdd(&gw1_[i][ 3], va        * go);
        gpuAtomicAdd(&gw1_[i][ 4], b_n       * go);
        gpuAtomicAdd(&gw1_[i][ 5], vb        * go);
        gpuAtomicAdd(&gw1_[i][ 6], a_xor_b   * go);
        gpuAtomicAdd(&gw1_[i][ 7], a_or_b    * go);
        gpuAtomicAdd(&gw1_[i][ 8], n_or      * go);
        gpuAtomicAdd(&gw1_[i][ 9], n_xor     * go);
        gpuAtomicAdd(&gw1_[i][10], (scalar_t(1) - vb)      * go);
        gpuAtomicAdd(&gw1_[i][11], (scalar_t(1) - vb + ab) * go);
        gpuAtomicAdd(&gw1_[i][12], (scalar_t(1) - va)      * go);
        gpuAtomicAdd(&gw1_[i][13], (scalar_t(1) - va + ab) * go);
        gpuAtomicAdd(&gw1_[i][14], (scalar_t(1) - ab)      * go);
        gpuAtomicAdd(&gw1_[i][15], go);
    }
}

/*************************************************************************************************/
/** TRIPLE LOGIC LAYER FUSED FORWARD (group-aware, x -> h1 -> h2 -> y) **************************/
/*************************************************************************************************/

template <typename scalar_t>
__global__ void logic_triple_forward_kernel_grouped(
    // x: [N, D0]
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> x,

    // Layer 1: D0 -> D1
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> a1,
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> b1,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> w1,  // [D1,16]

    // Layer 2: D1 -> D2
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> a2,
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> b2,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> w2,  // [D2,16]

    // Layer 3: D2 -> D3
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> a3,
    torch::PackedTensorAccessor32<int64_t, 1, at::RestrictPtrTraits> b3,
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> w3,  // [D3,16]

    // y: [N, D3]
    torch::PackedTensorAccessor32<scalar_t, 2, at::RestrictPtrTraits> y,

    // dimensions
    int D0, int D1, int D2, int D3,
    int groups   // k
){
    extern __shared__ char smem_raw[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);

    const int n = blockIdx.x;    // patch index
    const int g = blockIdx.y;    // group index
    const int tid = threadIdx.x;

    const int N = x.size(0);
    if (n >= N || g >= groups) return;

    // group별 local dim
    const int D1_g = D1 / groups;
    const int D2_g = D2 / groups;
    const int D3_g = D3 / groups;

    const int o1_start = g * D1_g;  // global offset in layer1 output
    const int o2_start = g * D2_g;  // layer2
    const int o3_start = g * D3_g;  // layer3/output

    // shared memory: 2 * max(D1_g, D2_g)
    const int max_nodes_g = max(D1_g, D2_g);
    scalar_t* buf0 = smem;                    // size >= D1_g or D2_g
    scalar_t* buf1 = smem + max_nodes_g;      // size >= D1_g or D2_g

    // -------------------------------------------
    // Layer 1: x[n,:D0] -> h1_g[0..D1_g)
    // -------------------------------------------
    for (int i_local = tid; i_local < D1_g; i_local += blockDim.x) {
        const int i = o1_start + i_local;    // global node index in [o1_start, o1_start + D1_g)
        const int64_t ia = a1[i];
        const int64_t ib = b1[i];

        if (ia < 0 || ia >= D0 || ib < 0 || ib >= D0) {
            buf0[i_local] = scalar_t(0);
            continue;
        }
        const scalar_t va = x[n][ia];
        const scalar_t vb = x[n][ib];
        const scalar_t* w = &w1[i][0];

        buf0[i_local] = gate_forward_16<scalar_t>(w, va, vb);
    }
    __syncthreads();

    // -------------------------------------------
    // Layer 2: h1_g[0..D1_g) -> h2_g[0..D2_g)
    // Pairwise connection: i_local -> (2*i_local) % D1_g, (2*i_local+1) % D1_g
    // -------------------------------------------
    for (int i_local = tid; i_local < D2_g; i_local += blockDim.x) {
        const int i = o2_start + i_local;   // global node index in [o2_start, o2_start + D2_g)
        
        // Pairwise: 직접 계산 (인덱스 배열 불필요)
        const int ia_local = (2 * i_local) % D1_g;
        const int ib_local = (2 * i_local + 1) % D1_g;

        const scalar_t va = buf0[ia_local];
        const scalar_t vb = buf0[ib_local];
        const scalar_t* w = &w2[i][0];

        buf1[i_local] = gate_forward_16<scalar_t>(w, va, vb);
    }
    __syncthreads();

    // -------------------------------------------
    // Layer 3: h2_g[0..D2_g) -> y[n, group slice]
    // Pairwise connection: i_local -> (2*i_local) % D2_g, (2*i_local+1) % D2_g
    // -------------------------------------------
    for (int i_local = tid; i_local < D3_g; i_local += blockDim.x) {
        const int i = o3_start + i_local;
        
        // Pairwise: 직접 계산 (인덱스 배열 불필요)
        const int ia_local = (2 * i_local) % D2_g;
        const int ib_local = (2 * i_local + 1) % D2_g;

        const scalar_t va = buf1[ia_local];
        const scalar_t vb = buf1[ib_local];
        const scalar_t* w = &w3[i][0];

        y[n][i] = gate_forward_16<scalar_t>(w, va, vb);
    }
}

torch::Tensor logic_triple_forward_cuda(
    torch::Tensor x,   // [N, D0]
    torch::Tensor a1, torch::Tensor b1, torch::Tensor w1,  // [D1], [D1], [D1,16]
    torch::Tensor a2, torch::Tensor b2, torch::Tensor w2,  // [D2], [D2], [D2,16]
    torch::Tensor a3, torch::Tensor b3, torch::Tensor w3,  // [D3], [D3], [D3,16]
    int groups                                             // k (num groups)
){
    CHECK_INPUT(x);
    CHECK_INPUT(w1);
    CHECK_INPUT(w2);
    CHECK_INPUT(w3);
    CHECK_INPUT(a1);
    CHECK_INPUT(b1);
    CHECK_INPUT(a2);
    CHECK_INPUT(b2);
    CHECK_INPUT(a3);
    CHECK_INPUT(b3);

    TORCH_CHECK(x.dim() == 2,  "x must be [N, D0]");
    TORCH_CHECK(w1.dim() == 2, "w1 must be [D1,16]");
    TORCH_CHECK(w2.dim() == 2, "w2 must be [D2,16]");
    TORCH_CHECK(w3.dim() == 2, "w3 must be [D3,16]");
    TORCH_CHECK(a1.dim() == 1 && b1.dim() == 1, "a1,b1 must be [D1]");
    TORCH_CHECK(a2.dim() == 1 && b2.dim() == 1, "a2,b2 must be [D2]");
    TORCH_CHECK(a3.dim() == 1 && b3.dim() == 1, "a3,b3 must be [D3]");

    const int64_t N  = x.size(0);
    const int64_t D0 = x.size(1);
    const int64_t D1 = a1.size(0);
    const int64_t D2 = a2.size(0);
    const int64_t D3 = a3.size(0);

    TORCH_CHECK(w1.size(0) == D1 && w1.size(1) == 16, "w1 shape mismatch");
    TORCH_CHECK(w2.size(0) == D2 && w2.size(1) == 16, "w2 shape mismatch");
    TORCH_CHECK(w3.size(0) == D3 && w3.size(1) == 16, "w3 shape mismatch");

    TORCH_CHECK(D1 % groups == 0 && D2 % groups == 0 && D3 % groups == 0,
                "logic_triple_forward_cuda: D1,D2,D3 must be divisible by groups. "
                "Got D1=", D1, ", D2=", D2, ", D3=", D3, ", groups=", groups);

    auto y = torch::empty({N, D3}, x.options());

    const int threads = 128;
    const dim3 block(threads);
    const dim3 grid(N, groups);    // (patch, group)

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(), "logic_triple_forward_cuda", [&]{
            const int D1_g = (int)(D1 / groups);
            const int D2_g = (int)(D2 / groups);
            const int max_nodes_g = std::max(D1_g, D2_g);
            const size_t shmem = 2 * max_nodes_g * sizeof(scalar_t);

            // (선택) shared mem 한도 체크
            int dev;
            cudaGetDevice(&dev);
            int max_shmem_bytes = 0;
            cudaDeviceGetAttribute(&max_shmem_bytes,
                                   cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            if (max_shmem_bytes == 0) {
                cudaDeviceGetAttribute(&max_shmem_bytes,
                                       cudaDevAttrMaxSharedMemoryPerBlock, dev);
            }
            TORCH_CHECK(shmem <= (size_t)max_shmem_bytes,
                        "logic_triple_forward_cuda: required shared memory (", shmem,
                        ") > device limit (", max_shmem_bytes,
                        "). Reduce out_channels or groups, or fallback to unfused.");

            logic_triple_forward_kernel_grouped<scalar_t><<<grid, block, shmem>>>(
                x.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                a1.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                b1.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                w1.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                a2.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                b2.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                w2.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                a3.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                b3.packed_accessor32<int64_t, 1, at::RestrictPtrTraits>(),
                w3.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                y.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),
                (int)D0, (int)D1, (int)D2, (int)D3,
                groups
            );
        }
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return y;
}
