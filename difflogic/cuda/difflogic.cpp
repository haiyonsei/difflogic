#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <ATen/core/TensorAccessor.h>   // <-- 추가 (packed_accessor*, RestrictPtrTraits)
#include <vector>
#include <cuda_runtime.h>


namespace py = pybind11;

// --- Existing function declarations ---
torch::Tensor logic_layer_cuda_forward(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
);
torch::Tensor logic_layer_cuda_backward_w(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_y
);
torch::Tensor logic_layer_cuda_backward_x(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor grad_y,
    torch::Tensor given_x_indices_of_y_start,
    torch::Tensor given_x_indices_of_y
);
torch::Tensor logic_layer_cuda_eval(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
);
std::tuple<torch::Tensor, int> tensor_packbits_cuda(
    torch::Tensor t,
    const int bit_count
);
torch::Tensor groupbitsum(
    torch::Tensor b,
    const int pad_len,
    const int k
);
torch::Tensor weighted_groupbitsum(
    torch::Tensor b,
    const int pad_len,
    const int k,
    torch::Tensor weights
);
torch::Tensor pruned_groupbitsum(
    torch::Tensor b,
    const int pad_len,
    const int k,
    torch::Tensor group_sizes
);

// ▼▼▼ Add declaration for the new pruned_weighted_groupbitsum function ▼▼▼
torch::Tensor pruned_weighted_groupbitsum(
    torch::Tensor b,
    const int pad_len,
    const int k,
    torch::Tensor group_sizes,
    torch::Tensor weights
);


// === Host launcher for FusedTreeConvLayer (ORPool 없이) ===
torch::Tensor fused_tree_conv_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weights_L,
    const torch::Tensor& a_idx_L,
    const torch::Tensor& b_idx_L,
    const torch::Tensor& nodes_per_level
);

std::vector<torch::Tensor> fused_tree_conv_backward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& grad_out,
    const torch::Tensor& weights_L,
    const torch::Tensor& a_idx_L,
    const torch::Tensor& b_idx_L,
    const torch::Tensor& nodes_per_level
);

// === Host launcher for forward (per-level version) ===
torch::Tensor fused_forward_ablevels_cuda(
    const torch::Tensor& x_padded,
    const torch::Tensor& weights_L,
    const torch::Tensor& a_idx_L,
    const torch::Tensor& b_idx_L,
    const torch::Tensor& leaf_ici,
    const torch::Tensor& leaf_ipx,
    const torch::Tensor& leaf_ipy,
    const torch::Tensor& nodes_per_level,
    int out_h, int out_w, int kernel_size, int stride, int groups, int in_channels_per_group);

std::vector<torch::Tensor> fused_backward_ablevels_cuda(
    const torch::Tensor& x_padded,
    const torch::Tensor& grad_out,
    const torch::Tensor& weights_L,
    const torch::Tensor& a_idx_L,
    const torch::Tensor& b_idx_L,
    const torch::Tensor& leaf_ici,
    const torch::Tensor& leaf_ipx,
    const torch::Tensor& leaf_ipy,
    const torch::Tensor& nodes_per_level,
    int out_h, int out_w, int kernel_size, int stride, int groups, int in_channels_per_group);

// BlockEfficientCrossbarLayer CUDA functions
torch::Tensor block_efficient_crossbar_forward(
    torch::Tensor x,
    torch::Tensor w_sparse,
    int num_blocks,
    int block_size,
    int out_per_block
);

torch::Tensor block_efficient_crossbar_backward_w(
    torch::Tensor x,
    torch::Tensor grad_y,
    int num_blocks,
    int block_size,
    int out_per_block
);

torch::Tensor block_efficient_crossbar_backward_x(
    torch::Tensor w_sparse,
    torch::Tensor grad_y,
    int num_blocks,
    int block_size,
    int out_per_block
);

torch::Tensor logic_triple_forward_cuda(
    torch::Tensor x,
    torch::Tensor a1, torch::Tensor b1, torch::Tensor w1,
    torch::Tensor a2, torch::Tensor b2, torch::Tensor w2,
    torch::Tensor a3, torch::Tensor b3, torch::Tensor w3,
    int groups
);

std::vector<torch::Tensor> logic_triple_backward_cuda(
    torch::Tensor x, torch::Tensor grad_y,
    torch::Tensor a1, torch::Tensor b1, torch::Tensor w1,
    torch::Tensor a2, torch::Tensor b2, torch::Tensor w2,
    torch::Tensor a3, torch::Tensor b3, torch::Tensor w3,
    int groups
);




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // --- Existing bindings ---
    m.def(
        "forward",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w) {        
            return logic_layer_cuda_forward(x, a, b, w);
        },
        "logic layer forward (CUDA)");
    m.def(
        "backward_w", [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor grad_y) {
            return logic_layer_cuda_backward_w(x, a, b, grad_y);
        },
        "logic layer backward w (CUDA)");
    m.def(
        "backward_x",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w, torch::Tensor grad_y, torch::Tensor given_x_indices_of_y_start, torch::Tensor given_x_indices_of_y) {
            return logic_layer_cuda_backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y);
        },
        "logic layer backward x (CUDA)");
    m.def(
        "eval",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor w) {
            return logic_layer_cuda_eval(x, a, b, w);
        },
        "logic layer eval (CUDA)");
    m.def(
        "tensor_packbits_cuda",
        [](torch::Tensor t, const int bit_count) {
            return tensor_packbits_cuda(t, bit_count);
        },
        "ltensor_packbits_cuda (CUDA)");
    m.def(
        "groupbitsum",
        [](torch::Tensor b, const int pad_len, const unsigned int k) {
            if (b.size(0) % k != 0) {
                throw py::value_error("in_dim (" + std::to_string(b.size(0)) + ") has to be divisible by k (" + std::to_string(k) + ") but it is not");
            }
            return groupbitsum(b, pad_len, k);
        },
        "groupbitsum (CUDA)");
    m.def(
        "weighted_groupbitsum",
        [](torch::Tensor b, const int pad_len, const unsigned int k, torch::Tensor weights) {
            if (b.size(0) % k != 0) {
                throw py::value_error("in_dim (" + std::to_string(b.size(0)) + ") must be divisible by k (" + std::to_string(k) + ")");
            }
            const int features_per_group = b.size(0) / k;
            if (weights.dim() != 2 || weights.size(0) != k || weights.size(1) != features_per_group) {
                throw py::value_error("Weights tensor has incorrect shape. Expected (" + std::to_string(k) + ", " + std::to_string(features_per_group) + "), but got (" + std::to_string(weights.size(0)) + ", " + std::to_string(weights.size(1)) + ")");
            }
            return weighted_groupbitsum(b, pad_len, k, weights);
        },
        "weighted_groupbitsum (CUDA)"
    );
    m.def(
        "pruned_groupbitsum",
        [](torch::Tensor b, const int pad_len, const unsigned int k, torch::Tensor group_sizes) {
            return pruned_groupbitsum(b, pad_len, k, group_sizes);
        },
        "pruned_groupbitsum with variable group sizes (CUDA)"
    );

    // ▼▼▼ Add the Python binding for pruned_weighted_groupbitsum ▼▼▼
    m.def(
        "pruned_weighted_groupbitsum",
        [](torch::Tensor b, const int pad_len, const unsigned int k, torch::Tensor group_sizes, torch::Tensor weights) {
            if (group_sizes.dim() != 1 || group_sizes.size(0) != k) {
                throw py::value_error("group_sizes must be a 1D tensor of size k (" + std::to_string(k) + ")");
            }
            const int64_t total_features = torch::sum(group_sizes).item<int64_t>();
            if (b.size(0) != total_features) {
                throw py::value_error("The sum of group_sizes (" + std::to_string(total_features) + ") must match the number of features in input tensor b (" + std::to_string(b.size(0)) + ")");
            }
            if (weights.dim() != 1 || weights.numel() != total_features) {
                throw py::value_error("weights must be a 1D tensor with a size equal to the total number of features (" + std::to_string(total_features) + ")");
            }
            return pruned_weighted_groupbitsum(b, pad_len, k, group_sizes, weights);
        },
        "Pruned and weighted groupbitsum with variable group sizes (CUDA)"
    );

    // === Fused LogicTree Conv + ORPool (Per-level version) ===
    // This is the current implementation used by FusedTreeConvORPoolLayer
    m.def(
        "logictree_forward_ablevels",
        [](const torch::Tensor& x_padded,
           const torch::Tensor& weights_L,
           const torch::Tensor& a_idx_L,
           const torch::Tensor& b_idx_L,
           const torch::Tensor& leaf_ici,
           const torch::Tensor& leaf_ipx,
           const torch::Tensor& leaf_ipy,
           const torch::Tensor& nodes_per_level,
           int out_h, int out_w,
           int kernel_size, int stride,
           int groups, int in_channels_per_group) {
            // 최소한의 검증 (CHECK_INPUT 매크로 쓰지 말고 TORCH_CHECK)
            TORCH_CHECK(x_padded.is_cuda(), "x_padded must be CUDA");
            TORCH_CHECK(weights_L.is_cuda() && a_idx_L.is_cuda() && b_idx_L.is_cuda(), "level tensors must be CUDA");
            TORCH_CHECK(leaf_ici.is_cuda() && leaf_ipx.is_cuda() && leaf_ipy.is_cuda(), "leaf tensors must be CUDA");
            TORCH_CHECK(nodes_per_level.is_cuda(), "nodes_per_level must be CUDA");
            return fused_forward_ablevels_cuda(
                x_padded, weights_L, a_idx_L, b_idx_L,
                leaf_ici, leaf_ipx, leaf_ipy, nodes_per_level,
                out_h, out_w, kernel_size, stride, groups, in_channels_per_group
            );
        },
        "Fused LogicTree+ORPool forward (per-level a,b indices; recomputes winners in backward)"
    );
    
    m.def(
        "logictree_backward_ablevels",
    [](const torch::Tensor& x_padded,
        const torch::Tensor& grad_out,
        const torch::Tensor& weights_L,
        const torch::Tensor& a_idx_L,
        const torch::Tensor& b_idx_L,
        const torch::Tensor& leaf_ici,
        const torch::Tensor& leaf_ipx,
        const torch::Tensor& leaf_ipy,
        const torch::Tensor& nodes_per_level,
        int out_h, int out_w,
        int kernel_size, int stride,
        int groups, int in_channels_per_group) {
            TORCH_CHECK(x_padded.is_cuda() && grad_out.is_cuda(), "x_padded/grad_out must be CUDA");
            return fused_backward_ablevels_cuda(
                x_padded, grad_out, weights_L, a_idx_L, b_idx_L,
                leaf_ici, leaf_ipx, leaf_ipy, nodes_per_level,
                out_h, out_w, kernel_size, stride, groups, in_channels_per_group
            );
        },
        "Fused LogicTree+ORPool backward (recomputes ORPool winner; returns grads for x and weights)"
    );
    
    m.def(
        "fused_tree_conv_forward",
        [](const torch::Tensor& x,
           const torch::Tensor& weights_L,
           const torch::Tensor& a_idx_L,
           const torch::Tensor& b_idx_L,
           const torch::Tensor& nodes_per_level) {
            TORCH_CHECK(x.is_cuda(), "x must be CUDA");
            TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
            TORCH_CHECK(weights_L.is_cuda() && a_idx_L.is_cuda() && b_idx_L.is_cuda(), "level tensors must be CUDA");
            TORCH_CHECK(nodes_per_level.is_cuda(), "nodes_per_level must be CUDA");
            return fused_tree_conv_forward_cuda(x, weights_L, a_idx_L, b_idx_L, nodes_per_level);
        },
        "Fused TreeConv forward (no ORPool, per-patch)"
    );
    
    m.def(
        "fused_tree_conv_backward",
        [](const torch::Tensor& x,
           const torch::Tensor& grad_out,
           const torch::Tensor& weights_L,
           const torch::Tensor& a_idx_L,
           const torch::Tensor& b_idx_L,
           const torch::Tensor& nodes_per_level) {
            TORCH_CHECK(x.is_cuda() && grad_out.is_cuda(), "x/grad_out must be CUDA");
            TORCH_CHECK(x.is_contiguous() && grad_out.is_contiguous(), "x/grad_out must be contiguous");
            return fused_tree_conv_backward_cuda(x, grad_out, weights_L, a_idx_L, b_idx_L, nodes_per_level);
        },
        "Fused TreeConv backward (no ORPool)"
    );
        
    // BlockEfficientCrossbarLayer bindings
    m.def(
        "block_efficient_crossbar_forward",
        [](torch::Tensor x, torch::Tensor w_sparse, int num_blocks, int block_size, int out_per_block) {
            TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
            TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
            TORCH_CHECK(w_sparse.is_cuda(), "w_sparse must be a CUDA tensor");
            TORCH_CHECK(w_sparse.is_contiguous(), "w_sparse must be contiguous");
            return block_efficient_crossbar_forward(x, w_sparse, num_blocks, block_size, out_per_block);
        },
        "BlockEfficientCrossbarLayer forward (CUDA)"
    );
    
    m.def(
        "block_efficient_crossbar_backward_w",
        [](torch::Tensor x, torch::Tensor grad_y, int num_blocks, int block_size, int out_per_block) {
            TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
            TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
            TORCH_CHECK(grad_y.is_cuda(), "grad_y must be a CUDA tensor");
            TORCH_CHECK(grad_y.is_contiguous(), "grad_y must be contiguous");
            return block_efficient_crossbar_backward_w(x, grad_y, num_blocks, block_size, out_per_block);
        },
        "BlockEfficientCrossbarLayer backward weights (CUDA)"
    );
    
    m.def(
        "block_efficient_crossbar_backward_x",
        [](torch::Tensor w_sparse, torch::Tensor grad_y, int num_blocks, int block_size, int out_per_block) {
            TORCH_CHECK(w_sparse.is_cuda(), "w_sparse must be a CUDA tensor");
            TORCH_CHECK(w_sparse.is_contiguous(), "w_sparse must be contiguous");
            TORCH_CHECK(grad_y.is_cuda(), "grad_y must be a CUDA tensor");
            TORCH_CHECK(grad_y.is_contiguous(), "grad_y must be contiguous");
            return block_efficient_crossbar_backward_x(w_sparse, grad_y, num_blocks, block_size, out_per_block);
        },
        "BlockEfficientCrossbarLayer backward input (CUDA)"
    );

    m.def(
        "logic_triple_forward",
        [](torch::Tensor x,
           torch::Tensor a1, torch::Tensor b1, torch::Tensor w1,
           torch::Tensor a2, torch::Tensor b2, torch::Tensor w2,
           torch::Tensor a3, torch::Tensor b3, torch::Tensor w3,
           int groups) {
            // Shared memory 체크 디버깅 코드
            const int64_t D1 = a1.size(0);
            const int64_t D2 = a2.size(0);
            const int64_t D1_g = D1 / groups;
            const int64_t D2_g = D2 / groups;
            const int64_t max_nodes = std::max(D1_g, D2_g);
            const size_t shmem = 2 * max_nodes * x.element_size();
            
            int dev;
            cudaGetDevice(&dev);
            int max_shmem_bytes = 0;
            cudaDeviceGetAttribute(&max_shmem_bytes,
                                   cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            if (max_shmem_bytes == 0) {
                cudaDeviceGetAttribute(&max_shmem_bytes,
                                       cudaDevAttrMaxSharedMemoryPerBlock, dev);
            }
            if (shmem > (size_t)max_shmem_bytes) {
                TORCH_CHECK(false,
                    "logic_triple_forward_cuda: required shared memory (", shmem,
                    ") > device limit (", max_shmem_bytes,
                    "). Reduce out_channels or fallback to unfused TreeConvLayer.");
            }
            
            return logic_triple_forward_cuda(
                x, a1, b1, w1, a2, b2, w2, a3, b3, w3, 
                groups
            );
        },
        "Fused triple LogicLayer forward (x -> h1 -> h2 -> y)"
    );
    m.def(
        "logic_triple_backward",
        [](torch::Tensor x, torch::Tensor grad_y,
           torch::Tensor a1, torch::Tensor b1, torch::Tensor w1,
           torch::Tensor a2, torch::Tensor b2, torch::Tensor w2,
           torch::Tensor a3, torch::Tensor b3, torch::Tensor w3,
           int groups) {
            // Shared memory 체크 디버깅 코드
            const int64_t D1 = a1.size(0);
            const int64_t D2 = a2.size(0);
            const int64_t D1_g = D1 / groups;
            const int64_t D2_g = D2 / groups;
            const size_t shmem = (D1_g + D2_g + D1_g + D2_g) * x.element_size();  // h1,h2,gh1,gh2
            
            int dev;
            cudaGetDevice(&dev);
            int max_shmem_bytes = 0;
            cudaDeviceGetAttribute(&max_shmem_bytes,
                                   cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            if (max_shmem_bytes == 0) {
                cudaDeviceGetAttribute(&max_shmem_bytes,
                                       cudaDevAttrMaxSharedMemoryPerBlock, dev);
            }
            if (shmem > (size_t)max_shmem_bytes) {
                TORCH_CHECK(false,
                    "logic_triple_backward_cuda: required shared memory (", shmem,
                    ") > device limit (", max_shmem_bytes,
                    "). Reduce out_channels or fallback to unfused TreeConvLayer.");
            }
            
            return logic_triple_backward_cuda(
                x, grad_y,
                a1, b1, w1,
                a2, b2, w2,
                a3, b3, w3,
                groups
            );
        },
        "Fused triple LogicLayer backward"
    );


}
