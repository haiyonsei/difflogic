import torch
import triton
import triton.language as tl
from typing import Optional

# ---------------------------------------------------------------------------
# 1. Training Mode Kernels (Forward, Backward)
# ---------------------------------------------------------------------------

@triton.jit
def logic_layer_forward_kernel(
    # Pointers to tensors
    X_ptr, Y_ptr, A_idx_ptr, B_idx_ptr, W_ptr,
    # Tensor dimensions
    in_size, batch_size, out_size,
    # Strides
    stride_x_0, stride_x_1,
    stride_y_0, stride_y_1,
    stride_a_0,
    stride_b_0,
    stride_w_0, stride_w_1,
    # Meta-parameters
    BLOCK_SIZE_OUT: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
):
    pid_out = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)

    offsets_out = pid_out * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    offsets_batch = pid_batch * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)

    mask_out = offsets_out < out_size
    mask_batch = offsets_batch < batch_size

    a_indices = tl.load(A_idx_ptr + offsets_out * stride_a_0, mask=mask_out)
    b_indices = tl.load(B_idx_ptr + offsets_out * stride_b_0, mask=mask_out)

    a_ = tl.load(X_ptr + a_indices[:, None] * stride_x_0 + offsets_batch[None, :] * stride_x_1, 
                 mask=mask_out[:, None] & mask_batch[None, :], other=0.0)
    b_ = tl.load(X_ptr + b_indices[:, None] * stride_x_0 + offsets_batch[None, :] * stride_x_1,
                 mask=mask_out[:, None] & mask_batch[None, :], other=0.0)

    # --- MODIFICATION START ---
    # Load each required weight column explicitly to avoid 2D slicing issues.
    # Each wN will be a 1D block of shape (BLOCK_SIZE_OUT,).
    # We need to expand their dimensions for broadcasting with a_ and b_.
    w1 = tl.load(W_ptr + offsets_out * stride_w_0 + 1 * stride_w_1, mask=mask_out)[:, None]
    w2 = tl.load(W_ptr + offsets_out * stride_w_0 + 2 * stride_w_1, mask=mask_out)[:, None]
    w3 = tl.load(W_ptr + offsets_out * stride_w_0 + 3 * stride_w_1, mask=mask_out)[:, None]
    w4 = tl.load(W_ptr + offsets_out * stride_w_0 + 4 * stride_w_1, mask=mask_out)[:, None]
    w5 = tl.load(W_ptr + offsets_out * stride_w_0 + 5 * stride_w_1, mask=mask_out)[:, None]
    w6 = tl.load(W_ptr + offsets_out * stride_w_0 + 6 * stride_w_1, mask=mask_out)[:, None]
    w7 = tl.load(W_ptr + offsets_out * stride_w_0 + 7 * stride_w_1, mask=mask_out)[:, None]
    w8 = tl.load(W_ptr + offsets_out * stride_w_0 + 8 * stride_w_1, mask=mask_out)[:, None]
    w9 = tl.load(W_ptr + offsets_out * stride_w_0 + 9 * stride_w_1, mask=mask_out)[:, None]
    w10 = tl.load(W_ptr + offsets_out * stride_w_0 + 10 * stride_w_1, mask=mask_out)[:, None]
    w11 = tl.load(W_ptr + offsets_out * stride_w_0 + 11 * stride_w_1, mask=mask_out)[:, None]
    w12 = tl.load(W_ptr + offsets_out * stride_w_0 + 12 * stride_w_1, mask=mask_out)[:, None]
    w13 = tl.load(W_ptr + offsets_out * stride_w_0 + 13 * stride_w_1, mask=mask_out)[:, None]
    w14 = tl.load(W_ptr + offsets_out * stride_w_0 + 14 * stride_w_1, mask=mask_out)[:, None]
    w15 = tl.load(W_ptr + offsets_out * stride_w_0 + 15 * stride_w_1, mask=mask_out)[:, None]
 
    a_mul_b = a_ * b_
    
    # Use the individually loaded weight columns in the calculation.
    y = (
        (w1 * a_mul_b + w2 * (a_ - a_mul_b)) +
        (w3 * a_ + w4 * (b_ - a_mul_b)) +
        (w5 * b_ + w6 * (a_ + b_ - 2.0 * a_mul_b)) +
        (w7 * (a_ + b_ - a_mul_b) + w8 * (1.0 - (a_ + b_ - a_mul_b))) +
        (w9 * (1.0 - (a_ + b_ - 2.0 * a_mul_b)) + w10 * (1.0 - b_)) +
        (w11 * (1.0 - b_ + a_mul_b) + w12 * (1.0 - a_)) +
        (w13 * (1.0 - a_ + a_mul_b) + w14 * (1.0 - a_mul_b)) +
        w15
    )
    # --- MODIFICATION END ---
    
    y_ptrs = Y_ptr + offsets_out[:, None] * stride_y_0 + offsets_batch[None, :] * stride_y_1
    tl.store(y_ptrs, y, mask=mask_out[:, None] & mask_batch[None, :])

@triton.jit
def logic_layer_backward_w_kernel(
    X_ptr, A_idx_ptr, B_idx_ptr, GradY_ptr, GradW_ptr,
    in_size, batch_size, out_size,
    stride_x_0, stride_x_1,
    stride_a_0,
    stride_b_0,
    stride_grady_0, stride_grady_1,
    stride_gradw_0, stride_gradw_1,
    BLOCK_SIZE_BATCH: tl.constexpr,
):
    pid_out = tl.program_id(axis=0) 
    
    idx_a = tl.load(A_idx_ptr + pid_out * stride_a_0)
    idx_b = tl.load(B_idx_ptr + pid_out * stride_b_0)
    
    grad_w_ab = 0.0
    grad_w_a = 0.0
    grad_w_b = 0.0
    grad_w_const = 0.0
    
    for batch_start in range(0, batch_size, BLOCK_SIZE_BATCH):
        offsets_batch = batch_start + tl.arange(0, BLOCK_SIZE_BATCH)
        mask_batch = offsets_batch < batch_size
        
        a_ = tl.load(X_ptr + idx_a * stride_x_0 + offsets_batch * stride_x_1, mask=mask_batch, other=0.0)
        b_ = tl.load(X_ptr + idx_b * stride_x_0 + offsets_batch * stride_x_1, mask=mask_batch, other=0.0)
        grad_y_ = tl.load(GradY_ptr + pid_out * stride_grady_0 + offsets_batch * stride_grady_1, mask=mask_batch, other=0.0)
        
        # --- MODIFICATION START ---
        # Add axis=0 to all tl.sum calls
        grad_w_ab += tl.sum((a_ * b_) * grad_y_, axis=0)
        grad_w_a += tl.sum(a_ * grad_y_, axis=0)
        grad_w_b += tl.sum(b_ * grad_y_, axis=0)
        grad_w_const += tl.sum(grad_y_, axis=0)
        # --- MODIFICATION END ---

    # The rest of the kernel remains the same
    grad_w = tl.full((16,), 0.0, dtype=tl.float32)
    grad_w = tl.where(tl.arange(0, 16) == 1, grad_w_ab, grad_w)
    grad_w = tl.where(tl.arange(0, 16) == 2, grad_w_a - grad_w_ab, grad_w)
    grad_w = tl.where(tl.arange(0, 16) == 3, grad_w_a, grad_w)
    grad_w = tl.where(tl.arange(0, 16) == 4, grad_w_b - grad_w_ab, grad_w)
    grad_w = tl.where(tl.arange(0, 16) == 5, grad_w_b, grad_w)
    grad_w = tl.where(tl.arange(0, 16) == 6, grad_w_a + grad_w_b - 2 * grad_w_ab, grad_w)
    grad_w = tl.where(tl.arange(0, 16) == 7, grad_w_a + grad_w_b - grad_w_ab, grad_w)
    grad_w = tl.where(tl.arange(0, 16) == 8, grad_w_const - (grad_w_a + grad_w_b - grad_w_ab), grad_w)
    grad_w = tl.where(tl.arange(0, 16) == 9, grad_w_const - (grad_w_a + grad_w_b - 2 * grad_w_ab), grad_w)
    grad_w = tl.where(tl.arange(0, 16) == 10, grad_w_const - grad_w_b, grad_w)
    grad_w = tl.where(tl.arange(0, 16) == 11, grad_w_const - grad_w_b + grad_w_ab, grad_w)
    grad_w = tl.where(tl.arange(0, 16) == 12, grad_w_const - grad_w_a, grad_w)
    grad_w = tl.where(tl.arange(0, 16) == 13, grad_w_const - grad_w_a + grad_w_ab, grad_w)
    grad_w = tl.where(tl.arange(0, 16) == 14, grad_w_const - grad_w_ab, grad_w)
    grad_w = tl.where(tl.arange(0, 16) == 15, grad_w_const, grad_w)
    
    gradw_ptrs = GradW_ptr + pid_out * stride_gradw_0 + tl.arange(0, 16) * stride_gradw_1
    tl.store(gradw_ptrs, grad_w)

@triton.jit
def logic_layer_backward_x_kernel(
    X_ptr, W_ptr, A_idx_ptr, B_idx_ptr, GradY_ptr, GradX_ptr,
    in_size, batch_size, out_size,
    stride_x_0, stride_x_1,
    stride_w_0, stride_w_1,
    stride_a_0,
    stride_b_0,
    stride_grady_0, stride_grady_1,
    stride_gradx_0, stride_gradx_1,
    BLOCK_SIZE_BATCH: tl.constexpr,
):
    pid_out = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)

    idx_a = tl.load(A_idx_ptr + pid_out * stride_a_0)
    idx_b = tl.load(B_idx_ptr + pid_out * stride_b_0)
    
    a_ = tl.load(X_ptr + idx_a * stride_x_0 + pid_batch * stride_x_1)
    b_ = tl.load(X_ptr + idx_b * stride_x_0 + pid_batch * stride_x_1)

    # --- MODIFICATION START ---
    # Load each required weight individually to avoid 1D slicing issues.
    w1 = tl.load(W_ptr + pid_out * stride_w_0 + 1 * stride_w_1)
    w2 = tl.load(W_ptr + pid_out * stride_w_0 + 2 * stride_w_1)
    w3 = tl.load(W_ptr + pid_out * stride_w_0 + 3 * stride_w_1)
    w4 = tl.load(W_ptr + pid_out * stride_w_0 + 4 * stride_w_1)
    w5 = tl.load(W_ptr + pid_out * stride_w_0 + 5 * stride_w_1)
    w6 = tl.load(W_ptr + pid_out * stride_w_0 + 6 * stride_w_1)
    w7 = tl.load(W_ptr + pid_out * stride_w_0 + 7 * stride_w_1)
    w8 = tl.load(W_ptr + pid_out * stride_w_0 + 8 * stride_w_1)
    w9 = tl.load(W_ptr + pid_out * stride_w_0 + 9 * stride_w_1)
    w10 = tl.load(W_ptr + pid_out * stride_w_0 + 10 * stride_w_1)
    w11 = tl.load(W_ptr + pid_out * stride_w_0 + 11 * stride_w_1)
    w12 = tl.load(W_ptr + pid_out * stride_w_0 + 12 * stride_w_1)
    w13 = tl.load(W_ptr + pid_out * stride_w_0 + 13 * stride_w_1)
    w14 = tl.load(W_ptr + pid_out * stride_w_0 + 14 * stride_w_1)
    # --- MODIFICATION END ---
    
    grad_y_ = tl.load(GradY_ptr + pid_out * stride_grady_0 + pid_batch * stride_grady_1)

    # Use the individually loaded weights in the calculation.
    dy_da = (
        (w1 * b_ + w2 * (1.0 - b_) + w3) +
        (w4 * -b_ + w6 * (1.0 - 2.0 * b_) + w7 * (1.0 - b_)) +
        (w8 * (b_ - 1.0) + w9 * (2.0 * b_ - 1.0) + w11 * b_) +
        (-w12 + w13 * (b_ - 1.0) + w14 * -b_)
    )

    dy_db = (
        (w1 * a_ + w2 * -a_ + w4 * (1.0 - a_)) +
        (w5 + w6 * (1.0 - 2.0 * a_) + w7 * (1.0 - a_)) +
        (w8 * (a_ - 1.0) + w9 * (2.0 * a_ - 1.0) - w10) +
        (w11 * (a_ - 1.0) + w13 * a_ + w14 * -a_)
    )
    
    grad_contribution_a = grad_y_ * dy_da
    grad_contribution_b = grad_y_ * dy_db
    
    grad_x_ptr_a = GradX_ptr + idx_a * stride_gradx_0 + pid_batch * stride_gradx_1
    tl.atomic_add(grad_x_ptr_a, grad_contribution_a)
    
    grad_x_ptr_b = GradX_ptr + idx_b * stride_gradx_0 + pid_batch * stride_gradx_1
    tl.atomic_add(grad_x_ptr_b, grad_contribution_b)


# ---------------------------------------------------------------------------
# 2. Inference Mode Kernel
# ---------------------------------------------------------------------------

@triton.jit
def logic_layer_eval_kernel(
    X_ptr, Y_ptr, A_idx_ptr, B_idx_ptr, W_ptr,
    in_size, batch_size, out_size,
    stride_x_0, stride_x_1,
    stride_y_0, stride_y_1,
    stride_a_0,
    stride_b_0,
    stride_w_0,
    # Meta-parameters
    BLOCK_SIZE_OUT: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
):
    """
    Triton kernel for the inference pass (eval). Operates on integer types.
    """
    pid_out = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)

    offsets_out = pid_out * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    offsets_batch = pid_batch * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)

    mask_out = offsets_out < out_size
    mask_batch = offsets_batch < batch_size

    a_indices = tl.load(A_idx_ptr + offsets_out * stride_a_0, mask=mask_out)
    b_indices = tl.load(B_idx_ptr + offsets_out * stride_b_0, mask=mask_out)

    a_ = tl.load(X_ptr + a_indices[:, None] * stride_x_0 + offsets_batch[None, :] * stride_x_1,
                 mask=mask_out[:, None] & mask_batch[None, :], other=0)
    b_ = tl.load(X_ptr + b_indices[:, None] * stride_x_0 + offsets_batch[None, :] * stride_x_1,
                 mask=mask_out[:, None] & mask_batch[None, :], other=0)
    
    # w contains the operator index (0-15)
    op_idx = tl.load(W_ptr + offsets_out * stride_w_0, mask=mask_out)

    # Perform the selected bitwise operation
    # Note: Triton JIT does not have a switch statement. We use tl.where chains.
    # The compiler is smart enough to optimize this.
    y = tl.zeros(a_.shape, dtype=a_.dtype)
    y = tl.where(op_idx[:, None] == 0, 0, y)
    y = tl.where(op_idx[:, None] == 1, a_ & b_, y)
    y = tl.where(op_idx[:, None] == 2, a_ & ~b_, y)
    y = tl.where(op_idx[:, None] == 3, a_, y)
    y = tl.where(op_idx[:, None] == 4, ~a_ & b_, y)
    y = tl.where(op_idx[:, None] == 5, b_, y)
    y = tl.where(op_idx[:, None] == 6, a_ ^ b_, y)
    y = tl.where(op_idx[:, None] == 7, a_ | b_, y)
    y = tl.where(op_idx[:, None] == 8, ~(a_ | b_), y)
    y = tl.where(op_idx[:, None] == 9, ~(a_ ^ b_), y)
    y = tl.where(op_idx[:, None] == 10, ~b_, y)
    y = tl.where(op_idx[:, None] == 11, a_ | ~b_, y)
    y = tl.where(op_idx[:, None] == 12, ~a_, y)
    y = tl.where(op_idx[:, None] == 13, ~a_ | b_, y)
    y = tl.where(op_idx[:, None] == 14, ~(a_ & b_), y)
    y = tl.where(op_idx[:, None] == 15, -1, y) # ~0 in 2's complement is -1

    y_ptrs = Y_ptr + offsets_out[:, None] * stride_y_0 + offsets_batch[None, :] * stride_y_1
    tl.store(y_ptrs, y, mask=mask_out[:, None] & mask_batch[None, :])


# ---------------------------------------------------------------------------
# 3. Utility Kernels (Packbits, GroupBitSum)
# ---------------------------------------------------------------------------

@triton.jit
def tensor_packbits_kernel(
    T_ptr, B_ptr,
    in_rows, in_cols,
    out_rows, out_cols,
    stride_t_0, stride_t_1,
    stride_b_0, stride_b_1,
    BIT_COUNT: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    """
    Packs a boolean tensor into an integer tensor.
    """
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    offsets_row = pid_row * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_col = pid_col * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    
    mask_row = offsets_row < out_rows
    mask_col = offsets_col < out_cols
    
    # Initialize packed value accumulator
    packed_val = tl.zeros((BLOCK_SIZE_ROW, BLOCK_SIZE_COL), dtype=B_ptr.dtype.element_ty)
    
    # Iterate through bits to pack
    for i in range(BIT_COUNT):
        t_col_idx = (offsets_col[None, :] * BIT_COUNT) + i
        mask_t_col = t_col_idx < in_cols
        
        # Load boolean value
        t_ptr = T_ptr + offsets_row[:, None] * stride_t_0 + t_col_idx * stride_t_1
        bit = tl.load(t_ptr, mask=mask_row[:, None] & mask_t_col, other=False).to(packed_val.dtype)
        
        # Shift bit into position and OR with accumulator
        packed_val |= (bit << i)

    # Store the packed integer
    b_ptr = B_ptr + offsets_row[:, None] * stride_b_0 + offsets_col[None, :] * stride_b_1
    tl.store(b_ptr, packed_val, mask=mask_row[:, None] & mask_col[None, :])


@triton.jit
def groupbitsum_kernel(
    B_ptr, T_ptr,
    in_rows, in_cols,
    out_rows, out_cols,
    stride_b_0, stride_b_1,
    stride_t_0, stride_t_1,
    BIT_COUNT: tl.constexpr,
    CLASS_SIZE: tl.constexpr,
    BLOCK_SIZE_OUT_ROW: tl.constexpr,
    BLOCK_SIZE_OUT_COL: tl.constexpr,
):
    """
    Unpacks bits from an integer tensor and sums them in groups.
    """
    pid_row = tl.program_id(axis=0) # out_rows
    pid_col = tl.program_id(axis=1) # out_cols

    offsets_out_row = pid_row * BLOCK_SIZE_OUT_ROW + tl.arange(0, BLOCK_SIZE_OUT_ROW)
    offsets_out_col = pid_col * BLOCK_SIZE_OUT_COL + tl.arange(0, BLOCK_SIZE_OUT_COL)

    mask_out_row = offsets_out_row < out_rows
    mask_out_col = offsets_out_col < out_cols
    
    # Determine the source column in `b` and the bit position
    b_col_idx = offsets_out_col[None, :] // BIT_COUNT
    bit_pos = offsets_out_col[None, :] % BIT_COUNT
    bit_mask = (1 << bit_pos).to(B_ptr.dtype.element_ty)

    # Initialize sum accumulator
    sum_val = tl.zeros((BLOCK_SIZE_OUT_ROW, BLOCK_SIZE_OUT_COL), dtype=T_ptr.dtype.element_ty)
    
    # Iterate through the elements of a class to sum their bits
    for i in range(CLASS_SIZE):
        b_row_idx = offsets_out_row[:, None] * CLASS_SIZE + i
        
        # Load the packed integer value
        b_ptr = B_ptr + b_row_idx * stride_b_0 + b_col_idx * stride_b_1
        packed_val = tl.load(b_ptr, mask=mask_out_row[:, None] & mask_out_col[None, :], other=0)
        
        # Check if the bit is set and add to sum
        is_bit_set = (packed_val & bit_mask) != 0
        sum_val += is_bit_set.to(sum_val.dtype)

    # Store the result
    t_ptr = T_ptr + offsets_out_row[:, None] * stride_t_0 + offsets_out_col[None, :] * stride_t_1
    tl.store(t_ptr, sum_val, mask=mask_out_row[:, None] & mask_out_col[None, :])

# ---------------------------------------------------------------------------
# 4. Python Wrapper Functions
# ---------------------------------------------------------------------------

def check_input(x):
    assert x.is_cuda, f"{x} must be a CUDA tensor"
    assert x.is_contiguous(), f"{x} must be contiguous"

def logic_layer_forward_triton(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    check_input(x); check_input(a); check_input(b); check_input(w)
    in_size, batch_size = x.shape
    out_size = w.shape[0]
    
    y = torch.empty((out_size, batch_size), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (
        triton.cdiv(out_size, META['BLOCK_SIZE_OUT']),
        triton.cdiv(batch_size, META['BLOCK_SIZE_BATCH'])
    )
    
    logic_layer_forward_kernel[grid](
        x, y, a, b, w,
        in_size, batch_size, out_size,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        a.stride(0),
        b.stride(0),
        w.stride(0), w.stride(1),
        BLOCK_SIZE_OUT=32, BLOCK_SIZE_BATCH=32,
    )
    return y

def logic_layer_backward_w_triton(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, grad_y: torch.Tensor) -> torch.Tensor:
    check_input(x); check_input(a); check_input(b); check_input(grad_y)
    in_size, batch_size = x.shape
    out_size = grad_y.shape[0]

    grad_w = torch.empty((out_size, 16), device=x.device, dtype=x.dtype)
    
    grid = (out_size,) # One program per output neuron
    
    logic_layer_backward_w_kernel[grid](
        x, a, b, grad_y, grad_w,
        in_size, batch_size, out_size,
        x.stride(0), x.stride(1),
        a.stride(0),
        b.stride(0),
        grad_y.stride(0), grad_y.stride(1),
        grad_w.stride(0), grad_w.stride(1),
        BLOCK_SIZE_BATCH=256,
    )
    return grad_w

def logic_layer_backward_x_triton(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, grad_y: torch.Tensor) -> torch.Tensor:
    check_input(x); check_input(a); check_input(b); check_input(w); check_input(grad_y)
    in_size, batch_size = x.shape
    out_size = grad_y.shape[0]

    # grad_x is initialized to zeros for the atomic adds
    grad_x = torch.zeros_like(x)
    
    # Grid is over the grad_y tensor dimensions
    grid = (out_size, batch_size)

    logic_layer_backward_x_kernel[grid](
        x, w, a, b, grad_y, grad_x,
        in_size, batch_size, out_size,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        a.stride(0),
        b.stride(0),
        grad_y.stride(0), grad_y.stride(1),
        grad_x.stride(0), grad_x.stride(1),
        BLOCK_SIZE_BATCH=1, # Not used, but required by signature
    )
    return grad_x

def logic_layer_eval_triton(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    check_input(x); check_input(a); check_input(b); check_input(w)
    in_size, batch_size = x.shape
    out_size = w.shape[0]

    y = torch.empty((out_size, batch_size), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(out_size, META['BLOCK_SIZE_OUT']),
        triton.cdiv(batch_size, META['BLOCK_SIZE_BATCH'])
    )

    logic_layer_eval_kernel[grid](
        x, y, a, b, w,
        in_size, batch_size, out_size,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        a.stride(0),
        b.stride(0),
        w.stride(0),
        BLOCK_SIZE_OUT=32, BLOCK_SIZE_BATCH=32,
    )
    return y

def tensor_packbits_triton(t: torch.Tensor, bit_count: int) -> tuple[torch.Tensor, int]:
    check_input(t)
    assert t.dtype == torch.bool, "Input tensor must be of type bool"
    assert bit_count in [8, 16, 32, 64], "`bit_count` has to be in {8, 16, 32, 64}"
    
    dtype_map = {8: torch.int8, 16: torch.int16, 32: torch.int32, 64: torch.int64}
    out_dtype = dtype_map[bit_count]

    in_rows, in_cols = t.shape
    out_cols = triton.cdiv(in_cols, bit_count)
    pad_len = (out_cols * bit_count - in_cols) % bit_count

    b = torch.zeros((in_rows, out_cols), device=t.device, dtype=out_dtype)
    
    grid = lambda META: (
        triton.cdiv(in_rows, META['BLOCK_SIZE_ROW']),
        triton.cdiv(out_cols, META['BLOCK_SIZE_COL'])
    )

    tensor_packbits_kernel[grid](
        t, b,
        in_rows, in_cols,
        in_rows, out_cols,
        t.stride(0), t.stride(1),
        b.stride(0), b.stride(1),
        BIT_COUNT=bit_count,
        BLOCK_SIZE_ROW=32,
        BLOCK_SIZE_COL=32,
    )
    return b, pad_len

def groupbitsum_triton(b: torch.Tensor, pad_len: int, k: int) -> torch.Tensor:
    check_input(b)
    in_rows, in_cols_packed = b.shape
    bit_count = b.element_size() * 8
    
    assert in_rows % k == 0, f"in_dim ({in_rows}) has to be divisible by k ({k})"
    
    out_rows = k
    out_cols = in_cols_packed * bit_count - pad_len
    class_size = in_rows // k
    
    t = torch.empty((out_rows, out_cols), device=b.device, dtype=torch.int32)
    
    grid = lambda META: (
        triton.cdiv(out_rows, META['BLOCK_SIZE_OUT_ROW']),
        triton.cdiv(out_cols, META['BLOCK_SIZE_OUT_COL'])
    )
    
    groupbitsum_kernel[grid](
        b, t,
        in_rows, in_cols_packed,
        out_rows, out_cols,
        b.stride(0), b.stride(1),
        t.stride(0), t.stride(1),
        BIT_COUNT=bit_count,
        CLASS_SIZE=class_size,
        BLOCK_SIZE_OUT_ROW=32,
        BLOCK_SIZE_OUT_COL=32,
    )
    return t.transpose(0, 1).contiguous()






@triton.jit
def pruned_weighted_groupbitsum_kernel(
    B_ptr, T_ptr, W_ptr,
    Group_start_indices_ptr,
    in_rows, out_cols,
    stride_b_0, stride_b_1,
    stride_t_0, stride_t_1,
    stride_w_0,
    BIT_COUNT: tl.constexpr,
    BLOCK_SIZE_OUT_COL: tl.constexpr,
    IS_WEIGHTED: tl.constexpr,
):
    pid_k = tl.program_id(axis=0)       # out_rows (k)
    pid_batch = tl.program_id(axis=1)   # out_cols

    offsets_batch = pid_batch * BLOCK_SIZE_OUT_COL + tl.arange(0, BLOCK_SIZE_OUT_COL)
    mask_batch = offsets_batch < out_cols

    b_col_idx = offsets_batch // BIT_COUNT
    bit_pos = offsets_batch % BIT_COUNT
    bit_mask = (1 << bit_pos).to(B_ptr.dtype.element_ty)

    sum_val = tl.zeros((BLOCK_SIZE_OUT_COL,), dtype=T_ptr.dtype.element_ty)
    
    start_idx = tl.load(Group_start_indices_ptr + pid_k)
    end_idx = tl.load(Group_start_indices_ptr + pid_k + 1)
    
    for i in range(start_idx, end_idx):
        b_row_idx = i
        
        b_ptr = B_ptr + b_row_idx * stride_b_0 + b_col_idx * stride_b_1
        packed_val = tl.load(b_ptr, mask=mask_batch, other=0)
        
        is_bit_set = (packed_val & bit_mask) != 0
        
        if IS_WEIGHTED:
            weight = tl.load(W_ptr + i * stride_w_0)
            sum_val += is_bit_set * weight
        else:
            sum_val += is_bit_set.to(sum_val.dtype)

    t_ptr = T_ptr + pid_k * stride_t_0 + offsets_batch * stride_t_1
    tl.store(t_ptr, sum_val, mask=mask_batch)

def pruned_weighted_groupbitsum_triton(b: torch.Tensor, k: int, group_sizes: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    in_rows, in_cols_packed = b.shape
    bit_count = b.element_size() * 8
    
    out_rows = k
    out_cols = in_cols_packed * bit_count # Assuming no padding for simplicity here
    
    # Create start indices for groups
    group_start_indices = torch.cat([torch.tensor([0], device=b.device), torch.cumsum(group_sizes, dim=0)]).to(torch.int32)

    t = torch.empty((out_rows, out_cols), device=b.device, dtype=torch.float32)

    grid = (k, triton.cdiv(out_cols, 1024))
    
    is_weighted = weights is not None
    if not is_weighted:
        # Create a dummy tensor for weights if not provided
        weights = torch.empty(0, device=b.device, dtype=torch.float32)

    pruned_weighted_groupbitsum_kernel[grid](
        b, t, weights,
        group_start_indices,
        in_rows, out_cols,
        b.stride(0), b.stride(1),
        t.stride(0), t.stride(1),
        weights.stride(0) if is_weighted else 0,
        BIT_COUNT=bit_count,
        BLOCK_SIZE_OUT_COL=1024,
        IS_WEIGHTED=is_weighted,
    )
    return t.transpose(0, 1).contiguous()



@triton.jit
def fused_logictree_conv_orpool_kernel(
    # Tensors
    X_padded_ptr, Y_ptr, Weights_ptr,
    Input_channel_indices_ptr, Input_pos_x_indices_ptr, Input_pos_y_indices_ptr,
    # Dimensions
    batch_size, in_channels, height, width, 
    out_channels, out_h_pool, out_w_pool, out_h_conv, out_w_conv,
    # Conv parameters
    kernel_size, stride, padding, groups, in_channels_per_group,
    # Strides for tensor access
    stride_xp_b, stride_xp_c, stride_xp_h, stride_xp_w,
    stride_y_b, stride_y_c, stride_y_h, stride_y_w,
    stride_w_oc, stride_w_g, stride_w_op,
    stride_ici_oc, stride_ici_inp,
    stride_ipx_oc, stride_ipx_inp,
    stride_ipy_oc, stride_ipy_inp,
    # Constants
    BLOCK_SIZE_W: tl.constexpr,
    NUM_W_BLOCKS: tl.constexpr,
):
    # 3D Grid Setup
    pid_b = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_spatial = tl.program_id(axis=2)

    # Recover original H, W block IDs from flattened spatial ID
    pid_h_out = pid_spatial // NUM_W_BLOCKS
    pid_w_block = pid_spatial % NUM_W_BLOCKS
    
    offsets_w_out = pid_w_block * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Base mask for pooled output width
    w_out_mask = offsets_w_out < out_w_pool

    # Early exit for padded grid rows
    if pid_h_out >= out_h_pool:
        return

    max_val = tl.zeros((BLOCK_SIZE_W,), dtype=X_padded_ptr.dtype.element_ty) - 1e9
    SHOULD_PRINT = False
    #(pid_b == 0 and pid_k == 0 and pid_h_out == 0 and pid_w_block == 0)
    #if SHOULD_PRINT:
    #    triton.device_print("\n\n--- TRITON DEBUG START (Layer 0, B0, K0, H_pool0, W_pool0) ---")

    # 2x2 pooling is done by these outer loops
    for h_offset in range(2):
        for w_offset in range(2):
            # Calculate convolution output coordinates
            conv_h = pid_h_out * 2 + h_offset
            
            # Use `if` block instead of `continue` for height boundary check
            if conv_h < out_h_conv:
                conv_w = offsets_w_out * 2 + w_offset
                
                # Create additional mask for convolution output width
                conv_w_mask = conv_w < out_w_conv
                # The final load mask must satisfy both conditions
                final_load_mask = w_out_mask & conv_w_mask

                patch_start_h = conv_h * stride
                patch_start_w = conv_w * stride
                group_idx = (pid_k * groups) // out_channels
                group_in_start = group_idx * in_channels_per_group
                
                # Base pointers for indices and data
                ici_base_ptr = Input_channel_indices_ptr + pid_k * stride_ici_oc
                ipx_base_ptr = Input_pos_x_indices_ptr + pid_k * stride_ipx_oc
                ipy_base_ptr = Input_pos_y_indices_ptr + pid_k * stride_ipy_oc
                b_ptr_base = X_padded_ptr + pid_b * stride_xp_b

                # --- DEBUG START ---
               # if SHOULD_PRINT:
                #    triton.device_print("Processing conv (h,w): ", conv_h, w_offset) # conv_w는 벡터이므로 w_offset으로 위치 특정


                # Individually load indices and then data for all 8 inputs
                # Input 0
                abs_y_0 = tl.load(ipy_base_ptr + 0 * stride_ipy_inp) + patch_start_h; 
                abs_x_0 = tl.load(ipx_base_ptr + 0 * stride_ipx_inp) + patch_start_w; 
                idx_c_0 = tl.load(ici_base_ptr + 0 * stride_ici_inp) + group_in_start
                ptr_0 = b_ptr_base + idx_c_0 * stride_xp_c + abs_y_0 * stride_xp_h + abs_x_0 * stride_xp_w; input_0 = tl.load(ptr_0, mask=final_load_mask, other=0.0)
                
                # Input 1
                abs_y_1 = tl.load(ipy_base_ptr + 1 * stride_ipy_inp) + patch_start_h; 
                abs_x_1 = tl.load(ipx_base_ptr + 1 * stride_ipx_inp) + patch_start_w; 
                idx_c_1 = tl.load(ici_base_ptr + 1 * stride_ici_inp) + group_in_start
                ptr_1 = b_ptr_base + idx_c_1 * stride_xp_c + abs_y_1 * stride_xp_h + abs_x_1 * stride_xp_w; input_1 = tl.load(ptr_1, mask=final_load_mask, other=0.0)

                # ... (나머지 6개 입력에 대해서도 동일하게 적용)
                abs_y_2 = tl.load(ipy_base_ptr + 2 * stride_ipy_inp) + patch_start_h; 
                abs_x_2 = tl.load(ipx_base_ptr + 2 * stride_ipx_inp) + patch_start_w; 
                idx_c_2 = tl.load(ici_base_ptr + 2 * stride_ici_inp) + group_in_start
                ptr_2 = b_ptr_base + idx_c_2 * stride_xp_c + abs_y_2 * stride_xp_h + abs_x_2 * stride_xp_w; input_2 = tl.load(ptr_2, mask=final_load_mask, other=0.0)

                abs_y_3 = tl.load(ipy_base_ptr + 3 * stride_ipy_inp) + patch_start_h; 
                abs_x_3 = tl.load(ipx_base_ptr + 3 * stride_ipx_inp) + patch_start_w; 
                idx_c_3 = tl.load(ici_base_ptr + 3 * stride_ici_inp) + group_in_start
                ptr_3 = b_ptr_base + idx_c_3 * stride_xp_c + abs_y_3 * stride_xp_h + abs_x_3 * stride_xp_w; input_3 = tl.load(ptr_3, mask=final_load_mask, other=0.0)

                abs_y_4 = tl.load(ipy_base_ptr + 4 * stride_ipy_inp) + patch_start_h; 
                abs_x_4 = tl.load(ipx_base_ptr + 4 * stride_ipx_inp) + patch_start_w; 
                idx_c_4 = tl.load(ici_base_ptr + 4 * stride_ici_inp) + group_in_start
                ptr_4 = b_ptr_base + idx_c_4 * stride_xp_c + abs_y_4 * stride_xp_h + abs_x_4 * stride_xp_w; input_4 = tl.load(ptr_4, mask=final_load_mask, other=0.0)

                abs_y_5 = tl.load(ipy_base_ptr + 5 * stride_ipy_inp) + patch_start_h; 
                abs_x_5 = tl.load(ipx_base_ptr + 5 * stride_ipx_inp) + patch_start_w; 
                idx_c_5 = tl.load(ici_base_ptr + 5 * stride_ici_inp) + group_in_start
                ptr_5 = b_ptr_base + idx_c_5 * stride_xp_c + abs_y_5 * stride_xp_h + abs_x_5 * stride_xp_w; input_5 = tl.load(ptr_5, mask=final_load_mask, other=0.0)

                abs_y_6 = tl.load(ipy_base_ptr + 6 * stride_ipy_inp) + patch_start_h; 
                abs_x_6 = tl.load(ipx_base_ptr + 6 * stride_ipx_inp) + patch_start_w; 
                idx_c_6 = tl.load(ici_base_ptr + 6 * stride_ici_inp) + group_in_start
                ptr_6 = b_ptr_base + idx_c_6 * stride_xp_c + abs_y_6 * stride_xp_h + abs_x_6 * stride_xp_w; input_6 = tl.load(ptr_6, mask=final_load_mask, other=0.0)

                abs_y_7 = tl.load(ipy_base_ptr + 7 * stride_ipy_inp) + patch_start_h; 
                abs_x_7 = tl.load(ipx_base_ptr + 7 * stride_ipx_inp) + patch_start_w; 
                idx_c_7 = tl.load(ici_base_ptr + 7 * stride_ici_inp) + group_in_start
                ptr_7 = b_ptr_base + idx_c_7 * stride_xp_c + abs_y_7 * stride_xp_h + abs_x_7 * stride_xp_w; input_7 = tl.load(ptr_7, mask=final_load_mask, other=0.0)

                # ▲▲▲ 수정된 부분 ▲▲▲

                '''
                                # --- DEBUG START ---
                if SHOULD_PRINT:
                    # BLOCK_SIZE_W 벡터 전체를 출력합니다.
                    triton.device_print("[Triton] Input 0 Block:", input_0)
                    triton.device_print("[Triton] Input 1 Block:", input_1)
                    triton.device_print("[Triton] Input 2 Block:", input_2)
                    triton.device_print("[Triton] Input 3 Block:", input_3)
                    triton.device_print("[Triton] Input 4 Block:", input_4)
                    triton.device_print("[Triton] Input 5 Block:", input_5)
                    triton.device_print("[Triton] Input 6 Block:", input_6)
                    triton.device_print("[Triton] Input 7 Block:", input_7)
                # --- DEBUG END ---
                # --- DEBUG END ---
                '''
                # Unrolled logic tree calculation
                w_ptr_base = Weights_ptr + pid_k * stride_w_oc
                
                # --- Node 1_0 ---
                w0_ptr = w_ptr_base + 0 * stride_w_g; w0_1 = tl.load(w0_ptr + 1 * stride_w_op); w0_2 = tl.load(w0_ptr + 2 * stride_w_op); w0_3 = tl.load(w0_ptr + 3 * stride_w_op); w0_4 = tl.load(w0_ptr + 4 * stride_w_op); w0_5 = tl.load(w0_ptr + 5 * stride_w_op); w0_6 = tl.load(w0_ptr + 6 * stride_w_op); w0_7 = tl.load(w0_ptr + 7 * stride_w_op); w0_8 = tl.load(w0_ptr + 8 * stride_w_op); w0_9 = tl.load(w0_ptr + 9 * stride_w_op); w0_10 = tl.load(w0_ptr + 10 * stride_w_op); w0_11 = tl.load(w0_ptr + 11 * stride_w_op); w0_12 = tl.load(w0_ptr + 12 * stride_w_op); w0_13 = tl.load(w0_ptr + 13 * stride_w_op); w0_14 = tl.load(w0_ptr + 14 * stride_w_op); w0_15 = tl.load(w0_ptr + 15 * stride_w_op)
                a_mul_b_0 = input_0 * input_1
                node_1_0 = (w0_1*a_mul_b_0 + w0_2*(input_0 - a_mul_b_0) + w0_3*input_0 + w0_4*(input_1 - a_mul_b_0) + w0_5*input_1 + w0_6*(input_0 + input_1 - 2.0*a_mul_b_0) + w0_7*(input_0 + input_1 - a_mul_b_0) + w0_8*(1.0 - (input_0 + input_1 - a_mul_b_0)) + w0_9*(1.0 - (input_0 + input_1 - 2.0*a_mul_b_0)) + w0_10*(1.0 - input_1) + w0_11*(1.0 - input_1 + a_mul_b_0) + w0_12*(1.0 - input_0) + w0_13*(1.0 - input_0 + a_mul_b_0) + w0_14*(1.0 - a_mul_b_0) + w0_15)

                # --- Node 1_1 ---
                w1_ptr = w_ptr_base + 1 * stride_w_g; w1_1 = tl.load(w1_ptr + 1 * stride_w_op); w1_2 = tl.load(w1_ptr + 2 * stride_w_op); w1_3 = tl.load(w1_ptr + 3 * stride_w_op); w1_4 = tl.load(w1_ptr + 4 * stride_w_op); w1_5 = tl.load(w1_ptr + 5 * stride_w_op); w1_6 = tl.load(w1_ptr + 6 * stride_w_op); w1_7 = tl.load(w1_ptr + 7 * stride_w_op); w1_8 = tl.load(w1_ptr + 8 * stride_w_op); w1_9 = tl.load(w1_ptr + 9 * stride_w_op); w1_10 = tl.load(w1_ptr + 10 * stride_w_op); w1_11 = tl.load(w1_ptr + 11 * stride_w_op); w1_12 = tl.load(w1_ptr + 12 * stride_w_op); w1_13 = tl.load(w1_ptr + 13 * stride_w_op); w1_14 = tl.load(w1_ptr + 14 * stride_w_op); w1_15 = tl.load(w1_ptr + 15 * stride_w_op)
                a_mul_b_1 = input_2 * input_3
                node_1_1 = (w1_1*a_mul_b_1 + w1_2*(input_2 - a_mul_b_1) + w1_3*input_2 + w1_4*(input_3 - a_mul_b_1) + w1_5*input_3 + w1_6*(input_2 + input_3 - 2.0*a_mul_b_1) + w1_7*(input_2 + input_3 - a_mul_b_1) + w1_8*(1.0 - (input_2 + input_3 - a_mul_b_1)) + w1_9*(1.0 - (input_2 + input_3 - 2.0*a_mul_b_1)) + w1_10*(1.0 - input_3) + w1_11*(1.0 - input_3 + a_mul_b_1) + w1_12*(1.0 - input_2) + w1_13*(1.0 - input_2 + a_mul_b_1) + w1_14*(1.0 - a_mul_b_1) + w1_15)

                # --- Node 1_2 ---
                w2_ptr = w_ptr_base + 2 * stride_w_g; w2_1 = tl.load(w2_ptr + 1 * stride_w_op); w2_2 = tl.load(w2_ptr + 2 * stride_w_op); w2_3 = tl.load(w2_ptr + 3 * stride_w_op); w2_4 = tl.load(w2_ptr + 4 * stride_w_op); w2_5 = tl.load(w2_ptr + 5 * stride_w_op); w2_6 = tl.load(w2_ptr + 6 * stride_w_op); w2_7 = tl.load(w2_ptr + 7 * stride_w_op); w2_8 = tl.load(w2_ptr + 8 * stride_w_op); w2_9 = tl.load(w2_ptr + 9 * stride_w_op); w2_10 = tl.load(w2_ptr + 10 * stride_w_op); w2_11 = tl.load(w2_ptr + 11 * stride_w_op); w2_12 = tl.load(w2_ptr + 12 * stride_w_op); w2_13 = tl.load(w2_ptr + 13 * stride_w_op); w2_14 = tl.load(w2_ptr + 14 * stride_w_op); w2_15 = tl.load(w2_ptr + 15 * stride_w_op)
                a_mul_b_2 = input_4 * input_5; node_1_2 = (w2_1*a_mul_b_2 + w2_2*(input_4 - a_mul_b_2) + w2_3*input_4 + w2_4*(input_5 - a_mul_b_2) + w2_5*input_5 + w2_6*(input_4 + input_5 - 2.0*a_mul_b_2) + w2_7*(input_4 + input_5 - a_mul_b_2) + w2_8*(1.0 - (input_4 + input_5 - a_mul_b_2)) + w2_9*(1.0 - (input_4 + input_5 - 2.0*a_mul_b_2)) + w2_10*(1.0 - input_5) + w2_11*(1.0 - input_5 + a_mul_b_2) + w2_12*(1.0 - input_4) + w2_13*(1.0 - input_4 + a_mul_b_2) + w2_14*(1.0 - a_mul_b_2) + w2_15)
                
                # --- Node 1_3 ---
                w3_ptr = w_ptr_base + 3 * stride_w_g; w3_1 = tl.load(w3_ptr + 1 * stride_w_op); w3_2 = tl.load(w3_ptr + 2 * stride_w_op); w3_3 = tl.load(w3_ptr + 3 * stride_w_op); w3_4 = tl.load(w3_ptr + 4 * stride_w_op); w3_5 = tl.load(w3_ptr + 5 * stride_w_op); w3_6 = tl.load(w3_ptr + 6 * stride_w_op); w3_7 = tl.load(w3_ptr + 7 * stride_w_op); w3_8 = tl.load(w3_ptr + 8 * stride_w_op); w3_9 = tl.load(w3_ptr + 9 * stride_w_op); w3_10 = tl.load(w3_ptr + 10 * stride_w_op); w3_11 = tl.load(w3_ptr + 11 * stride_w_op); w3_12 = tl.load(w3_ptr + 12 * stride_w_op); w3_13 = tl.load(w3_ptr + 13 * stride_w_op); w3_14 = tl.load(w3_ptr + 14 * stride_w_op); w3_15 = tl.load(w3_ptr + 15 * stride_w_op)
                a_mul_b_3 = input_6 * input_7; node_1_3 = (w3_1*a_mul_b_3 + w3_2*(input_6 - a_mul_b_3) + w3_3*input_6 + w3_4*(input_7 - a_mul_b_3) + w3_5*input_7 + w3_6*(input_6 + input_7 - 2.0*a_mul_b_3) + w3_7*(input_6 + input_7 - a_mul_b_3) + w3_8*(1.0 - (input_6 + input_7 - a_mul_b_3)) + w3_9*(1.0 - (input_6 + input_7 - 2.0*a_mul_b_3)) + w3_10*(1.0 - input_7) + w3_11*(1.0 - input_7 + a_mul_b_3) + w3_12*(1.0 - input_6) + w3_13*(1.0 - input_6 + a_mul_b_3) + w3_14*(1.0 - a_mul_b_3) + w3_15)
                
                # --- Node 2_0 ---
                w4_ptr = w_ptr_base + 4 * stride_w_g; w4_1 = tl.load(w4_ptr + 1 * stride_w_op); w4_2 = tl.load(w4_ptr + 2 * stride_w_op); w4_3 = tl.load(w4_ptr + 3 * stride_w_op); w4_4 = tl.load(w4_ptr + 4 * stride_w_op); w4_5 = tl.load(w4_ptr + 5 * stride_w_op); w4_6 = tl.load(w4_ptr + 6 * stride_w_op); w4_7 = tl.load(w4_ptr + 7 * stride_w_op); w4_8 = tl.load(w4_ptr + 8 * stride_w_op); w4_9 = tl.load(w4_ptr + 9 * stride_w_op); w4_10 = tl.load(w4_ptr + 10 * stride_w_op); w4_11 = tl.load(w4_ptr + 11 * stride_w_op); w4_12 = tl.load(w4_ptr + 12 * stride_w_op); w4_13 = tl.load(w4_ptr + 13 * stride_w_op); w4_14 = tl.load(w4_ptr + 14 * stride_w_op); w4_15 = tl.load(w4_ptr + 15 * stride_w_op)
                a_i, b_i = node_1_0, node_1_1; a_mul_b = a_i * b_i; node_2_0 = (w4_1*a_mul_b + w4_2*(a_i - a_mul_b) + w4_3*a_i + w4_4*(b_i - a_mul_b) + w4_5*b_i + w4_6*(a_i + b_i - 2.0*a_mul_b) + w4_7*(a_i + b_i - a_mul_b) + w4_8*(1.0 - (a_i + b_i - a_mul_b)) + w4_9*(1.0 - (a_i + b_i - 2.0*a_mul_b)) + w4_10*(1.0 - b_i) + w4_11*(1.0 - b_i + a_mul_b) + w4_12*(1.0 - a_i) + w4_13*(1.0 - a_i + a_mul_b) + w4_14*(1.0 - a_mul_b) + w4_15)

                # --- Node 2_1 ---
                w5_ptr = w_ptr_base + 5 * stride_w_g; w5_1 = tl.load(w5_ptr + 1 * stride_w_op); w5_2 = tl.load(w5_ptr + 2 * stride_w_op); w5_3 = tl.load(w5_ptr + 3 * stride_w_op); w5_4 = tl.load(w5_ptr + 4 * stride_w_op); w5_5 = tl.load(w5_ptr + 5 * stride_w_op); w5_6 = tl.load(w5_ptr + 6 * stride_w_op); w5_7 = tl.load(w5_ptr + 7 * stride_w_op); w5_8 = tl.load(w5_ptr + 8 * stride_w_op); w5_9 = tl.load(w5_ptr + 9 * stride_w_op); w5_10 = tl.load(w5_ptr + 10 * stride_w_op); w5_11 = tl.load(w5_ptr + 11 * stride_w_op); w5_12 = tl.load(w5_ptr + 12 * stride_w_op); w5_13 = tl.load(w5_ptr + 13 * stride_w_op); w5_14 = tl.load(w5_ptr + 14 * stride_w_op); w5_15 = tl.load(w5_ptr + 15 * stride_w_op)
                a_i, b_i = node_1_2, node_1_3; a_mul_b = a_i * b_i; node_2_1 = (w5_1*a_mul_b + w5_2*(a_i - a_mul_b) + w5_3*a_i + w5_4*(b_i - a_mul_b) + w5_5*b_i + w5_6*(a_i + b_i - 2.0*a_mul_b) + w5_7*(a_i + b_i - a_mul_b) + w5_8*(1.0 - (a_i + b_i - a_mul_b)) + w5_9*(1.0 - (a_i + b_i - 2.0*a_mul_b)) + w5_10*(1.0 - b_i) + w5_11*(1.0 - b_i + a_mul_b) + w5_12*(1.0 - a_i) + w5_13*(1.0 - a_i + a_mul_b) + w5_14*(1.0 - a_mul_b) + w5_15)

                # --- Root Node ---
                w6_ptr = w_ptr_base + 6 * stride_w_g; w6_1 = tl.load(w6_ptr + 1 * stride_w_op); w6_2 = tl.load(w6_ptr + 2 * stride_w_op); w6_3 = tl.load(w6_ptr + 3 * stride_w_op); w6_4 = tl.load(w6_ptr + 4 * stride_w_op); w6_5 = tl.load(w6_ptr + 5 * stride_w_op); w6_6 = tl.load(w6_ptr + 6 * stride_w_op); w6_7 = tl.load(w6_ptr + 7 * stride_w_op); w6_8 = tl.load(w6_ptr + 8 * stride_w_op); w6_9 = tl.load(w6_ptr + 9 * stride_w_op); w6_10 = tl.load(w6_ptr + 10 * stride_w_op); w6_11 = tl.load(w6_ptr + 11 * stride_w_op); w6_12 = tl.load(w6_ptr + 12 * stride_w_op); w6_13 = tl.load(w6_ptr + 13 * stride_w_op); w6_14 = tl.load(w6_ptr + 14 * stride_w_op); w6_15 = tl.load(w6_ptr + 15 * stride_w_op)
                a_i, b_i = node_2_0, node_2_1; a_mul_b = a_i * b_i; conv_out = (w6_1*a_mul_b + w6_2*(a_i - a_mul_b) + w6_3*a_i + w6_4*(b_i - a_mul_b) + w6_5*b_i + w6_6*(a_i + b_i - 2.0*a_mul_b) + w6_7*(a_i + b_i - a_mul_b) + w6_8*(1.0 - (a_i + b_i - a_mul_b)) + w6_9*(1.0 - (a_i + b_i - 2.0*a_mul_b)) + w6_10*(1.0 - b_i) + w6_11*(1.0 - b_i + a_mul_b) + w6_12*(1.0 - a_i) + w6_13*(1.0 - a_i + a_mul_b) + w6_14*(1.0 - a_mul_b) + w6_15)
                

                                # --- DEBUG START ---
                
                
                #if SHOULD_PRINT:
                #    triton.device_print("[Triton] Tree Level 1 Output Block (node_1_0):", node_1_0)
                #    triton.device_print("[Triton] Tree Level 2 Output Block (node_2_0):", node_2_0)
                #    triton.device_print("[Triton] Final Conv Output Block:", conv_out)
                # --- DEBUG END ---

                max_val = tl.where(conv_w_mask, tl.maximum(max_val, conv_out), max_val)


    #if SHOULD_PRINT:
    #    triton.device_print("[Triton] Final Pooled Output Block:", max_val)
    #    triton.device_print("--- TRITON DEBUG END ---")

    y_ptr_final = Y_ptr + pid_b * stride_y_b + pid_k * stride_y_c + pid_h_out * stride_y_h + offsets_w_out * stride_y_w
    tl.store(y_ptr_final, max_val, mask=w_out_mask)




@triton.jit
def fused_logictree_conv_backward_kernel(
    # --- Input Tensors ---
    Grad_Y_ptr, Y_ptr, X_padded_ptr, W_sm_ptr,
    Input_channel_indices_ptr, Input_pos_x_indices_ptr, Input_pos_y_indices_ptr,
    
    # --- Output Tensors ---
    Grad_X_padded_ptr, Grad_Weights_ptr,
    
    # --- Dimensions & Parameters ---
    batch_size, out_channels, out_h_pool, out_w_pool, out_h_conv, out_w_conv,
    kernel_size, stride, padding, groups, in_channels_per_group,
    
    # --- Strides ---
    stride_gy_b, stride_gy_c, stride_gy_h, stride_gy_w,
    stride_y_b, stride_y_c, stride_y_h, stride_y_w,
    stride_xp_b, stride_xp_c, stride_xp_h, stride_xp_w,
    stride_w_oc, stride_w_g, stride_w_op,
    stride_gxp_b, stride_gxp_c, stride_gxp_h, stride_gxp_w,
    stride_gw_oc, stride_gw_g, stride_gw_op,
    stride_ici_oc, stride_ici_inp,
    stride_ipx_oc, stride_ipx_inp,
    stride_ipy_oc, stride_ipy_inp,
    
    # --- Compile-time Constants ---
    BLOCK_SIZE_W: tl.constexpr,
    NUM_W_BLOCKS: tl.constexpr,
    NUM_INPUTS_PER_TREE: tl.constexpr,
    TREE_DEPTH: tl.constexpr,
):
    # Section 1: Grid and PID setup
    pid_b = tl.program_id(axis=0); pid_k = tl.program_id(axis=1); pid_spatial = tl.program_id(axis=2)
    pid_h_out = pid_spatial // NUM_W_BLOCKS; pid_w_block = pid_spatial % NUM_W_BLOCKS
    offsets_w_out = pid_w_block * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    w_out_mask = offsets_w_out < out_w_pool
    if pid_h_out >= out_h_pool:
        return

    # Section 2: Max-Pooling Backward Prep
    y_final_ptr = Y_ptr + pid_b*stride_y_b + pid_k*stride_y_c + pid_h_out*stride_y_h + offsets_w_out
    grad_y_final_ptr = Grad_Y_ptr + pid_b*stride_gy_b + pid_k*stride_gy_c + pid_h_out*stride_gy_h + offsets_w_out
    y_final = tl.load(y_final_ptr, mask=w_out_mask, other=0.0)
    grad_y_final = tl.load(grad_y_final_ptr, mask=w_out_mask, other=0.0)

    # Section 3: Gradient Path Finding and Backpropagation
    for h_offset in range(2):
        for w_offset in range(2):
            conv_h = pid_h_out * 2 + h_offset
            if conv_h < out_h_conv:
                conv_w = offsets_w_out * 2 + w_offset
                conv_w_mask = conv_w < out_w_conv
                final_load_mask = w_out_mask & conv_w_mask

                # --- 3A: Forward Recalculation (using your verified unrolled method) ---
                patch_start_h = conv_h*stride; patch_start_w = conv_w*stride
                group_idx = (pid_k*groups)//out_channels; group_in_start = group_idx*in_channels_per_group
                ici_base_ptr = Input_channel_indices_ptr + pid_k*stride_ici_oc
                ipx_base_ptr = Input_pos_x_indices_ptr + pid_k*stride_ipx_oc
                ipy_base_ptr = Input_pos_y_indices_ptr + pid_k*stride_ipy_oc
                xp_base_ptr = X_padded_ptr + pid_b*stride_xp_b
                
                # Unrolled input loading
                ptr_0=xp_base_ptr+(tl.load(ici_base_ptr+0*stride_ici_inp)+group_in_start)*stride_xp_c+(tl.load(ipy_base_ptr+0*stride_ipy_inp)+patch_start_h)*stride_xp_h+(tl.load(ipx_base_ptr+0*stride_ipx_inp)+patch_start_w)*stride_xp_w; input_0=tl.load(ptr_0,mask=final_load_mask,other=0.0)
                ptr_1=xp_base_ptr+(tl.load(ici_base_ptr+1*stride_ici_inp)+group_in_start)*stride_xp_c+(tl.load(ipy_base_ptr+1*stride_ipy_inp)+patch_start_h)*stride_xp_h+(tl.load(ipx_base_ptr+1*stride_ipx_inp)+patch_start_w)*stride_xp_w; input_1=tl.load(ptr_1,mask=final_load_mask,other=0.0)
                ptr_2=xp_base_ptr+(tl.load(ici_base_ptr+2*stride_ici_inp)+group_in_start)*stride_xp_c+(tl.load(ipy_base_ptr+2*stride_ipy_inp)+patch_start_h)*stride_xp_h+(tl.load(ipx_base_ptr+2*stride_ipx_inp)+patch_start_w)*stride_xp_w; input_2=tl.load(ptr_2,mask=final_load_mask,other=0.0)
                ptr_3=xp_base_ptr+(tl.load(ici_base_ptr+3*stride_ici_inp)+group_in_start)*stride_xp_c+(tl.load(ipy_base_ptr+3*stride_ipy_inp)+patch_start_h)*stride_xp_h+(tl.load(ipx_base_ptr+3*stride_ipx_inp)+patch_start_w)*stride_xp_w; input_3=tl.load(ptr_3,mask=final_load_mask,other=0.0)
                ptr_4=xp_base_ptr+(tl.load(ici_base_ptr+4*stride_ici_inp)+group_in_start)*stride_xp_c+(tl.load(ipy_base_ptr+4*stride_ipy_inp)+patch_start_h)*stride_xp_h+(tl.load(ipx_base_ptr+4*stride_ipx_inp)+patch_start_w)*stride_xp_w; input_4=tl.load(ptr_4,mask=final_load_mask,other=0.0)
                ptr_5=xp_base_ptr+(tl.load(ici_base_ptr+5*stride_ici_inp)+group_in_start)*stride_xp_c+(tl.load(ipy_base_ptr+5*stride_ipy_inp)+patch_start_h)*stride_xp_h+(tl.load(ipx_base_ptr+5*stride_ipx_inp)+patch_start_w)*stride_xp_w; input_5=tl.load(ptr_5,mask=final_load_mask,other=0.0)
                ptr_6=xp_base_ptr+(tl.load(ici_base_ptr+6*stride_ici_inp)+group_in_start)*stride_xp_c+(tl.load(ipy_base_ptr+6*stride_ipy_inp)+patch_start_h)*stride_xp_h+(tl.load(ipx_base_ptr+6*stride_ipx_inp)+patch_start_w)*stride_xp_w; input_6=tl.load(ptr_6,mask=final_load_mask,other=0.0)
                ptr_7=xp_base_ptr+(tl.load(ici_base_ptr+7*stride_ici_inp)+group_in_start)*stride_xp_c+(tl.load(ipy_base_ptr+7*stride_ipy_inp)+patch_start_h)*stride_xp_h+(tl.load(ipx_base_ptr+7*stride_ipx_inp)+patch_start_w)*stride_xp_w; input_7=tl.load(ptr_7,mask=final_load_mask,other=0.0)


            
# ==========================================================
                #       [시작] 트리 중간 노드 값 재계산 (Unrolled)
                # ==========================================================
                # 이 로직은 Forward Pass와 동일하며, Backward Pass에서 중간 값들을 참조하기 위해 필요합니다.
                # 모든 가중치(w)는 Softmax를 통과한 w_sm 값을 사용합니다.
                w_sm_base_ptr = W_sm_ptr + pid_k * stride_w_oc
                
                # --- 레벨 1 노드 (4개) ---
                # Node 1_0 (inputs: 0, 1)
                w0_ptr = w_sm_base_ptr + 0 * stride_w_g
                w0_1 = tl.load(w0_ptr + 1 * stride_w_op);  w0_2 = tl.load(w0_ptr + 2 * stride_w_op);  w0_3 = tl.load(w0_ptr + 3 * stride_w_op);  w0_4 = tl.load(w0_ptr + 4 * stride_w_op);  w0_5 = tl.load(w0_ptr + 5 * stride_w_op);  w0_6 = tl.load(w0_ptr + 6 * stride_w_op);  w0_7 = tl.load(w0_ptr + 7 * stride_w_op);  w0_8 = tl.load(w0_ptr + 8 * stride_w_op);  w0_9 = tl.load(w0_ptr + 9 * stride_w_op);  w0_10 = tl.load(w0_ptr + 10 * stride_w_op); w0_11 = tl.load(w0_ptr + 11 * stride_w_op); w0_12 = tl.load(w0_ptr + 12 * stride_w_op); w0_13 = tl.load(w0_ptr + 13 * stride_w_op); w0_14 = tl.load(w0_ptr + 14 * stride_w_op); w0_15 = tl.load(w0_ptr + 15 * stride_w_op)
                a_mul_b_0 = input_0 * input_1
                node_1_0 = w0_1*a_mul_b_0 + w0_2*(input_0-a_mul_b_0) + w0_3*input_0 + w0_4*(input_1-a_mul_b_0) + w0_5*input_1 + w0_6*(input_0+input_1-2*a_mul_b_0) + w0_7*(input_0+input_1-a_mul_b_0) + w0_8*(1-(input_0+input_1-a_mul_b_0)) + w0_9*(1-(input_0+input_1-2*a_mul_b_0)) + w0_10*(1-input_1) + w0_11*(1-input_1+a_mul_b_0) + w0_12*(1-input_0) + w0_13*(1-input_0+a_mul_b_0) + w0_14*(1-a_mul_b_0) + w0_15

                # Node 1_1 (inputs: 2, 3)
                w1_ptr = w_sm_base_ptr + 1 * stride_w_g
                w1_1 = tl.load(w1_ptr + 1 * stride_w_op);  w1_2 = tl.load(w1_ptr + 2 * stride_w_op);  w1_3 = tl.load(w1_ptr + 3 * stride_w_op);  w1_4 = tl.load(w1_ptr + 4 * stride_w_op);  w1_5 = tl.load(w1_ptr + 5 * stride_w_op);  w1_6 = tl.load(w1_ptr + 6 * stride_w_op);  w1_7 = tl.load(w1_ptr + 7 * stride_w_op);  w1_8 = tl.load(w1_ptr + 8 * stride_w_op);  w1_9 = tl.load(w1_ptr + 9 * stride_w_op);  w1_10 = tl.load(w1_ptr + 10 * stride_w_op); w1_11 = tl.load(w1_ptr + 11 * stride_w_op); w1_12 = tl.load(w1_ptr + 12 * stride_w_op); w1_13 = tl.load(w1_ptr + 13 * stride_w_op); w1_14 = tl.load(w1_ptr + 14 * stride_w_op); w1_15 = tl.load(w1_ptr + 15 * stride_w_op)
                a_mul_b_1 = input_2 * input_3
                node_1_1 = w1_1*a_mul_b_1 + w1_2*(input_2-a_mul_b_1) + w1_3*input_2 + w1_4*(input_3-a_mul_b_1) + w1_5*input_3 + w1_6*(input_2+input_3-2*a_mul_b_1) + w1_7*(input_2+input_3-a_mul_b_1) + w1_8*(1-(input_2+input_3-a_mul_b_1)) + w1_9*(1-(input_2+input_3-2*a_mul_b_1)) + w1_10*(1-input_3) + w1_11*(1-input_3+a_mul_b_1) + w1_12*(1-input_2) + w1_13*(1-input_2+a_mul_b_1) + w1_14*(1-a_mul_b_1) + w1_15

                # Node 1_2 (inputs: 4, 5)
                w2_ptr = w_sm_base_ptr + 2 * stride_w_g
                w2_1 = tl.load(w2_ptr + 1 * stride_w_op);  w2_2 = tl.load(w2_ptr + 2 * stride_w_op);  w2_3 = tl.load(w2_ptr + 3 * stride_w_op);  w2_4 = tl.load(w2_ptr + 4 * stride_w_op);  w2_5 = tl.load(w2_ptr + 5 * stride_w_op);  w2_6 = tl.load(w2_ptr + 6 * stride_w_op);  w2_7 = tl.load(w2_ptr + 7 * stride_w_op);  w2_8 = tl.load(w2_ptr + 8 * stride_w_op);  w2_9 = tl.load(w2_ptr + 9 * stride_w_op);  w2_10 = tl.load(w2_ptr + 10 * stride_w_op); w2_11 = tl.load(w2_ptr + 11 * stride_w_op); w2_12 = tl.load(w2_ptr + 12 * stride_w_op); w2_13 = tl.load(w2_ptr + 13 * stride_w_op); w2_14 = tl.load(w2_ptr + 14 * stride_w_op); w2_15 = tl.load(w2_ptr + 15 * stride_w_op)
                a_mul_b_2 = input_4 * input_5
                node_1_2 = w2_1*a_mul_b_2 + w2_2*(input_4-a_mul_b_2) + w2_3*input_4 + w2_4*(input_5-a_mul_b_2) + w2_5*input_5 + w2_6*(input_4+input_5-2*a_mul_b_2) + w2_7*(input_4+input_5-a_mul_b_2) + w2_8*(1-(input_4+input_5-a_mul_b_2)) + w2_9*(1-(input_4+input_5-2*a_mul_b_2)) + w2_10*(1-input_5) + w2_11*(1-input_5+a_mul_b_2) + w2_12*(1-input_4) + w2_13*(1-input_4+a_mul_b_2) + w2_14*(1-a_mul_b_2) + w2_15

                # Node 1_3 (inputs: 6, 7)
                w3_ptr = w_sm_base_ptr + 3 * stride_w_g
                w3_1 = tl.load(w3_ptr + 1 * stride_w_op);  w3_2 = tl.load(w3_ptr + 2 * stride_w_op);  w3_3 = tl.load(w3_ptr + 3 * stride_w_op);  w3_4 = tl.load(w3_ptr + 4 * stride_w_op);  w3_5 = tl.load(w3_ptr + 5 * stride_w_op);  w3_6 = tl.load(w3_ptr + 6 * stride_w_op);  w3_7 = tl.load(w3_ptr + 7 * stride_w_op);  w3_8 = tl.load(w3_ptr + 8 * stride_w_op);  w3_9 = tl.load(w3_ptr + 9 * stride_w_op);  w3_10 = tl.load(w3_ptr + 10 * stride_w_op); w3_11 = tl.load(w3_ptr + 11 * stride_w_op); w3_12 = tl.load(w3_ptr + 12 * stride_w_op); w3_13 = tl.load(w3_ptr + 13 * stride_w_op); w3_14 = tl.load(w3_ptr + 14 * stride_w_op); w3_15 = tl.load(w3_ptr + 15 * stride_w_op)
                a_mul_b_3 = input_6 * input_7
                node_1_3 = w3_1*a_mul_b_3 + w3_2*(input_6-a_mul_b_3) + w3_3*input_6 + w3_4*(input_7-a_mul_b_3) + w3_5*input_7 + w3_6*(input_6+input_7-2*a_mul_b_3) + w3_7*(input_6+input_7-a_mul_b_3) + w3_8*(1-(input_6+input_7-a_mul_b_3)) + w3_9*(1-(input_6+input_7-2*a_mul_b_3)) + w3_10*(1-input_7) + w3_11*(1-input_7+a_mul_b_3) + w3_12*(1-input_6) + w3_13*(1-input_6+a_mul_b_3) + w3_14*(1-a_mul_b_3) + w3_15

                # --- 레벨 2 노드 (2개) ---
                # Node 2_0 (inputs: node_1_0, node_1_1)
                w4_ptr = w_sm_base_ptr + 4 * stride_w_g
                w4_1 = tl.load(w4_ptr + 1 * stride_w_op);  w4_2 = tl.load(w4_ptr + 2 * stride_w_op);  w4_3 = tl.load(w4_ptr + 3 * stride_w_op);  w4_4 = tl.load(w4_ptr + 4 * stride_w_op);  w4_5 = tl.load(w4_ptr + 5 * stride_w_op);  w4_6 = tl.load(w4_ptr + 6 * stride_w_op);  w4_7 = tl.load(w4_ptr + 7 * stride_w_op);  w4_8 = tl.load(w4_ptr + 8 * stride_w_op);  w4_9 = tl.load(w4_ptr + 9 * stride_w_op);  w4_10 = tl.load(w4_ptr + 10 * stride_w_op); w4_11 = tl.load(w4_ptr + 11 * stride_w_op); w4_12 = tl.load(w4_ptr + 12 * stride_w_op); w4_13 = tl.load(w4_ptr + 13 * stride_w_op); w4_14 = tl.load(w4_ptr + 14 * stride_w_op); w4_15 = tl.load(w4_ptr + 15 * stride_w_op)
                a_mul_b_4 = node_1_0 * node_1_1
                node_2_0 = w4_1*a_mul_b_4 + w4_2*(node_1_0-a_mul_b_4) + w4_3*node_1_0 + w4_4*(node_1_1-a_mul_b_4) + w4_5*node_1_1 + w4_6*(node_1_0+node_1_1-2*a_mul_b_4) + w4_7*(node_1_0+node_1_1-a_mul_b_4) + w4_8*(1-(node_1_0+node_1_1-a_mul_b_4)) + w4_9*(1-(node_1_0+node_1_1-2*a_mul_b_4)) + w4_10*(1-node_1_1) + w4_11*(1-node_1_1+a_mul_b_4) + w4_12*(1-node_1_0) + w4_13*(1-node_1_0+a_mul_b_4) + w4_14*(1-a_mul_b_4) + w4_15

                # Node 2_1 (inputs: node_1_2, node_1_3)
                w5_ptr = w_sm_base_ptr + 5 * stride_w_g
                w5_1 = tl.load(w5_ptr + 1 * stride_w_op);  w5_2 = tl.load(w5_ptr + 2 * stride_w_op);  w5_3 = tl.load(w5_ptr + 3 * stride_w_op);  w5_4 = tl.load(w5_ptr + 4 * stride_w_op);  w5_5 = tl.load(w5_ptr + 5 * stride_w_op);  w5_6 = tl.load(w5_ptr + 6 * stride_w_op);  w5_7 = tl.load(w5_ptr + 7 * stride_w_op);  w5_8 = tl.load(w5_ptr + 8 * stride_w_op);  w5_9 = tl.load(w5_ptr + 9 * stride_w_op);  w5_10 = tl.load(w5_ptr + 10 * stride_w_op); w5_11 = tl.load(w5_ptr + 11 * stride_w_op); w5_12 = tl.load(w5_ptr + 12 * stride_w_op); w5_13 = tl.load(w5_ptr + 13 * stride_w_op); w5_14 = tl.load(w5_ptr + 14 * stride_w_op); w5_15 = tl.load(w5_ptr + 15 * stride_w_op)
                a_mul_b_5 = node_1_2 * node_1_3
                node_2_1 = w5_1*a_mul_b_5 + w5_2*(node_1_2-a_mul_b_5) + w5_3*node_1_2 + w5_4*(node_1_3-a_mul_b_5) + w5_5*node_1_3 + w5_6*(node_1_2+node_1_3-2*a_mul_b_5) + w5_7*(node_1_2+node_1_3-a_mul_b_5) + w5_8*(1-(node_1_2+node_1_3-a_mul_b_5)) + w5_9*(1-(node_1_2+node_1_3-2*a_mul_b_5)) + w5_10*(1-node_1_3) + w5_11*(1-node_1_3+a_mul_b_5) + w5_12*(1-node_1_2) + w5_13*(1-node_1_2+a_mul_b_5) + w5_14*(1-a_mul_b_5) + w5_15

                # --- 루트 노드 (1개) ---
                # Root Node (inputs: node_2_0, node_2_1)
                w6_ptr = w_sm_base_ptr + 6 * stride_w_g
                w6_1 = tl.load(w6_ptr + 1 * stride_w_op);  w6_2 = tl.load(w6_ptr + 2 * stride_w_op);  w6_3 = tl.load(w6_ptr + 3 * stride_w_op);  w6_4 = tl.load(w6_ptr + 4 * stride_w_op);  w6_5 = tl.load(w6_ptr + 5 * stride_w_op);  w6_6 = tl.load(w6_ptr + 6 * stride_w_op);  w6_7 = tl.load(w6_ptr + 7 * stride_w_op);  w6_8 = tl.load(w6_ptr + 8 * stride_w_op);  w6_9 = tl.load(w6_ptr + 9 * stride_w_op);  w6_10 = tl.load(w6_ptr + 10 * stride_w_op); w6_11 = tl.load(w6_ptr + 11 * stride_w_op); w6_12 = tl.load(w6_ptr + 12 * stride_w_op); w6_13 = tl.load(w6_ptr + 13 * stride_w_op); w6_14 = tl.load(w6_ptr + 14 * stride_w_op); w6_15 = tl.load(w6_ptr + 15 * stride_w_op)
                a_mul_b_6 = node_2_0 * node_2_1
                conv_out = w6_1*a_mul_b_6 + w6_2*(node_2_0-a_mul_b_6) + w6_3*node_2_0 + w6_4*(node_2_1-a_mul_b_6) + w6_5*node_2_1 + w6_6*(node_2_0+node_2_1-2*a_mul_b_6) + w6_7*(node_2_0+node_2_1-a_mul_b_6) + w6_8*(1-(node_2_0+node_2_1-a_mul_b_6)) + w6_9*(1-(node_2_0+node_2_1-2*a_mul_b_6)) + w6_10*(1-node_2_1) + w6_11*(1-node_2_1+a_mul_b_6) + w6_12*(1-node_2_0) + w6_13*(1-node_2_0+a_mul_b_6) + w6_14*(1-a_mul_b_6) + w6_15
                # ==========================================================
                #       [끝] 트리 중간 노드 값 재계산 (Unrolled)
                # ==========================================================                
                # ==========================================================
                #       [시작] 3B: 실제 역전파 계산 (Unrolled)
                # ==========================================================
                is_max_path = (conv_out >= y_final - 1e-4) # 부동소수점 오차에 강건한 비교
                grad_conv_out = tl.where(is_max_path & final_load_mask, grad_y_final, 0.0)

                # --- 루트 노드(6)의 역전파 ---
                # 입력: node_2_0, node_2_1. 그래디언트: grad_conv_out. 가중치: w6
                a, b = node_2_0, node_2_1
                w_ptr = w_sm_base_ptr + 6 * stride_w_g
                w_grad_ptr = Grad_Weights_ptr + pid_k*stride_gw_oc + 6*stride_gw_g
                w6_1=tl.load(w_ptr+1*stride_w_op); w6_2=tl.load(w_ptr+2*stride_w_op); w6_3=tl.load(w_ptr+3*stride_w_op); w6_4=tl.load(w_ptr+4*stride_w_op); w6_5=tl.load(w_ptr+5*stride_w_op); w6_6=tl.load(w_ptr+6*stride_w_op); w6_7=tl.load(w_ptr+7*stride_w_op); w6_8=tl.load(w_ptr+8*stride_w_op); w6_9=tl.load(w_ptr+9*stride_w_op); w6_10=tl.load(w_ptr+10*stride_w_op); w6_11=tl.load(w_ptr+11*stride_w_op); w6_12=tl.load(w_ptr+12*stride_w_op); w6_13=tl.load(w_ptr+13*stride_w_op); w6_14=tl.load(w_ptr+14*stride_w_op)
                dy_da_6 = (w6_1*b + w6_2*(1-b) + w6_3) + (w6_4*(-b) + w6_6*(1-2*b) + w6_7*(1-b)) + (w6_8*(b-1) + w6_9*(2*b-1) + w6_11*b) + (-w6_12 + w6_13*(b-1) + w6_14*(-b))
                dy_db_6 = (w6_1*a + w6_2*(-a) + w6_4*(1-a)) + (w6_5 + w6_6*(1-2*a) + w6_7*(1-a)) + (w6_8*(a-1) + w6_9*(2*a-1) - w6_10) + (w6_11*(a-1) + w6_13*a + w6_14*(-a))
                grad_node_2_0 = grad_conv_out * dy_da_6
                grad_node_2_1 = grad_conv_out * dy_db_6
                grad_w6_ab=tl.sum((a*b)*grad_conv_out,0); grad_w6_a=tl.sum(a*grad_conv_out,0); grad_w6_b=tl.sum(b*grad_conv_out,0); grad_w6_const=tl.sum(grad_conv_out,0)
                grad_w6=tl.zeros((16,),dtype=grad_w6_ab.dtype); grad_w6=tl.where(tl.arange(0,16)==1,grad_w6_ab,grad_w6); grad_w6=tl.where(tl.arange(0,16)==2,grad_w6_a-grad_w6_ab,grad_w6); grad_w6=tl.where(tl.arange(0,16)==3,grad_w6_a,grad_w6); grad_w6=tl.where(tl.arange(0,16)==4,grad_w6_b-grad_w6_ab,grad_w6); grad_w6=tl.where(tl.arange(0,16)==5,grad_w6_b,grad_w6); grad_w6=tl.where(tl.arange(0,16)==6,grad_w6_a+grad_w6_b-2*grad_w6_ab,grad_w6); grad_w6=tl.where(tl.arange(0,16)==7,grad_w6_a+grad_w6_b-grad_w6_ab,grad_w6); grad_w6=tl.where(tl.arange(0,16)==8,grad_w6_const-(grad_w6_a+grad_w6_b-grad_w6_ab),grad_w6); grad_w6=tl.where(tl.arange(0,16)==9,grad_w6_const-(grad_w6_a+grad_w6_b-2*grad_w6_ab),grad_w6); grad_w6=tl.where(tl.arange(0,16)==10,grad_w6_const-grad_w6_b,grad_w6); grad_w6=tl.where(tl.arange(0,16)==11,grad_w6_const-grad_w6_b+grad_w6_ab,grad_w6); grad_w6=tl.where(tl.arange(0,16)==12,grad_w6_const-grad_w6_a,grad_w6); grad_w6=tl.where(tl.arange(0,16)==13,grad_w6_const-grad_w6_a+grad_w6_ab,grad_w6); grad_w6=tl.where(tl.arange(0,16)==14,grad_w6_const-grad_w6_ab,grad_w6); grad_w6=tl.where(tl.arange(0,16)==15,grad_w6_const,grad_w6)
                tl.atomic_add(w_grad_ptr+tl.arange(0,16)*stride_gw_op,grad_w6)

                # --- 레벨 2 노드(4, 5)의 역전파 ---
                # 노드 5
                a, b = node_1_2, node_1_3
                grad_parent = grad_node_2_1
                w_ptr = w_sm_base_ptr + 5 * stride_w_g
                w_grad_ptr = Grad_Weights_ptr + pid_k*stride_gw_oc + 5*stride_gw_g
                w5_1=tl.load(w_ptr+1*stride_w_op); w5_2=tl.load(w_ptr+2*stride_w_op); w5_3=tl.load(w_ptr+3*stride_w_op); w5_4=tl.load(w_ptr+4*stride_w_op); w5_5=tl.load(w_ptr+5*stride_w_op); w5_6=tl.load(w_ptr+6*stride_w_op); w5_7=tl.load(w_ptr+7*stride_w_op); w5_8=tl.load(w_ptr+8*stride_w_op); w5_9=tl.load(w_ptr+9*stride_w_op); w5_10=tl.load(w_ptr+10*stride_w_op); w5_11=tl.load(w_ptr+11*stride_w_op); w5_12=tl.load(w_ptr+12*stride_w_op); w5_13=tl.load(w_ptr+13*stride_w_op); w5_14=tl.load(w_ptr+14*stride_w_op);
                dy_da_5 = (w5_1*b + w5_2*(1-b) + w5_3) + (w5_4*(-b) + w5_6*(1-2*b) + w5_7*(1-b)) + (w5_8*(b-1) + w5_9*(2*b-1) + w5_11*b) + (-w5_12 + w5_13*(b-1) + w5_14*(-b))
                dy_db_5 = (w5_1*a + w5_2*(-a) + w5_4*(1-a)) + (w5_5 + w5_6*(1-2*a) + w5_7*(1-a)) + (w5_8*(a-1) + w5_9*(2*a-1) - w5_10) + (w5_11*(a-1) + w5_13*a + w5_14*(-a))
                grad_node_1_2 = grad_parent * dy_da_5
                grad_node_1_3 = grad_parent * dy_db_5
                grad_w5_ab=tl.sum((a*b)*grad_parent,0); grad_w5_a=tl.sum(a*grad_parent,0); grad_w5_b=tl.sum(b*grad_parent,0); grad_w5_const=tl.sum(grad_parent,0)
                grad_w5=tl.zeros((16,),dtype=grad_w5_ab.dtype); grad_w5=tl.where(tl.arange(0,16)==1,grad_w5_ab,grad_w5); grad_w5=tl.where(tl.arange(0,16)==2,grad_w5_a-grad_w5_ab,grad_w5); grad_w5=tl.where(tl.arange(0,16)==3,grad_w5_a,grad_w5); grad_w5=tl.where(tl.arange(0,16)==4,grad_w5_b-grad_w5_ab,grad_w5); grad_w5=tl.where(tl.arange(0,16)==5,grad_w5_b,grad_w5); grad_w5=tl.where(tl.arange(0,16)==6,grad_w5_a+grad_w5_b-2*grad_w5_ab,grad_w5); grad_w5=tl.where(tl.arange(0,16)==7,grad_w5_a+grad_w5_b-grad_w5_ab,grad_w5); grad_w5=tl.where(tl.arange(0,16)==8,grad_w5_const-(grad_w5_a+grad_w5_b-grad_w5_ab),grad_w5); grad_w5=tl.where(tl.arange(0,16)==9,grad_w5_const-(grad_w5_a+grad_w5_b-2*grad_w5_ab),grad_w5); grad_w5=tl.where(tl.arange(0,16)==10,grad_w5_const-grad_w5_b,grad_w5); grad_w5=tl.where(tl.arange(0,16)==11,grad_w5_const-grad_w5_b+grad_w5_ab,grad_w5); grad_w5=tl.where(tl.arange(0,16)==12,grad_w5_const-grad_w5_a,grad_w5); grad_w5=tl.where(tl.arange(0,16)==13,grad_w5_const-grad_w5_a+grad_w5_ab,grad_w5); grad_w5=tl.where(tl.arange(0,16)==14,grad_w5_const-grad_w5_ab,grad_w5); grad_w5=tl.where(tl.arange(0,16)==15,grad_w5_const,grad_w5)
                tl.atomic_add(w_grad_ptr+tl.arange(0,16)*stride_gw_op,grad_w5)

                # 노드 4
                a, b = node_1_0, node_1_1
                grad_parent = grad_node_2_0
                w_ptr = w_sm_base_ptr + 4 * stride_w_g
                w_grad_ptr = Grad_Weights_ptr + pid_k*stride_gw_oc + 4*stride_gw_g
                w4_1=tl.load(w_ptr+1*stride_w_op); w4_2=tl.load(w_ptr+2*stride_w_op); w4_3=tl.load(w_ptr+3*stride_w_op); w4_4=tl.load(w_ptr+4*stride_w_op); w4_5=tl.load(w_ptr+5*stride_w_op); w4_6=tl.load(w_ptr+6*stride_w_op); w4_7=tl.load(w_ptr+7*stride_w_op); w4_8=tl.load(w_ptr+8*stride_w_op); w4_9=tl.load(w_ptr+9*stride_w_op); w4_10=tl.load(w_ptr+10*stride_w_op); w4_11=tl.load(w_ptr+11*stride_w_op); w4_12=tl.load(w_ptr+12*stride_w_op); w4_13=tl.load(w_ptr+13*stride_w_op); w4_14=tl.load(w_ptr+14*stride_w_op);
                dy_da_4 = (w4_1*b + w4_2*(1-b) + w4_3) + (w4_4*(-b) + w4_6*(1-2*b) + w4_7*(1-b)) + (w4_8*(b-1) + w4_9*(2*b-1) + w4_11*b) + (-w4_12 + w4_13*(b-1) + w4_14*(-b))
                dy_db_4 = (w4_1*a + w4_2*(-a) + w4_4*(1-a)) + (w4_5 + w4_6*(1-2*a) + w4_7*(1-a)) + (w4_8*(a-1) + w4_9*(2*a-1) - w4_10) + (w4_11*(a-1) + w4_13*a + w4_14*(-a))
                grad_node_1_0 = grad_parent * dy_da_4
                grad_node_1_1 = grad_parent * dy_db_4
                grad_w4_ab=tl.sum((a*b)*grad_parent,0); grad_w4_a=tl.sum(a*grad_parent,0); grad_w4_b=tl.sum(b*grad_parent,0); grad_w4_const=tl.sum(grad_parent,0)
                grad_w4=tl.zeros((16,),dtype=grad_w4_ab.dtype); grad_w4=tl.where(tl.arange(0,16)==1,grad_w4_ab,grad_w4); grad_w4=tl.where(tl.arange(0,16)==2,grad_w4_a-grad_w4_ab,grad_w4); grad_w4=tl.where(tl.arange(0,16)==3,grad_w4_a,grad_w4); grad_w4=tl.where(tl.arange(0,16)==4,grad_w4_b-grad_w4_ab,grad_w4); grad_w4=tl.where(tl.arange(0,16)==5,grad_w4_b,grad_w4); grad_w4=tl.where(tl.arange(0,16)==6,grad_w4_a+grad_w4_b-2*grad_w4_ab,grad_w4); grad_w4=tl.where(tl.arange(0,16)==7,grad_w4_a+grad_w4_b-grad_w4_ab,grad_w4); grad_w4=tl.where(tl.arange(0,16)==8,grad_w4_const-(grad_w4_a+grad_w4_b-grad_w4_ab),grad_w4); grad_w4=tl.where(tl.arange(0,16)==9,grad_w4_const-(grad_w4_a+grad_w4_b-2*grad_w4_ab),grad_w4); grad_w4=tl.where(tl.arange(0,16)==10,grad_w4_const-grad_w4_b,grad_w4); grad_w4=tl.where(tl.arange(0,16)==11,grad_w4_const-grad_w4_b+grad_w4_ab,grad_w4); grad_w4=tl.where(tl.arange(0,16)==12,grad_w4_const-grad_w4_a,grad_w4); grad_w4=tl.where(tl.arange(0,16)==13,grad_w4_const-grad_w4_a+grad_w4_ab,grad_w4); grad_w4=tl.where(tl.arange(0,16)==14,grad_w4_const-grad_w4_ab,grad_w4); grad_w4=tl.where(tl.arange(0,16)==15,grad_w4_const,grad_w4)
                tl.atomic_add(w_grad_ptr+tl.arange(0,16)*stride_gw_op,grad_w4)

                # --- 레벨 1 노드(0, 1, 2, 3)의 역전파 ---
                # 노드 3
                a, b = input_6, input_7
                grad_parent = grad_node_1_3
                w_ptr = w_sm_base_ptr + 3 * stride_w_g
                w_grad_ptr = Grad_Weights_ptr + pid_k*stride_gw_oc + 3*stride_gw_g
                w3_1=tl.load(w_ptr+1*stride_w_op); w3_2=tl.load(w_ptr+2*stride_w_op); w3_3=tl.load(w_ptr+3*stride_w_op); w3_4=tl.load(w_ptr+4*stride_w_op); w3_5=tl.load(w_ptr+5*stride_w_op); w3_6=tl.load(w_ptr+6*stride_w_op); w3_7=tl.load(w_ptr+7*stride_w_op); w3_8=tl.load(w_ptr+8*stride_w_op); w3_9=tl.load(w_ptr+9*stride_w_op); w3_10=tl.load(w_ptr+10*stride_w_op); w3_11=tl.load(w_ptr+11*stride_w_op); w3_12=tl.load(w_ptr+12*stride_w_op); w3_13=tl.load(w_ptr+13*stride_w_op); w3_14=tl.load(w_ptr+14*stride_w_op);
                dy_da_3 = (w3_1*b + w3_2*(1-b) + w3_3) + (w3_4*(-b) + w3_6*(1-2*b) + w3_7*(1-b)) + (w3_8*(b-1) + w3_9*(2*b-1) + w3_11*b) + (-w3_12 + w3_13*(b-1) + w3_14*(-b))
                dy_db_3 = (w3_1*a + w3_2*(-a) + w3_4*(1-a)) + (w3_5 + w3_6*(1-2*a) + w3_7*(1-a)) + (w3_8*(a-1) + w3_9*(2*a-1) - w3_10) + (w3_11*(a-1) + w3_13*a + w3_14*(-a))
                grad_input_6 = grad_parent * dy_da_3
                grad_input_7 = grad_parent * dy_db_3
                grad_w3_ab=tl.sum((a*b)*grad_parent,0); grad_w3_a=tl.sum(a*grad_parent,0); grad_w3_b=tl.sum(b*grad_parent,0); grad_w3_const=tl.sum(grad_parent,0)
                grad_w3=tl.zeros((16,),dtype=grad_w3_ab.dtype); grad_w3=tl.where(tl.arange(0,16)==1,grad_w3_ab,grad_w3); grad_w3=tl.where(tl.arange(0,16)==2,grad_w3_a-grad_w3_ab,grad_w3); grad_w3=tl.where(tl.arange(0,16)==3,grad_w3_a,grad_w3); grad_w3=tl.where(tl.arange(0,16)==4,grad_w3_b-grad_w3_ab,grad_w3); grad_w3=tl.where(tl.arange(0,16)==5,grad_w3_b,grad_w3); grad_w3=tl.where(tl.arange(0,16)==6,grad_w3_a+grad_w3_b-2*grad_w3_ab,grad_w3); grad_w3=tl.where(tl.arange(0,16)==7,grad_w3_a+grad_w3_b-grad_w3_ab,grad_w3); grad_w3=tl.where(tl.arange(0,16)==8,grad_w3_const-(grad_w3_a+grad_w3_b-grad_w3_ab),grad_w3); grad_w3=tl.where(tl.arange(0,16)==9,grad_w3_const-(grad_w3_a+grad_w3_b-2*grad_w3_ab),grad_w3); grad_w3=tl.where(tl.arange(0,16)==10,grad_w3_const-grad_w3_b,grad_w3); grad_w3=tl.where(tl.arange(0,16)==11,grad_w3_const-grad_w3_b+grad_w3_ab,grad_w3); grad_w3=tl.where(tl.arange(0,16)==12,grad_w3_const-grad_w3_a,grad_w3); grad_w3=tl.where(tl.arange(0,16)==13,grad_w3_const-grad_w3_a+grad_w3_ab,grad_w3); grad_w3=tl.where(tl.arange(0,16)==14,grad_w3_const-grad_w3_ab,grad_w3); grad_w3=tl.where(tl.arange(0,16)==15,grad_w3_const,grad_w3)
                tl.atomic_add(w_grad_ptr+tl.arange(0,16)*stride_gw_op,grad_w3)

                # 노드 2
                a, b = input_4, input_5
                grad_parent = grad_node_1_2
                w_ptr = w_sm_base_ptr + 2 * stride_w_g
                w_grad_ptr = Grad_Weights_ptr + pid_k*stride_gw_oc + 2*stride_gw_g
                w2_1=tl.load(w_ptr+1*stride_w_op); w2_2=tl.load(w_ptr+2*stride_w_op); w2_3=tl.load(w_ptr+3*stride_w_op); w2_4=tl.load(w_ptr+4*stride_w_op); w2_5=tl.load(w_ptr+5*stride_w_op); w2_6=tl.load(w_ptr+6*stride_w_op); w2_7=tl.load(w_ptr+7*stride_w_op); w2_8=tl.load(w_ptr+8*stride_w_op); w2_9=tl.load(w_ptr+9*stride_w_op); w2_10=tl.load(w_ptr+10*stride_w_op); w2_11=tl.load(w_ptr+11*stride_w_op); w2_12=tl.load(w_ptr+12*stride_w_op); w2_13=tl.load(w_ptr+13*stride_w_op); w2_14=tl.load(w_ptr+14*stride_w_op);
                dy_da_2 = (w2_1*b + w2_2*(1-b) + w2_3) + (w2_4*(-b) + w2_6*(1-2*b) + w2_7*(1-b)) + (w2_8*(b-1) + w2_9*(2*b-1) + w2_11*b) + (-w2_12 + w2_13*(b-1) + w2_14*(-b))
                dy_db_2 = (w2_1*a + w2_2*(-a) + w2_4*(1-a)) + (w2_5 + w2_6*(1-2*a) + w2_7*(1-a)) + (w2_8*(a-1) + w2_9*(2*a-1) - w2_10) + (w2_11*(a-1) + w2_13*a + w2_14*(-a))
                grad_input_4 = grad_parent * dy_da_2
                grad_input_5 = grad_parent * dy_db_2
                grad_w2_ab=tl.sum((a*b)*grad_parent,0); grad_w2_a=tl.sum(a*grad_parent,0); grad_w2_b=tl.sum(b*grad_parent,0); grad_w2_const=tl.sum(grad_parent,0)
                grad_w2=tl.zeros((16,),dtype=grad_w2_ab.dtype); grad_w2=tl.where(tl.arange(0,16)==1,grad_w2_ab,grad_w2); grad_w2=tl.where(tl.arange(0,16)==2,grad_w2_a-grad_w2_ab,grad_w2); grad_w2=tl.where(tl.arange(0,16)==3,grad_w2_a,grad_w2); grad_w2=tl.where(tl.arange(0,16)==4,grad_w2_b-grad_w2_ab,grad_w2); grad_w2=tl.where(tl.arange(0,16)==5,grad_w2_b,grad_w2); grad_w2=tl.where(tl.arange(0,16)==6,grad_w2_a+grad_w2_b-2*grad_w2_ab,grad_w2); grad_w2=tl.where(tl.arange(0,16)==7,grad_w2_a+grad_w2_b-grad_w2_ab,grad_w2); grad_w2=tl.where(tl.arange(0,16)==8,grad_w2_const-(grad_w2_a+grad_w2_b-grad_w2_ab),grad_w2); grad_w2=tl.where(tl.arange(0,16)==9,grad_w2_const-(grad_w2_a+grad_w2_b-2*grad_w2_ab),grad_w2); grad_w2=tl.where(tl.arange(0,16)==10,grad_w2_const-grad_w2_b,grad_w2); grad_w2=tl.where(tl.arange(0,16)==11,grad_w2_const-grad_w2_b+grad_w2_ab,grad_w2); grad_w2=tl.where(tl.arange(0,16)==12,grad_w2_const-grad_w2_a,grad_w2); grad_w2=tl.where(tl.arange(0,16)==13,grad_w2_const-grad_w2_a+grad_w2_ab,grad_w2); grad_w2=tl.where(tl.arange(0,16)==14,grad_w2_const-grad_w2_ab,grad_w2); grad_w2=tl.where(tl.arange(0,16)==15,grad_w2_const,grad_w2)
                tl.atomic_add(w_grad_ptr+tl.arange(0,16)*stride_gw_op,grad_w2)

                # 노드 1
                a, b = input_2, input_3
                grad_parent = grad_node_1_1
                w_ptr = w_sm_base_ptr + 1 * stride_w_g
                w_grad_ptr = Grad_Weights_ptr + pid_k*stride_gw_oc + 1*stride_gw_g
                w1_1=tl.load(w_ptr+1*stride_w_op); w1_2=tl.load(w_ptr+2*stride_w_op); w1_3=tl.load(w_ptr+3*stride_w_op); w1_4=tl.load(w_ptr+4*stride_w_op); w1_5=tl.load(w_ptr+5*stride_w_op); w1_6=tl.load(w_ptr+6*stride_w_op); w1_7=tl.load(w_ptr+7*stride_w_op); w1_8=tl.load(w_ptr+8*stride_w_op); w1_9=tl.load(w_ptr+9*stride_w_op); w1_10=tl.load(w_ptr+10*stride_w_op); w1_11=tl.load(w_ptr+11*stride_w_op); w1_12=tl.load(w_ptr+12*stride_w_op); w1_13=tl.load(w_ptr+13*stride_w_op); w1_14=tl.load(w_ptr+14*stride_w_op);
                dy_da_1 = (w1_1*b + w1_2*(1-b) + w1_3) + (w1_4*(-b) + w1_6*(1-2*b) + w1_7*(1-b)) + (w1_8*(b-1) + w1_9*(2*b-1) + w1_11*b) + (-w1_12 + w1_13*(b-1) + w1_14*(-b))
                dy_db_1 = (w1_1*a + w1_2*(-a) + w1_4*(1-a)) + (w1_5 + w1_6*(1-2*a) + w1_7*(1-a)) + (w1_8*(a-1) + w1_9*(2*a-1) - w1_10) + (w1_11*(a-1) + w1_13*a + w1_14*(-a))
                grad_input_2 = grad_parent * dy_da_1
                grad_input_3 = grad_parent * dy_db_1
                grad_w1_ab=tl.sum((a*b)*grad_parent,0); grad_w1_a=tl.sum(a*grad_parent,0); grad_w1_b=tl.sum(b*grad_parent,0); grad_w1_const=tl.sum(grad_parent,0)
                grad_w1=tl.zeros((16,),dtype=grad_w1_ab.dtype); grad_w1=tl.where(tl.arange(0,16)==1,grad_w1_ab,grad_w1); grad_w1=tl.where(tl.arange(0,16)==2,grad_w1_a-grad_w1_ab,grad_w1); grad_w1=tl.where(tl.arange(0,16)==3,grad_w1_a,grad_w1); grad_w1=tl.where(tl.arange(0,16)==4,grad_w1_b-grad_w1_ab,grad_w1); grad_w1=tl.where(tl.arange(0,16)==5,grad_w1_b,grad_w1); grad_w1=tl.where(tl.arange(0,16)==6,grad_w1_a+grad_w1_b-2*grad_w1_ab,grad_w1); grad_w1=tl.where(tl.arange(0,16)==7,grad_w1_a+grad_w1_b-grad_w1_ab,grad_w1); grad_w1=tl.where(tl.arange(0,16)==8,grad_w1_const-(grad_w1_a+grad_w1_b-grad_w1_ab),grad_w1); grad_w1=tl.where(tl.arange(0,16)==9,grad_w1_const-(grad_w1_a+grad_w1_b-2*grad_w1_ab),grad_w1); grad_w1=tl.where(tl.arange(0,16)==10,grad_w1_const-grad_w1_b,grad_w1); grad_w1=tl.where(tl.arange(0,16)==11,grad_w1_const-grad_w1_b+grad_w1_ab,grad_w1); grad_w1=tl.where(tl.arange(0,16)==12,grad_w1_const-grad_w1_a,grad_w1); grad_w1=tl.where(tl.arange(0,16)==13,grad_w1_const-grad_w1_a+grad_w1_ab,grad_w1); grad_w1=tl.where(tl.arange(0,16)==14,grad_w1_const-grad_w1_ab,grad_w1); grad_w1=tl.where(tl.arange(0,16)==15,grad_w1_const,grad_w1)
                tl.atomic_add(w_grad_ptr+tl.arange(0,16)*stride_gw_op,grad_w1)

                # 노드 0
                a, b = input_0, input_1
                grad_parent = grad_node_1_0
                w_ptr = w_sm_base_ptr + 0 * stride_w_g
                w_grad_ptr = Grad_Weights_ptr + pid_k*stride_gw_oc + 0*stride_gw_g
                w0_1=tl.load(w_ptr+1*stride_w_op); w0_2=tl.load(w_ptr+2*stride_w_op); w0_3=tl.load(w_ptr+3*stride_w_op); w0_4=tl.load(w_ptr+4*stride_w_op); w0_5=tl.load(w_ptr+5*stride_w_op); w0_6=tl.load(w_ptr+6*stride_w_op); w0_7=tl.load(w_ptr+7*stride_w_op); w0_8=tl.load(w_ptr+8*stride_w_op); w0_9=tl.load(w_ptr+9*stride_w_op); w0_10=tl.load(w_ptr+10*stride_w_op); w0_11=tl.load(w_ptr+11*stride_w_op); w0_12=tl.load(w_ptr+12*stride_w_op); w0_13=tl.load(w_ptr+13*stride_w_op); w0_14=tl.load(w_ptr+14*stride_w_op);
                dy_da_0 = (w0_1*b + w0_2*(1-b) + w0_3) + (w0_4*(-b) + w0_6*(1-2*b) + w0_7*(1-b)) + (w0_8*(b-1) + w0_9*(2*b-1) + w0_11*b) + (-w0_12 + w0_13*(b-1) + w0_14*(-b))
                dy_db_0 = (w0_1*a + w0_2*(-a) + w0_4*(1-a)) + (w0_5 + w0_6*(1-2*a) + w0_7*(1-a)) + (w0_8*(a-1) + w0_9*(2*a-1) - w0_10) + (w0_11*(a-1) + w0_13*a + w0_14*(-a))
                grad_input_0 = grad_parent * dy_da_0
                grad_input_1 = grad_parent * dy_db_0
                grad_w0_ab=tl.sum((a*b)*grad_parent,0); grad_w0_a=tl.sum(a*grad_parent,0); grad_w0_b=tl.sum(b*grad_parent,0); grad_w0_const=tl.sum(grad_parent,0)
                grad_w0=tl.zeros((16,),dtype=grad_w0_ab.dtype); grad_w0=tl.where(tl.arange(0,16)==1,grad_w0_ab,grad_w0); grad_w0=tl.where(tl.arange(0,16)==2,grad_w0_a-grad_w0_ab,grad_w0); grad_w0=tl.where(tl.arange(0,16)==3,grad_w0_a,grad_w0); grad_w0=tl.where(tl.arange(0,16)==4,grad_w0_b-grad_w0_ab,grad_w0); grad_w0=tl.where(tl.arange(0,16)==5,grad_w0_b,grad_w0); grad_w0=tl.where(tl.arange(0,16)==6,grad_w0_a+grad_w0_b-2*grad_w0_ab,grad_w0); grad_w0=tl.where(tl.arange(0,16)==7,grad_w0_a+grad_w0_b-grad_w0_ab,grad_w0); grad_w0=tl.where(tl.arange(0,16)==8,grad_w0_const-(grad_w0_a+grad_w0_b-grad_w0_ab),grad_w0); grad_w0=tl.where(tl.arange(0,16)==9,grad_w0_const-(grad_w0_a+grad_w0_b-2*grad_w0_ab),grad_w0); grad_w0=tl.where(tl.arange(0,16)==10,grad_w0_const-grad_w0_b,grad_w0); grad_w0=tl.where(tl.arange(0,16)==11,grad_w0_const-grad_w0_b+grad_w0_ab,grad_w0); grad_w0=tl.where(tl.arange(0,16)==12,grad_w0_const-grad_w0_a,grad_w0); grad_w0=tl.where(tl.arange(0,16)==13,grad_w0_const-grad_w0_a+grad_w0_ab,grad_w0); grad_w0=tl.where(tl.arange(0,16)==14,grad_w0_const-grad_w0_ab,grad_w0); grad_w0=tl.where(tl.arange(0,16)==15,grad_w0_const,grad_w0)
                tl.atomic_add(w_grad_ptr+tl.arange(0,16)*stride_gw_op,grad_w0)

                # --- 3C: 입력 X에 대한 최종 그래디언트 누적 ---
                grad_x_base_ptr = Grad_X_padded_ptr + pid_b * stride_gxp_b
                
                ptr_grad_0 = grad_x_base_ptr + (tl.load(ici_base_ptr + 0*stride_ici_inp) + group_in_start)*stride_gxp_c + (tl.load(ipy_base_ptr + 0*stride_ipy_inp) + patch_start_h)*stride_gxp_h + (tl.load(ipx_base_ptr + 0*stride_ipx_inp) + patch_start_w)*stride_gxp_w
                tl.atomic_add(ptr_grad_0, grad_input_0, mask=final_load_mask)
                ptr_grad_1 = grad_x_base_ptr + (tl.load(ici_base_ptr + 1*stride_ici_inp) + group_in_start)*stride_gxp_c + (tl.load(ipy_base_ptr + 1*stride_ipy_inp) + patch_start_h)*stride_gxp_h + (tl.load(ipx_base_ptr + 1*stride_ipx_inp) + patch_start_w)*stride_gxp_w
                tl.atomic_add(ptr_grad_1, grad_input_1, mask=final_load_mask)
                ptr_grad_2 = grad_x_base_ptr + (tl.load(ici_base_ptr + 2*stride_ici_inp) + group_in_start)*stride_gxp_c + (tl.load(ipy_base_ptr + 2*stride_ipy_inp) + patch_start_h)*stride_gxp_h + (tl.load(ipx_base_ptr + 2*stride_ipx_inp) + patch_start_w)*stride_gxp_w
                tl.atomic_add(ptr_grad_2, grad_input_2, mask=final_load_mask)
                ptr_grad_3 = grad_x_base_ptr + (tl.load(ici_base_ptr + 3*stride_ici_inp) + group_in_start)*stride_gxp_c + (tl.load(ipy_base_ptr + 3*stride_ipy_inp) + patch_start_h)*stride_gxp_h + (tl.load(ipx_base_ptr + 3*stride_ipx_inp) + patch_start_w)*stride_gxp_w
                tl.atomic_add(ptr_grad_3, grad_input_3, mask=final_load_mask)
                ptr_grad_4 = grad_x_base_ptr + (tl.load(ici_base_ptr + 4*stride_ici_inp) + group_in_start)*stride_gxp_c + (tl.load(ipy_base_ptr + 4*stride_ipy_inp) + patch_start_h)*stride_gxp_h + (tl.load(ipx_base_ptr + 4*stride_ipx_inp) + patch_start_w)*stride_gxp_w
                tl.atomic_add(ptr_grad_4, grad_input_4, mask=final_load_mask)
                ptr_grad_5 = grad_x_base_ptr + (tl.load(ici_base_ptr + 5*stride_ici_inp) + group_in_start)*stride_gxp_c + (tl.load(ipy_base_ptr + 5*stride_ipy_inp) + patch_start_h)*stride_gxp_h + (tl.load(ipx_base_ptr + 5*stride_ipx_inp) + patch_start_w)*stride_gxp_w
                tl.atomic_add(ptr_grad_5, grad_input_5, mask=final_load_mask)
                ptr_grad_6 = grad_x_base_ptr + (tl.load(ici_base_ptr + 6*stride_ici_inp) + group_in_start)*stride_gxp_c + (tl.load(ipy_base_ptr + 6*stride_ipy_inp) + patch_start_h)*stride_gxp_h + (tl.load(ipx_base_ptr + 6*stride_ipx_inp) + patch_start_w)*stride_gxp_w
                tl.atomic_add(ptr_grad_6, grad_input_6, mask=final_load_mask)
                ptr_grad_7 = grad_x_base_ptr + (tl.load(ici_base_ptr + 7*stride_ici_inp) + group_in_start)*stride_gxp_c + (tl.load(ipy_base_ptr + 7*stride_ipy_inp) + patch_start_h)*stride_gxp_h + (tl.load(ipx_base_ptr + 7*stride_ipx_inp) + patch_start_w)*stride_gxp_w
                tl.atomic_add(ptr_grad_7, grad_input_7, mask=final_load_mask)
                # ==========================================================
                #       [끝] 3B & 3C: 실제 역전파 계산 및 누적
                # ==========================================================
