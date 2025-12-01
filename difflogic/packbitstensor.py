import torch
import numpy as np
from typing import Optional

# 기존 CUDA C++ 확장 임포트
import difflogic_cuda

# Triton 백엔드 래퍼 함수 임포트 (이전 답변에서 구현했다고 가정)
from .difflogic_triton import (
    tensor_packbits_triton,
    pruned_weighted_groupbitsum_triton
)


class PackBitsTensor:
    """
    A tensor class for efficiently storing and manipulating boolean tensors on the GPU
    by packing bits into integer types.

    This version supports both legacy CUDA C++ ('cuda') and modern Triton ('triton')
    backends for its core operations.
    """
    def __init__(self,
                 t: torch.BoolTensor,
                 bit_count: int = 32,
                 device: str = 'cuda',
                 implementation: str = 'cuda'):
        """
        :param t: The input boolean tensor (batch, features).
        :param bit_count: The number of bits for packing (e.g., 32 for int32).
        :param device: The device to use ('cuda').
        :param implementation: The backend to use ('cuda' or 'triton').
        """
        assert len(t.shape) == 2, f"Input tensor must be 2D, but got shape {t.shape}"
        assert device == 'cuda', "PackBitsTensor is only supported on CUDA devices."
        assert implementation in ['cuda', 'triton'], f"Unsupported implementation: {implementation}"

        self.bit_count = bit_count
        self.device = device
        self.implementation = implementation

        t_transposed = t.to(device).T.contiguous()

        if self.implementation == 'triton':
            self.t, self.pad_len = tensor_packbits_triton(t_transposed, self.bit_count)
        elif self.implementation == 'cuda':
            self.t, self.pad_len = difflogic_cuda.tensor_packbits_cuda(t_transposed, self.bit_count)

    def group_sum(self, k: int) -> torch.Tensor:
        """Performs a grouped sum over features with equal group sizes."""
        if self.implementation == 'triton':
            in_dim = self.t.shape[0]
            assert in_dim % k == 0, f"Input dim ({in_dim}) must be divisible by k ({k})."
            group_size = in_dim // k
            # Create a tensor representing equal group sizes
            group_sizes = torch.full((k,), group_size, device=self.device, dtype=torch.int)
            return pruned_weighted_groupbitsum_triton(self.t, k, group_sizes, weights=None)
        elif self.implementation == 'cuda':
            return difflogic_cuda.groupbitsum(self.t, self.pad_len, k)

    def pruned_group_sum(self, k: int, group_sizes: torch.Tensor) -> torch.Tensor:
        """Performs a grouped sum over features with variable group sizes."""
        if self.implementation == 'triton':
            return pruned_weighted_groupbitsum_triton(self.t, k, group_sizes, weights=None)
        elif self.implementation == 'cuda':
            return difflogic_cuda.pruned_groupbitsum(self.t, self.pad_len, k, group_sizes)

    def weighted_group_sum(self, k: int, weights: torch.Tensor) -> torch.Tensor:
        """Performs a weighted grouped sum over features with equal group sizes."""
        if self.implementation == 'triton':
            in_dim = self.t.shape[0]
            assert in_dim % k == 0, f"Input dim ({in_dim}) must be divisible by k ({k})."
            group_size = in_dim // k
            # Create a tensor representing equal group sizes
            group_sizes = torch.full((k,), group_size, device=self.device, dtype=torch.int)
            return pruned_weighted_groupbitsum_triton(self.t, k, group_sizes, weights=weights)
        elif self.implementation == 'cuda':
            return difflogic_cuda.weighted_groupbitsum(self.t, self.pad_len, k, weights)

    def pruned_weighted_group_sum(self, k: int, group_sizes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Performs a weighted grouped sum over features with variable group sizes."""
        if self.implementation == 'triton':
            return pruned_weighted_groupbitsum_triton(self.t, k, group_sizes, weights=weights)
        elif self.implementation == 'cuda':
            return difflogic_cuda.pruned_weighted_groupbitsum(self.t, self.pad_len, k, group_sizes, weights)

    # flatten and __repr__ methods remain unchanged
    def flatten(self, start_dim=0, end_dim=-1, **kwargs):
        return self

    def _get_member_repr(self, member):
        if len(member) <= 4:
            result = [(np.binary_repr(integer, width=self.bit_count))[::-1] for integer in member]
            return ' '.join(result)
        first_three = [(np.binary_repr(integer, width=self.bit_count))[::-1] for integer in member[:3]]
        sep = "..."
        final = np.binary_repr(member[-1], width=self.bit_count)[::-1]
        return f"{' '.join(first_three)} {sep} {final}"

    def __repr__(self):
        t_cpu = self.t.cpu().numpy()
        return '\n'.join([self._get_member_repr(item) for item in t_cpu])