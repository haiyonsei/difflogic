import torch
import difflogic_cuda
import numpy as np
from .functional import bin_op_s, get_unique_connections, GradFactor, get_pairwise_connections, get_unique_connections_in_groups, get_unique_connections_in_channel, get_random_connections_in_groups
from .packbitstensor import PackBitsTensor
from typing import Optional
import torch.nn as nn
from .difflogic_triton import *
import torch
import triton
import triton.language as tl
from typing import Optional
import torch.nn.functional as F
########################################################################################################################


########################################################################################################################
################  AUTOGRAD FUNCTION FOR TRITON
########################################################################################################################
class LogicLayerTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w):
        ctx.save_for_backward(x, a, b, w)
        # Call the Triton forward wrapper
        return logic_layer_forward_triton(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, w = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]: # grad for x
            # Call the Triton backward_x wrapper
            grad_x = logic_layer_backward_x_triton(x, a, b, w, grad_y)
        if ctx.needs_input_grad[3]: # grad for w
            # Call the Triton backward_w wrapper
            grad_w = logic_layer_backward_w_triton(x, a, b, grad_y)
        
        # The order of returned grads must match the order of inputs to forward()
        return grad_x, None, None, grad_w



class LogicLayer(torch.nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None, # 기본값 None
            connections: str = 'unique',
            hard_weights: bool = False,
            ste: bool = True,
            init: str = None,
            tau: float = 1.0,
            k: int = None,
            k_in: int = None,
    ):
        super().__init__()
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, 16, device=device))   

        self.init = init
        if self.init == "residual":
            with torch.no_grad():
                self.weights.fill_(0.0)
                feedforward_gate_idx = 3
                self.weights[:, feedforward_gate_idx] = 5.0 #approximately 90%
        elif self.init == "orpool":
            with torch.no_grad():
                self.weights.fill_(0.0)
                feedforward_gate_idx = 7
                self.weights[:, feedforward_gate_idx] = 5.0 #approximately 90%


        self.in_dim, self.out_dim, self.device = in_dim, out_dim, device
        self.grad_factor, self.hard_weights, self.ste = grad_factor, hard_weights, ste
        self.tau = tau
        self.k = k
        if k_in is not None:
            self.k_in = k_in
        else:
            self.k_in = k
        self.implementation = implementation
        if self.implementation is None and device == 'cuda':
            # cuda를 기본 CUDA 구현으로 설정
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        
        # 사용 가능한 구현 목록에 'triton' 추가
        assert self.implementation in ['cuda', 'python', 'triton'], self.implementation
        if self.implementation == 'triton':
            assert device == 'cuda', "Triton implementation is only available for CUDA devices."

        self.connections = connections
        assert self.connections in ['random', 'unique', 'pairwise', 'unique_in_groups', 'unique_in_channel', 'random_in_groups'], self.connections
        self.indices = self.get_connections(self.connections, device)

        # ❗ Triton의 backward_x는 atomic_add를 사용하므로 이 인덱스 맵이 필요 없음
        if self.implementation == 'cuda':
            # 기존 CUDA C++ 구현을 위한 인덱스 맵핑
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_0_np = self.indices[0].cpu().numpy()
            indices_1_np = self.indices[1].cpu().numpy()
            for y in range(out_dim):
                given_x_indices_of_y[indices_0_np[y]].append(y)
                given_x_indices_of_y[indices_1_np[y]].append(y)
            self.given_x_indices_of_y_start = torch.tensor(
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), device=device, dtype=torch.int64)
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist], dtype=torch.int64, device=device)

        self.num_neurons = out_dim
        self.num_weights = out_dim
        self._last_p = None

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            assert not self.training, 'PackBitsTensor is not supported for training mode.'
            assert self.device == 'cuda', 'PackBitsTensor is only supported for CUDA.'
            if self.implementation == 'triton':
                return self.forward_triton_eval(x)
            elif self.implementation == 'cuda':
                return self.forward_cuda_eval(x)
        else:
            if self.grad_factor != 1.:
                x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == 'triton':
            return self.forward_triton(x)
        elif self.implementation == 'cuda':
            return self.forward_cuda(x)
        elif self.implementation == 'python':
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)

    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        if self.indices[0].dtype == torch.int64 or self.indices[1].dtype == torch.int64:
            self.indices = self.indices[0].long(), self.indices[1].long()

        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            if self.ste: #STE
                weights = torch.nn.functional.softmax(self.weights/self.tau, dim=-1)
                weights.retain_grad()
                self._last_p = weights
                
                w_hard = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
                weights_ste = w_hard.detach() + (weights - weights.detach()) 
                weights = weights_ste
                #weights = torch.nn.functional.gumbel_softmax(self.weights, tau=self.tau, hard=True, dim=-1) 
            else:
                if self.hard_weights:
                    weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
                else:
                    weights = torch.nn.functional.softmax(self.weights/self.tau, dim=-1)
                    weights.retain_grad()
                    self._last_p = weights


            x = bin_op_s(a, b, weights)
        else:
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
            x = bin_op_s(a, b, weights)
        
        return x

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == 'cuda', x.device
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1)
        x = x.contiguous()

        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices

        if self.training:
            # [수정된 로직 시작]
            w = None
            if self.ste:
                # STE가 켜져 있으면 항상 soft + hard 조합 사용
                w_soft = torch.nn.functional.softmax(self.weights/self.tau, dim=-1).to(x.dtype)
                w_hard = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
                w = w_hard.detach() + (w_soft - w_soft.detach())
            elif self.hard_weights:
                # STE가 꺼져있고 hard_weights가 켜져 있으면 one_hot 사용
                w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            else:
                # 둘 다 꺼져 있으면 softmax 사용
                w = torch.nn.functional.softmax(self.weights/self.tau, dim=-1).to(x.dtype)
                w.retain_grad()
                self._last_p = w
            # [수정된 로직 끝]

            x = LogicLayerCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            # 평가 시에는 기존과 동일하게 one_hot 사용
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                x = LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)
        
        return x


    def forward_cuda_eval(self, x: PackBitsTensor):
        """
        WARNING: this is an in-place operation.

        :param x:
        :return:
        """
        assert not self.training
        assert isinstance(x, PackBitsTensor)
        assert x.t.shape[0] == self.in_dim, (x.t.shape, self.in_dim)

        a, b = self.indices
        w = self.weights.argmax(-1).to(torch.uint8)
        x.t = difflogic_cuda.eval(x.t, a, b, w)

        return x

    def forward_triton(self, x):
        assert x.device.type == 'cuda' and x.ndim == 2
        x = x.transpose(0, 1).contiguous()
        assert x.shape[0] == self.in_dim

        a, b = self.indices

        if self.training:
            w = torch.nn.functional.softmax(self.weights / self.tau, dim=-1).to(x.dtype)
            if self.ste:
                w_hard = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
                w = w_hard.detach() + (w - w.detach())
            # Triton autograd function 호출
            x = LogicLayerTritonFunction.apply(x, a, b, w).transpose(0, 1)
        else:
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                # 추론 시에는 autograd function을 거칠 필요 없이 바로 커널 호출 가능
                # return logic_layer_forward_triton(x, a, b, w).transpose(0, 1)
                # 일관성을 위해 Function 사용
                x = LogicLayerTritonFunction.apply(x, a, b, w).transpose(0, 1)
        
        return x

    def forward_triton_eval(self, x: PackBitsTensor):
        assert not self.training
        assert isinstance(x, PackBitsTensor) and x.t.shape[0] == self.in_dim

        a, b = self.indices
        w = self.weights.argmax(-1).to(torch.uint8)
        # Triton eval 커널 래퍼 호출
        x.t = logic_layer_eval_triton(x.t, a, b, w)
        return x

    def forward_p(self, x, p):
        """
        p-space 실험용 forward:
        - self.weights / STE / tau 등을 전혀 쓰지 않고,
        - 외부에서 주어진 gate 확률 p를 그대로 사용해서
        forward_python과 동일한 방식으로 a, b를 뽑아 bin_op_s를 수행한다.

        Args:
            x: 입력 activation, shape [..., in_dim]
            p: gate 확률 텐서, shape [out_dim, 16]
            (예: p = softmax(self.weights / self.tau) 형태로 미리 계산한 값)
        """
        # forward_python과 동일한 shape 체크
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        # indices dtype 정리 (int64 → long), forward_python과 동일
        if self.indices[0].dtype == torch.int64 or self.indices[1].dtype == torch.int64:
            self.indices = self.indices[0].long(), self.indices[1].long()

        # a, b 추출: [..., out_dim]
        a, b = x[..., self.indices[0]], x[..., self.indices[1]]

        # p shape / dtype 정리
        assert p.shape[0] == self.out_dim and p.shape[1] == 16, \
            f"p shape must be [out_dim, 16], got {p.shape}, expected [{self.out_dim}, 16]"

        # x와 device/dtype 맞춰줌
        p_used = p.to(dtype=a.dtype, device=a.device)

        # training/eval 여부와 상관 없이, 주어진 p를 그대로 사용
        x_out = bin_op_s(a, b, p_used)

        return x_out




    def extra_repr(self):
        return '{}, {}, {}, {}'.format(self.in_dim, self.out_dim, 'train' if self.training else 'eval', self.connections)

    def get_connections(self, connections, device='cuda'):
        #assert self.out_dim * 2 >= self.in_dim, 'The number of neurons ({}) must not be smaller than half of the ' \
        #                                        'number of inputs ({}) because otherwise not all inputs could be ' \
        #                                       'used or considered.'.format(self.out_dim, self.in_dim)
        if connections == 'random':
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        elif connections == 'unique':
            return get_unique_connections(self.in_dim, self.out_dim, device)
        elif connections == 'unique_in_groups':
            return get_unique_connections_in_groups(self.in_dim, self.out_dim, self.k, device)
        elif connections == 'unique_in_channel':
            return get_unique_connections_in_channel(self.in_dim, self.out_dim, self.k, device)
        elif connections == 'pairwise':
            return get_pairwise_connections(self.in_dim, self.out_dim, device)
        elif connections == 'random_in_groups' :
            return get_random_connections_in_groups(self.in_dim, self.out_dim, self.k_in, self.k, device)
        else:
            raise ValueError(connections)



########################################################################################################################


class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """
    def __init__(self, k: int, tau: float = 1., beta: float = 0.0, device='cuda', noise_prob: float = 0.0):
        """

        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device:
        """
        super().__init__()
        self.k = k
        self.tau = tau  
        self.device = device
        self.noise_prob = noise_prob
        self.beta = beta
    def forward(self, x):
        if self.training and self.noise_prob > 0.0:
            # (same shape) Bernoulli(p) mask
            m = torch.bernoulli(
                    torch.full_like(x, self.noise_prob)
                ).detach()                # mask 자체는 gradient X
            # XOR : 0→1, 1→0  (x must be 0/1)
            x = (x + m) % 2
            # ─ 실수 0‥1 입력을 “반전”하고 싶으면 대신 ↓ 사용
            #x = x * (1 - m) + (1 - x) * m


        if isinstance(x, PackBitsTensor):
            return x.group_sum(self.k)

        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        y = x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau
        return y + self.beta

    def extra_repr(self):
        return 'k={}, tau={}'.format(self.k, self.tau)

    

class WeightedGroupSum(nn.Module):
    """
    [UPDATED] 'ste' 플래그를 추가하여 두 가지 학습 모드를 지원합니다.
    - ste=True: STE를 사용한 Hard-quantization 학습 (기존 방식)
    - ste=False: STE를 사용하지 않는 Soft-weight 학습
    - eval 모드에서는 항상 Hard-quantization으로 동작합니다.
    """
    def __init__(self, k, in_dim, tau=1.0, beta=0.0,
                 reg_type="l1", init="ones", w_max=None, device='cuda',
                 ste: bool = True): # <-- [신규] ste 인자 추가
        super().__init__()
        self.k, self.tau, self.beta = k, tau, beta
        self.in_dim = in_dim
        self.reg_type, self.w_max = reg_type, w_max
        self.device = device
        self._init = init
        self.ste = ste # <-- [신규] ste 플래그 저장
        
        g = self.in_dim // self.k
        w0 = torch.ones(self.k, g, device=self.device) if self._init == "ones" else torch.randn(self.k, g, device=self.device)
        self.weight_raw = nn.Parameter(w0, requires_grad=True)

    @staticmethod
    def _ste_round(w):
        """Straight-Through Estimator for the rounding function."""
        return (w.round() - w).detach() + w

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            # ... (PackBitsTensor 관련 로직은 동일)
            pass
            
        g = x.shape[-1] // self.k

        # 연산에 사용할 가중치를 결정하는 로직
        if self.training:
            # --- 학습 모드 ---
            # weight_raw를 0 이상으로 유지
            continuous_weights = self.weight_raw.clamp(min=0)
            
            if self.ste:
                # STE 모드: 순전파에는 hard-quantized 값을, 역전파에는 STE를 사용
                weights_for_forward = self._ste_round(continuous_weights)
            else:
                # Soft 모드: 순전파에 continuous 값을 그대로 사용
                weights_for_forward = continuous_weights
        else:
            # --- 평가 모드 ---
            # 항상 hard-quantized 값을 사용하며, 그래디언트 추적 안 함
            with torch.no_grad():
                weights_for_forward = self.weight_raw.clamp(min=0).round()

        # w_max가 설정된 경우, 최대값 제한 적용
        if self.w_max is not None:
            weights_for_forward = weights_for_forward.clamp(max=self.w_max)

        # 최종 연산
        x = x.reshape(*x.shape[:-1], self.k, g)
        y = (x * weights_for_forward).sum(-1) / self.tau
        return y + self.beta

    def reg_loss(self):
        # L1 정규화는 항상 연속 파라미터인 weight_raw에 적용
        if self.reg_type == "l1":
            return self.weight_raw.abs().sum()
        elif self.reg_type == "l2":
            return (self.weight_raw ** 2).sum()
        return torch.tensor(0.0, device=self.weight_raw.device)



class PrunedGroupSum(nn.Module):
    """
    The PrunedGroupSum module.
    Handles variable group sizes and matches the GroupSum interface.
    """
    def __init__(self, k: int, group_sizes: torch.Tensor, tau: float = 1., beta: float = 0.0, noise_prob: float = 0.0):
        """
        :param k: number of intended real valued outputs, e.g., number of classes
        :param group_sizes: a 1D integer tensor containing the size of each group.
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param beta: a bias term added to the output.
        :param noise_prob: probability of flipping bits during training.
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.beta = beta
        self.noise_prob = noise_prob
        
        # register_buffer ensures the tensor is part of the module's state
        # and moves with it (e.g., .to(device)), but is not a trainable parameter.
        self.register_buffer('group_sizes', group_sizes)
        
        # Calculate total input dimension from the sum of group sizes.
        self.in_dim = int(group_sizes.sum().item())

    def forward(self, x: torch.Tensor | PackBitsTensor) -> torch.Tensor:
        # 1. (Optional) Apply noise during training, identical to GroupSum.
        if self.training and self.noise_prob > 0.0:
            # Create a Bernoulli mask with the same shape as the input.
            m = torch.bernoulli(torch.full_like(x, self.noise_prob)).detach()
            # Flip bits using XOR logic (for 0/1 inputs).
            x = (x + m) % 2

        # 2. Perform the group sum based on input type.
        # Path for optimized, bit-packed tensors
        if isinstance(x, PackBitsTensor):
            y = x.pruned_group_sum(self.k, self.group_sizes)
        # Path for regular torch.Tensors
        else:
            # Use torch.split for variable group sizes.
            split_sizes = self.group_sizes.tolist()
            chunks = torch.split(x, split_sizes, dim=-1)
            # Sum each chunk and stack the results.
            summed_chunks = [chunk.sum(dim=-1) for chunk in chunks]
            y = torch.stack(summed_chunks, dim=-1)

        # 3. Apply temperature scaling and bias, then return.
        # The CUDA kernel returns integers, so ensure the result is float.
        return y.float() / self.tau + self.beta

    def extra_repr(self) -> str:
        # Provides a descriptive string for printing the module.
        return (f'k={self.k}, in_dim={self.in_dim}, tau={self.tau}, beta={self.beta}, '
                f'noise_prob={self.noise_prob}')

class PrunedWeightedGroupSum(nn.Module):
    """
    The PrunedWeightedGroupSum module.
    
    This module performs a weighted sum over groups of features with variable sizes,
    which is essential for models that have been physically pruned. It combines the
    functionality of WeightedGroupSum (learnable weights) and PrunedGroupSum 
    (variable group sizes).
    """
    def __init__(self, 
                 k: int, 
                 group_sizes: torch.Tensor,
                 weights: torch.Tensor,
                 tau: float = 1.0, 
                 beta: float = 0.0, 
                 noise_prob: float = 0.0,
                 reg_type: str = "l1", 
                 w_max: float = None):
        """
        :param k: The number of output groups (e.g., number of classes).
        :param group_sizes: A 1D integer tensor specifying the number of features in each group.
        :param weights: A 1D float tensor containing the weights for all active features, concatenated.
                        The total number of weights must equal the sum of `group_sizes`.
        :param tau: The softmax temperature for scaling the output.
        :param beta: A learnable bias term added to the output.
        :param noise_prob: The probability of flipping input bits during training.
        :param reg_type: The type of regularization loss to apply to the weights ('l1' or 'l2').
        :param w_max: The maximum value to which weights are clamped.
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.beta = beta
        self.noise_prob = noise_prob
        self.reg_type = reg_type
        self.w_max = w_max

        # Ensure group_sizes is a 1D tensor
        assert group_sizes.dim() == 1, "group_sizes must be a 1D tensor."
        
        # Register group_sizes as a buffer. It's part of the module's state 
        # but not a trainable parameter.
        self.register_buffer('group_sizes', group_sizes.to(torch.int))
        
        # The total input dimension is the sum of all group sizes.
        self.in_dim = int(self.group_sizes.sum().item())

        # The number of weights must match the total number of input features.
        assert weights.numel() == self.in_dim, \
            f"Number of weights ({weights.numel()}) must match the sum of group_sizes ({self.in_dim})."

        # Register the 1D weights tensor as a trainable parameter.
        self.weight_raw = nn.Parameter(weights.clone().to(torch.float))

    @staticmethod
    def _ste_round(w: torch.Tensor) -> torch.Tensor:
        """Straight-Through Estimator for rounding."""
        return (w.round() - w).detach() + w

    def forward(self, x: torch.Tensor | PackBitsTensor) -> torch.Tensor:
        """
        Performs the forward pass for the pruned weighted group sum.
        """
        # Apply quantization and constraints to the weights.
        self.weight_raw.data.clamp_(min=0)
        w_q = self._ste_round(self.weight_raw)
        if self.w_max is not None:
            w_q.clamp_(max=self.w_max)

        # Handle optimized PackBitsTensor input by calling the CUDA kernel.
        if isinstance(x, PackBitsTensor):
            # Assumes a 'pruned_weighted_group_sum' method exists on PackBitsTensor
            # This method will call the 'pruned_weighted_groupbitsum' CUDA function.
            y = x.pruned_weighted_group_sum(self.k, self.group_sizes, w_q)
        
        # Handle standard torch.Tensor input.
        else:
            # Split the input tensor and the weights tensor according to the variable group sizes.
            x_chunks = torch.split(x, self.group_sizes.tolist(), dim=-1)
            w_chunks = torch.split(w_q, self.group_sizes.tolist(), dim=0)
            
            # Calculate the weighted sum for each chunk.
            summed_chunks = [
                (x_chunk * w_chunk).sum(dim=-1) 
                for x_chunk, w_chunk in zip(x_chunks, w_chunks)
            ]
            
            # Stack the results to form the final output tensor.
            y = torch.stack(summed_chunks, dim=-1)

        # Apply temperature scaling and bias.
        return y.float() / self.tau + self.beta

    def reg_loss(self) -> torch.Tensor:
        """
        Calculates the regularization loss for the weights, same as in WeightedGroupSum.
        """
        if self.reg_type == "l1":
            return self.weight_raw.abs().sum()
        elif self.reg_type == "l2":
            return (self.weight_raw ** 2).sum()
        return torch.tensor(0.0, device=self.weight_raw.device)

    def extra_repr(self) -> str:
        """
        Provides a descriptive string for printing the module.
        """
        return (f'k={self.k}, in_dim={self.in_dim}, tau={self.tau}, beta={self.beta}, '
                f'reg_type={self.reg_type}, noise_prob={self.noise_prob}')










class MaskedGroupSum(nn.Module):
    """
    The MaskedGroupSum module.

    This module learns a binary (0/1) mask to apply to input features before
    performing a group summation. The mask is learned via logits and a
    Straight-Through Estimator (STE) for binarization. This allows for feature
    selection within each group.
    """
    def __init__(self, 
                 k: int, 
                 in_dim: int, 
                 initial_T: float = 1.0,  # 초기 온도 T
                 output_scale: float = 1.0,
                 beta: float = 0.0,
                 device: str = 'cuda'):
        """
        :param k: 그룹의 수 (출력의 수).
        :param in_dim: 전체 입력 피처의 차원.
        :param initial_T: 어닐링을 시작할 초기 온도 값.
        :param output_scale: 합산된 결과에 적용될 스케일링 값.
        :param beta: 결과에 더해질 편향(bias) 값.
        :param device: 텐서를 위치시킬 장치.
        """
        super().__init__()
        
        # --- 기본 매개변수 ---
        self.k = k
        self.in_dim = in_dim
        self.output_scale = output_scale
        self.beta = beta
        self.device = device

        if in_dim % k != 0:
            raise ValueError(f"in_dim ({in_dim})은 k ({k})로 나누어 떨어져야 합니다.")
        
        self.group_size = in_dim // k

        # --- 학습 가능한 마스크 로짓 ---
        # 안정적인 학습을 위해 확률 대신 로짓을 학습합니다.
        # 5.0으로 초기화하여 시그모이드 통과 시 1에 가까운 값에서 시작하도록 설정.
        initial_logit_value = 0.0 
        mask_logits = torch.full(
            (self.k, self.group_size), 
            fill_value=initial_logit_value, 
            device=self.device
        )
        self.mask_logits = nn.Parameter(mask_logits, requires_grad=True)
        
        # --- 온도(T) ---
        # 온도는 학습 과정에서 변경되므로 nn.Parameter가 아닌 버퍼(buffer)로 등록합니다.
        self.register_buffer('T', torch.tensor(initial_T, device=self.device))

    def anneal_temperature(self, new_T: float):
        """
        학습 루프에서 온도를 업데이트하기 위한 헬퍼 함수.
        """
        self.T.fill_(new_T)

    def get_mask(self, hard: bool = False) -> torch.Tensor:
        """
        현재 마스크를 반환하는 함수.
        :param hard: True일 경우, 0 또는 1의 이진 마스크를 반환 (추론 시 사용).
                     False일 경우, 0과 1 사이의 소프트 마스크를 반환 (학습 시 사용).
        """
        # 1. 온도 T를 이용해 로짓 스케일링
        scaled_logits = self.mask_logits / self.T
        
        # 2. 시그모이드 함수를 통과시켜 소프트 마스크 생성
        soft_mask = torch.sigmoid(scaled_logits)
        
        if hard:
            # 추론 시에는 반올림하여 완전한 이진 마스크로 만듭니다.
            return torch.round(soft_mask)
        return soft_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        학습된 마스크를 적용하고 그룹 합산을 수행합니다.
        """
        # 입력 x를 그룹 형태로 변환: (..., k, group_size)
        x_reshaped = x.reshape(*x.shape[:-1], self.k, self.group_size)

        # --- 마스크 생성 및 적용 ---
        # 학습 중에는 그래디언트 흐름을 위해 소프트 마스크를 사용합니다.
        mask = self.get_mask(hard=False)
        x_masked = x_reshaped * mask

        # --- 합산 및 스케일링 ---
        y = x_masked.sum(dim=-1)
        y = y / self.output_scale + self.beta
        
        return y


    def extra_repr(self) -> str:
        """
        Provides a string representation for the module's configuration.
        """
        return f'k={self.k}, in_dim={self.in_dim}, tau={self.tau}, reg_type={self.reg_type}'

 


########################################################################################################################


class LogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y)
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda.backward_w(x, a, b, grad_y)
        return grad_x, None, None, grad_w, None, None, None


########################################################################################################################



class FusedLogicTreeConvFunction(torch.autograd.Function):
    """
    Triton 커널을 사용하여 Fused 연산을 수행하고, PyTorch 자동 미분 시스템에
    수동으로 구현한 backward pass를 연결합니다.
    """
    @staticmethod
    def forward(ctx, x, weights,
                input_channel_indices, input_pos_x_indices, input_pos_y_indices,
                kernel_size, stride, padding, groups, tree_depth):

        # 입력 텐서는 CUDA에 있어야 하며 contiguous해야 합니다.
        assert x.is_cuda and x.is_contiguous()
        assert weights.is_cuda and weights.is_contiguous()

        # 패딩 적용
        if padding > 0:
            x_padded = F.pad(x, (padding, padding, padding, padding))
        else:
            x_padded = x

        # 차원 계산
        batch_size, _, height, width = x.shape
        in_channels_total = x_padded.shape[1]
        out_channels = weights.shape[0]
        in_channels_per_group = in_channels_total // groups

        out_h_conv = (height + 2 * padding - kernel_size) // stride + 1
        out_w_conv = (width + 2 * padding - kernel_size) // stride + 1
        out_h_pool = out_h_conv // 2
        out_w_pool = out_w_conv // 2

        # 최종 출력 텐서 생성
        Y = torch.empty((batch_size, out_channels, out_h_pool, out_w_pool), device=x.device, dtype=x.dtype)
        
        # 그리드 및 블록 크기 설정
        BLOCK_SIZE_W = 32
        num_w_blocks = triton.cdiv(out_w_pool, BLOCK_SIZE_W)
        grid = (batch_size, out_channels, out_h_pool * num_w_blocks)
        
        # Y 텐서를 실제 계산 결과로 채웁니다.
        fused_logictree_conv_orpool_kernel[grid](
            # Tensors
            x_padded, Y, weights,
            input_channel_indices, input_pos_x_indices, input_pos_y_indices,
            # Dimensions
            batch_size, in_channels_total, height, width,
            out_channels, out_h_pool, out_w_pool, out_h_conv, out_w_conv,
            # Conv parameters
            kernel_size, stride, padding, groups, in_channels_per_group,
            # Strides for tensor access
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2), x_padded.stride(3),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            weights.stride(0), weights.stride(1), weights.stride(2),
            input_channel_indices.stride(0), input_channel_indices.stride(1),
            input_pos_x_indices.stride(0), input_pos_x_indices.stride(1),
            input_pos_y_indices.stride(0), input_pos_y_indices.stride(1),
            # Constants
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            NUM_W_BLOCKS=num_w_blocks,
        )

        # Backward Pass에서 사용할 텐서와 파라미터들을 ctx에 저장
        ctx.save_for_backward(x_padded, weights, Y,
                              input_channel_indices, input_pos_x_indices, input_pos_y_indices)
        ctx.dims = (out_h_conv, out_w_conv)
        ctx.params = (kernel_size, stride, padding, groups, tree_depth, in_channels_per_group)
        
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        # grad_Y는 Y와 동일한 shape를 가지는, 다음 레이어로부터 온 그래디언트입니다.
        if not grad_Y.is_contiguous():
            grad_Y = grad_Y.contiguous()

        # Forward에서 저장했던 텐서와 파라미터들을 불러옵니다.
        x_padded, weights, Y, input_channel_indices, input_pos_x_indices, input_pos_y_indices = ctx.saved_tensors
        out_h_conv, out_w_conv = ctx.dims
        kernel_size, stride, padding, groups, tree_depth, in_channels_per_group = ctx.params

        # 차원 정보
        batch_size, out_channels, out_h_pool, out_w_pool = Y.shape
        _, in_channels_total, height_pad, width_pad = x_padded.shape

        # 그래디언트를 저장할 출력 텐서들을 0으로 초기화합니다.
        grad_x_padded = torch.zeros_like(x_padded)
        grad_weights = torch.zeros_like(weights)

        # 그리드 및 블록 크기 설정 (Forward와 동일)
        BLOCK_SIZE_W = 32
        num_w_blocks = triton.cdiv(out_w_pool, BLOCK_SIZE_W)
        grid = (batch_size, out_channels, out_h_pool * num_w_blocks)
        
        # Backward 커널 호출
        fused_logictree_conv_backward_kernel[grid](
            # 입력 텐서 포인터
            grad_Y, Y, x_padded, weights,
            input_channel_indices, input_pos_x_indices, input_pos_y_indices,
            # 출력 텐서 포인터
            grad_x_padded, grad_weights,
            # 텐서 차원 및 파라미터
            batch_size, out_channels, out_h_pool, out_w_pool, out_h_conv, out_w_conv,
            kernel_size, stride, padding, groups, in_channels_per_group,
            # 스트라이드 정보
            *grad_Y.stride(), *Y.stride(), *x_padded.stride(), *weights.stride(),
            *grad_x_padded.stride(), *grad_weights.stride(),
            *input_channel_indices.stride(), *input_pos_x_indices.stride(), *input_pos_y_indices.stride(),
            # 컴파일 타임 상수
            BLOCK_SIZE_W=BLOCK_SIZE_W,
            NUM_W_BLOCKS=num_w_blocks,
            NUM_INPUTS_PER_TREE=2**tree_depth,
            TREE_DEPTH=tree_depth,
        )

        # 패딩된 x의 그래디언트에서 패딩 부분을 제거
        if padding > 0:
            grad_x = grad_x_padded[:, :, padding:-padding, padding:-padding]
        else:
            grad_x = grad_x_padded
        
        # forward 함수의 입력 순서에 맞춰 그래디언트를 반환합니다.
        # 참고: 이 return 문의 개수는 forward의 입력 인자 개수와 정확히 일치해야 합니다.
        return (
            grad_x,          # x에 대한 그래디언트
            grad_weights,    # weights에 대한 그래디언트
            None,            # input_channel_indices
            None,            # input_pos_x_indices
            None,            # input_pos_y_indices
            None,            # kernel_size
            None,            # stride
            None,            # padding
            None,            # groups
            None,            # tree_depth
        )


class FusedLogicTreeBlock(nn.Module):
    """
    LogicTreeConv2d와 ORPool2d의 연산을 하나의 Fused Kernel로 통합한 모듈.
    내부적으로 FusedLogicTreeConvFunction을 호출하여 backpropagation을 지원합니다.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, tree_depth: int,
                 stride: int = 1, padding: int = 0, groups: int = 1, init: str = 'residual', ste: bool = True, tau: float = 1.0, implementation: str = 'triton'):
        super().__init__()
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError('in_channels and out_channels must be divisible by groups')

        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, kernel_size
        self.stride, self.padding, self.groups = stride, padding, groups
        self.tree_depth, self.init, self.ste= tree_depth, init, ste
        self.tau = tau # __init__에서 tau를 올바르게 저장하도록 수정

        self.num_inputs_per_tree = 2 ** self.tree_depth
        self.num_gates_per_tree = self.num_inputs_per_tree - 1
        
        self.weights = nn.Parameter(torch.randn(self.out_channels, self.num_gates_per_tree, 16))

        if self.init == 'residual':
            with torch.no_grad():
                self.weights.fill_(0.0)
                feedforward_gate_idx = 3
                self.weights[:, :, feedforward_gate_idx] = 5.0

        self.in_channels_per_group = self.in_channels // self.groups
        
        # 인덱스 파라미터 생성 (학습되지 않음)
        if self.groups > 1:
            two_channels_per_tree = torch.randint(0, self.in_channels_per_group, (self.out_channels, 2))
            selector_indices = torch.randint(0, 2, (self.out_channels, self.num_inputs_per_tree))
            final_channel_indices = torch.gather(two_channels_per_tree, 1, selector_indices)
            self.input_channel_indices = nn.Parameter(final_channel_indices, requires_grad=False)
        else:
            self.input_channel_indices = nn.Parameter(
                torch.randint(0, self.in_channels_per_group, (self.out_channels, self.num_inputs_per_tree)),
                requires_grad=False
            )
        self.input_pos_x_indices = nn.Parameter(
            torch.randint(0, self.kernel_size, (self.out_channels, self.num_inputs_per_tree)),
            requires_grad=False
        )
        self.input_pos_y_indices = nn.Parameter(
            torch.randint(0, self.kernel_size, (self.out_channels, self.num_inputs_per_tree)),
            requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Train/Eval 모드에 따라 가중치를 다르게 처리하고,
        Contiguous 입력 보장 후 FusedLogicTreeConvFunction에 전달합니다.
        """
        x = x.contiguous()

        if self.training:
            # 학습 시: Softmax와 STE를 적용한 가중치 w를 생성
            w = torch.nn.functional.softmax(self.weights / self.tau, dim=-1).to(x.dtype)
            if self.ste:
                w_hard = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
                w = w_hard.detach() + (w - w.detach())
            
            return FusedLogicTreeConvFunction.apply(
                x, w,
                self.input_channel_indices, self.input_pos_x_indices, self.input_pos_y_indices,
                self.kernel_size, self.stride, self.padding, self.groups, self.tree_depth
            )

        else: # 추론 시
            # 추론 시: One-hot 가중치 w를 생성
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            
            with torch.no_grad():
                return FusedLogicTreeConvFunction.apply(
                    x, w,
                    self.input_channel_indices, self.input_pos_x_indices, self.input_pos_y_indices,
                    self.kernel_size, self.stride, self.padding, self.groups, self.tree_depth
                )