import torch
import difflogic_cuda
import numpy as np
from .functional import bin_op_s, get_unique_connections, GradFactor
from .packbitstensor import PackBitsTensor
from typing import Optional
import torch.nn as nn

########################################################################################################################


class LogicLayer(torch.nn.Module):
    """
    The core module for differentiable logic gate networks. Provides a differentiable logic gate layer.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
            hard_weights: bool = False,
            ste: bool = True,
    ):
        """
        :param in_dim:      input dimensionality of the layer
        :param out_dim:     output dimensionality of the layer
        :param device:      device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python'). cuda is around 100x faster than python
        :param connections: method for initializing the connectivity of the logic gate net
        """
        super().__init__()
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, 16, device=device))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        self.hard_weights = hard_weights
        """
        The CUDA implementation is the fast implementation. As the name implies, the cuda implementation is only 
        available for device='cuda'. The `python` implementation exists for 2 reasons:
        1. To provide an easy-to-understand implementation of differentiable logic gate networks 
        2. To provide a CPU implementation of differentiable logic gate networks 
        """
        self.implementation = implementation
        if self.implementation is None and device == 'cuda':
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        assert self.implementation in ['cuda', 'python'], self.implementation
    

        self.connections = connections
        assert self.connections in ['random', 'unique'], self.connections
        self.indices = self.get_connections(self.connections, device)

        if self.implementation == 'cuda':
            """
            Defining additional indices for improving the efficiency of the backward of the CUDA implementation.
            """
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
        self.ste = ste
        self.tau = 1.0

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            assert not self.training, 'PackBitsTensor is not supported for the differentiable training mode.'
            assert self.device == 'cuda', 'PackBitsTensor is only supported for CUDA, not for {}. ' \
                                          'If you want fast inference on CPU, please use CompiledDiffLogicModel.' \
                                          ''.format(self.device)

        else:
            if self.grad_factor != 1.:
                x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == 'cuda':
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
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
                w_hard = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
                weights_ste = w_hard.detach() + (weights - weights.detach()) 
                weights = weights_ste
                #weights = torch.nn.functional.gumbel_softmax(self.weights, tau=self.tau, hard=True, dim=-1) 
            else:
                if self.hard_weights:
                    weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
                else:
                    weights = torch.nn.functional.softmax(self.weights/self.tau, dim=-1)

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
            # Use softmax instead of one_hot to ensure gradients are properly propagated
            w = torch.nn.functional.softmax(self.weights/self.tau, dim=-1).to(x.dtype)
            if self.ste:
                w_hard = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
                w = w_hard.detach() + (w - w.detach())
            return LogicLayerCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                return LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)

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

    def extra_repr(self):
        return '{}, {}, {}'.format(self.in_dim, self.out_dim, 'train' if self.training else 'eval')

    def get_connections(self, connections, device='cuda'):
        assert self.out_dim * 2 >= self.in_dim, 'The number of neurons ({}) must not be smaller than half of the ' \
                                                'number of inputs ({}) because otherwise not all inputs could be ' \
                                                'used or considered.'.format(self.out_dim, self.in_dim)
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

    

# ----- WeightedGroupSum (이전 답변의 구현 가정) -------------------------
class WeightedGroupSum(nn.Module):
    def __init__(self, k, input_dim, tau=1.0, beta=0.0, noise_prob=0.0,
                 reg_type="l1", init="ones", w_max=None, device='cuda'):
        super().__init__()
        self.k, self.tau, self.beta = k, tau, beta
        self.input_dim = input_dim
        self.noise_prob, self.reg_type, self.w_max = noise_prob, reg_type, w_max
        self.device = device
        self._init = init
        w0 = torch.ones(self.k, self.input_dim//self.k, device=self.device) \
                if self._init == "ones" else torch.randn(self.k, g, device=self.device)
        self.weight_raw = nn.Parameter(w0, requires_grad=True)

    @staticmethod
    def _ste_round(w):           # STE quantization
        return (w.round() - w).detach() + w

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            raise NotImplementedError
        g = x.shape[-1] // self.k

        # force weight_raw to be greater than or equal to 0
        self.weight_raw.data.clamp_(min=0)
        w_q = self._ste_round(self.weight_raw)
        if self.w_max is not None:
            w_q = w_q.clamp_(0, self.w_max)


        x = x.reshape(*x.shape[:-1], self.k, g)
        y = (x * w_q).sum(-1) / self.tau
        return y + self.beta

    def reg_loss(self):
        if self.reg_type == "l1":
            return self.weight_raw.abs().sum()
        elif self.reg_type == "l2":
            return (self.weight_raw ** 2).sum()
        return torch.tensor(0.0, device=self.weight_raw.device)


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
