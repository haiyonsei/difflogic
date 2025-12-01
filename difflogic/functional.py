import torch
import numpy as np

BITS_TO_NP_DTYPE = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}


# | id | Operator             | AB=00 | AB=01 | AB=10 | AB=11 |
# |----|----------------------|-------|-------|-------|-------|
# | 0  | 0                    | 0     | 0     | 0     | 0     |
# | 1  | A and B              | 0     | 0     | 0     | 1     |
# | 2  | not(A implies B)     | 0     | 0     | 1     | 0     |
# | 3  | A                    | 0     | 0     | 1     | 1     |
# | 4  | not(B implies A)     | 0     | 1     | 0     | 0     |
# | 5  | B                    | 0     | 1     | 0     | 1     |
# | 6  | A xor B              | 0     | 1     | 1     | 0     |
# | 7  | A or B               | 0     | 1     | 1     | 1     |
# | 8  | not(A or B)          | 1     | 0     | 0     | 0     |
# | 9  | not(A xor B)         | 1     | 0     | 0     | 1     |
# | 10 | not(B)               | 1     | 0     | 1     | 0     |
# | 11 | B implies A          | 1     | 0     | 1     | 1     |
# | 12 | not(A)               | 1     | 1     | 0     | 0     |
# | 13 | A implies B          | 1     | 1     | 0     | 1     |
# | 14 | not(A and B)         | 1     | 1     | 1     | 0     |
# | 15 | 1                    | 1     | 1     | 1     | 1     |

def bin_op(a, b, i):
    assert a[0].shape == b[0].shape, (a[0].shape, b[0].shape)
    if a.shape[0] > 1:
        assert a[1].shape == b[1].shape, (a[1].shape, b[1].shape)

    if i == 0:
        return torch.zeros_like(a)
    elif i == 1:
        return a * b
    elif i == 2:
        return a - a * b
    elif i == 3:
        return a
    elif i == 4:
        return b - a * b
    elif i == 5:
        return b
    elif i == 6:
        return a + b - 2 * a * b
    elif i == 7:
        return a + b - a * b
    elif i == 8:
        return 1 - (a + b - a * b)
    elif i == 9:
        return 1 - (a + b - 2 * a * b)
    elif i == 10:
        return 1 - b
    elif i == 11:
        return 1 - b + a * b
    elif i == 12:
        return 1 - a
    elif i == 13:
        return 1 - a + a * b
    elif i == 14:
        return 1 - a * b
    elif i == 15:
        return torch.ones_like(a)


def bin_op_s(a, b, i_s):
    r = torch.zeros_like(a)
    for i in range(16):
        u = bin_op(a, b, i)
        r = r + i_s[..., i] * u
    return r


########################################################################################################################


def get_unique_connections(in_dim, out_dim, device='cuda'):
    #assert out_dim * 2 >= in_dim, 'The number of neurons ({}) must not be smaller than half of the number of inputs ' \
    #                              '({}) because otherwise not all inputs could be used or considered.'.format(
    #    out_dim, in_dim
    #)

    x = torch.arange(in_dim).long().unsqueeze(0)

    # Take pairs (0, 1), (2, 3), (4, 5), ...
    a, b = x[..., ::2], x[..., 1::2]
    if a.shape[-1] != b.shape[-1]:
        m = min(a.shape[-1], b.shape[-1])
        a = a[..., :m]
        b = b[..., :m]

    # If this was not enough, take pairs (1, 2), (3, 4), (5, 6), ...
    if a.shape[-1] < out_dim:
        a_, b_ = x[..., 1::2], x[..., 2::2]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        if a.shape[-1] != b.shape[-1]:
            m = min(a.shape[-1], b.shape[-1])
            a = a[..., :m]
            b = b[..., :m]

    # If this was not enough, take pairs with offsets >= 2:
    offset = 2
    while out_dim > a.shape[-1] > offset:
        a_, b_ = x[..., :-offset], x[..., offset:]
        a = torch.cat([a, a_], dim=-1)
        b = torch.cat([b, b_], dim=-1)
        offset += 1
        assert a.shape[-1] == b.shape[-1], (a.shape[-1], b.shape[-1])

    if a.shape[-1] >= out_dim:
        a = a[..., :out_dim]
        b = b[..., :out_dim]
    else:
        assert False, (a.shape[-1], offset, out_dim)

    perm = torch.randperm(out_dim)

    a = a[:, perm].squeeze(0)
    b = b[:, perm].squeeze(0)

    a, b = a.to(torch.int64), b.to(torch.int64)
    a, b = a.to(device), b.to(device)
    a, b = a.contiguous(), b.contiguous()
    return a, b

def get_unique_connections_in_groups(in_dim, out_dim, k, device='cuda'):
    """
    in_dim을 k개의 그룹으로 나누고, 각 그룹 내에서만 unique connection을 형성합니다.
    """
    # 각 차원이 k로 나누어 떨어지는지 확인
    assert in_dim % k == 0, f"in_dim {in_dim}은 k {k}로 나누어 떨어져야 합니다."
    assert out_dim % k == 0, f"out_dim {out_dim}은 k {k}로 나누어 떨어져야 합니다."

    # 그룹별 차원 계산
    in_dim_group = in_dim // k
    out_dim_group = out_dim // k

    all_a = []
    all_b = []

    # k개의 그룹에 대해 반복
    for i in range(k):
        # 현재 그룹의 입력 인덱스 오프셋 계산
        offset = i * in_dim_group
        
        # 그룹 내에서 고유 연결 생성 (get_unique_connections 함수 호출)
        # 단, 입력 인덱스에 오프셋을 더해 전체 차원에서의 위치를 맞춰줌
        group_a, group_b = get_unique_connections(in_dim_group, out_dim_group, device=device)
        
        all_a.append(group_a + offset)
        all_b.append(group_b + offset)

    # 모든 그룹의 결과를 하나의 텐서로 결합
    final_a = torch.cat(all_a, dim=0)
    final_b = torch.cat(all_b, dim=0)
    
    return final_a.contiguous(), final_b.contiguous()


def get_random_connections_in_groups(in_dim, out_dim, k_in, k_out, device='cuda'):
    """
    in_dim을 k_in개의 그룹으로, out_dim을 k_out개의 그룹으로 나눕니다.
    각 출력 그룹은 입력 그룹 중 하나에 매핑됩니다 (k_in < k_out인 경우 공유).
    각 출력 그룹에 대해, 매핑된 입력 그룹 내에서 고유한 뉴런을 '무작위 비복원 추출'하여
    필요한 만큼의 연결(쌍)을 생성합니다.

    Args:
        in_dim (int): 입력 차원.
        out_dim (int): 출력 차원.
        k_in (int): 입력 그룹의 수.
        k_out (int): 출력 그룹의 수.
        device (str): 텐서를 생성할 장치.
    """
    # 각 차원이 해당 k 값으로 나누어 떨어지는지 확인
    assert in_dim % k_in == 0, f"in_dim {in_dim}은 k_in {k_in}으로 나누어 떨어져야 합니다."
    assert out_dim % k_out == 0, f"out_dim {out_dim}은 k_out {k_out}으로 나누어 떨어져야 합니다."

    # 그룹별 차원 계산
    in_dim_group = in_dim // k_in
    out_dim_group = out_dim // k_out

    # [핵심] 각 그룹 내에서 고유한 쌍을 만들 충분한 입력이 있는지 확인
    # out_dim_group개의 쌍을 만들려면, 2 * out_dim_group 개의 고유한 입력 뉴런이 필요합니다.
    assert in_dim_group >= 2 * out_dim_group, \
        f"Group in_dim ({in_dim_group}) must be at least 2 * group out_dim ({out_dim_group}) to create unique pairs."

    all_a = []
    all_b = []

    # k_out개의 출력 그룹에 대해 반복
    for i in range(k_out):
        # 1. [변경점] 현재 출력 그룹(i)이 연결될 입력 그룹 인덱스를 결정합니다.
        # k_in이 k_out보다 작으면, 입력 그룹이 공유됩니다 (예: i % k_in).
        # input_group_idx = i % k_in

        # 제안하신 방식의 코드
        sharing_factor = k_out // k_in
        input_group_idx = i // sharing_factor
                
        # 2. 해당 입력 그룹의 인덱스 범위를 정의합니다.
        offset = input_group_idx * in_dim_group
        group_inputs = torch.arange(offset, offset + in_dim_group, device=device)
        
        # 3. 입력 그룹 내 인덱스를 무작위로 섞습니다 (비복원 추출).
        shuffled_indices = group_inputs[torch.randperm(in_dim_group, device=device)]
        
        # 4. 현재 출력 그룹에 필요한 만큼(out_dim_group * 2)의 고유한 인덱스를 선택합니다.
        num_to_select = out_dim_group * 2
        selected_indices = shuffled_indices[:num_to_select]
        
        # 5. 선택된 인덱스를 두 그룹으로 나누어 (a, b) 쌍을 만듭니다.
        pairs = selected_indices.view(2, out_dim_group)
        group_a = pairs[0]
        group_b = pairs[1]
        
        all_a.append(group_a)
        all_b.append(group_b)

    # 모든 출력 그룹의 결과를 하나의 텐서로 결합
    final_a = torch.cat(all_a, dim=0)
    final_b = torch.cat(all_b, dim=0)
    
    return final_a.contiguous(), final_b.contiguous()


def get_unique_connections_in_channel(in_dim, out_dim, k, device='cuda'):
    """
    in_dim을 k개의 그룹으로 나누고, 각 그룹 내에서만 unique connection을 형성합니다.
    """
    # 각 차원이 k로 나누어 떨어지는지 확인
    # 각 차원이 k 및 2로 나누어 떨어지는지 확인
    assert in_dim % (k * 2) == 0, f"in_dim {in_dim}은 k*2 ({k*2})로 나누어 떨어져야 합니다."
    assert out_dim % k == 0, f"out_dim {out_dim}은 k {k}로 나누어 떨어져야 합니다."

    # 그룹별 차원 계산
    in_dim_group = in_dim // k
    out_dim_group = out_dim // k
    
    # 각 그룹은 다시 2개의 '채널 그룹'으로 나뉨
    sub_group_dim = in_dim_group // 2

    all_a = []
    all_b = []

    # k개의 그룹에 대해 반복
    for i in range(k):
        # 1. 현재 그룹의 전체 오프셋 계산
        offset = i * in_dim_group
        
        # 2. 고정된 공간 패턴 생성 (패치 부분)
        # sub_group_dim 내에서 규칙 기반으로 고유한 쌍을 만듭니다.
        # 이 패턴은 항상 동일하게 생성됩니다 (예: (0,1), (2,3), ...).
        group_a_base, group_b_base = get_unique_connections(sub_group_dim, out_dim_group, device=device)
        
        # 3. 무작위 채널 그룹 선택
        # 0 또는 1을 랜덤하게 선택하여 앞 그룹을 쓸지, 뒤 그룹을 쓸지 결정합니다.
        channel_indice = torch.randint(0, 2, (out_dim_group,), device=device)

        # 4. 최종 인덱스 조합
        # channel_indice가 1일 경우, sub_group_dim 만큼의 오프셋을 더해줍니다.
        choice_offset = channel_indice * sub_group_dim
        
        final_a = group_a_base + offset + choice_offset
        final_b = group_b_base + offset + choice_offset

        all_a.append(final_a)
        all_b.append(final_b)

    # 모든 그룹의 결과를 하나의 텐서로 결합
    final_a = torch.cat(all_a, dim=0)
    final_b = torch.cat(all_b, dim=0)
    
    # 최종적으로 그룹 간 쌍의 순서를 한 번 더 섞어줌
    perm = torch.randperm(out_dim, device=device)
    final_a = final_a[perm]
    final_b = final_b[perm]

    return final_a.contiguous(), final_b.contiguous()





def get_pairwise_connections(in_dim: int, out_dim: int, device: str) -> torch.Tensor:
    """
    i번째 출력 뉴런을 (2*i)번째와 (2*i+1)번째 입력 뉴런에 연결하는
    고정된(deterministic) 인덱스를 생성합니다.
    모듈로 연산을 사용하여 인덱스가 in_dim을 초과하지 않도록 합니다.
    """
    indices_base = torch.arange(out_dim, device=device) * 2
    indices_a = indices_base % in_dim
    indices_b = (indices_base + 1) % in_dim
    return torch.stack([indices_a, indices_b])



########################################################################################################################


class GradFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f):
        ctx.f = f
        return x

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y * ctx.f, None


########################################################################################################################



