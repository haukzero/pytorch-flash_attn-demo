import torch
from torch.nn import functional as F

BLOCK_SIZE = 256


def ori_attn(Q, K, V):
    scale = 1 / (Q.shape[-1] ** 0.5)
    a = Q @ K.transpose(-1, -2) * scale
    a = F.softmax(a, dim=-1)
    o = a @ V
    return o


def flash_attention_v1_forward(Q, K, V):
    scale = 1 / (K.shape[-1] ** 0.5)
    Q = Q * scale

    O = torch.zeros_like(Q)  # (bsz, num_heads, seq_len, head_dim)
    l = torch.zeros(Q.shape[:-1])[..., None]  # (bsz, num_heads, seq_len, 1)
    m = torch.ones(Q.shape[:-1])[..., None] * -torch.inf  # (bsz, num_heads, seq_len, 1)
    l, m = l.to(Q.device), m.to(Q.device)

    Br, Bc = min(BLOCK_SIZE, Q.shape[-1]), BLOCK_SIZE
    Q_BLOCKS = torch.split(Q, Br, dim=-2)  # (Tr, bsz, num_heads, Br, head_dim)
    K_BLOCKS = torch.split(K, Bc, dim=-2)  # (Tc, bsz, num_heads, Bc, head_dim)
    V_BLOCKS = torch.split(V, Bc, dim=-2)  # (Tc, bsz, num_heads, Bc, head_dim)
    Tr, Tc = len(Q_BLOCKS), len(K_BLOCKS)
    O_BLOCKS = list(torch.split(O, Br, dim=-2))  # (Tr, bsz, num_heads, Br, head_dim)
    l_BLOCKS = list(torch.split(l, Br, dim=-2))  # (Tr, bsz, num_heads, Br, 1)
    m_BLOCKS = list(torch.split(m, Br, dim=-2))  # (Tr, bsz, num_heads, Br, 1)

    for j in range(Tc):
        Kj, Vj = K_BLOCKS[j], V_BLOCKS[j]  # (bsz, num_heads, Bc, head_dim)

        for i in range(Tr):
            Qi = Q_BLOCKS[i]  # (bsz, num_heads, Br, head_dim)
            li = l_BLOCKS[i]  # (bsz, num_heads, Br, 1)
            mi = m_BLOCKS[i]  # (bsz, num_heads, Br, 1)

            S_ij = Qi @ Kj.transpose(-1, -2)  # (bsz, num_heads, Br, Bc)

            m_ij = torch.max(S_ij, dim=-1, keepdim=True)[0]  # (bsz, num_heads, Br, 1)
            P_ij = torch.exp(S_ij - m_ij)  # (bsz, num_heads, Br, Bc)
            l_ij = torch.sum(P_ij, dim=-1, keepdim=True)  # (bsz, num_heads, Br, 1)

            mi_new = torch.max(mi, m_ij)  # (bsz, num_heads, Br, 1)
            li_new = li * torch.exp(mi - mi_new) + l_ij * torch.exp(
                m_ij - mi_new
            )  # (bsz, num_heads, Br, 1)

            O_BLOCKS[i] = (
                O_BLOCKS[i] * li * torch.exp(mi - mi_new)
                + torch.exp(m_ij - mi_new) * (P_ij @ Vj)
            ) / li_new
            l_BLOCKS[i] = li_new
            m_BLOCKS[i] = mi_new

    O = torch.cat(O_BLOCKS, dim=-2)
    l = torch.cat(l_BLOCKS, dim=-2)
    m = torch.cat(m_BLOCKS, dim=-2)
    return O, l, m


def flash_attn_v1(Q, K, V):
    O, _, _ = flash_attention_v1_forward(Q, K, V)
    return O


def flash_attention_v2_forward(Q, K, V):
    scale = 1 / (K.shape[-1] ** 0.5)
    Q = Q * scale

    O = torch.zeros_like(Q)  # (bsz, num_heads, seq_len, head_dim)
    l = torch.zeros(Q.shape[:-1])[..., None]  # (bsz, num_heads, seq_len, 1)
    m = torch.ones(Q.shape[:-1])[..., None] * -torch.inf  # (bsz, num_heads, seq_len, 1)
    l, m = l.to(Q.device), m.to(Q.device)

    Br, Bc = min(BLOCK_SIZE, Q.shape[-1]), BLOCK_SIZE
    Q_BLOCKS = torch.split(Q, Br, dim=-2)  # (Tr, bsz, num_heads, Br, head_dim)
    K_BLOCKS = torch.split(K, Bc, dim=-2)  # (Tc, bsz, num_heads, Bc, head_dim)
    V_BLOCKS = torch.split(V, Bc, dim=-2)  # (Tc, bsz, num_heads, Bc, head_dim)
    Tr, Tc = len(Q_BLOCKS), len(K_BLOCKS)
    O_BLOCKS = list(torch.split(O, Br, dim=-2))  # (Tr, bsz, num_heads, Br, head_dim)
    l_BLOCKS = list(torch.split(l, Br, dim=-2))  # (Tr, bsz, num_heads, Br, 1)
    m_BLOCKS = list(torch.split(m, Br, dim=-2))  # (Tr, bsz, num_heads, Br, 1)

    # 以 Q 作为主循环
    for i in range(Tr):
        Qi = Q_BLOCKS[i]  # (bsz, num_heads, Br, head_dim)
        Oi = O_BLOCKS[i]  # (bsz, num_heads, Br, head_dim)
        li = l_BLOCKS[i]  # (bsz, num_heads, Br, 1)
        mi = m_BLOCKS[i]  # (bsz, num_heads, Br, 1)

        for j in range(Tc):
            Kj, Vj = K_BLOCKS[j], V_BLOCKS[j]  # (bsz, num_heads, Bc, head_dim)

            Sij = Qi @ Kj.transpose(-1, -2)  # (bsz, num_heads, Br, Bc)
            m_ij = torch.max(Sij, dim=-1, keepdim=True)[0]
            mi_new = torch.max(mi, m_ij)  # (bsz, num_heads, Br, 1)
            Pij = torch.exp(Sij - mi_new)  # (bsz, num_heads, Br, Bc)
            li = torch.exp(mi - mi_new) * li + torch.sum(Pij, dim=-1, keepdim=True)
            Oi = Oi * torch.exp(mi - mi_new) + Pij @ Vj
            mi = mi_new

        # 最后再做一次 scale
        O_BLOCKS[i] = Oi / li
        m_BLOCKS[i] = mi
        l_BLOCKS[i] = li

    O = torch.cat(O_BLOCKS, dim=-2)
    l = torch.cat(l_BLOCKS, dim=-2)
    m = torch.cat(m_BLOCKS, dim=-2)
    return O, l, m


def flash_attn_v2(Q, K, V):
    O, _, _ = flash_attention_v2_forward(Q, K, V)
    return O


def check(func, shape, decode=False, eps=1e-6, seed=None, *args, **kwargs):
    if decode:
        dim = len(shape)
        if dim == 2:
            q_shape = (1, shape[1])
        elif dim == 3:
            q_shape = (shape[0], 1, shape[2])
        elif dim == 4:
            q_shape = (shape[0], shape[1], 1, shape[3])
        else:
            raise ValueError("Invalid shape")
    else:
        q_shape = shape
    if seed is not None:
        torch.manual_seed(seed)
    Q = torch.randn(q_shape).cuda()
    K = torch.randn(shape).cuda()
    V = torch.randn(shape).cuda()
    y1 = ori_attn(Q, K, V)
    y2 = func(Q, K, V, *args, **kwargs)
    if torch.allclose(y1, y2, atol=eps):
        print("Passed")
    else:
        print("Failed")


if __name__ == "__main__":
    check(
        func=flash_attn_v2,
        shape=(1, 8, 512, 1024),
        decode=True,
        seed=0,
    )
