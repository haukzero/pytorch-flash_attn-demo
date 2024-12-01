import math
import torch
from torch.nn import functional as F


def ori_softmax(x: torch.Tensor):
    return x.exp() / x.exp().sum()


def safe_softmax(x: torch.Tensor):
    m = x.max()
    return (x - m).exp() / (x - m).exp().sum()


def online_softmax(x: torch.Tensor):
    assert x.dim() == 1
    m, l = -torch.inf, 0
    for i in range(x.shape[0]):
        new_m = max(m, x[i])
        l = l * math.exp(m - new_m) + math.exp(x[i] - new_m)
        m = new_m
    return (x - m).exp() / l


def online_softmax_block(x: torch.Tensor, num_blocks: int = 2):
    assert x.dim() == 1 and x.shape[0] % num_blocks == 0
    x_blocks = x.view(num_blocks, -1)
    m = torch.full((num_blocks,), -torch.inf, device=x.device)
    l = torch.zeros_like(m)
    m_global, l_global = -torch.inf, 0
    for i in range(num_blocks):
        x_block = x_blocks[i]
        for j in range(x_block.shape[-1]):
            new_m = max(m[i], x_block[j])
            l[i] = l[i] * math.exp(m[i] - new_m) + math.exp(x_block[j] - new_m)
            m[i] = new_m
        m_global = max(m_global, m[i])
    for i in range(num_blocks):
        l_global += l[i] * math.exp(m[i] - m_global)
    return (x - m_global).exp() / l_global


def online_softmax_block_2d(x: torch.Tensor, num_blocks: int = 2):
    assert x.dim() == 2 and x.shape[1] % num_blocks == 0
    bsz, hidden_dim = x.shape
    block_size = hidden_dim // num_blocks
    x_blocks = x.chunk(num_blocks, dim=1)  # (num_blocks, bsz, block_size)
    m = torch.full((num_blocks, bsz), -torch.inf, device=x.device)
    l = torch.zeros_like(m)
    global_m = torch.full((bsz, 1), -torch.inf, device=x.device)
    global_l = torch.zeros_like(global_m)
    for i, x_block in enumerate(x_blocks):
        # x_block: (bsz, block_size)
        for j in range(block_size):
            new_m = torch.max(m[i], x_block[:, j])
            l[i] = l[i] * torch.exp(m[i] - new_m) + torch.exp(x_block[:, j] - new_m)
            m[i] = new_m
        global_m = torch.max(global_m, m[i].view(bsz, 1))
    for i in range(num_blocks):
        global_l += l[i].view(bsz, 1) * torch.exp(m[i].view(bsz, 1) - global_m)
    return (x - global_m).exp() / global_l


def check(func, shape=128, device="cpu", eps=1e-16, seed=None, *args, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(shape, device=device)
    y1 = F.softmax(x, dim=-1)
    y2 = func(x, *args, **kwargs)
    if torch.allclose(y1, y2, atol=eps):
        print("Passed")
    else:
        print("Failed")


if __name__ == "__main__":
    check(
        func=online_softmax_block_2d,
        shape=(4, 1024),
        seed=0,
        device="cuda:0",
        num_blocks=8,
    )
