# PyTorch Flash-Attn Demo

从 `softmax` 出发, 用 `pytorch` 模拟 `flash-attn v1/v2` 中的主要算法

## Project Structure

- [`softmax.py`](./softmax.py): softmax 实现
- [`softmax.pdf`](./softmax.pdf): softmax 公式推导原理
- [`attention.py`](./attention.py): 包含原始 `attention` 计算和 `flash_attention_v1`, `flash_attention_v2` 计算的实现
    - 为简明起见, 算法中并没有添加 `mask` 和 `dropped out`
- [`flash_attn_v1.pdf`](./flash_attn_v1.pdf): `flash_attention_v1` 的核心算法原理公式推导
- [`flash_attn_v2.pdf`](./flash_attn_v2.pdf): `flash_attention_v2` 的核心算法原理公式推导以及其在 v1 版本上的改进