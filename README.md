# ContinuousBatching in 100 line

just

```python
def flash_attn_varlen_func(
    q, # (total_q, nheads, headdim)
    k, # (total_k, nheads, headdim)
    v, # (total_v, nheads, headdim)
    cu_seqlens_q, # (batch_size + 1)
    cu_seqlens_k, # (batch_size + 1)
    max_seqlen_q, # 所有序列中最长的q的长度
    max_seqlen_k, # 所有序列中最长的k的长度
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    return_attn_probs=False,
):
    pass
```

detail see [my note](https://github.com/66RING/Notes/blob/master/universe/ml/flash_attn_varlen_batcing_api_usage.md)

try:

```python
import torch
from flash_attn import flash_attn_varlen_func, flash_attn_func

def main():
    dtype = torch.bfloat16
    HEAD = 2
    HEAD_DIM = 2
    seqlens = [1, 2, 3, 4]
    query = torch.empty(0, HEAD, HEAD_DIM, dtype=dtype).cuda()
    key = torch.empty(0, HEAD, HEAD_DIM, dtype=dtype).cuda()
    value = torch.empty(0, HEAD, HEAD_DIM, dtype=dtype).cuda()

    querys = []
    keys = []
    values = []
    for l in seqlens:
        q = torch.rand(l, HEAD, HEAD_DIM, dtype=dtype).cuda()
        k = torch.rand(l, HEAD, HEAD_DIM, dtype=dtype).cuda()
        v = torch.rand(l, HEAD, HEAD_DIM, dtype=dtype).cuda()
        querys.append(q)
        keys.append(k)
        values.append(v)
        query = torch.cat([query, q], dim=0)
        key = torch.cat([key, k], dim=0)
        value = torch.cat([value, v], dim=0)

    print("===Standard===")
    for q, k, v in zip(querys, keys, values):
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        out = flash_attn_func(q, k, v)
        print(out)
    print("=========\n")

    seq_len = torch.tensor(seqlens, dtype=torch.int32).cuda()
    # NOTE: flash_attn_varlen_func这个接口需要(bs + 1)长度的cu_seqlens_q和cu_seqlens_k
    prefill_start_pos = torch.cumsum(seq_len, dim=0, dtype=torch.int32) - seq_len
    prefill_start_pos = torch.cat([prefill_start_pos, torch.tensor([torch.sum(seq_len)], dtype=torch.int32, device="cuda")], dim=0)
    print(prefill_start_pos.shape)
    print(prefill_start_pos)

    print(query.shape, key.shape, value.shape)
    cu_seqlens_q = prefill_start_pos
    cu_seqlens_k = prefill_start_pos
    max_seqlen_q = max(seqlens)
    max_seqlen_k = max(seqlens)

    out = flash_attn_varlen_func(query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    acc = 0

    print("===Varlen===")
    for l in seqlens:
        print(out[acc:acc+l])
        acc += l
    print("=========\n")

if __name__ == "__main__":
    main()
```
