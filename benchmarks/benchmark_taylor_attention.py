import argparse
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import taylor_attention


def baseline_attention(q, k, v):
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("b h i j, b h j d -> b h i d", attn, v)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.time()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.time() - start) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim-head", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-lens", default="1024,4096,8192,16384")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--P", type=int, default=4)
    parser.add_argument("--block-q", type=int, default=512)
    parser.add_argument("--block-k", type=int, default=512)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x]

    print(f"Device: {device} dtype: {dtype} heads: {args.heads} dim_head: {args.dim_head}")
    for n in seq_lens:
        q = torch.randn(args.batch, args.heads, n, args.dim_head, device=device, dtype=dtype)
        k = torch.randn(args.batch, args.heads, n, args.dim_head, device=device, dtype=dtype)
        v = torch.randn(args.batch, args.heads, n, args.dim_head, device=device, dtype=dtype)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def run_baseline():
            return baseline_attention(q, k, v)

        base_time = _bench(run_baseline, args.warmup, args.iters, device)
        base_mem = None
        if device.type == "cuda":
            base_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        config = {
            "enabled": True,
            "P": args.P,
            "min_tokens": 0,
            "max_feature_dim_R": 500000,
            "block_size_q": args.block_q,
            "block_size_k": args.block_k,
            "eps": 1e-6,
            "fallback_on_negative": False,
            "allow_cross_attention": True,
            "max_head_dim": 256,
        }

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def run_taylor():
            return taylor_attention.taylor_attention(q, k, v, args.heads, skip_reshape=True, config=config)

        taylor_time = _bench(run_taylor, args.warmup, args.iters, device)
        taylor_mem = None
        if device.type == "cuda":
            taylor_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        print(f"N={n} baseline={base_time * 1000:.2f}ms taylor={taylor_time * 1000:.2f}ms")
        if base_mem is not None and taylor_mem is not None:
            print(f"  peak_mem_mb baseline={base_mem:.2f} taylor={taylor_mem:.2f}")


if __name__ == "__main__":
    main()
