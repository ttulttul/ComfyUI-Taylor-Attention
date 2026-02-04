import argparse
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import taylor_attention


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
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--patch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim-head", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--depth-single", type=int, default=2)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--P", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    try:
        sys.path.append("../ComfyUI")
        from comfy.ldm.flux.model import Flux
        import comfy.ops
    except Exception as exc:
        raise SystemExit(f"Unable to import ComfyUI Flux model: {exc}")

    hidden = args.heads * args.dim_head
    params = dict(
        in_channels=4,
        out_channels=4,
        vec_in_dim=None,
        context_in_dim=hidden,
        hidden_size=hidden,
        mlp_ratio=4.0,
        num_heads=args.heads,
        depth=args.depth,
        depth_single_blocks=args.depth_single,
        axes_dim=[args.dim_head],
        theta=10000,
        patch_size=args.patch,
        qkv_bias=False,
        guidance_embed=False,
        txt_ids_dims=[],
        global_modulation=False,
        mlp_silu_act=False,
        ops_bias=True,
        default_ref_method="offset",
        ref_index_scale=1.0,
        yak_mlp=False,
        txt_norm=False,
    )

    model = Flux(dtype=dtype, device=device, operations=comfy.ops.disable_weight_init, **params).to(device)

    h_tokens = args.height // args.patch
    w_tokens = args.width // args.patch
    tokens = h_tokens * w_tokens

    img = torch.randn(1, tokens, params["in_channels"] * args.patch * args.patch, device=device, dtype=dtype)
    img_ids = torch.zeros(1, tokens, len(params["axes_dim"]), device=device, dtype=torch.float32)
    txt = torch.randn(1, args.context_len, params["context_in_dim"], device=device, dtype=dtype)
    txt_ids = torch.zeros(1, args.context_len, len(params["axes_dim"]), device=device, dtype=torch.float32)
    timesteps = torch.zeros(1, device=device, dtype=dtype)

    def run_baseline():
        return model.forward_orig(img, img_ids, txt, txt_ids, timesteps, None, None, None, transformer_options={})

    config = {
        "enabled": True,
        "P": args.P,
        "min_tokens": 0,
        "max_feature_dim_R": 500000,
        "block_size_q": 512,
        "block_size_k": 512,
        "eps": 1e-6,
        "fallback_on_negative": False,
        "allow_cross_attention": True,
        "max_head_dim": 256,
    }

    def run_taylor():
        topts = {
            "optimized_attention_override": taylor_attention.taylor_attention_override,
            "taylor_attention": config,
        }
        return model.forward_orig(img, img_ids, txt, txt_ids, timesteps, None, None, None, transformer_options=topts)

    base_time = _bench(run_baseline, args.warmup, args.iters, device)
    taylor_time = _bench(run_taylor, args.warmup, args.iters, device)

    print(f"Tokens={tokens} baseline={base_time * 1000:.2f}ms taylor={taylor_time * 1000:.2f}ms")


if __name__ == "__main__":
    main()
