# SATA-Speedup: Taylor Attention Backend for ComfyUI

This custom node adds a runtime-switchable attention backend for diffusion transformers (Flux/Flux2) using a truncated Taylor expansion with a symmetry-aware feature map. It is designed for large token counts where quadratic attention becomes expensive.

## Install

1. Copy this folder into your ComfyUI `custom_nodes` directory.
2. Restart ComfyUI.
3. Use the **Taylor Attention Backend** node from the `advanced/attention` category.

## Usage

- Add **Taylor Attention Backend** to your graph.
- Set `backend` to `taylor` to enable, or `standard` to disable.
- Key parameters:
  - `P`: number of Taylor terms (default 4).
  - `min_tokens`: only use Taylor when tokens >= this threshold (default 10000).
  - `max_feature_dim_R`: safety cap for feature dimension.
  - `block_size_q` / `block_size_k`: block sizes for memory control.
  - `fallback_on_negative`: fallback to standard attention if denominators are too small.

When enabled, the node injects `optimized_attention_override` into `transformer_options`, so Flux attention calls will route through the Taylor backend and fall back if unsupported masks or stability issues are detected.

## Tests

```bash
pytest -q
RUN_LONG_TESTS=1 pytest -q  # includes 1024/4096 token tests
```

## Benchmarks

Microbenchmark (resolution-dependent via token count):

```bash
python benchmarks/benchmark_taylor_attention.py --device cuda --dtype float16 --seq-lens 1024,4096,8192
```

Flux forward-step benchmark (synthetic weights):

```bash
python benchmarks/benchmark_flux_forward.py --device cuda --dtype float16 --height 1024 --width 1024
```

## Notes

- Taylor attention is approximate and may fall back to standard attention if unsupported masks are detected or denominators become unstable.
- Large head dimensions can make feature expansion prohibitively large; `max_feature_dim_R` and `max_head_dim` guard against this.
