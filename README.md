# Taylor Attention Backend for ComfyUI

This *HIGHLY EXPERIMENTAL* and *TOTALLY_UNPROVEN* custom node adds a
runtime-switchable attention backend for diffusion transformers (Flux/Flux2)
using a truncated Taylor expansion with a symmetry-aware feature map. It is
designed for large token counts where quadratic attention becomes expensive.

## Install

1. Copy this folder into your ComfyUI `custom_nodes` directory.
2. Restart ComfyUI.
3. Use the **Taylor Attention Backend** node from the `advanced/attention` category.

## Usage

- Add **Taylor Attention Backend** to your graph.
- Set `backend` to `taylor` to enable, or `standard` to disable.
- Key parameters:
  - `P`: number of Taylor terms (default 4).
  - `min_tokens`: only use Taylor when tokens >= this threshold (default 100).
  - `max_feature_dim_R`: safety cap for feature dimension (default 370000).
  - `sub_head_blocks`: split each head into smaller blocks to reduce feature expansion (default 4).
  - `qk_normalize`: L2-normalize queries/keys before Taylor features (default false).
  - `scale_mul`: additional scale multiplier for q·k before Taylor (default 1.0).
  - `block_size_q` / `block_size_k`: block sizes for memory control (defaults 32 / 16).
  - `fallback_on_negative`: fallback to standard attention if denominators are too small.
  - `force_fp32`: accumulate Taylor features in fp32 for stability (default false).
  - `memory_reserve`: ask ComfyUI to free VRAM before Taylor attention allocations.
  - `memory_reserve_factor`: safety multiplier for the VRAM estimate.
  - `memory_reserve_log`: log reserved VRAM estimates (default true).
  - `early_probe`: run a denominator probe before full Taylor compute (default true).
  - `probe_samples`: number of queries sampled for the probe (default 16).
  - `denom_fp32`: compute denominators in fp32 to reduce underflow (default true).
  - `quality_check`: log a sampled softmax vs Taylor comparison per call (default true).
  - `quality_check_samples`: number of sampled queries per call (default 16).
  - `quality_check_log_every`: log every N Taylor calls (default 1).
  - `log_shapes`: log Taylor attention shapes (default true).
  - `log_fallbacks`: log Taylor fallbacks (default true).

When enabled, the node injects `optimized_attention_override` into `transformer_options`, so Flux attention calls will route through the Taylor backend and fall back if unsupported masks or stability issues are detected.

## Performance Optimizations

Diffusion transformers (Flux/Flux2) run at very different scales than LLMs (large spatial token counts, fixed head dims), so Taylor attention can blow past `max_feature_dim_R` even for modest `P`. The `sub_head_blocks` option is the main way to keep Taylor feasible at these scales.

- `sub_head_blocks` splits each attention head into smaller blocks and runs Taylor on each block independently.
  - This reduces the feature expansion size `R` dramatically while keeping `P` fixed.
  - It is an approximation: more blocks = less accuracy, but much lower VRAM and faster compute.
- Rule of thumb: pick the smallest `sub_head_blocks` that makes `feature_dim(block_dim, P) <= max_feature_dim_R` where `block_dim = head_dim / sub_head_blocks`.

Example (Flux head_dim=128, P=4, max_feature_dim_R=60000):
- sub_head_blocks=4 -> block_dim=32 -> R=52,360 (fits)
- sub_head_blocks=8 -> block_dim=16 -> R=3,876 (very safe, more approximate)

Start with `sub_head_blocks=4` for Flux/Flux2 and only increase if you still hit `feature_dim_too_large` or VRAM pressure. Use `quality_check` to sample accuracy when you change this.

## Tests

```bash
pytest -q
RUN_LONG_TESTS=1 pytest -q  # includes 1024/4096 token tests
```

### Using uv

```bash
uv sync --extra test
uv run pytest
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
- Denominator stats are logged for every Taylor call to help debug stability issues.
- Large head dimensions can make feature expansion prohibitively large; `max_feature_dim_R` and `max_head_dim` guard against this.
