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
  - `qk_norm_clip`: clip L2 norm of queries/keys (default 0, disables).
  - `qk_norm_power`: soften q/k magnitude by dividing by ||q||^power (default 0, disables).
  - `qk_norm_sigma_max`: only apply q/k normalization when sigma <= this value (default 0, disabled).
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
  - `denom_fallback_frac_limit`: fallback only if denom<=eps exceeds this fraction (default 0).
  - `fused_kernel`: use a Triton fused kernel to stream Taylor feature chunks (CUDA only).
  - `fused_feature_chunk_size`: number of Taylor features processed per chunk when fused (default 8192).
  - `fused_value_chunk_size`: value dimension chunking for fused path (0 = full).
  - `s_store_bf16`: store S chunks in bf16 to reduce memory (may reduce accuracy).
  - `taylor_sigma_max`: only run Taylor when sigma <= this value (0 disables).
  - `taylor_layer_start` / `taylor_layer_end`: only run Taylor when block_index is within this inclusive range (-1 disables each bound).
  - `auto_tune`: stochastic search for q/k scaling during early steps.
  - `auto_tune_steps`: number of steps to search (default 1).
  - `auto_tune_candidates`: candidates per step (default 8).
  - `auto_tune_quality_samples`: samples per candidate for scoring (default 4).
  - `auto_tune_seed`: RNG seed for reproducible candidate selection (default 0).
  - `auto_tune_qk_norm_power_min/max`, `auto_tune_qk_norm_clip_min/max`, `auto_tune_scale_mul_min/max`: search ranges.
  - `auto_tune_max_tokens`: token cap during tuning (default 512).
  - `quality_check`: sampled softmax vs Taylor comparison is always computed against unmodified attention.
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
If P=4 is unstable without changing model behavior, try `qk_norm_clip` or `qk_norm_power` before enabling full `qk_normalize`.

### Fused Kernel Notes

If you need to run without `sub_head_blocks`, enable `fused_kernel` and set `sub_head_blocks=1`. This switches to a streaming implementation that avoids allocating the full feature tensor. It requires Triton (`pip install triton`) and CUDA; otherwise it falls back to torch ops.

For P>=5, the fused path now streams feature indices on the GPU to avoid huge Python-side tuple construction.

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
- Taylor logs one per sampling step, including aggregated denominator/quality stats and key config values.
- Step logs include fallback reasons so you can tell why Taylor was skipped.
- Step stats now include q/k norm and sampled q·k percentile diagnostics to gauge how far attention is from the Taylor-friendly regime.
- Step stats now include `quality_raw` (vs unmodified attention) and `quality_eff` (vs modified attention) summaries.
- Quality stats always compare against unmodified attention; `quality_raw` and `quality_eff` share the same sampled indices for comparability.
- Adjust `quality_check_samples` to control cost.
- `fused_full_kernel` uses a fully fused Triton path that avoids Python feature loops by precomputing feature tables; it is fastest when feature_dim is modest (e.g., sub-head blocks) but consumes full S/Z memory.

## Hybrid Local/Global Attention (Flux RoPE)

The `HybridTaylorAttentionBackend` node patches Flux's attention function at runtime per diffusion-model call (via model pre-run/cleanup callbacks) to support a hybrid strategy:

- **Local exact attention** (with RoPE) in a sliding window.
- **Global low-dim Taylor approximation** using pre-RoPE Q/K projected to a small dimension.
- Optional `log_quality_stats` computes 8 sampled hybrid-vs-exact comparisons per attention call and logs a single aggregate after sampling completes.
- Local window can be scheduled by sigma via `local_window_min/max` and `local_window_sigma_low/high` (window shrinks as sigma decreases using a smoothstep curve; use `local_window` when schedule is disabled). A value of `0` means "full attention", so scheduled endpoints use the current sequence length as the full window.
- Quality stats logs include the hybrid config parameters used for the run.
- When `log_quality_stats` is enabled, the results are also appended to `output/hybrid-attention-results.jsonl` with config + meta (sigma, shapes, inferred latent resolution, etc.).

## Flux2TTR (Distilled TTR Attention)

The `Flux2TTR` node now uses a **hybrid kernel-regression attention** replacement for Flux single blocks:

- Branch A: normalized kernel regression attention (positive feature map, fp32 KV/Ksum accumulation, chunked by query/key).
- Branch B: exact softmax residual over a small landmark set (learnable gate `alpha`).
- Final output: `kernel_out + alpha * landmark_out`.

- Inputs:
  - `model` (`MODEL`)
  - `latents` (`LATENT`) and `conditioning` (`CONDITIONING`) (kept for workflow compatibility; calibration now uses real runtime capture mode)
  - `learning_rate`, `steps` (default `512`)
  - `training` toggle
  - `training_preview_ttr` (when training, emit student/TTR output for preview instead of teacher passthrough)
  - `comet_enabled`, `comet_project_name`, `comet_workspace`, `comet_api_key` (optional Comet telemetry)
  - `checkpoint_path`
  - `feature_dim` (must be a multiple of 256 and at least 128)
  - `query_chunk_size`, `key_chunk_size`
  - `landmark_count`, `text_tokens_guess`
  - `alpha_init`, `alpha_lr_multiplier`, `phi_lr_multiplier`
  - `training_query_token_cap`, `replay_buffer_size`, `train_steps_per_call`
  - `huber_beta`, `grad_clip_norm`
  - `readiness_threshold`, `readiness_min_updates`
  - `layer_start` / `layer_end` (optional single-block index range to patch)
  - `inference_mixed_precision` (use bf16/fp16 inference path on CUDA)
- Outputs:
  - patched `MODEL` with Flux attention routed through per-layer HKR student modules
  - `loss_value` from load/runtime state

Behavior:
- `training=true`: runs online distillation during sampler execution, with **query-only subsampling** (`q_sub`) and **full k/v context** for student forward passes. Samples are stored in per-layer replay buffers and optimized with Smooth-L1/Huber loss.
- `training=false`: loads checkpointed HKR layers and uses student attention only for layers that pass readiness checks.
- Readiness is tracked per layer via EMA loss + minimum update count. Layers that are not ready fall back to native Flux attention.
- Unsupported full per-query masks (`[B,H,Nq,Nk]` style) explicitly fail closed to native attention.
- During model execution, Flux attention is patched on pre-run and restored on cleanup; single-block calls route to per-layer `Flux2HKRAttnLayer` instances keyed by `block_index`.
- Checkpoint format is `flux2_ttr_v2` and stores layer weights plus readiness/EMA metadata for fail-closed inference.
- When Comet logging is enabled, Flux2TTR logs per-layer distillation metrics (loss/mse/nmse/cosine/ema/ready) each update.

Speed tips:
- Distill once, then run with `training=false` for normal sampling.
- Keep `feature_dim=256` unless quality demands a higher value.
- Start with `query_chunk_size=256`, `key_chunk_size=1024`; tune upward for speed if VRAM allows.
- Keep `training_query_token_cap` in the `64-256` range for practical replay memory during online distillation (`128` default).
- `replay_buffer_size` now defaults to `8` (instead of large buffers) to keep VRAM stable during online training.
- Replay samples are offloaded to CPU in reduced precision by default, then moved back to GPU per optimization step.
- Use `layer_start` / `layer_end` to patch only late single blocks as a cheap quality/speed tradeoff.
- Keep `inference_mixed_precision=true` on CUDA for the fastest inference path.
- Flux2TTR now asks ComfyUI to reserve VRAM ahead of each TTR call (Taylor-style `free_memory` reservation) using a `1.1x` safety factor over estimated need.
- `enable_memory_reserve` is now **off by default** for Flux2TTR because aggressive mid-run offload requests can cause CPU/CUDA device mismatches in downstream linear layers on some ComfyUI runs.
- Note: this reservation is advisory/offload-oriented (via `free_memory`), not a persistent allocation; Flux2TTR now releases runtime GPU state and unregisters runtime objects at cleanup to avoid VRAM buildup across repeated runs.
- If a cached graph references a missing Flux2TTR runtime ID, the node now attempts to recover runtime state from its saved config + `checkpoint_path` instead of immediately falling back.
- Training at `feature_dim=256` typically needs roughly a few GB of extra VRAM; reservation is intended to offload earlier nodes before HKR allocations.
- If training hits OOM, Flux2TTR now auto-reduces training pressure (`training_query_token_cap`, chunk sizes, landmarks) and clears the active layer replay buffer before disabling training.
- If training still OOMs, Flux2TTR disables training for the run and falls back gracefully (teacher passthrough or preview fallback) instead of crashing generation.
- If checkpoint loss is still high, Flux2TTR will fail closed to native attention fallback in inference mode instead of emitting low-quality garbage output.
- During online distillation, Flux2TTR logs progress every 10 training updates with current loss so you can tune `steps`.
- If you only want fastest distillation and don't need visual feedback, set `training_preview_ttr=false` to stay in teacher passthrough during training runs.
- `comet_api_key` can be left blank to use the `COMET_API_KEY` environment variable.

## Clocked Sweep Values

The `ClockedSweepValues` node maps a clock list to evenly distributed sweep values. Provide a clock list (any list of floats; length defines output length) and a list of values to sweep. The output is a float list (ComfyUI list output) the same length as the clock, split into equal contiguous segments per value. You can also enter a single integer string (e.g., `30`) to create a 1..N clock, or leave the clock blank to infer length from the values list.

## Combinations

The `Combinations` node takes up to four value lists (A–D) and returns float-list outputs that cycle each list to cover all combinations. With A=[1,2,3] and B=[4,5], the outputs are:
- A_out = [1,2,3,1,2,3]
- B_out = [4,5,4,5,4,5]

Each additional list increases the output length multiplicatively.

Example: clock length 100, values `[1, 2, 3]` → 33 entries of 1, 33 of 2, 34 of 3.

Inputs accept JSON arrays, comma/space-separated strings, or list inputs.

This is intended for very large images where full attention is prohibitive. It is an approximate inference hack and will change model behavior.
- Large head dimensions can make feature expansion prohibitively large; `max_feature_dim_R` and `max_head_dim` guard against this.
