# Learnings

- ComfyUI's attention functions are wrapped with `optimized_attention_override` via `transformer_options`, which lets a custom node swap attention backends without patching core code.
- Flux attention calls `optimized_attention(..., skip_reshape=True)` with `q/k/v` shaped `[B, H, N, D]` after RoPE, so backend overrides should handle that layout.
- Taylor attention now estimates its own activation memory and calls ComfyUI's `model_management.free_memory` to prompt offloading before large allocations.
- Early-probe and fp32 denominator options help avoid slow Taylor fallbacks when denominators go unstable.
- Sub-head block Taylor splits each head into smaller blocks to reduce feature dimension while keeping P fixed.
- Defaults now target diffusion-scale workloads (low min_tokens, sub-head blocks, and tighter block sizes).
- Taylor now aggregates denominator/quality stats per sampling step and logs a single summary per step.
- Added qk_normalize and scale_mul knobs to stabilize P=4 Taylor attention by shrinking q·k values.
- Added qk_norm_clip and qk_norm_power to stabilize P=4 without fully normalizing Q/K.
- Denominator fallbacks can now be gated by a fraction threshold to avoid over-triggering.
- Quality stats are now computed against unmodified attention, even when Q/K are adjusted.
- Auto-tune mode can search q/k scaling during early steps and lock in a best config.
- Step logs now include q/k norm and sampled q·k percentile diagnostics to measure regime mismatch.
- Added a Triton fused-kernel path that streams Taylor feature chunks to avoid full feature tensor allocation.
