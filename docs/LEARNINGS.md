# Learnings

- ComfyUI's attention functions are wrapped with `optimized_attention_override` via `transformer_options`, which lets a custom node swap attention backends without patching core code.
- Flux attention calls `optimized_attention(..., skip_reshape=True)` with `q/k/v` shaped `[B, H, N, D]` after RoPE, so backend overrides should handle that layout.
- Taylor attention now estimates its own activation memory and calls ComfyUI's `model_management.free_memory` to prompt offloading before large allocations.
- Early-probe and fp32 denominator options help avoid slow Taylor fallbacks when denominators go unstable.
- Sub-head block Taylor splits each head into smaller blocks to reduce feature dimension while keeping P fixed.
- Defaults now target diffusion-scale workloads (low min_tokens, sub-head blocks, and tighter block sizes).
- Denominator stats are now logged on every Taylor call for stability debugging.
