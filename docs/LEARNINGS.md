# Learnings

- ComfyUI's attention functions are wrapped with `optimized_attention_override` via `transformer_options`, which lets a custom node swap attention backends without patching core code.
- Flux attention calls `optimized_attention(..., skip_reshape=True)` with `q/k/v` shaped `[B, H, N, D]` after RoPE, so backend overrides should handle that layout.
