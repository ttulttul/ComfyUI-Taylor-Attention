import logging
from typing import Dict, Any

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

import taylor_attention
import hybrid_attention

logger = logging.getLogger(__name__)


def _build_config(
    backend: str,
    P: int,
    min_tokens: int,
    max_feature_dim_R: int,
    block_size_q: int,
    block_size_k: int,
    eps: float,
    fallback_on_negative: bool,
    allow_cross_attention: bool,
    max_head_dim: int,
    sub_head_blocks: int,
    qk_normalize: bool,
    qk_norm_clip: float,
    qk_norm_power: float,
    qk_norm_sigma_max: float,
    scale_mul: float,
    force_fp32: bool,
    memory_reserve: bool,
    memory_reserve_factor: float,
    memory_reserve_log: bool,
    early_probe: bool,
    probe_samples: int,
    denom_fp32: bool,
    denom_fallback_frac_limit: float,
    fused_kernel: bool,
    fused_full_kernel: bool,
    fused_feature_chunk_size: int,
    fused_value_chunk_size: int,
    s_store_bf16: bool,
    taylor_sigma_max: float,
    taylor_layer_start: int,
    taylor_layer_end: int,
    auto_tune: bool,
    auto_tune_steps: int,
    auto_tune_candidates: int,
    auto_tune_quality_samples: int,
    auto_tune_seed: int,
    auto_tune_qk_norm_power_min: float,
    auto_tune_qk_norm_power_max: float,
    auto_tune_qk_norm_clip_min: float,
    auto_tune_qk_norm_clip_max: float,
    auto_tune_scale_mul_min: float,
    auto_tune_scale_mul_max: float,
    auto_tune_max_tokens: int,
    quality_check: bool,
    quality_check_samples: int,
    quality_check_log_every: int,
    log_shapes: bool,
    log_fallbacks: bool,
) -> Dict[str, Any]:
    if backend != "taylor":
        return {"enabled": False}
    return {
        "enabled": True,
        "P": int(P),
        "min_tokens": int(min_tokens),
        "max_feature_dim_R": int(max_feature_dim_R),
        "block_size_q": int(block_size_q),
        "block_size_k": int(block_size_k),
        "eps": float(eps),
        "fallback_on_negative": bool(fallback_on_negative),
        "allow_cross_attention": bool(allow_cross_attention),
        "max_head_dim": int(max_head_dim),
        "sub_head_blocks": int(sub_head_blocks),
        "qk_normalize": bool(qk_normalize),
        "qk_norm_clip": float(qk_norm_clip),
        "qk_norm_power": float(qk_norm_power),
        "qk_norm_sigma_max": float(qk_norm_sigma_max),
        "scale_mul": float(scale_mul),
        "force_fp32": bool(force_fp32),
        "memory_reserve": bool(memory_reserve),
        "memory_reserve_factor": float(memory_reserve_factor),
        "memory_reserve_log": bool(memory_reserve_log),
        "early_probe": bool(early_probe),
        "probe_samples": int(probe_samples),
        "denom_fp32": bool(denom_fp32),
        "denom_fallback_frac_limit": float(denom_fallback_frac_limit),
        "fused_kernel": bool(fused_kernel),
        "fused_full_kernel": bool(fused_full_kernel),
        "fused_feature_chunk_size": int(fused_feature_chunk_size),
        "fused_value_chunk_size": int(fused_value_chunk_size),
        "s_store_bf16": bool(s_store_bf16),
        "taylor_sigma_max": float(taylor_sigma_max),
        "taylor_layer_start": int(taylor_layer_start),
        "taylor_layer_end": int(taylor_layer_end),
        "auto_tune": bool(auto_tune),
        "auto_tune_steps": int(auto_tune_steps),
        "auto_tune_candidates": int(auto_tune_candidates),
        "auto_tune_quality_samples": int(auto_tune_quality_samples),
        "auto_tune_seed": int(auto_tune_seed),
        "auto_tune_qk_norm_power_min": float(auto_tune_qk_norm_power_min),
        "auto_tune_qk_norm_power_max": float(auto_tune_qk_norm_power_max),
        "auto_tune_qk_norm_clip_min": float(auto_tune_qk_norm_clip_min),
        "auto_tune_qk_norm_clip_max": float(auto_tune_qk_norm_clip_max),
        "auto_tune_scale_mul_min": float(auto_tune_scale_mul_min),
        "auto_tune_scale_mul_max": float(auto_tune_scale_mul_max),
        "auto_tune_max_tokens": int(auto_tune_max_tokens),
        "quality_check": bool(quality_check),
        "quality_check_samples": int(quality_check_samples),
        "quality_check_log_every": int(quality_check_log_every),
        "log_shapes": bool(log_shapes),
        "log_fallbacks": bool(log_fallbacks),
    }


def _build_hybrid_config(
    enabled: bool,
    local_window: int,
    local_chunk: int,
    global_dim: int,
    global_P: int,
    global_weight: float,
    global_sigma_low: float,
    global_sigma_high: float,
    global_scale_mul: float,
    global_norm_power: float,
    global_norm_clip: float,
    use_pca: bool,
    pca_samples: int,
    allow_cross_attention: bool,
    layer_start: int,
    layer_end: int,
    eps: float,
    force_fp32: bool,
    log_steps: bool,
) -> Dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "local_window": int(local_window),
        "local_chunk": int(local_chunk),
        "global_dim": int(global_dim),
        "global_P": int(global_P),
        "global_weight": float(global_weight),
        "global_sigma_low": float(global_sigma_low),
        "global_sigma_high": float(global_sigma_high),
        "global_scale_mul": float(global_scale_mul),
        "global_norm_power": float(global_norm_power),
        "global_norm_clip": float(global_norm_clip),
        "use_pca": bool(use_pca),
        "pca_samples": int(pca_samples),
        "allow_cross_attention": bool(allow_cross_attention),
        "layer_start": int(layer_start),
        "layer_end": int(layer_end),
        "eps": float(eps),
        "force_fp32": bool(force_fp32),
        "log_steps": bool(log_steps),
    }


class TaylorAttentionBackend(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TaylorAttentionBackend",
            display_name="Taylor Attention Backend",
            category="advanced/attention",
            description="Toggle Taylor-approximated attention for large-token diffusion transformers.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("backend", options=["standard", "taylor"], default="taylor"),
                io.Int.Input("P", default=4, min=1, max=8, step=1, tooltip="Number of Taylor terms."),
                io.Int.Input("min_tokens", default=100, min=0, max=200000, step=1, tooltip="Enable Taylor only when tokens >= min_tokens."),
                io.Int.Input("max_feature_dim_R", default=370000, min=1, max=1000000000, step=1, tooltip="Safety cap for feature dimension R."),
                io.Int.Input("block_size_q", default=32, min=1, max=8192, step=1),
                io.Int.Input("block_size_k", default=16, min=1, max=8192, step=1),
                io.Float.Input("eps", default=1e-6, min=1e-12, max=1e-2, step=1e-6),
                io.Boolean.Input("fallback_on_negative", default=True, tooltip="Fallback if denominators are too small."),
                io.Boolean.Input("allow_cross_attention", default=True),
                io.Int.Input("max_head_dim", default=128, min=1, max=512, step=1),
                io.Int.Input("sub_head_blocks", default=4, min=1, max=16, step=1, tooltip="Split each head into sub-blocks to reduce feature dimension."),
                io.Boolean.Input("qk_normalize", default=False, tooltip="L2-normalize queries/keys before Taylor features."),
                io.Float.Input("qk_norm_clip", default=0.0, min=0.0, max=100.0, step=0.5, tooltip="Clip L2 norm of queries/keys (0 disables)."),
                io.Float.Input("qk_norm_power", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Soften q/k magnitude by dividing by ||q||^power."),
                io.Float.Input("qk_norm_sigma_max", default=0.0, min=0.0, max=50.0, step=0.01, tooltip="Only apply q/k normalization when sigma <= this value (0 disables)."),
                io.Float.Input("scale_mul", default=1.0, min=0.0, max=4.0, step=0.05, tooltip="Additional scale multiplier for q·k before Taylor."),
                io.Boolean.Input("force_fp32", default=False, tooltip="Accumulate Taylor features in fp32 for stability."),
                io.Boolean.Input("memory_reserve", default=True, tooltip="Ask ComfyUI to free VRAM for Taylor attention."),
                io.Float.Input("memory_reserve_factor", default=1.1, min=1.0, max=4.0, step=0.05, tooltip="Safety multiplier for reserved VRAM estimate."),
                io.Boolean.Input("memory_reserve_log", default=True, tooltip="Log reserved VRAM estimates."),
                io.Boolean.Input("early_probe", default=True, tooltip="Probe denominators before full Taylor compute."),
                io.Int.Input("probe_samples", default=16, min=1, max=64, step=1, tooltip="Queries sampled for early probe."),
                io.Boolean.Input("denom_fp32", default=True, tooltip="Compute denominators in fp32 for stability."),
                io.Float.Input("denom_fallback_frac_limit", default=0.0, min=0.0, max=1.0, step=0.0001, tooltip="Fallback only if denom <= eps exceeds this fraction (0 = any)."),
                io.Boolean.Input("fused_kernel", default=False, tooltip="Use Triton fused kernel to stream features (CUDA only)."),
                io.Boolean.Input("fused_full_kernel", default=False, tooltip="Use fully fused Triton kernel without Python feature loops (CUDA only)."),
                io.Int.Input("fused_feature_chunk_size", default=8192, min=128, max=200000, step=128, tooltip="Feature chunk size for fused kernel streaming."),
                io.Int.Input("fused_value_chunk_size", default=0, min=0, max=4096, step=32, tooltip="Value dimension chunk size (0 = full)."),
                io.Boolean.Input("s_store_bf16", default=False, tooltip="Store S chunks in bf16 to reduce memory."),
                io.Float.Input("taylor_sigma_max", default=0.0, min=0.0, max=50.0, step=0.01, tooltip="Only run Taylor when sigma <= this value (0 disables)."),
                io.Int.Input("taylor_layer_start", default=-1, min=-1, max=512, step=1, tooltip="Only run Taylor on block_index >= this value (-1 disables)."),
                io.Int.Input("taylor_layer_end", default=-1, min=-1, max=512, step=1, tooltip="Only run Taylor on block_index <= this value (-1 disables)."),
                io.Boolean.Input("auto_tune", default=False, tooltip="Auto-tune q/k scaling during early steps."),
                io.Int.Input("auto_tune_steps", default=1, min=0, max=4, step=1, tooltip="Number of steps to search for a better config."),
                io.Int.Input("auto_tune_candidates", default=8, min=1, max=32, step=1, tooltip="Number of candidates per step."),
                io.Int.Input("auto_tune_quality_samples", default=4, min=1, max=32, step=1, tooltip="Samples used per candidate for quality scoring."),
                io.Int.Input("auto_tune_seed", default=0, min=0, max=100000, step=1),
                io.Float.Input("auto_tune_qk_norm_power_min", default=0.2, min=0.0, max=1.0, step=0.05),
                io.Float.Input("auto_tune_qk_norm_power_max", default=0.7, min=0.0, max=1.0, step=0.05),
                io.Float.Input("auto_tune_qk_norm_clip_min", default=8.0, min=0.0, max=100.0, step=0.5),
                io.Float.Input("auto_tune_qk_norm_clip_max", default=20.0, min=0.0, max=100.0, step=0.5),
                io.Float.Input("auto_tune_scale_mul_min", default=0.8, min=0.0, max=4.0, step=0.05),
                io.Float.Input("auto_tune_scale_mul_max", default=1.0, min=0.0, max=4.0, step=0.05),
                io.Int.Input("auto_tune_max_tokens", default=512, min=0, max=8192, step=1, tooltip="Max tokens used during auto-tune (0 = full)."),
                io.Boolean.Input("quality_check", default=True, tooltip="Log a sampled accuracy check vs softmax."),
                io.Int.Input("quality_check_samples", default=16, min=1, max=64, step=1, tooltip="Number of sampled queries per call."),
                io.Int.Input("quality_check_log_every", default=1, min=1, max=1000, step=1, tooltip="Log every N Taylor calls."),
                io.Boolean.Input("log_shapes", default=True),
                io.Boolean.Input("log_fallbacks", default=True),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        backend: str,
        P: int,
        min_tokens: int,
        max_feature_dim_R: int,
        block_size_q: int,
        block_size_k: int,
        eps: float,
        fallback_on_negative: bool,
        allow_cross_attention: bool,
        max_head_dim: int,
        sub_head_blocks: int,
        qk_normalize: bool,
        qk_norm_clip: float,
        qk_norm_power: float,
        qk_norm_sigma_max: float,
        scale_mul: float,
        force_fp32: bool,
        memory_reserve: bool,
        memory_reserve_factor: float,
        memory_reserve_log: bool,
        early_probe: bool,
        probe_samples: int,
        denom_fp32: bool,
        denom_fallback_frac_limit: float,
        fused_kernel: bool,
        fused_full_kernel: bool,
        fused_feature_chunk_size: int,
        fused_value_chunk_size: int,
        s_store_bf16: bool,
        taylor_sigma_max: float,
        taylor_layer_start: int,
        taylor_layer_end: int,
        auto_tune: bool,
        auto_tune_steps: int,
        auto_tune_candidates: int,
        auto_tune_quality_samples: int,
        auto_tune_seed: int,
        auto_tune_qk_norm_power_min: float,
        auto_tune_qk_norm_power_max: float,
        auto_tune_qk_norm_clip_min: float,
        auto_tune_qk_norm_clip_max: float,
        auto_tune_scale_mul_min: float,
        auto_tune_scale_mul_max: float,
        auto_tune_max_tokens: int,
        quality_check: bool,
        quality_check_samples: int,
        quality_check_log_every: int,
        log_shapes: bool,
        log_fallbacks: bool,
    ) -> io.NodeOutput:
        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})

        if backend == "taylor":
            prev_override = transformer_options.get("optimized_attention_override")
            transformer_options["taylor_attention_prev_override"] = prev_override
            transformer_options["optimized_attention_override"] = taylor_attention.taylor_attention_override
            transformer_options["taylor_attention"] = _build_config(
                backend,
                P,
                min_tokens,
                max_feature_dim_R,
                block_size_q,
                block_size_k,
                eps,
                fallback_on_negative,
                allow_cross_attention,
                max_head_dim,
                sub_head_blocks,
                qk_normalize,
                qk_norm_clip,
                qk_norm_power,
                qk_norm_sigma_max,
                scale_mul,
                force_fp32,
                memory_reserve,
                memory_reserve_factor,
                memory_reserve_log,
                early_probe,
                probe_samples,
                denom_fp32,
                denom_fallback_frac_limit,
                fused_kernel,
                fused_full_kernel,
                fused_feature_chunk_size,
                fused_value_chunk_size,
                s_store_bf16,
                taylor_sigma_max,
                taylor_layer_start,
                taylor_layer_end,
                auto_tune,
                auto_tune_steps,
                auto_tune_candidates,
                auto_tune_quality_samples,
                auto_tune_seed,
                auto_tune_qk_norm_power_min,
                auto_tune_qk_norm_power_max,
                auto_tune_qk_norm_clip_min,
                auto_tune_qk_norm_clip_max,
                auto_tune_scale_mul_min,
                auto_tune_scale_mul_max,
                auto_tune_max_tokens,
                quality_check,
                quality_check_samples,
                quality_check_log_every,
                log_shapes,
                log_fallbacks,
            )
            logger.info("Enabled Taylor attention backend.")
        else:
            if transformer_options.get("optimized_attention_override") is taylor_attention.taylor_attention_override:
                transformer_options["optimized_attention_override"] = transformer_options.pop("taylor_attention_prev_override", None)
            transformer_options.pop("taylor_attention", None)
            logger.info("Disabled Taylor attention backend.")

        return io.NodeOutput(m)


class HybridTaylorAttentionBackend(io.ComfyNode):
    @classmethod
    def define(cls) -> io.NodeDef:
        return io.NodeDef(
            node_id="HybridTaylorAttentionBackend",
            display_name="Hybrid Taylor Attention Backend",
            category="advanced/attention",
            description="Hybrid local exact attention with global low-dim Taylor approximation (Flux RoPE-compatible).",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=False, tooltip="Enable hybrid attention override for Flux."),
                io.Int.Input("local_window", default=512, min=0, max=8192, step=1, tooltip="Local window radius for exact attention (0 = full)."),
                io.Int.Input("local_chunk", default=256, min=1, max=4096, step=1, tooltip="Query chunk size for local attention."),
                io.Int.Input("global_dim", default=16, min=1, max=128, step=1, tooltip="Projection dimension for global approximation."),
                io.Int.Input("global_P", default=2, min=1, max=4, step=1, tooltip="Taylor order for global approximation."),
                io.Float.Input("global_weight", default=0.1, min=0.0, max=4.0, step=0.01, tooltip="Scale applied to global approximation."),
                io.Float.Input("global_sigma_low", default=0.0, min=0.0, max=50.0, step=0.01, tooltip="Sigma below which global weight is 0."),
                io.Float.Input("global_sigma_high", default=0.0, min=0.0, max=50.0, step=0.01, tooltip="Sigma above which global weight is full (0 disables ramp)."),
                io.Float.Input("global_scale_mul", default=1.0, min=0.0, max=4.0, step=0.01, tooltip="Scale multiplier for global q·k before Taylor."),
                io.Float.Input("global_norm_power", default=0.0, min=0.0, max=1.0, step=0.05, tooltip="Global q/k norm power (0 disables)."),
                io.Float.Input("global_norm_clip", default=0.0, min=0.0, max=100.0, step=0.5, tooltip="Global q/k norm clip (0 disables)."),
                io.Boolean.Input("use_pca", default=True, tooltip="Compute PCA projection for global approximation."),
                io.Int.Input("pca_samples", default=2048, min=0, max=65536, step=256, tooltip="Samples used for PCA projection (0 = full)."),
                io.Boolean.Input("allow_cross_attention", default=True, tooltip="Allow hybrid attention on cross-attention."),
                io.Int.Input("layer_start", default=-1, min=-1, max=512, step=1, tooltip="Only run hybrid on block_index >= this value (-1 disables)."),
                io.Int.Input("layer_end", default=-1, min=-1, max=512, step=1, tooltip="Only run hybrid on block_index <= this value (-1 disables)."),
                io.Float.Input("eps", default=1e-6, min=1e-12, max=1e-2, step=1e-6),
                io.Boolean.Input("force_fp32", default=True, tooltip="Use fp32 for global approximation."),
                io.Boolean.Input("log_steps", default=True, tooltip="Log hybrid attention stats per step."),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled: bool,
        local_window: int,
        local_chunk: int,
        global_dim: int,
        global_P: int,
        global_weight: float,
        global_sigma_low: float,
        global_sigma_high: float,
        global_scale_mul: float,
        global_norm_power: float,
        global_norm_clip: float,
        use_pca: bool,
        pca_samples: int,
        allow_cross_attention: bool,
        layer_start: int,
        layer_end: int,
        eps: float,
        force_fp32: bool,
        log_steps: bool,
    ) -> io.NodeOutput:
        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})

        if enabled:
            hybrid_attention.enable_hybrid_attention()
            transformer_options["hybrid_taylor_attention"] = _build_hybrid_config(
                enabled,
                local_window,
                local_chunk,
                global_dim,
                global_P,
                global_weight,
                global_sigma_low,
                global_sigma_high,
                global_scale_mul,
                global_norm_power,
                global_norm_clip,
                use_pca,
                pca_samples,
                allow_cross_attention,
                layer_start,
                layer_end,
                eps,
                force_fp32,
                log_steps,
            )
        else:
            transformer_options.pop("hybrid_taylor_attention", None)
            hybrid_attention.disable_hybrid_attention()

        return io.NodeOutput(m)


class TaylorAttentionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TaylorAttentionBackend, HybridTaylorAttentionBackend]


async def comfy_entrypoint() -> TaylorAttentionExtension:
    return TaylorAttentionExtension()
