from __future__ import annotations

import logging
import math
import os
from typing import Dict, Any

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import comfy.patcher_extension as patcher_extension

import taylor_attention
import hybrid_attention
import flux2_ttr
import sweep_utils

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
    local_window_min: int,
    local_window_max: int,
    local_window_sigma_low: float,
    local_window_sigma_high: float,
    local_chunk: int,
    prefix_tokens: int,
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
    log_quality_stats: bool,
) -> Dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "local_window": int(local_window),
        "local_window_min": int(local_window_min),
        "local_window_max": int(local_window_max),
        "local_window_sigma_low": float(local_window_sigma_low),
        "local_window_sigma_high": float(local_window_sigma_high),
        "local_chunk": int(local_chunk),
        "prefix_tokens": int(prefix_tokens),
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
        "log_quality_stats": bool(log_quality_stats),
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
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HybridTaylorAttentionBackend",
            display_name="Hybrid Taylor Attention Backend",
            category="advanced/attention",
            description="Hybrid local exact attention with global low-dim Taylor approximation (Flux RoPE-compatible).",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=False, tooltip="Enable hybrid attention override for Flux."),
                io.Int.Input("local_window", default=512, min=0, max=8192, step=1, tooltip="Default local window radius for exact attention (0 = full)."),
                io.Int.Input("local_window_min", default=512, min=0, max=8192, step=1, tooltip="Minimum local window when scheduling by sigma."),
                io.Int.Input("local_window_max", default=512, min=0, max=8192, step=1, tooltip="Maximum local window when scheduling by sigma."),
                io.Float.Input("local_window_sigma_low", default=0.0, min=0.0, max=50.0, step=0.01, tooltip="Sigma below which local window is min (0 disables schedule)."),
                io.Float.Input("local_window_sigma_high", default=0.0, min=0.0, max=50.0, step=0.01, tooltip="Sigma above which local window is max (0 disables schedule)."),
                io.Int.Input("local_chunk", default=256, min=1, max=1000000000, step=1, tooltip="Query chunk size for local attention."),
                io.Int.Input("prefix_tokens", default=0, min=0, max=8192, step=1, tooltip="Always include the first N tokens (e.g., text tokens) in local attention."),
                io.Int.Input("global_dim", default=16, min=1, max=128, step=1, tooltip="Projection dimension for global approximation."),
                io.Int.Input("global_P", default=2, min=1, max=4, step=1, tooltip="Taylor order for global approximation."),
                io.Float.Input("global_weight", default=0.1, min=0.0, max=4.0, step=0.01, tooltip="Scale applied to global approximation."),
                io.Float.Input("global_sigma_low", default=0.0, min=0.0, max=50.0, step=0.01, tooltip="Sigma below which global weight is full."),
                io.Float.Input("global_sigma_high", default=0.0, min=0.0, max=50.0, step=0.01, tooltip="Sigma above which global weight is 0 (0 disables ramp)."),
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
                io.Boolean.Input("log_quality_stats", default=False, tooltip="Log hybrid-vs-exact quality stats after sampling completes."),
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
        local_window_min: int,
        local_window_max: int,
        local_window_sigma_low: float,
        local_window_sigma_high: float,
        local_chunk: int,
        prefix_tokens: int,
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
        log_quality_stats: bool,
    ) -> io.NodeOutput:
        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})

        if enabled:
            transformer_options["hybrid_taylor_attention"] = _build_hybrid_config(
                enabled,
                local_window,
                local_window_min,
                local_window_max,
                local_window_sigma_low,
                local_window_sigma_high,
                local_chunk,
                prefix_tokens,
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
                log_quality_stats,
            )
            callback_key = "hybrid_taylor_attention"
            m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_PRE_RUN, callback_key)
            m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_CLEANUP, callback_key)
            m.add_callback_with_key(
                patcher_extension.CallbacksMP.ON_PRE_RUN,
                callback_key,
                hybrid_attention.pre_run_callback,
            )
            m.add_callback_with_key(
                patcher_extension.CallbacksMP.ON_CLEANUP,
                callback_key,
                hybrid_attention.cleanup_callback,
            )
        else:
            transformer_options.pop("hybrid_taylor_attention", None)
            callback_key = "hybrid_taylor_attention"
            m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_PRE_RUN, callback_key)
            m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_CLEANUP, callback_key)

        return io.NodeOutput(m)


class Flux2TTR(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Flux2TTR",
            display_name="Flux2TTR",
            category="advanced/attention",
            description="Replace Flux single-block attention with hybrid kernel-regression + landmark residual attention.",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("latents"),
                io.Conditioning.Input("conditioning"),
                io.Float.Input("learning_rate", default=1e-4, min=1e-7, max=1.0, step=1e-7),
                io.Int.Input("steps", default=512, min=0, max=200000, step=1),
                io.Boolean.Input("training", default=True, tooltip="Train TTR layers by distillation when enabled."),
                io.Boolean.Input(
                    "training_preview_ttr",
                    default=True,
                    tooltip="When training, output TTR student attention for visual preview instead of teacher passthrough.",
                ),
                io.Boolean.Input(
                    "comet_enabled",
                    default=True,
                    tooltip="Log per-layer distillation metrics to Comet during training.",
                ),
                io.String.Input(
                    "comet_project_name",
                    default="ttr-distillation",
                    multiline=False,
                    tooltip="Comet project name used when metric logging is enabled.",
                ),
                io.String.Input(
                    "comet_workspace",
                    default="ken-simpson",
                    multiline=False,
                    tooltip="Comet workspace used when metric logging is enabled.",
                ),
                io.String.Input(
                    "comet_api_key",
                    default="",
                    multiline=False,
                    tooltip="Optional Comet API key override. Leave blank to use COMET_API_KEY env var.",
                ),
                io.String.Input(
                    "checkpoint_path",
                    default="",
                    multiline=False,
                    tooltip="Checkpoint file to load/save TTR layer weights.",
                ),
                io.Int.Input(
                    "feature_dim",
                    default=256,
                    min=128,
                    max=8192,
                    step=256,
                    tooltip="Kernel feature dimension (must be a multiple of 256).",
                ),
                io.Int.Input(
                    "query_chunk_size",
                    default=256,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Query chunk size for kernel attention evaluation.",
                ),
                io.Int.Input(
                    "key_chunk_size",
                    default=1024,
                    min=1,
                    max=8192,
                    step=1,
                    tooltip="Key chunk size for kernel KV/Ksum accumulation.",
                ),
                io.Int.Input(
                    "landmark_count",
                    default=128,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip="Number of landmark keys used for exact softmax residual.",
                ),
                io.Int.Input(
                    "text_tokens_guess",
                    default=77,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip="Assumed number of text tokens at the start of sequence for landmark selection.",
                ),
                io.Float.Input(
                    "alpha_init",
                    default=0.1,
                    min=0.0,
                    max=10.0,
                    step=1e-3,
                    tooltip="Initial residual gate for landmark softmax branch.",
                ),
                io.Float.Input(
                    "alpha_lr_multiplier",
                    default=5.0,
                    min=0.0,
                    max=100.0,
                    step=1e-3,
                    tooltip="Learning-rate multiplier for landmark gate alpha.",
                ),
                io.Float.Input(
                    "phi_lr_multiplier",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=1e-3,
                    tooltip="Learning-rate multiplier for kernel feature networks.",
                ),
                io.Int.Input(
                    "training_query_token_cap",
                    default=256,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Max number of query tokens per replay sample; keys/values always stay full length.",
                ),
                io.Int.Input(
                    "replay_buffer_size",
                    default=64,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Replay buffer capacity per layer for distillation samples.",
                ),
                io.Int.Input(
                    "train_steps_per_call",
                    default=1,
                    min=1,
                    max=32,
                    step=1,
                    tooltip="Number of replay optimization steps run per attention call.",
                ),
                io.Float.Input(
                    "huber_beta",
                    default=0.05,
                    min=1e-6,
                    max=10.0,
                    step=1e-4,
                    tooltip="Smooth-L1 beta for distillation loss.",
                ),
                io.Float.Input(
                    "grad_clip_norm",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=1e-3,
                    tooltip="Global gradient clipping norm (0 disables clipping).",
                ),
                io.Float.Input(
                    "readiness_threshold",
                    default=0.12,
                    min=0.0,
                    max=10.0,
                    step=1e-4,
                    tooltip="Enable student inference for a layer only when EMA loss is below this threshold.",
                ),
                io.Int.Input(
                    "readiness_min_updates",
                    default=24,
                    min=0,
                    max=100000,
                    step=1,
                    tooltip="Minimum replay updates before a layer can be marked ready.",
                ),
                io.Boolean.Input(
                    "enable_memory_reserve",
                    default=False,
                    tooltip="Call ComfyUI free_memory before HKR attention allocations (can offload aggressively).",
                ),
                io.Int.Input(
                    "layer_start",
                    default=-1,
                    min=-1,
                    max=512,
                    step=1,
                    tooltip="Only apply TTR to single blocks with index >= layer_start (-1 disables).",
                ),
                io.Int.Input(
                    "layer_end",
                    default=-1,
                    min=-1,
                    max=512,
                    step=1,
                    tooltip="Only apply TTR to single blocks with index <= layer_end (-1 disables).",
                ),
                io.Boolean.Input(
                    "inference_mixed_precision",
                    default=True,
                    tooltip="Use input dtype (bf16/fp16) for TTR inference on CUDA for speed.",
                ),
            ],
            outputs=[io.Model.Output(), io.Float.Output("loss_value")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        latents,
        conditioning,
        learning_rate: float,
        steps: int,
        training: bool,
        training_preview_ttr: bool,
        comet_enabled: bool,
        comet_project_name: str,
        comet_workspace: str,
        comet_api_key: str,
        checkpoint_path: str,
        feature_dim: int,
        query_chunk_size: int,
        key_chunk_size: int,
        landmark_count: int,
        text_tokens_guess: int,
        alpha_init: float,
        alpha_lr_multiplier: float,
        phi_lr_multiplier: float,
        training_query_token_cap: int,
        replay_buffer_size: int,
        train_steps_per_call: int,
        huber_beta: float,
        grad_clip_norm: float,
        readiness_threshold: float,
        readiness_min_updates: int,
        enable_memory_reserve: bool,
        layer_start: int,
        layer_end: int,
        inference_mixed_precision: bool,
    ) -> io.NodeOutput:
        feature_dim = flux2_ttr.validate_feature_dim(feature_dim)
        checkpoint_path = (checkpoint_path or "").strip()
        train_steps = int(steps)

        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})

        prev_cfg = transformer_options.get("flux2_ttr")
        if isinstance(prev_cfg, dict):
            prev_runtime = prev_cfg.get("runtime_id")
            if isinstance(prev_runtime, str):
                flux2_ttr.unregister_runtime(prev_runtime)

        runtime = flux2_ttr.Flux2TTRRuntime(
            feature_dim=feature_dim,
            learning_rate=float(learning_rate),
            training=bool(training),
            steps=train_steps,
            scan_chunk_size=int(query_chunk_size),
            key_chunk_size=int(key_chunk_size),
            landmark_count=int(landmark_count),
            text_tokens_guess=int(text_tokens_guess),
            alpha_init=float(alpha_init),
            alpha_lr_multiplier=float(alpha_lr_multiplier),
            phi_lr_multiplier=float(phi_lr_multiplier),
            training_query_token_cap=int(training_query_token_cap),
            replay_buffer_size=int(replay_buffer_size),
            train_steps_per_call=int(train_steps_per_call),
            huber_beta=float(huber_beta),
            grad_clip_norm=float(grad_clip_norm),
            readiness_threshold=float(readiness_threshold),
            readiness_min_updates=int(readiness_min_updates),
            enable_memory_reserve=bool(enable_memory_reserve),
            layer_start=int(layer_start),
            layer_end=int(layer_end),
            inference_mixed_precision=bool(inference_mixed_precision),
            training_preview_ttr=bool(training_preview_ttr),
            comet_enabled=bool(comet_enabled),
            comet_project_name=str(comet_project_name or "ttr-distillation"),
            comet_workspace=str(comet_workspace or "ken-simpson"),
            comet_api_key=str(comet_api_key or ""),
        )
        runtime.register_layer_specs(flux2_ttr.infer_flux_single_layer_specs(m))

        if training:
            if checkpoint_path and os.path.isfile(checkpoint_path):
                logger.info("Flux2TTR: loading existing checkpoint before online distillation: %s", checkpoint_path)
                runtime.load_checkpoint(checkpoint_path)
            runtime.training_mode = True
            runtime.training_enabled = train_steps > 0
            runtime.training_steps_total = max(0, train_steps)
            runtime.steps_remaining = max(0, train_steps)
            runtime.training_updates_done = 0
            loss_value = float(runtime.last_loss) if not math.isnan(runtime.last_loss) else 0.0
        else:
            if not checkpoint_path:
                raise ValueError("Flux2TTR: checkpoint_path is required when training is disabled.")
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Flux2TTR: checkpoint not found: {checkpoint_path}")
            runtime.load_checkpoint(checkpoint_path)
            runtime.training_mode = False
            runtime.training_enabled = False
            runtime.training_steps_total = 0
            runtime.steps_remaining = 0
            runtime.training_updates_done = 0
            loss_value = float(runtime.last_loss) if not math.isnan(runtime.last_loss) else 0.0

        runtime_id = flux2_ttr.register_runtime(runtime)
        transformer_options["flux2_ttr"] = {
            "enabled": True,
            "runtime_id": runtime_id,
            "training": runtime.training_enabled,
            "training_mode": runtime.training_mode,
            "training_preview_ttr": runtime.training_preview_ttr,
            "comet_enabled": runtime.comet_enabled,
            "comet_project_name": runtime.comet_project_name,
            "comet_workspace": runtime.comet_workspace,
            "training_steps_total": int(runtime.training_steps_total),
            "training_steps_remaining": int(runtime.steps_remaining),
            "learning_rate": float(learning_rate),
            "feature_dim": feature_dim,
            "query_chunk_size": int(query_chunk_size),
            "scan_chunk_size": int(query_chunk_size),
            "key_chunk_size": int(key_chunk_size),
            "landmark_count": int(landmark_count),
            "text_tokens_guess": int(text_tokens_guess),
            "alpha_init": float(alpha_init),
            "alpha_lr_multiplier": float(alpha_lr_multiplier),
            "phi_lr_multiplier": float(phi_lr_multiplier),
            "training_query_token_cap": int(training_query_token_cap),
            "replay_buffer_size": int(replay_buffer_size),
            "train_steps_per_call": int(train_steps_per_call),
            "huber_beta": float(huber_beta),
            "grad_clip_norm": float(grad_clip_norm),
            "readiness_threshold": float(readiness_threshold),
            "readiness_min_updates": int(readiness_min_updates),
            "enable_memory_reserve": bool(enable_memory_reserve),
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "inference_mixed_precision": bool(inference_mixed_precision),
            "max_safe_inference_loss": float(runtime.max_safe_inference_loss),
            "checkpoint_path": checkpoint_path,
        }

        callback_key = "flux2_ttr"
        m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_PRE_RUN, callback_key)
        m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_CLEANUP, callback_key)
        m.add_callback_with_key(
            patcher_extension.CallbacksMP.ON_PRE_RUN,
            callback_key,
            flux2_ttr.pre_run_callback,
        )
        m.add_callback_with_key(
            patcher_extension.CallbacksMP.ON_CLEANUP,
            callback_key,
            flux2_ttr.cleanup_callback,
        )

        logger.info(
            (
                "Flux2TTR configured: training_mode=%s training_preview_ttr=%s comet_enabled=%s "
                "training_steps=%d feature_dim=%d q_chunk=%d k_chunk=%d landmarks=%d "
                "replay=%d train_steps_per_call=%d readiness=(%.6g,%d) reserve=%s layer_range=[%d,%d] "
                "mixed_precision=%s checkpoint=%s loss=%.6g"
            ),
            training,
            bool(training_preview_ttr),
            bool(comet_enabled),
            train_steps,
            feature_dim,
            int(query_chunk_size),
            int(key_chunk_size),
            int(landmark_count),
            int(replay_buffer_size),
            int(train_steps_per_call),
            float(readiness_threshold),
            int(readiness_min_updates),
            bool(enable_memory_reserve),
            int(layer_start),
            int(layer_end),
            bool(inference_mixed_precision),
            checkpoint_path if checkpoint_path else "<none>",
            float(loss_value),
        )
        return io.NodeOutput(m, float(loss_value))


class ClockedSweepValues(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ClockedSweepValues",
            display_name="Clocked Sweep Values",
            category="advanced/scheduling",
            description="Map a clock list to evenly distributed sweep values.",
            inputs=[
                io.MultiType.Input(
                    io.String.Input(
                        "clock",
                        multiline=True,
                        placeholder="0, 1, 2 or 30",
                        tooltip="Clock list (length defines output length). Accepts JSON list, comma/space-separated values, a list input, or a single integer string to create 1..N.",
                    ),
                    [io.AnyType],
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "values",
                        multiline=True,
                        placeholder="0.1, 0.2, 0.3",
                        tooltip="Values to sweep across the clock (JSON list, comma/space-separated, or list input). If clock is blank, its length is inferred from values.",
                    ),
                    [io.AnyType],
                ),
            ],
            outputs=[io.Float.Output("values", is_output_list=True)],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clock, values) -> io.NodeOutput:
        clock_list, values_list = sweep_utils.parse_clock_and_values(clock, values)
        output = sweep_utils.build_clocked_sweep(clock_list, values_list)
        logger.info(
            "Clocked sweep built: clock=%d values=%d output=%d",
            len(clock_list),
            len(values_list),
            len(output),
        )
        return io.NodeOutput(output)


class Combinations(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Combinations",
            display_name="Combinations",
            category="advanced/scheduling",
            description="Generate repeated lists that cover all combinations of provided values.",
            inputs=[
                io.MultiType.Input(
                    io.String.Input(
                        "a",
                        multiline=True,
                        placeholder="1, 2, 3",
                        tooltip="Values for A (JSON list, comma/space-separated, or list input).",
                    ),
                    [io.AnyType],
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "b",
                        multiline=True,
                        placeholder="4, 5",
                        tooltip="Values for B (optional).",
                    ),
                    [io.AnyType],
                    optional=True,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "c",
                        multiline=True,
                        placeholder="",
                        tooltip="Values for C (optional).",
                    ),
                    [io.AnyType],
                    optional=True,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "d",
                        multiline=True,
                        placeholder="",
                        tooltip="Values for D (optional).",
                    ),
                    [io.AnyType],
                    optional=True,
                ),
            ],
            outputs=[
                io.Float.Output("a_out", is_output_list=True),
                io.Float.Output("b_out", is_output_list=True),
                io.Float.Output("c_out", is_output_list=True),
                io.Float.Output("d_out", is_output_list=True),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, a, b=None, c=None, d=None) -> io.NodeOutput:
        values = []
        names = []
        a_list = sweep_utils.to_float_list(a, "a")
        if not a_list:
            raise ValueError("Combinations: 'a' must contain at least one value.")
        values.append(a_list)
        names.append("a")
        for label, item in (("b", b), ("c", c), ("d", d)):
            if item is None or item == "":
                continue
            parsed = sweep_utils.to_float_list(item, label)
            if parsed:
                values.append(parsed)
                names.append(label)

        outputs = sweep_utils.build_combinations(values)
        out_map = dict(zip(names, outputs))
        return io.NodeOutput(
            out_map.get("a", []),
            out_map.get("b", []),
            out_map.get("c", []),
            out_map.get("d", []),
        )


class TaylorAttentionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TaylorAttentionBackend, HybridTaylorAttentionBackend, Flux2TTR, ClockedSweepValues, Combinations]


async def comfy_entrypoint() -> TaylorAttentionExtension:
    return TaylorAttentionExtension()
