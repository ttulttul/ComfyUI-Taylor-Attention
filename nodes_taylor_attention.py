import logging
from typing import Dict, Any

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

import taylor_attention

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
    scale_mul: float,
    force_fp32: bool,
    memory_reserve: bool,
    memory_reserve_factor: float,
    memory_reserve_log: bool,
    early_probe: bool,
    probe_samples: int,
    denom_fp32: bool,
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
        "scale_mul": float(scale_mul),
        "force_fp32": bool(force_fp32),
        "memory_reserve": bool(memory_reserve),
        "memory_reserve_factor": float(memory_reserve_factor),
        "memory_reserve_log": bool(memory_reserve_log),
        "early_probe": bool(early_probe),
        "probe_samples": int(probe_samples),
        "denom_fp32": bool(denom_fp32),
        "quality_check": bool(quality_check),
        "quality_check_samples": int(quality_check_samples),
        "quality_check_log_every": int(quality_check_log_every),
        "log_shapes": bool(log_shapes),
        "log_fallbacks": bool(log_fallbacks),
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
                io.Int.Input("max_feature_dim_R", default=370000, min=1, max=500000, step=1, tooltip="Safety cap for feature dimension R."),
                io.Int.Input("block_size_q", default=32, min=1, max=8192, step=1),
                io.Int.Input("block_size_k", default=16, min=1, max=8192, step=1),
                io.Float.Input("eps", default=1e-6, min=1e-12, max=1e-2, step=1e-6),
                io.Boolean.Input("fallback_on_negative", default=True, tooltip="Fallback if denominators are too small."),
                io.Boolean.Input("allow_cross_attention", default=True),
                io.Int.Input("max_head_dim", default=128, min=1, max=512, step=1),
                io.Int.Input("sub_head_blocks", default=4, min=1, max=16, step=1, tooltip="Split each head into sub-blocks to reduce feature dimension."),
                io.Boolean.Input("qk_normalize", default=False, tooltip="L2-normalize queries/keys before Taylor features."),
                io.Float.Input("scale_mul", default=1.0, min=0.0, max=4.0, step=0.05, tooltip="Additional scale multiplier for q·k before Taylor."),
                io.Boolean.Input("force_fp32", default=False, tooltip="Accumulate Taylor features in fp32 for stability."),
                io.Boolean.Input("memory_reserve", default=True, tooltip="Ask ComfyUI to free VRAM for Taylor attention."),
                io.Float.Input("memory_reserve_factor", default=1.1, min=1.0, max=4.0, step=0.05, tooltip="Safety multiplier for reserved VRAM estimate."),
                io.Boolean.Input("memory_reserve_log", default=True, tooltip="Log reserved VRAM estimates."),
                io.Boolean.Input("early_probe", default=True, tooltip="Probe denominators before full Taylor compute."),
                io.Int.Input("probe_samples", default=16, min=1, max=64, step=1, tooltip="Queries sampled for early probe."),
                io.Boolean.Input("denom_fp32", default=True, tooltip="Compute denominators in fp32 for stability."),
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
        scale_mul: float,
        force_fp32: bool,
        memory_reserve: bool,
        memory_reserve_factor: float,
        memory_reserve_log: bool,
        early_probe: bool,
        probe_samples: int,
        denom_fp32: bool,
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
                scale_mul,
                force_fp32,
                memory_reserve,
                memory_reserve_factor,
                memory_reserve_log,
                early_probe,
                probe_samples,
                denom_fp32,
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


class TaylorAttentionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TaylorAttentionBackend]


async def comfy_entrypoint() -> TaylorAttentionExtension:
    return TaylorAttentionExtension()
