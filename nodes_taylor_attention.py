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
                io.Combo.Input("backend", options=["standard", "taylor"], default="standard"),
                io.Int.Input("P", default=4, min=1, max=8, step=1, tooltip="Number of Taylor terms."),
                io.Int.Input("min_tokens", default=10000, min=0, max=200000, step=1, tooltip="Enable Taylor only when tokens >= min_tokens."),
                io.Int.Input("max_feature_dim_R", default=60000, min=1, max=500000, step=1, tooltip="Safety cap for feature dimension R."),
                io.Int.Input("block_size_q", default=512, min=1, max=8192, step=1),
                io.Int.Input("block_size_k", default=512, min=1, max=8192, step=1),
                io.Float.Input("eps", default=1e-6, min=1e-12, max=1e-2, step=1e-6),
                io.Boolean.Input("fallback_on_negative", default=True, tooltip="Fallback if denominators are too small."),
                io.Boolean.Input("allow_cross_attention", default=True),
                io.Int.Input("max_head_dim", default=128, min=1, max=512, step=1),
                io.Boolean.Input("log_shapes", default=False),
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
