import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

import taylor_sym_features

logger = logging.getLogger(__name__)


class TaylorAttentionFallback(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class TaylorAttentionConfig:
    enabled: bool = True
    P: int = 4
    min_tokens: int = 10000
    max_feature_dim_R: int = 60000
    block_size_q: int = 512
    block_size_k: int = 512
    eps: float = 1e-6
    fallback_on_negative: bool = True
    allow_cross_attention: bool = True
    max_head_dim: int = 128
    force_fp32: bool = True
    log_shapes: bool = False
    log_fallbacks: bool = True
    log_every: int = 100


_GLOBAL_STATS: Dict[str, Any] = {
    "calls": 0,
    "taylor_calls": 0,
    "fallback_calls": 0,
    "fallback_reasons": {},
}


def _update_stats(reason: Optional[str] = None, used_taylor: bool = False) -> None:
    _GLOBAL_STATS["calls"] += 1
    if used_taylor:
        _GLOBAL_STATS["taylor_calls"] += 1
    if reason is not None:
        _GLOBAL_STATS["fallback_calls"] += 1
        reasons = _GLOBAL_STATS["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1


def _maybe_log_stats(config: TaylorAttentionConfig) -> None:
    if config.log_every <= 0:
        return
    if _GLOBAL_STATS["calls"] % config.log_every == 0:
        logger.info(
            "Taylor attention stats: calls=%s taylor=%s fallback=%s reasons=%s",
            _GLOBAL_STATS["calls"],
            _GLOBAL_STATS["taylor_calls"],
            _GLOBAL_STATS["fallback_calls"],
            _GLOBAL_STATS["fallback_reasons"],
        )


def _resolve_config(config: Optional[Dict[str, Any]]) -> TaylorAttentionConfig:
    if config is None:
        return TaylorAttentionConfig(enabled=False)
    cfg = TaylorAttentionConfig()
    for key, value in config.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _log_shapes_once(config: TaylorAttentionConfig, transformer_options: Optional[dict], q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor], scale: float, skip_reshape: bool) -> None:
    if not config.log_shapes:
        return
    if transformer_options is not None:
        already = transformer_options.get("taylor_attention_logged_shapes", False)
        if already:
            return
        transformer_options["taylor_attention_logged_shapes"] = True
    logger.info(
        "Taylor attention shapes q=%s k=%s v=%s dtype=%s scale=%s skip_reshape=%s mask=%s",
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        q.dtype,
        scale,
        skip_reshape,
        None if mask is None else tuple(mask.shape),
    )


def _mask_is_binary(mask: torch.Tensor) -> bool:
    if mask.dtype == torch.bool:
        return True
    if mask.is_floating_point():
        return bool(torch.all((mask == 0) | (mask == 1)))
    return bool(torch.all((mask == 0) | (mask == 1)))


def _normalize_key_mask(mask: torch.Tensor, batch: int, heads: int, n_q: int, n_k: int) -> torch.Tensor:
    if not _mask_is_binary(mask):
        raise TaylorAttentionFallback("unsupported_mask_values")

    if mask.dtype != torch.bool:
        mask = mask != 0

    if mask.ndim == 2:
        if mask.shape[1] != n_k:
            raise TaylorAttentionFallback("unsupported_mask_shape")
        if mask.shape[0] not in (1, batch):
            raise TaylorAttentionFallback("unsupported_mask_shape")
        if mask.shape[0] == 1 and batch > 1:
            mask = mask.expand(batch, n_k)
        return mask

    if mask.ndim == 3:
        if mask.shape[-1] != n_k:
            raise TaylorAttentionFallback("unsupported_mask_shape")
        if mask.shape[1] == 1:
            mask = mask[:, 0, :]
        elif mask.shape[1] == n_q:
            if not torch.all(mask == mask[:, :1, :]):
                raise TaylorAttentionFallback("unsupported_mask_per_query")
            mask = mask[:, 0, :]
        else:
            raise TaylorAttentionFallback("unsupported_mask_shape")
        return mask

    if mask.ndim == 4:
        if mask.shape[-1] != n_k:
            raise TaylorAttentionFallback("unsupported_mask_shape")
        if mask.shape[1] not in (1, heads):
            raise TaylorAttentionFallback("unsupported_mask_shape")
        if mask.shape[1] == heads and not torch.all(mask == mask[:, :1, :, :]):
            raise TaylorAttentionFallback("unsupported_mask_per_head")
        if mask.shape[2] == 1:
            mask = mask[:, 0, 0, :]
        elif mask.shape[2] == n_q:
            if not torch.all(mask == mask[:, :1, :1, :]):
                raise TaylorAttentionFallback("unsupported_mask_per_query")
            mask = mask[:, 0, 0, :]
        else:
            raise TaylorAttentionFallback("unsupported_mask_shape")
        return mask

    raise TaylorAttentionFallback("unsupported_mask_shape")


def _reshape_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    skip_reshape: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
    if skip_reshape:
        if q.ndim != 4:
            raise TaylorAttentionFallback("unsupported_q_shape")
        if k.ndim != 4 or v.ndim != 4:
            raise TaylorAttentionFallback("unsupported_kv_shape")
        batch, q_heads, n_q, dim_head = q.shape
        if q_heads != heads:
            raise TaylorAttentionFallback("heads_mismatch")
        if k.shape[0] != batch or v.shape[0] != batch:
            raise TaylorAttentionFallback("batch_mismatch")
        if k.shape[1] != heads or v.shape[1] != heads:
            raise TaylorAttentionFallback("heads_mismatch")
        if k.shape[3] != dim_head or v.shape[3] != dim_head:
            raise TaylorAttentionFallback("head_dim_mismatch")
        n_k = k.shape[2]
        return q, k, v, batch, heads, n_q, n_k

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise TaylorAttentionFallback("unsupported_qkv_shape")
    batch, n_q, inner_dim = q.shape
    if inner_dim % heads != 0:
        raise TaylorAttentionFallback("head_dim_mismatch")
    dim_head = inner_dim // heads
    if k.shape[0] != batch or v.shape[0] != batch:
        raise TaylorAttentionFallback("batch_mismatch")
    if k.shape[2] != inner_dim or v.shape[2] != inner_dim:
        raise TaylorAttentionFallback("kv_dim_mismatch")
    n_k = k.shape[1]
    q = q.view(batch, n_q, heads, dim_head).permute(0, 2, 1, 3)
    k = k.view(batch, n_k, heads, dim_head).permute(0, 2, 1, 3)
    v = v.view(batch, n_k, heads, dim_head).permute(0, 2, 1, 3)
    return q, k, v, batch, heads, n_q, n_k


def taylor_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: Optional[torch.Tensor] = None,
    attn_precision: Optional[torch.dtype] = None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    config: Optional[Dict[str, Any]] = None,
    transformer_options: Optional[dict] = None,
    **kwargs,
) -> torch.Tensor:
    cfg = _resolve_config(config)
    if not cfg.enabled:
        raise TaylorAttentionFallback("disabled")

    q, k, v, batch, heads, n_q, n_k = _reshape_inputs(q, k, v, heads, skip_reshape)

    if not cfg.allow_cross_attention and n_q != n_k:
        raise TaylorAttentionFallback("cross_attention_disabled")

    tokens = max(n_q, n_k)
    if tokens < cfg.min_tokens:
        raise TaylorAttentionFallback("below_min_tokens")

    dim_head = q.shape[-1]
    if dim_head > cfg.max_head_dim:
        raise TaylorAttentionFallback("head_dim_too_large")

    feature_dim = taylor_sym_features.feature_dim(dim_head, cfg.P)
    if feature_dim > cfg.max_feature_dim_R:
        raise TaylorAttentionFallback("feature_dim_too_large")

    scale = dim_head ** -0.5
    _log_shapes_once(cfg, transformer_options, q, k, v, mask, scale, skip_reshape)

    if mask is not None:
        key_mask = _normalize_key_mask(mask, batch, heads, n_q, n_k)
        key_mask = key_mask.to(device=q.device, dtype=torch.float32)
        key_mask = key_mask[:, None, :]
    else:
        key_mask = None

    specs = taylor_sym_features.get_feature_specs(dim_head, cfg.P, q.device)

    dtype_accum = torch.float32 if cfg.force_fp32 else q.dtype
    v_dtype = v.dtype

    s = torch.zeros((batch, heads, feature_dim, v.shape[-1]), dtype=dtype_accum, device=q.device)
    z = torch.zeros((batch, heads, feature_dim), dtype=dtype_accum, device=q.device)

    sqrt_betas = []
    for spec in specs:
        beta = (scale ** spec.degree) / math.factorial(spec.degree)
        sqrt_betas.append(math.sqrt(beta))

    block_k = max(1, cfg.block_size_k)
    offset = 0
    for spec, sqrt_beta in zip(specs, sqrt_betas):
        m_p = spec.indices.shape[0]
        for start in range(0, n_k, block_k):
            end = min(start + block_k, n_k)
            k_blk = k[:, :, start:end, :]
            v_blk = v[:, :, start:end, :]
            k_blk_f = k_blk.to(dtype=dtype_accum)
            v_blk_f = v_blk.to(dtype=dtype_accum)
            phi_k = taylor_sym_features.eval_phi(k_blk_f, spec.indices)
            psi_k = phi_k * spec.sqrt_w * sqrt_beta
            if key_mask is not None:
                mask_blk = key_mask[:, :, start:end]
                psi_k = psi_k * mask_blk[..., None]
            s[:, :, offset:offset + m_p, :] += torch.einsum("b h n r, b h n d -> b h r d", psi_k, v_blk_f)
            z[:, :, offset:offset + m_p] += psi_k.sum(dim=2)
        offset += m_p

    out = torch.empty((batch, heads, n_q, v.shape[-1]), dtype=dtype_accum, device=q.device)
    block_q = max(1, cfg.block_size_q)

    for start in range(0, n_q, block_q):
        end = min(start + block_q, n_q)
        q_blk = q[:, :, start:end, :]
        q_blk_f = q_blk.to(dtype=dtype_accum)
        num = torch.zeros((batch, heads, end - start, v.shape[-1]), dtype=dtype_accum, device=q.device)
        den = torch.zeros((batch, heads, end - start), dtype=dtype_accum, device=q.device)
        offset = 0
        for spec, sqrt_beta in zip(specs, sqrt_betas):
            m_p = spec.indices.shape[0]
            phi_q = taylor_sym_features.eval_phi(q_blk_f, spec.indices)
            psi_q = phi_q * spec.sqrt_w * sqrt_beta
            num += torch.einsum("b h n r, b h r d -> b h n d", psi_q, s[:, :, offset:offset + m_p, :])
            den += torch.einsum("b h n r, b h r -> b h n", psi_q, z[:, :, offset:offset + m_p])
            offset += m_p

        if torch.isnan(den).any() or torch.isinf(den).any():
            raise TaylorAttentionFallback("denominator_invalid")
        if cfg.fallback_on_negative and torch.any(den <= cfg.eps):
            raise TaylorAttentionFallback("denominator_too_small")
        den = torch.clamp(den, min=cfg.eps)
        out[:, :, start:end, :] = num / den[..., None]

    out = out.to(dtype=v_dtype)
    if skip_output_reshape:
        return out

    out = out.permute(0, 2, 1, 3).reshape(batch, n_q, heads * dim_head)
    return out


def taylor_attention_override(original_func, *args, **kwargs):
    transformer_options = kwargs.get("transformer_options", None)
    config_dict = None
    if transformer_options is not None:
        config_dict = transformer_options.get("taylor_attention")
    cfg = _resolve_config(config_dict)
    if not cfg.enabled:
        _update_stats()
        return original_func(*args, **kwargs)

    try:
        out = taylor_attention(*args, config=config_dict, **kwargs)
        _update_stats(used_taylor=True)
        _maybe_log_stats(cfg)
        return out
    except TaylorAttentionFallback as exc:
        _update_stats(reason=exc.reason)
        _maybe_log_stats(cfg)
        if cfg.log_fallbacks:
            logger.warning("Taylor attention fallback: %s", exc.reason)
        return original_func(*args, **kwargs)
