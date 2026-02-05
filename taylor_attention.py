import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

import taylor_sym_features

try:
    from comfy import model_management
except Exception:
    model_management = None

logger = logging.getLogger(__name__)


class TaylorAttentionFallback(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class TaylorAttentionConfig:
    enabled: bool = True
    P: int = 4
    min_tokens: int = 100
    max_feature_dim_R: int = 370000
    block_size_q: int = 32
    block_size_k: int = 16
    eps: float = 1e-6
    fallback_on_negative: bool = True
    allow_cross_attention: bool = True
    max_head_dim: int = 128
    sub_head_blocks: int = 4
    force_fp32: bool = False
    memory_reserve: bool = True
    memory_reserve_factor: float = 1.1
    memory_reserve_log: bool = True
    early_probe: bool = True
    probe_samples: int = 16
    denom_fp32: bool = True
    log_shapes: bool = True
    log_fallbacks: bool = True
    log_every: int = 100
    quality_check: bool = True
    quality_check_samples: int = 16
    quality_check_log_every: int = 1


_GLOBAL_STATS: Dict[str, Any] = {
    "calls": 0,
    "taylor_calls": 0,
    "fallback_calls": 0,
    "fallback_reasons": {},
}
_QUALITY_CHECK_CALLS = 0


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


def _clone_config(config: Optional[Dict[str, Any]], cfg: TaylorAttentionConfig) -> Dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    return cfg.__dict__.copy()


def _quality_check_should_log(cfg: TaylorAttentionConfig) -> bool:
    global _QUALITY_CHECK_CALLS
    _QUALITY_CHECK_CALLS += 1
    if cfg.quality_check_log_every <= 0:
        return False
    return (_QUALITY_CHECK_CALLS % cfg.quality_check_log_every) == 0


def _run_quality_check(
    cfg: TaylorAttentionConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    key_mask_bool: Optional[torch.Tensor],
    scale: float,
) -> None:
    if not cfg.quality_check:
        return
    if not _quality_check_should_log(cfg):
        return

    batch, heads, n_q, dim_head = q.shape
    n_k = k.shape[2]
    samples = min(cfg.quality_check_samples, n_q)
    if samples <= 0:
        return

    idx = torch.randperm(n_q, device=q.device)[:samples]
    q_s = q[:, :, idx, :].float()
    k_f = k.float()
    v_f = v.float()

    scores = torch.einsum("b h s d, b h n d -> b h s n", q_s, k_f) * scale
    if key_mask_bool is not None:
        mask = key_mask_bool[:, None, None, :].to(device=scores.device)
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    exact = torch.einsum("b h s n, b h n d -> b h s d", attn, v_f)

    approx = out[:, :, idx, :].float()
    diff = (approx - exact).abs()
    mean_abs = diff.mean().item()
    max_abs = diff.max().item()
    mean_rel = (diff / (exact.abs() + 1e-6)).mean().item()

    logger.info(
        "Taylor quality check: samples=%s mean_abs=%.6f max_abs=%.6f mean_rel=%.6f",
        samples,
        mean_abs,
        max_abs,
        mean_rel,
    )





def _estimate_taylor_memory_bytes(batch: int, heads: int, n_q: int, d_val: int, feature_dim: int, m_max: int, block_q: int, block_k: int, dtype_size: int) -> int:
    core_elems = batch * heads * (feature_dim * d_val + feature_dim + n_q * d_val)
    temp_elems = batch * heads * (max(block_q, block_k) * m_max * 2)
    return int((core_elems + temp_elems) * dtype_size)



def _maybe_reserve_memory(
    cfg: TaylorAttentionConfig,
    q: torch.Tensor,
    v: torch.Tensor,
    feature_dim: int,
    specs,
    block_q: int,
    block_k: int,
    transformer_options: Optional[dict],
    dtype_accum: torch.dtype,
) -> None:
    if not cfg.memory_reserve:
        return
    if model_management is None:
        return
    if q.device.type == "cpu":
        return
    if len(specs) == 0:
        return

    batch, heads, n_q, _ = q.shape
    d_val = v.shape[-1]
    m_max = max(spec.indices.shape[0] for spec in specs)
    dtype_size = torch.tensor([], dtype=dtype_accum).element_size()
    mem_bytes = _estimate_taylor_memory_bytes(batch, heads, n_q, d_val, feature_dim, m_max, block_q, block_k, dtype_size)
    mem_bytes = int(mem_bytes * cfg.memory_reserve_factor)
    if mem_bytes <= 0:
        return

    if transformer_options is not None:
        key = (batch, heads, n_q, d_val, feature_dim, m_max, dtype_size, block_q, block_k, cfg.memory_reserve_factor)
        if transformer_options.get("taylor_attention_memory_reserved") == key:
            return
        transformer_options["taylor_attention_memory_reserved"] = key

    try:
        model_management.free_memory(mem_bytes, q.device)
        if cfg.memory_reserve_log:
            logger.info("Taylor attention reserved ~%.2f MB", mem_bytes / (1024 * 1024))
    except Exception as exc:
        logger.warning("Taylor attention reserve memory failed: %s", exc)





def _probe_denominators(
    cfg: TaylorAttentionConfig,
    q: torch.Tensor,
    z: torch.Tensor,
    specs,
    sqrt_betas,
    sqrt_ws,
    eps: float,
    dtype_accum: torch.dtype,
) -> None:
    if not cfg.early_probe:
        return
    n_q = q.shape[2]
    samples = min(cfg.probe_samples, n_q)
    if samples <= 0:
        return

    idx = torch.randperm(n_q, device=q.device)[:samples]
    q_probe = q[:, :, idx, :].to(dtype=dtype_accum)

    den_dtype = torch.float32 if cfg.denom_fp32 else dtype_accum
    den = torch.zeros((q.shape[0], q.shape[1], samples), dtype=den_dtype, device=q.device)

    offset = 0
    for spec, sqrt_beta, sqrt_w in zip(specs, sqrt_betas, sqrt_ws):
        m_p = spec.indices.shape[0]
        phi_q = taylor_sym_features.eval_phi(q_probe, spec.indices)
        psi_q = phi_q * sqrt_w * sqrt_beta
        z_slice = z[:, :, offset:offset + m_p]
        if cfg.denom_fp32:
            den += torch.einsum("b h n r, b h r -> b h n", psi_q.float(), z_slice.float())
        else:
            den += torch.einsum("b h n r, b h r -> b h n", psi_q, z_slice)
        offset += m_p

    probe_stats = _init_den_stats()
    _accum_den_stats(probe_stats, den, eps)
    _log_den_stats(probe_stats, prefix="Taylor denominator stats (probe)")

    if torch.isnan(den).any() or torch.isinf(den).any():
        raise TaylorAttentionFallback("denominator_invalid")
    if cfg.fallback_on_negative and torch.any(den <= eps):
        raise TaylorAttentionFallback("denominator_too_small")



def _init_den_stats() -> Dict[str, float]:
    return {"min": None, "max": None, "sum": 0.0, "count": 0, "le_eps": 0, "blocks": 0}


def _accum_den_stats(stats: Dict[str, float], den: torch.Tensor, eps: float) -> None:
    den_f = den.float()
    min_val = float(den_f.min().item())
    max_val = float(den_f.max().item())
    sum_val = float(den_f.sum().item())
    count = int(den_f.numel())
    le_eps = int((den_f <= eps).sum().item())

    if stats["min"] is None or min_val < stats["min"]:
        stats["min"] = min_val
    if stats["max"] is None or max_val > stats["max"]:
        stats["max"] = max_val
    stats["sum"] += sum_val
    stats["count"] += count
    stats["le_eps"] += le_eps
    stats["blocks"] += 1


def _log_den_stats(stats: Dict[str, float], prefix: str = "Taylor denominator stats") -> None:
    if stats["count"] <= 0:
        return
    mean_val = stats["sum"] / stats["count"]
    frac_le = stats["le_eps"] / stats["count"]
    logger.info(
        "%s: min=%.6g max=%.6g mean=%.6g frac_le_eps=%.6g blocks=%s",
        prefix,
        stats["min"],
        stats["max"],
        mean_val,
        frac_le,
        stats["blocks"],
    )


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
    sub_head_blocks = max(1, cfg.sub_head_blocks)
    if sub_head_blocks > 1:
        if dim_head % sub_head_blocks != 0:
            raise TaylorAttentionFallback("sub_head_block_mismatch")
        block_dim = dim_head // sub_head_blocks
        if block_dim > cfg.max_head_dim:
            raise TaylorAttentionFallback("head_dim_too_large")

        feature_dim = taylor_sym_features.feature_dim(block_dim, cfg.P)
        if feature_dim > cfg.max_feature_dim_R:
            if cfg.log_fallbacks:
                logger.warning(
                    "Taylor attention fallback: feature_dim_too_large (R=%s > max_feature_dim_R=%s) head_dim=%s block_dim=%s sub_head_blocks=%s P=%s",
                    feature_dim,
                    cfg.max_feature_dim_R,
                    dim_head,
                    block_dim,
                    sub_head_blocks,
                    cfg.P,
                )
            raise TaylorAttentionFallback("feature_dim_too_large")

        sub_config = _clone_config(config, cfg)
        sub_config["sub_head_blocks"] = 1
        sub_outputs = []
        for block_idx in range(sub_head_blocks):
            start = block_idx * block_dim
            end = start + block_dim
            sub_outputs.append(
                taylor_attention(
                    q[..., start:end],
                    k[..., start:end],
                    v[..., start:end],
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=True,
                    skip_output_reshape=True,
                    config=sub_config,
                    transformer_options=transformer_options,
                    **kwargs,
                )
            )
        out = torch.cat(sub_outputs, dim=-1)
        if skip_output_reshape:
            return out
        out = out.permute(0, 2, 1, 3).reshape(batch, n_q, heads * dim_head)
        return out

    if dim_head > cfg.max_head_dim:
        raise TaylorAttentionFallback("head_dim_too_large")

    feature_dim = taylor_sym_features.feature_dim(dim_head, cfg.P)
    if feature_dim > cfg.max_feature_dim_R:
        if cfg.log_fallbacks:
            logger.warning(
                "Taylor attention fallback: feature_dim_too_large (R=%s > max_feature_dim_R=%s) head_dim=%s P=%s",
                feature_dim,
                cfg.max_feature_dim_R,
                dim_head,
                cfg.P,
            )
        raise TaylorAttentionFallback("feature_dim_too_large")

    scale = dim_head ** -0.5
    _log_shapes_once(cfg, transformer_options, q, k, v, mask, scale, skip_reshape)

    dtype_accum = torch.float32 if cfg.force_fp32 else q.dtype
    v_dtype = v.dtype
    block_k = max(1, cfg.block_size_k)
    block_q = max(1, cfg.block_size_q)

    if mask is not None:
        key_mask_bool = _normalize_key_mask(mask, batch, heads, n_q, n_k).to(device=q.device)
        key_mask = key_mask_bool.to(dtype=dtype_accum)
        key_mask = key_mask[:, None, :]
    else:
        key_mask_bool = None
        key_mask = None

    specs = taylor_sym_features.get_feature_specs(dim_head, cfg.P, q.device)

    _maybe_reserve_memory(cfg, q, v, feature_dim, specs, block_q, block_k, transformer_options, dtype_accum)

    sqrt_betas = []
    sqrt_ws = []
    for spec in specs:
        beta = (scale ** spec.degree) / math.factorial(spec.degree)
        sqrt_betas.append(torch.tensor(math.sqrt(beta), dtype=dtype_accum, device=q.device))
        sqrt_ws.append(spec.sqrt_w.to(dtype=dtype_accum))

    if cfg.early_probe:
        z = torch.zeros((batch, heads, feature_dim), dtype=dtype_accum, device=q.device)
        offset = 0
        for spec, sqrt_beta, sqrt_w in zip(specs, sqrt_betas, sqrt_ws):
            m_p = spec.indices.shape[0]
            for start in range(0, n_k, block_k):
                end = min(start + block_k, n_k)
                k_blk = k[:, :, start:end, :]
                k_blk_f = k_blk.to(dtype=dtype_accum)
                phi_k = taylor_sym_features.eval_phi(k_blk_f, spec.indices)
                psi_k = phi_k * sqrt_w * sqrt_beta
                if key_mask is not None:
                    mask_blk = key_mask[:, :, start:end]
                    psi_k = psi_k * mask_blk[..., None]
                z[:, :, offset:offset + m_p] += psi_k.sum(dim=2)
            offset += m_p

        _probe_denominators(cfg, q, z, specs, sqrt_betas, sqrt_ws, cfg.eps, dtype_accum)

        s = torch.zeros((batch, heads, feature_dim, v.shape[-1]), dtype=dtype_accum, device=q.device)
        offset = 0
        for spec, sqrt_beta, sqrt_w in zip(specs, sqrt_betas, sqrt_ws):
            m_p = spec.indices.shape[0]
            for start in range(0, n_k, block_k):
                end = min(start + block_k, n_k)
                k_blk = k[:, :, start:end, :]
                v_blk = v[:, :, start:end, :]
                k_blk_f = k_blk.to(dtype=dtype_accum)
                v_blk_f = v_blk.to(dtype=dtype_accum)
                phi_k = taylor_sym_features.eval_phi(k_blk_f, spec.indices)
                psi_k = phi_k * sqrt_w * sqrt_beta
                if key_mask is not None:
                    mask_blk = key_mask[:, :, start:end]
                    psi_k = psi_k * mask_blk[..., None]
                s[:, :, offset:offset + m_p, :] += torch.einsum("b h n r, b h n d -> b h r d", psi_k, v_blk_f)
            offset += m_p
    else:
        s = torch.zeros((batch, heads, feature_dim, v.shape[-1]), dtype=dtype_accum, device=q.device)
        z = torch.zeros((batch, heads, feature_dim), dtype=dtype_accum, device=q.device)
        offset = 0
        for spec, sqrt_beta, sqrt_w in zip(specs, sqrt_betas, sqrt_ws):
            m_p = spec.indices.shape[0]
            for start in range(0, n_k, block_k):
                end = min(start + block_k, n_k)
                k_blk = k[:, :, start:end, :]
                v_blk = v[:, :, start:end, :]
                k_blk_f = k_blk.to(dtype=dtype_accum)
                v_blk_f = v_blk.to(dtype=dtype_accum)
                phi_k = taylor_sym_features.eval_phi(k_blk_f, spec.indices)
                psi_k = phi_k * sqrt_w * sqrt_beta
                if key_mask is not None:
                    mask_blk = key_mask[:, :, start:end]
                    psi_k = psi_k * mask_blk[..., None]
                s[:, :, offset:offset + m_p, :] += torch.einsum("b h n r, b h n d -> b h r d", psi_k, v_blk_f)
                z[:, :, offset:offset + m_p] += psi_k.sum(dim=2)
            offset += m_p

    out = torch.empty((batch, heads, n_q, v.shape[-1]), dtype=dtype_accum, device=q.device)

    den_dtype = torch.float32 if cfg.denom_fp32 else dtype_accum
    den_stats = _init_den_stats()

    for start in range(0, n_q, block_q):
        end = min(start + block_q, n_q)
        q_blk = q[:, :, start:end, :]
        q_blk_f = q_blk.to(dtype=dtype_accum)
        num = torch.zeros((batch, heads, end - start, v.shape[-1]), dtype=dtype_accum, device=q.device)
        den = torch.zeros((batch, heads, end - start), dtype=den_dtype, device=q.device)
        offset = 0
        for spec, sqrt_beta, sqrt_w in zip(specs, sqrt_betas, sqrt_ws):
            m_p = spec.indices.shape[0]
            phi_q = taylor_sym_features.eval_phi(q_blk_f, spec.indices)
            psi_q = phi_q * sqrt_w * sqrt_beta
            num += torch.einsum("b h n r, b h r d -> b h n d", psi_q, s[:, :, offset:offset + m_p, :])
            z_slice = z[:, :, offset:offset + m_p]
            if cfg.denom_fp32:
                den += torch.einsum("b h n r, b h r -> b h n", psi_q.float(), z_slice.float())
            else:
                den += torch.einsum("b h n r, b h r -> b h n", psi_q, z_slice)
            offset += m_p

        _accum_den_stats(den_stats, den, cfg.eps)
        probe_stats = _init_den_stats()
    _accum_den_stats(probe_stats, den, eps)
    _log_den_stats(probe_stats, prefix="Taylor denominator stats (probe)")

    if torch.isnan(den).any() or torch.isinf(den).any():
            _log_den_stats(den_stats, prefix="Taylor denominator stats (partial)")
            raise TaylorAttentionFallback("denominator_invalid")
        if cfg.fallback_on_negative and torch.any(den <= cfg.eps):
            _log_den_stats(den_stats, prefix="Taylor denominator stats (partial)")
            raise TaylorAttentionFallback("denominator_too_small")
        den = torch.clamp(den, min=cfg.eps)
        if den.dtype != dtype_accum:
            den = den.to(dtype_accum)
        out[:, :, start:end, :] = num / den[..., None]

    _log_den_stats(den_stats)

    _run_quality_check(cfg, q, k, v, out, key_mask_bool, scale)

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
