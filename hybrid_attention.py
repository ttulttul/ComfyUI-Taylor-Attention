from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

import taylor_sym_features

logger = logging.getLogger(__name__)

_ORIGINAL_FLUX_ATTENTION: Dict[str, Any] = {}
_PATCH_DEPTH = 0
_PCA_CACHE: Dict[Tuple[torch.device, int, int], torch.Tensor] = {}


@dataclass
class HybridAttentionConfig:
    enabled: bool = True
    local_window: int = 512
    local_chunk: int = 256
    prefix_tokens: int = 0
    global_dim: int = 16
    global_P: int = 2
    global_weight: float = 0.1
    global_sigma_low: float = 0.0
    global_sigma_high: float = 0.0
    global_scale_mul: float = 1.0
    global_norm_power: float = 0.0
    global_norm_clip: float = 0.0
    use_pca: bool = True
    pca_samples: int = 2048
    allow_cross_attention: bool = True
    layer_start: int = -1
    layer_end: int = -1
    eps: float = 1e-6
    force_fp32: bool = True
    log_steps: bool = True


def _resolve_config(config: Optional[Dict[str, Any]]) -> HybridAttentionConfig:
    if config is None:
        return HybridAttentionConfig(enabled=False)
    cfg = HybridAttentionConfig()
    for key, value in config.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _get_sigma(transformer_options: Optional[dict]) -> Optional[float]:
    if transformer_options is None:
        return None
    sigmas = transformer_options.get("sigmas")
    if sigmas is None:
        return None
    if torch.is_tensor(sigmas):
        if sigmas.numel() > 0:
            return float(sigmas.flatten()[0].item())
        return None
    try:
        return float(sigmas)
    except (TypeError, ValueError):
        return None


def _compute_global_weight(cfg: HybridAttentionConfig, sigma: Optional[float]) -> float:
    if sigma is None:
        return cfg.global_weight
    if cfg.global_sigma_high <= cfg.global_sigma_low or cfg.global_sigma_high <= 0:
        return cfg.global_weight
    if sigma <= cfg.global_sigma_low:
        return 0.0
    if sigma >= cfg.global_sigma_high:
        return cfg.global_weight
    return cfg.global_weight * (sigma - cfg.global_sigma_low) / (cfg.global_sigma_high - cfg.global_sigma_low)


def _slice_mask(mask: torch.Tensor, q_start: int, q_end: int, k_start: int, k_end: int) -> torch.Tensor:
    if mask is None:
        return None
    if mask.ndim == 2:
        return mask[:, k_start:k_end]
    if mask.ndim == 3:
        return mask[:, q_start:q_end, k_start:k_end]
    if mask.ndim == 4:
        return mask[:, :, q_start:q_end, k_start:k_end]
    return mask


def _project_pca(q: torch.Tensor, k: torch.Tensor, d_low: int, samples: int) -> torch.Tensor:
    device = q.device
    dim = q.shape[-1]
    d_low = min(d_low, dim)
    key = (device, dim, d_low)
    cached = _PCA_CACHE.get(key)
    if cached is not None:
        return cached

    qk = torch.cat([q, k], dim=2)
    flat = qk.reshape(-1, dim)
    if samples > 0 and flat.shape[0] > samples:
        idx = torch.randperm(flat.shape[0], device=flat.device)[:samples]
        flat = flat[idx]
    if flat.shape[0] < d_low or flat.shape[0] < 2:
        proj = torch.eye(dim, device=device, dtype=q.dtype)[:, :d_low]
        _PCA_CACHE[key] = proj
        logger.warning(
            "Hybrid attention PCA fallback: only %d samples for d_low=%d; using identity projection.",
            flat.shape[0],
            d_low,
        )
        return proj
    flat = flat.float()
    flat = flat - flat.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(flat, q=d_low, center=False)
    proj = v[:, :d_low].to(dtype=q.dtype, device=q.device)
    _PCA_CACHE[key] = proj
    return proj


def _apply_qk_norm(q: torch.Tensor, k: torch.Tensor, power: float, clip: float, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if power <= 0 and clip <= 0:
        return q, k
    q_f = q.float()
    k_f = k.float()
    q_norm = torch.norm(q_f, dim=-1, keepdim=True) + eps
    k_norm = torch.norm(k_f, dim=-1, keepdim=True) + eps
    if power > 0:
        q_f = q_f / (q_norm ** power)
        k_f = k_f / (k_norm ** power)
    if clip > 0:
        q_f = q_f * torch.clamp(clip / q_norm, max=1.0)
        k_f = k_f * torch.clamp(clip / k_norm, max=1.0)
    return q_f, k_f


def _taylor_feature_map(x: torch.Tensor, P: int, scale: float) -> torch.Tensor:
    device = x.device
    dtype = x.dtype
    dim = x.shape[-1]
    specs = taylor_sym_features.get_feature_specs(dim, P, device)
    features = []
    for spec in specs:
        phi = taylor_sym_features.eval_phi(x, spec.indices)
        sqrt_beta = math.sqrt((scale ** spec.degree) / math.factorial(spec.degree))
        psi = phi * spec.sqrt_w.to(dtype=phi.dtype) * sqrt_beta
        features.append(psi)
    if len(features) == 1:
        return features[0].to(dtype=dtype)
    return torch.cat(features, dim=-1).to(dtype=dtype)


def _global_taylor_attention(
    cfg: HybridAttentionConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: Optional[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    d_low = min(cfg.global_dim, q.shape[-1])
    if d_low <= 0:
        return torch.zeros_like(v)
    if cfg.use_pca:
        proj = _project_pca(q, k, d_low, cfg.pca_samples)
        q_low = q @ proj
        k_low = k @ proj
    else:
        q_low = q[..., :d_low]
        k_low = k[..., :d_low]

    q_low, k_low = _apply_qk_norm(q_low, k_low, cfg.global_norm_power, cfg.global_norm_clip, cfg.eps)

    scale_low = (q.shape[-1] ** -0.5) * cfg.global_scale_mul
    scale_low = math.sqrt(scale_low)
    q_low = q_low * scale_low
    k_low = k_low * scale_low

    phi_q = _taylor_feature_map(q_low, cfg.global_P, 1.0)
    phi_k = _taylor_feature_map(k_low, cfg.global_P, 1.0)

    if key_mask is not None:
        phi_k = phi_k * key_mask[..., None]

    kv_summary = torch.einsum("b h n r, b h n d -> b h r d", phi_k, v.float())
    global_out = torch.einsum("b h n r, b h r d -> b h n d", phi_q, kv_summary)
    k_sum = phi_k.sum(dim=2)
    denom = torch.einsum("b h n r, b h r -> b h n", phi_q, k_sum)
    denom = denom.clamp_min(cfg.eps)
    global_out = global_out / denom[..., None]
    return global_out.to(dtype=v.dtype)


def _local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: Optional[torch.Tensor],
    transformer_options: Optional[dict],
    window: int,
    chunk: int,
    prefix_tokens: int,
):
    from comfy.ldm.modules.attention import optimized_attention

    batch, heads, n_q, dim_head = q.shape
    n_k = k.shape[2]
    if window <= 0:
        return optimized_attention(q, k, v, heads, skip_reshape=True, mask=mask, transformer_options=transformer_options)

    out = torch.zeros((batch, n_q, heads * dim_head), device=q.device, dtype=q.dtype)
    chunk = max(1, chunk)
    window = max(1, window)
    for start in range(0, n_q, chunk):
        end = min(start + chunk, n_q)
        k_start = max(0, start - window)
        k_end = min(n_k, end + window)
        if prefix_tokens > 0 and k_start > prefix_tokens:
            k_prefix = k[:, :, :prefix_tokens, :]
            v_prefix = v[:, :, :prefix_tokens, :]
            k_local = k[:, :, k_start:k_end, :]
            v_local = v[:, :, k_start:k_end, :]
            k_cat = torch.cat([k_prefix, k_local], dim=2)
            v_cat = torch.cat([v_prefix, v_local], dim=2)
            if mask is not None:
                mask_prefix = _slice_mask(mask, start, end, 0, prefix_tokens)
                mask_local = _slice_mask(mask, start, end, k_start, k_end)
                if mask_prefix is not None and mask_local is not None:
                    mask_slice = torch.cat([mask_prefix, mask_local], dim=-1)
                else:
                    mask_slice = None
            else:
                mask_slice = None
        else:
            k_cat = k[:, :, k_start:k_end, :]
            v_cat = v[:, :, k_start:k_end, :]
            mask_slice = _slice_mask(mask, start, end, k_start, k_end)

        out[:, start:end, :] = optimized_attention(
            q[:, :, start:end, :],
            k_cat,
            v_cat,
            heads,
            skip_reshape=True,
            mask=mask_slice,
            transformer_options=transformer_options,
        )
    return out


def hybrid_attention(q, k, v, pe, mask=None, transformer_options=None):
    global _ORIGINAL_FLUX_ATTENTION
    if _ORIGINAL_FLUX_ATTENTION is None:
        raise RuntimeError("Hybrid attention not initialized")

    cfg = _resolve_config(transformer_options.get("hybrid_taylor_attention") if transformer_options else None)
    if not cfg.enabled:
        return _ORIGINAL_FLUX_ATTENTION(q, k, v, pe, mask=mask, transformer_options=transformer_options)

    if not cfg.allow_cross_attention and q.shape[2] != k.shape[2]:
        return _ORIGINAL_FLUX_ATTENTION(q, k, v, pe, mask=mask, transformer_options=transformer_options)

    if cfg.layer_start >= 0 or cfg.layer_end >= 0:
        block_index = None if transformer_options is None else transformer_options.get("block_index")
        if isinstance(block_index, int):
            if cfg.layer_start >= 0 and block_index < cfg.layer_start:
                return _ORIGINAL_FLUX_ATTENTION(q, k, v, pe, mask=mask, transformer_options=transformer_options)
            if cfg.layer_end >= 0 and block_index > cfg.layer_end:
                return _ORIGINAL_FLUX_ATTENTION(q, k, v, pe, mask=mask, transformer_options=transformer_options)

    sigma = _get_sigma(transformer_options)
    global_weight = _compute_global_weight(cfg, sigma)

    if pe is not None:
        from comfy.ldm.flux.math import apply_rope
        q_rope, k_rope = apply_rope(q, k, pe)
    else:
        q_rope, k_rope = q, k

    heads = q.shape[1]
    local_out = _local_attention(
        q_rope,
        k_rope,
        v,
        heads,
        mask,
        transformer_options,
        cfg.local_window,
        cfg.local_chunk,
        cfg.prefix_tokens,
    )

    if global_weight <= 0:
        if cfg.log_steps:
            logger.info(
                "Hybrid attention: sigma=%s local_window=%s prefix_tokens=%s global_weight=0 (global skipped)",
                sigma,
                cfg.local_window,
                cfg.prefix_tokens,
            )
        return local_out

    key_mask = None
    if mask is not None:
        try:
            import taylor_attention
            key_mask = taylor_attention._normalize_key_mask(mask, q.shape[0], q.shape[1], q.shape[2], k.shape[2])
        except Exception:
            key_mask = None

    q_global = q.float() if cfg.force_fp32 else q
    k_global = k.float() if cfg.force_fp32 else k
    v_global = v.float() if cfg.force_fp32 else v
    global_out = _global_taylor_attention(cfg, q_global, k_global, v_global, key_mask, scale=1.0)

    if local_out.ndim == 3:
        global_out = global_out.permute(0, 2, 1, 3).reshape(local_out.shape[0], local_out.shape[1], -1)
    if global_out.dtype != local_out.dtype:
        global_out = global_out.to(dtype=local_out.dtype)
    weight = torch.tensor(global_weight, dtype=local_out.dtype, device=local_out.device)
    out = local_out + weight * global_out
    if cfg.log_steps:
        logger.info(
            "Hybrid attention: sigma=%s local_window=%s prefix_tokens=%s global_dim=%s global_P=%s global_weight=%.3g",
            sigma,
            cfg.local_window,
            cfg.prefix_tokens,
            cfg.global_dim,
            cfg.global_P,
            global_weight,
        )
    return out


def enable_hybrid_attention() -> None:
    patch_flux_attention()


def disable_hybrid_attention() -> None:
    restore_flux_attention()


def patch_flux_attention() -> None:
    global _ORIGINAL_FLUX_ATTENTION, _PATCH_DEPTH
    if _PATCH_DEPTH == 0:
        import comfy.ldm.flux.math as flux_math
        import comfy.ldm.flux.layers as flux_layers
        _ORIGINAL_FLUX_ATTENTION["math"] = flux_math.attention
        _ORIGINAL_FLUX_ATTENTION["layers"] = flux_layers.attention
        flux_math.attention = hybrid_attention
        flux_layers.attention = hybrid_attention
        logger.info("Hybrid Taylor attention enabled (Flux attention patched).")
    _PATCH_DEPTH += 1


def restore_flux_attention() -> None:
    global _ORIGINAL_FLUX_ATTENTION, _PATCH_DEPTH
    if _PATCH_DEPTH <= 0:
        return
    _PATCH_DEPTH -= 1
    if _PATCH_DEPTH == 0 and _ORIGINAL_FLUX_ATTENTION:
        import comfy.ldm.flux.math as flux_math
        import comfy.ldm.flux.layers as flux_layers
        flux_math.attention = _ORIGINAL_FLUX_ATTENTION.get("math", flux_math.attention)
        flux_layers.attention = _ORIGINAL_FLUX_ATTENTION.get("layers", flux_layers.attention)
        _ORIGINAL_FLUX_ATTENTION = {}
        logger.info("Hybrid Taylor attention disabled (Flux attention restored).")


def hybrid_wrapper(executor, *args, **kwargs):
    transformer_options = kwargs.get("transformer_options")
    cfg = transformer_options.get("hybrid_taylor_attention") if transformer_options else None
    next_exec = executor._create_next_executor()
    if not cfg or not cfg.get("enabled", False):
        return next_exec.execute(*args, **kwargs)
    if kwargs.get("_hybrid_wrapper_active"):
        return next_exec.execute(*args, **kwargs)
    kwargs["_hybrid_wrapper_active"] = True
    patch_flux_attention()
    try:
        return next_exec.execute(*args, **kwargs)
    finally:
        kwargs.pop("_hybrid_wrapper_active", None)
        restore_flux_attention()


def pre_run_callback(patcher):
    transformer_options = getattr(patcher, "model_options", {}).get("transformer_options", {})
    if not transformer_options:
        transformer_options = getattr(getattr(patcher, "model", None), "model_options", {}).get("transformer_options", {})
    cfg = transformer_options.get("hybrid_taylor_attention")
    if not cfg or not cfg.get("enabled", False):
        return
    if cfg.get("log_steps", False):
        logger.info(
            "Hybrid attention pre-run: enabled local_window=%s prefix_tokens=%s global_dim=%s global_P=%s global_weight=%.3g",
            cfg.get("local_window"),
            cfg.get("prefix_tokens"),
            cfg.get("global_dim"),
            cfg.get("global_P"),
            cfg.get("global_weight"),
        )
    patch_flux_attention()


def cleanup_callback(patcher):
    transformer_options = getattr(patcher, "model_options", {}).get("transformer_options", {})
    if not transformer_options:
        transformer_options = getattr(getattr(patcher, "model", None), "model_options", {}).get("transformer_options", {})
    cfg = transformer_options.get("hybrid_taylor_attention")
    if not cfg or not cfg.get("enabled", False):
        return
    restore_flux_attention()
    if cfg.get("log_steps", False):
        logger.info("Hybrid attention cleanup: restored Flux attention")
