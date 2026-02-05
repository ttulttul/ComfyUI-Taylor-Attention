import logging
import math
import random
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
    qk_normalize: bool = False
    qk_norm_clip: float = 0.0
    qk_norm_power: float = 0.0
    scale_mul: float = 1.0
    force_fp32: bool = False
    memory_reserve: bool = True
    memory_reserve_factor: float = 1.1
    memory_reserve_log: bool = True
    early_probe: bool = True
    probe_samples: int = 16
    denom_fp32: bool = True
    denom_fallback_frac_limit: float = 0.0
    auto_tune: bool = False
    auto_tune_steps: int = 1
    auto_tune_candidates: int = 8
    auto_tune_quality_samples: int = 4
    auto_tune_seed: int = 0
    auto_tune_qk_norm_power_min: float = 0.2
    auto_tune_qk_norm_power_max: float = 0.7
    auto_tune_qk_norm_clip_min: float = 8.0
    auto_tune_qk_norm_clip_max: float = 20.0
    auto_tune_scale_mul_min: float = 0.8
    auto_tune_scale_mul_max: float = 1.0
    auto_tune_max_tokens: int = 512
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

def _init_quality_stats() -> Dict[str, float]:
    return {"sum_abs": 0.0, "sum_rel": 0.0, "max_abs": 0.0, "max_rel": 0.0, "samples": 0}


def _merge_quality_stats(dst: Dict[str, float], src: Optional[Dict[str, float]]) -> None:
    if not src:
        return
    dst["sum_abs"] += src["sum_abs"]
    dst["sum_rel"] += src["sum_rel"]
    dst["samples"] += src["samples"]
    if src["max_abs"] > dst["max_abs"]:
        dst["max_abs"] = src["max_abs"]
    if src["max_rel"] > dst["max_rel"]:
        dst["max_rel"] = src["max_rel"]


def _merge_den_stats(dst: Dict[str, float], src: Dict[str, float]) -> None:
    if src["count"] <= 0:
        return
    if dst["min"] is None or src["min"] < dst["min"]:
        dst["min"] = src["min"]
    if dst["max"] is None or src["max"] > dst["max"]:
        dst["max"] = src["max"]
    dst["sum"] += src["sum"]
    dst["count"] += src["count"]
    dst["le_eps"] += src["le_eps"]
    dst["blocks"] += src["blocks"]


def _config_summary(cfg: TaylorAttentionConfig) -> Dict[str, Any]:
    return {
        "qk_normalize": cfg.qk_normalize,
        "qk_norm_power": cfg.qk_norm_power,
        "qk_norm_clip": cfg.qk_norm_clip,
        "scale_mul": cfg.scale_mul,
        "sub_head_blocks": cfg.sub_head_blocks,
        "force_fp32": cfg.force_fp32,
        "denom_fp32": cfg.denom_fp32,
        "probe_samples": cfg.probe_samples,
        "denom_fallback_frac_limit": cfg.denom_fallback_frac_limit,
    }

_DIAG_SAMPLE_Q = 64
_DIAG_SAMPLE_K = 64
_DIAG_QUANTILES = torch.tensor([0.5, 0.9, 0.99])


def _summarize_stats(values: torch.Tensor) -> Optional[Dict[str, float]]:
    if values.numel() == 0:
        return None
    vals = values.float().flatten()
    if vals.numel() == 0:
        return None
    quantiles = torch.quantile(vals, _DIAG_QUANTILES.to(device=vals.device))
    return {
        "min": float(vals.min().item()),
        "max": float(vals.max().item()),
        "mean": float(vals.mean().item()),
        "p50": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
    }


def _format_stats(stats: Optional[Dict[str, float]]) -> str:
    if not stats:
        return "n/a"
    return (
        "min={min:.6g} mean={mean:.6g} p50={p50:.6g} "
        "p90={p90:.6g} p99={p99:.6g} max={max:.6g}"
    ).format(**stats)


def _sample_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    max_q: int = _DIAG_SAMPLE_Q,
    max_k: int = _DIAG_SAMPLE_K,
) -> torch.Tensor:
    n_q = q.shape[2]
    n_k = k.shape[2]
    q_count = min(max_q, n_q)
    k_count = min(max_k, n_k)
    q_idx = torch.randperm(n_q, device=q.device)[:q_count]
    k_idx = torch.randperm(n_k, device=k.device)[:k_count]
    q_s = q[:, :, q_idx, :].float()
    k_s = k[:, :, k_idx, :].float()
    dots = torch.einsum("b h q d, b h k d -> b h q k", q_s, k_s)
    return dots * scale


def _compute_diagnostics(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    q_eff: torch.Tensor,
    k_eff: torch.Tensor,
    scale_raw: float,
    scale_eff: float,
) -> Dict[str, str]:
    with torch.no_grad():
        q_norm_raw = torch.norm(q_raw.float(), dim=-1)
        k_norm_raw = torch.norm(k_raw.float(), dim=-1)
        q_norm_eff = torch.norm(q_eff.float(), dim=-1)
        k_norm_eff = torch.norm(k_eff.float(), dim=-1)
        qk_raw = _sample_qk(q_raw, k_raw, scale_raw)
        qk_eff = _sample_qk(q_eff, k_eff, scale_eff)
        return {
            "q_norm_raw": _format_stats(_summarize_stats(q_norm_raw)),
            "k_norm_raw": _format_stats(_summarize_stats(k_norm_raw)),
            "q_norm_eff": _format_stats(_summarize_stats(q_norm_eff)),
            "k_norm_eff": _format_stats(_summarize_stats(k_norm_eff)),
            "qk_raw": _format_stats(_summarize_stats(qk_raw)),
            "qk_eff": _format_stats(_summarize_stats(qk_eff)),
        }


def _sample_range(rng, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return float(min_val)
    return float(rng.uniform(min_val, max_val))


def _auto_tune_state(config: Dict[str, Any]) -> Dict[str, Any]:
    state = config.get("_auto_tune_state")
    if state is None:
        state = {
            "last_sigma": None,
            "step_index": 0,
            "tuned_this_step": False,
            "best_score": None,
            "best_config": None,
        }
        config["_auto_tune_state"] = state
    return state


def _maybe_auto_tune(
    config: Dict[str, Any],
    cfg: TaylorAttentionConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: Optional[torch.Tensor],
    attn_precision: Optional[torch.dtype],
    skip_reshape: bool,
    transformer_options: Optional[dict],
) -> None:
    if not cfg.auto_tune:
        return
    if cfg.auto_tune_steps <= 0 or cfg.auto_tune_candidates <= 0:
        return
    if cfg.qk_normalize:
        return

    state = _auto_tune_state(config)
    sigma = None
    if transformer_options is not None:
        sigmas = transformer_options.get("sigmas")
        if isinstance(sigmas, torch.Tensor) and sigmas.numel() > 0:
            sigma = float(sigmas.flatten()[0].item())

    if sigma is not None and sigma != state["last_sigma"]:
        state["last_sigma"] = sigma
        state["tuned_this_step"] = False
        if state["best_config"] is not None:
            state["step_index"] += 1

    if state["step_index"] >= cfg.auto_tune_steps:
        return
    if state["tuned_this_step"]:
        return

    rng = random.Random(cfg.auto_tune_seed + int(state["step_index"]))
    candidates = []

    # include current config as baseline
    candidates.append({
        "qk_norm_power": cfg.qk_norm_power,
        "qk_norm_clip": cfg.qk_norm_clip,
        "scale_mul": cfg.scale_mul,
    })

    for _ in range(cfg.auto_tune_candidates):
        candidates.append({
            "qk_norm_power": _sample_range(rng, cfg.auto_tune_qk_norm_power_min, cfg.auto_tune_qk_norm_power_max),
            "qk_norm_clip": _sample_range(rng, cfg.auto_tune_qk_norm_clip_min, cfg.auto_tune_qk_norm_clip_max),
            "scale_mul": _sample_range(rng, cfg.auto_tune_scale_mul_min, cfg.auto_tune_scale_mul_max),
        })

    best_score = state["best_score"]
    best_config = state["best_config"]

    q_eval, k_eval, v_eval, batch, heads_eval, n_q, n_k = _reshape_inputs(q, k, v, heads, skip_reshape)

    mask_eval = mask
    if mask_eval is not None and cfg.auto_tune_max_tokens > 0:
        mask_eval = _slice_mask(mask_eval, n_q, n_k, cfg.auto_tune_max_tokens)

    if cfg.auto_tune_max_tokens > 0:
        max_tokens = cfg.auto_tune_max_tokens
        q_eval = q_eval[:, :, :max_tokens, :]
        k_eval = k_eval[:, :, :max_tokens, :]
        v_eval = v_eval[:, :, :max_tokens, :]

    if mask_eval is not None:
        key_mask_bool = _normalize_key_mask(mask_eval, batch, heads_eval, q_eval.shape[2], k_eval.shape[2]).to(device=q_eval.device)
    else:
        key_mask_bool = None

    scale_base = q_eval.shape[-1] ** -0.5

    for candidate in candidates:
        cand_cfg = dict(config)
        cand_cfg.update(candidate)
        cand_cfg["auto_tune"] = False
        cand_cfg["quality_check_samples"] = cfg.auto_tune_quality_samples

        den_stats = _init_den_stats()
        try:
            out = taylor_attention(
                q_eval,
                k_eval,
                v_eval,
                heads_eval,
                mask=mask_eval,
                attn_precision=attn_precision,
                skip_reshape=True,
                skip_output_reshape=True,
                config=cand_cfg,
                transformer_options=None,
                skip_quality_stats=True,
                skip_step_log=True,
                den_stats_out=den_stats,
            )
        except TaylorAttentionFallback:
            continue
        except RuntimeError:
            continue

        quality_stats = _compute_quality_stats(_resolve_config(cand_cfg), q_eval, k_eval, v_eval, out, key_mask_bool, scale_base)
        if not quality_stats or quality_stats["samples"] == 0:
            continue

        mean_abs = quality_stats["sum_abs"] / quality_stats["samples"]
        denom_frac = (den_stats["le_eps"] / den_stats["count"]) if den_stats["count"] > 0 else 1.0
        score = mean_abs + (denom_frac * 10.0)

        if best_score is None or score < best_score:
            best_score = score
            best_config = candidate

    if best_config is not None:
        config.update(best_config)
        state["best_score"] = best_score
        state["best_config"] = best_config
        if transformer_options is not None:
            step_stats = transformer_options.get("taylor_step_stats")
            if step_stats is not None:
                step_stats["config"] = _config_summary(_resolve_config(config))
        logger.info(
            "Taylor auto-tune selected: power=%.3g clip=%.3g scale=%.3g score=%.6g",
            best_config["qk_norm_power"],
            best_config["qk_norm_clip"],
            best_config["scale_mul"],
            best_score,
        )

    state["tuned_this_step"] = True


def _get_step_stats(transformer_options: Optional[dict], cfg: TaylorAttentionConfig) -> Optional[Dict[str, Any]]:
    if transformer_options is None:
        return None
    stats = transformer_options.get("taylor_step_stats")
    if stats is None:
        stats = {
            "calls": 0,
            "taylor_calls": 0,
            "fallback_calls": 0,
            "denom_fallbacks": 0,
            "fallback_reasons": {},
            "den_stats": _init_den_stats(),
            "quality": _init_quality_stats(),
            "config": _config_summary(cfg),
        }
        transformer_options["taylor_step_stats"] = stats
    return stats


def _should_log_step(transformer_options: Optional[dict]) -> bool:
    if transformer_options is None:
        return True
    if transformer_options.get("taylor_step_logged"):
        return False
    block_type = transformer_options.get("block_type")
    total_blocks = transformer_options.get("total_blocks")
    block_index = transformer_options.get("block_index")
    if block_type == "single" and isinstance(total_blocks, int) and isinstance(block_index, int):
        return block_index == total_blocks - 1
    return False


def _maybe_add_diagnostics(
    step_stats: Optional[Dict[str, Any]],
    transformer_options: Optional[dict],
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    q_eff: torch.Tensor,
    k_eff: torch.Tensor,
    scale_raw: float,
    scale_eff: float,
) -> None:
    if step_stats is None:
        return
    if "diag" in step_stats:
        return
    if not _should_log_step(transformer_options):
        return
    step_stats["diag"] = _compute_diagnostics(q_raw, k_raw, q_eff, k_eff, scale_raw, scale_eff)


def _maybe_log_step_stats(transformer_options: Optional[dict], cfg: TaylorAttentionConfig, step_stats: Optional[Dict[str, Any]] = None) -> None:
    if step_stats is None:
        step_stats = _get_step_stats(transformer_options, cfg)
    if step_stats is None:
        return
    if not _should_log_step(transformer_options):
        return

    den_stats = step_stats["den_stats"]
    if den_stats["count"] > 0:
        den_mean = den_stats["sum"] / den_stats["count"]
        den_frac = den_stats["le_eps"] / den_stats["count"]
        den_min = den_stats["min"]
        den_max = den_stats["max"]
    else:
        den_mean = float("nan")
        den_frac = float("nan")
        den_min = float("nan")
        den_max = float("nan")

    quality = step_stats["quality"]
    if quality["samples"] > 0:
        q_mean_abs = quality["sum_abs"] / quality["samples"]
        q_mean_rel = quality["sum_rel"] / quality["samples"]
        q_max_abs = quality["max_abs"]
        q_max_rel = quality["max_rel"]
    else:
        q_mean_abs = float("nan")
        q_mean_rel = float("nan")
        q_max_abs = float("nan")
        q_max_rel = float("nan")

    calls = step_stats["calls"]
    denom_frac = (step_stats["denom_fallbacks"] / calls) if calls > 0 else 0.0

    sigma = None
    if transformer_options is not None:
        sigmas = transformer_options.get("sigmas")
        if isinstance(sigmas, torch.Tensor) and sigmas.numel() > 0:
            sigma = float(sigmas.flatten()[0].item())
        elif sigmas is not None:
            try:
                sigma = float(sigmas)
            except Exception:
                sigma = None

    cfg_summary = step_stats["config"]

    diag = step_stats.get("diag")
    diag_str = ""
    if diag:
        diag_str = (
            " diag[q_norm_raw={q_norm_raw} k_norm_raw={k_norm_raw} qk_raw={qk_raw} "
            "q_norm_eff={q_norm_eff} k_norm_eff={k_norm_eff} qk_eff={qk_eff}]"
        ).format(**diag)

    logger.info(
        "Taylor step stats: sigma=%s calls=%s taylor=%s fallback=%s denom_fallback_frac=%.6g "
        "den[min=%.6g max=%.6g mean=%.6g frac_le_eps=%.6g] "
        "quality[mean_abs=%.6g max_abs=%.6g mean_rel=%.6g max_rel=%.6g samples=%s] "
        "config[qk_normalize=%s qk_norm_power=%.3g qk_norm_clip=%.3g scale_mul=%.3g sub_head_blocks=%s force_fp32=%s denom_fp32=%s probe_samples=%s denom_fallback_frac_limit=%.3g]%s",
        sigma,
        calls,
        step_stats["taylor_calls"],
        step_stats["fallback_calls"],
        denom_frac,
        den_min,
        den_max,
        den_mean,
        den_frac,
        q_mean_abs,
        q_max_abs,
        q_mean_rel,
        q_max_rel,
        quality["samples"],
        cfg_summary["qk_normalize"],
        cfg_summary["qk_norm_power"],
        cfg_summary["qk_norm_clip"],
        cfg_summary["scale_mul"],
        cfg_summary["sub_head_blocks"],
        cfg_summary["force_fp32"],
        cfg_summary["denom_fp32"],
        cfg_summary["probe_samples"],
        cfg_summary["denom_fallback_frac_limit"],
        diag_str,
    )

    if transformer_options is not None:
        transformer_options["taylor_step_logged"] = True
        transformer_options.pop("taylor_step_stats", None)

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


def _compute_quality_stats(
    cfg: TaylorAttentionConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    key_mask_bool: Optional[torch.Tensor],
    scale: float,
) -> Optional[Dict[str, float]]:
    batch, heads, n_q, dim_head = q.shape
    samples = min(cfg.quality_check_samples, n_q)
    if samples <= 0:
        return None

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
    rel = diff / (exact.abs() + 1e-6)

    return {
        "sum_abs": float(diff.sum().item()),
        "sum_rel": float(rel.sum().item()),
        "max_abs": float(diff.max().item()),
        "max_rel": float(rel.max().item()),
        "samples": int(diff.numel()),
    }





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


def _slice_mask(mask: torch.Tensor, n_q: int, n_k: int, max_tokens: int) -> torch.Tensor:
    if max_tokens <= 0:
        return mask
    target_q = min(n_q, max_tokens)
    target_k = min(n_k, max_tokens)

    if mask.ndim == 2:
        return mask[:, :target_k]
    if mask.ndim == 3:
        m = mask
        if m.shape[1] == n_q and n_q > target_q:
            m = m[:, :target_q, :]
        if m.shape[2] == n_k and n_k > target_k:
            m = m[:, :, :target_k]
        return m
    if mask.ndim == 4:
        m = mask
        if m.shape[2] == n_q and n_q > target_q:
            m = m[:, :, :target_q, :]
        if m.shape[3] == n_k and n_k > target_k:
            m = m[:, :, :, :target_k]
        return m
    return mask


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
    skip_quality_stats: bool = False,
    skip_step_log: bool = False,
    den_stats_out: Optional[Dict[str, float]] = None,
    **kwargs,
) -> torch.Tensor:
    cfg = _resolve_config(config)
    if not cfg.enabled:
        raise TaylorAttentionFallback("disabled")

    step_stats = _get_step_stats(transformer_options, cfg)

    q, k, v, batch, heads, n_q, n_k = _reshape_inputs(q, k, v, heads, skip_reshape)

    if not cfg.allow_cross_attention and n_q != n_k:
        raise TaylorAttentionFallback("cross_attention_disabled")

    tokens = max(n_q, n_k)
    if tokens < cfg.min_tokens:
        raise TaylorAttentionFallback("below_min_tokens")

    dim_head = q.shape[-1]
    q_orig = q
    k_orig = k
    scale_base = dim_head ** -0.5
    scale = scale_base * cfg.scale_mul
    _log_shapes_once(cfg, transformer_options, q, k, v, mask, scale, skip_reshape)
    if mask is not None:
        key_mask_bool = _normalize_key_mask(mask, batch, heads, n_q, n_k).to(device=q.device)
    else:
        key_mask_bool = None
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
                    skip_quality_stats=True,
                    skip_step_log=True,
                    **kwargs,
                )
            )
        out = torch.cat(sub_outputs, dim=-1)
        if step_stats is not None:
            quality_stats = _compute_quality_stats(cfg, q_orig, k_orig, v, out, key_mask_bool, scale_base)
            _merge_quality_stats(step_stats["quality"], quality_stats)
            _maybe_log_step_stats(transformer_options, cfg, step_stats)
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

    dtype_accum = torch.float32 if cfg.force_fp32 else q.dtype
    v_dtype = v.dtype
    if cfg.qk_normalize or cfg.qk_norm_clip > 0 or cfg.qk_norm_power > 0:
        q_f = q.to(dtype=dtype_accum)
        k_f = k.to(dtype=dtype_accum)
        q_norm = torch.norm(q_f, dim=-1, keepdim=True) + cfg.eps
        k_norm = torch.norm(k_f, dim=-1, keepdim=True) + cfg.eps
        if cfg.qk_normalize:
            q = q_f / q_norm
            k = k_f / k_norm
        else:
            q = q_f
            k = k_f
            if cfg.qk_norm_power > 0:
                q = q / (q_norm ** cfg.qk_norm_power)
                k = k / (k_norm ** cfg.qk_norm_power)
            if cfg.qk_norm_clip > 0:
                q = q * torch.clamp(cfg.qk_norm_clip / q_norm, max=1.0)
                k = k * torch.clamp(cfg.qk_norm_clip / k_norm, max=1.0)
    block_k = max(1, cfg.block_size_k)
    block_q = max(1, cfg.block_size_q)

    if mask is not None:
        key_mask = key_mask_bool.to(dtype=dtype_accum)
        key_mask = key_mask[:, None, :]
    else:
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

        if torch.isnan(den).any() or torch.isinf(den).any():
            if step_stats is not None:
                _merge_den_stats(step_stats["den_stats"], den_stats)
            raise TaylorAttentionFallback("denominator_invalid")
        if cfg.fallback_on_negative:
            den_le = torch.sum(den <= cfg.eps).item()
            if cfg.denom_fallback_frac_limit > 0:
                den_frac = den_le / den.numel()
                if den_frac > cfg.denom_fallback_frac_limit:
                    if step_stats is not None:
                        _merge_den_stats(step_stats["den_stats"], den_stats)
                    raise TaylorAttentionFallback("denominator_too_small")
            else:
                if den_le > 0:
                    if step_stats is not None:
                        _merge_den_stats(step_stats["den_stats"], den_stats)
                    raise TaylorAttentionFallback("denominator_too_small")
        den = torch.clamp(den, min=cfg.eps)
        if den.dtype != dtype_accum:
            den = den.to(dtype_accum)
        out[:, :, start:end, :] = num / den[..., None]

    quality_stats = None if skip_quality_stats else _compute_quality_stats(cfg, q_orig, k_orig, v, out, key_mask_bool, scale_base)
    if den_stats_out is not None:
        _merge_den_stats(den_stats_out, den_stats)
    if step_stats is not None:
        step_stats["taylor_calls"] += 1
        _merge_den_stats(step_stats["den_stats"], den_stats)
        _merge_quality_stats(step_stats["quality"], quality_stats)
        _maybe_add_diagnostics(step_stats, transformer_options, q_orig, k_orig, q, k, scale_base, scale)
        if not skip_step_log:
            _maybe_log_step_stats(transformer_options, cfg, step_stats)

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

    step_stats = _get_step_stats(transformer_options, cfg)
    if step_stats is not None:
        step_stats["calls"] += 1

    try:
        if config_dict is not None:
            _maybe_auto_tune(config_dict, cfg, args[0], args[1], args[2], args[3], kwargs.get("mask", None), kwargs.get("attn_precision", None), kwargs.get("skip_reshape", False), transformer_options)
        out = taylor_attention(*args, config=config_dict, **kwargs)
        _update_stats(used_taylor=True)
        _maybe_log_stats(cfg)
        return out
    except TaylorAttentionFallback as exc:
        _update_stats(reason=exc.reason)
        _maybe_log_stats(cfg)
        if step_stats is not None:
            step_stats["fallback_calls"] += 1
            reasons = step_stats["fallback_reasons"]
            reasons[exc.reason] = reasons.get(exc.reason, 0) + 1
            if exc.reason == "denominator_too_small":
                step_stats["denom_fallbacks"] += 1
        _maybe_log_step_stats(transformer_options, cfg, step_stats)
        if cfg.log_fallbacks:
            logger.warning("Taylor attention fallback: %s", exc.reason)
        return original_func(*args, **kwargs)
