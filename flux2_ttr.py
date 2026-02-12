from __future__ import annotations

import logging
import math
import os
import random
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

import flux2_comet_logging
from flux2_ttr_embeddings import ScalarSinusoidalEmbedding

logger = logging.getLogger(__name__)

_ORIGINAL_FLUX_ATTENTION: Dict[str, Any] = {}
_PATCH_DEPTH = 0
_RUNTIME_REGISTRY: Dict[str, "Flux2TTRRuntime"] = {}
_RUNTIME_CHECKPOINT_CACHE: Dict[tuple[str, int, int, int, bool], "Flux2TTRRuntime"] = {}
_RUNTIME_CHECKPOINT_CACHE_LOCK = threading.Lock()
_TTR_COMET_NAMESPACE = "flux2_ttr_runtime"
_TTR_COMET_EXPERIMENTS: Dict[str, Any] = flux2_comet_logging.experiments_for(_TTR_COMET_NAMESPACE)
_TTR_COMET_LOGGED_PARAM_KEYS: set[str] = flux2_comet_logging.logged_param_keys_for(_TTR_COMET_NAMESPACE)
_MEMORY_RESERVE_FACTOR = 1.1
_EMPIRICAL_TRAINING_FLOOR_BYTES = 3 * 1024 * 1024 * 1024
_DISTILL_METRIC_EPS = 1e-8
_EMA_DECAY = 0.9
_MAX_SAFE_INFERENCE_LOSS = 0.5

_DEFAULT_Q_CHUNK = 256
_DEFAULT_K_CHUNK = 1024
_DEFAULT_TRAIN_QUERY_CAP = 128
_DEFAULT_REPLAY_BUFFER = 8
_DEFAULT_TRAIN_STEPS_PER_CALL = 1
_DEFAULT_HUBER_BETA = 0.05
_DEFAULT_LOSS_LOG_VAR_HUBER = 0.0
_DEFAULT_LOSS_LOG_VAR_COSINE = -1.0
_DEFAULT_READY_THRESHOLD = 0.12
_DEFAULT_READY_MIN_UPDATES = 24
_READINESS_HYSTERESIS = 1.2
_DEFAULT_ALPHA_LR_MUL = 5.0
_DEFAULT_PHI_LR_MUL = 1.0
_DEFAULT_GRAD_CLIP = 1.0
_DEFAULT_LANDMARK_FRACTION = 0.08
_DEFAULT_LANDMARK_MIN = 64
_DEFAULT_LANDMARK_MAX = 0
_OOM_RECOVERY_LANDMARK_MAX = 512
_DEFAULT_TEXT_TOKENS_GUESS = 77
_DEFAULT_REPLAY_OFFLOAD_CPU = True
_DEFAULT_REPLAY_STORAGE_DTYPE = "float16"
_DEFAULT_REPLAY_MAX_BYTES = 768 * 1024 * 1024
_DEFAULT_CFG_SCALE = 1.0
_DEFAULT_MIN_SWAP_LAYERS = 1
_DEFAULT_MAX_SWAP_LAYERS = -1
_DEFAULT_COMET_LOG_EVERY = 50
_DEFAULT_CONTROLLER_POLICY = "threshold"
_DEFAULT_CONTROLLER_TEMPERATURE = 1.0
_RUN_EMA_PERIODIC_FLUSH_UPDATES = 20
_CONTROLLER_POLICY_THRESHOLD = "threshold"
_CONTROLLER_POLICY_STOCHASTIC = "stochastic"
_COMET_AGG_METRICS = (
    "loss",
    "mse",
    "nmse",
    "cosine_similarity",
    "ema_cosine_dist",
    "ema_loss",
    "alpha_sigmoid",
    "sigma",
    "cfg_scale",
    "layers_swapped",
    "layers_total",
)


def _git_short_hash(length: int = 7) -> str:
    return flux2_comet_logging.git_short_hash(__file__, length=int(length))


def _generate_experiment_key() -> str:
    """Build a unique Comet experiment key from git HEAD + current time."""
    return flux2_comet_logging.generate_experiment_key(__file__)


def _generate_experiment_display_name() -> str:
    """Build a readable Comet display name as YYYY-MM-DD-HHMMSS-<git6>."""
    return flux2_comet_logging.generate_experiment_display_name(__file__)


try:
    from comfy import model_management
except Exception:
    model_management = None

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    import flux2_ttr_controller
except Exception:
    flux2_ttr_controller = None


@dataclass
class FluxLayerSpec:
    layer_key: str
    num_heads: int
    head_dim: int


@dataclass
class ReplaySample:
    q_sub: torch.Tensor
    k_full: torch.Tensor
    v_full: torch.Tensor
    teacher_sub: torch.Tensor
    key_mask: Optional[torch.Tensor]
    text_token_count: Optional[int]
    conditioning_token_count: Optional[int] = None
    sigma: Optional[float] = None
    cfg_scale: Optional[float] = None
    nbytes: int = 0
    created_step: int = 0


def validate_feature_dim(feature_dim: int) -> int:
    dim = int(feature_dim)
    if dim < 128:
        raise ValueError(f"Flux2TTR: feature_dim must be >= 128 (got {dim}).")
    if dim % 256 != 0:
        raise ValueError(f"Flux2TTR: feature_dim must be a multiple of 256 (got {dim}).")
    return dim


def _infer_comfyui_root_from_anchor(anchor_file: str) -> Path:
    anchor = Path(str(anchor_file)).resolve()
    for parent in (anchor.parent, *anchor.parents):
        if parent.name == "custom_nodes":
            return parent.parent
    return anchor.parent


def approximate_attention_models_dir(*, anchor_file: str = __file__, ensure_exists: bool = True) -> str:
    models_dir_value = getattr(folder_paths, "models_dir", None) if folder_paths is not None else None
    if models_dir_value:
        base_models_dir = Path(str(models_dir_value)).resolve()
    else:
        comfy_root = _infer_comfyui_root_from_anchor(anchor_file)
        base_models_dir = comfy_root / "models"

    target_dir = base_models_dir / "approximate_attention"
    if ensure_exists:
        os.makedirs(target_dir, exist_ok=True)
    return str(target_dir)


def resolve_default_checkpoint_path(
    checkpoint_path: str,
    *,
    default_filename: str,
    anchor_file: str = __file__,
    ensure_dir: bool = True,
) -> str:
    path = str(checkpoint_path or "").strip()
    if path:
        return path
    base_dir = approximate_attention_models_dir(anchor_file=anchor_file, ensure_exists=ensure_dir)
    return str(Path(base_dir) / str(default_filename))


def _checkpoint_feature_dim_from_payload(payload: dict[str, Any], checkpoint_path: str) -> int:
    fmt = payload.get("format")
    if fmt != "flux2_ttr_v2":
        raise ValueError(f"Flux2TTR: unsupported checkpoint format in {checkpoint_path}: {fmt!r}")
    try:
        return validate_feature_dim(int(payload.get("feature_dim", 0)))
    except Exception as exc:
        raise ValueError(
            f"Flux2TTR: invalid checkpoint feature_dim in {checkpoint_path}: {payload.get('feature_dim')!r}."
        ) from exc


def load_checkpoint_feature_dim(checkpoint_path: str) -> int:
    path = str(checkpoint_path or "").strip()
    if not path:
        raise ValueError("Flux2TTR: checkpoint_path must be set when reading checkpoint metadata.")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Flux2TTR: unsupported checkpoint payload in {path}.")
    return _checkpoint_feature_dim_from_payload(payload, path)


def runtime_checkpoint_cache_key(
    checkpoint_path: str,
    *,
    feature_dim: int,
    training: bool,
) -> tuple[str, int, int, int, bool]:
    path = str(checkpoint_path or "").strip()
    if not path:
        raise ValueError("Flux2TTR: checkpoint_path must be set when deriving cache key.")
    resolved_path = os.path.realpath(path)
    stat_result = os.stat(resolved_path)
    mtime_ns = int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)))
    return (
        resolved_path,
        mtime_ns,
        int(stat_result.st_size),
        int(feature_dim),
        bool(training),
    )


def get_cached_runtime_for_checkpoint(
    checkpoint_path: str,
    *,
    feature_dim: int,
    training: bool,
) -> Optional["Flux2TTRRuntime"]:
    try:
        cache_key = runtime_checkpoint_cache_key(
            checkpoint_path,
            feature_dim=int(feature_dim),
            training=bool(training),
        )
    except Exception:
        return None
    with _RUNTIME_CHECKPOINT_CACHE_LOCK:
        return _RUNTIME_CHECKPOINT_CACHE.get(cache_key)


def cache_runtime_for_checkpoint(
    checkpoint_path: str,
    *,
    feature_dim: int,
    training: bool,
    runtime: "Flux2TTRRuntime",
) -> None:
    try:
        cache_key = runtime_checkpoint_cache_key(
            checkpoint_path,
            feature_dim=int(feature_dim),
            training=bool(training),
        )
    except Exception as exc:
        logger.debug(
            "Flux2TTR: runtime checkpoint cache key unavailable for %r (%s).",
            checkpoint_path,
            exc,
        )
        return

    cache_path, _, _, cache_feature_dim, cache_training = cache_key
    with _RUNTIME_CHECKPOINT_CACHE_LOCK:
        stale_keys = [
            key
            for key in _RUNTIME_CHECKPOINT_CACHE
            if key[0] == cache_path
            and key[3] == cache_feature_dim
            and key[4] == cache_training
            and key != cache_key
        ]
        for key in stale_keys:
            _RUNTIME_CHECKPOINT_CACHE.pop(key, None)
        _RUNTIME_CHECKPOINT_CACHE[cache_key] = runtime

    if stale_keys:
        logger.debug(
            "Flux2TTR: runtime checkpoint cache refreshed for %s (removed %d stale entries).",
            cache_path,
            len(stale_keys),
        )
    else:
        logger.debug("Flux2TTR: cached runtime for checkpoint %s.", cache_path)


def clear_runtime_checkpoint_cache() -> None:
    with _RUNTIME_CHECKPOINT_CACHE_LOCK:
        _RUNTIME_CHECKPOINT_CACHE.clear()


def _normalize_landmark_min(landmark_min: int) -> int:
    return max(1, int(landmark_min))


def _normalize_landmark_max(landmark_max: int) -> int:
    # 0 means "unlimited landmark budget".
    return max(0, int(landmark_max))


def _bounded_landmark_count(raw_count: int, landmark_min: int, landmark_max: int) -> int:
    min_count = _normalize_landmark_min(landmark_min)
    max_count = _normalize_landmark_max(landmark_max)
    count = max(min_count, int(raw_count))
    if max_count <= 0:
        return count
    return min(max_count, count)


def _supports_key_padding_mask(mask: Optional[torch.Tensor], batch: int, n_query: int, n_key: int) -> bool:
    if mask is None:
        return True
    if mask.ndim == 2:
        return mask.shape == (batch, n_key)
    if mask.ndim == 3:
        # Support only [B,1,Nk] as key-padding broadcast.
        return mask.shape == (batch, 1, n_key)
    if mask.ndim == 4:
        # Support only [B,1,1,Nk] as key-padding broadcast.
        return mask.shape == (batch, 1, 1, n_key)
    return False


def _key_mask_from_mask(mask: Optional[torch.Tensor], batch: int, keys: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    mask_bool = mask if mask.dtype == torch.bool else mask != 0
    if mask_bool.ndim == 2 and mask_bool.shape == (batch, keys):
        return mask_bool
    if mask_bool.ndim == 3 and mask_bool.shape == (batch, 1, keys):
        return mask_bool[:, 0, :]
    if mask_bool.ndim == 4 and mask_bool.shape == (batch, 1, 1, keys):
        return mask_bool[:, 0, 0, :]
    return None


def _safe_key_mask(key_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if key_mask is None:
        return None
    if bool(key_mask.any(dim=-1).all()):
        return key_mask
    key_mask = key_mask.clone()
    empty_rows = ~key_mask.any(dim=-1)
    if empty_rows.any():
        key_mask[empty_rows, 0] = True
    return key_mask


def _softmax_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
    key_mask = _safe_key_mask(_key_mask_from_mask(mask, q.shape[0], k.shape[2]))
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask[:, None, None, :].to(device=scores.device), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("b h i j, b h j d -> b h i d", attn, v)


def _flatten_heads(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], x.shape[1] * x.shape[3])


def _unflatten_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    return x.view(x.shape[0], x.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)


def _estimate_flux2_ttr_memory_bytes(
    batch: int,
    heads: int,
    n_query: int,
    n_key: int,
    head_dim: int,
    feature_dim: int,
    q_chunk_size: int,
    k_chunk_size: int,
    dtype_size: int,
    training: bool,
    landmark_max: int = _DEFAULT_LANDMARK_MAX,
    text_count_estimate: int = _DEFAULT_TEXT_TOKENS_GUESS,
) -> int:
    bh = batch * heads
    nq = max(1, int(n_query))
    nk = max(1, int(n_key))
    q_chunk = min(nq, max(1, int(q_chunk_size)))
    k_chunk = min(nk, max(1, int(k_chunk_size)))

    # Kernel branch: KV [BH,F,D], Ksum [BH,F], plus q/k chunk temporaries.
    kv_elems = bh * feature_dim * head_dim
    ksum_elems = bh * feature_dim
    chunk_elems = bh * (k_chunk * feature_dim + k_chunk * head_dim + q_chunk * feature_dim + q_chunk * head_dim)

    # Landmark branch: q_chunk x landmarks score matrix and softmax output.
    # Conditioning tokens are always landmarks; image tokens still use the landmark budget.
    cond_landmarks = min(nk, max(0, int(text_count_estimate)))
    max_image_landmarks = max(0, nk - cond_landmarks)
    lm_max = _normalize_landmark_max(landmark_max)
    if lm_max <= 0:
        image_landmarks = max_image_landmarks
    else:
        image_landmarks = min(max_image_landmarks, lm_max)
    landmarks = min(nk, cond_landmarks + image_landmarks)
    landmark_elems = bh * (q_chunk * landmarks + landmarks * head_dim + q_chunk * head_dim)

    total = kv_elems + ksum_elems + chunk_elems + landmark_elems
    # Sigma/CFG conditioner params/activations are tiny but include a small fixed cushion.
    total += 16 * 1024
    if training:
        # Extra autograd + replay batch tensors.
        train_elems = bh * (nq * head_dim + nk * head_dim * 2)
        total += train_elems
        total = int(total * 1.6)
    return int(total * dtype_size)


def _maybe_reserve_memory(
    runtime: "Flux2TTRRuntime",
    q: torch.Tensor,
    k: torch.Tensor,
    transformer_options: Optional[dict],
    training: bool,
    dtype_accum: torch.dtype,
    layer_key: Optional[str] = None,
) -> None:
    if model_management is None:
        return
    if q.device.type == "cpu":
        return

    batch, heads, n_query, head_dim = q.shape
    n_key = int(k.shape[2])
    cond_count_estimate = runtime._infer_conditioning_token_count(transformer_options, n_key)
    if cond_count_estimate is None:
        cond_count_estimate = runtime._infer_text_token_count(transformer_options, n_key)
    if training:
        n_query = min(n_query, max(1, int(runtime.training_query_token_cap)))
    dtype_size = torch.tensor([], dtype=dtype_accum).element_size()

    mem_bytes = _estimate_flux2_ttr_memory_bytes(
        batch=batch,
        heads=heads,
        n_query=n_query,
        n_key=n_key,
        head_dim=head_dim,
        feature_dim=runtime.feature_dim,
        q_chunk_size=runtime.query_chunk_size,
        k_chunk_size=runtime.key_chunk_size,
        dtype_size=dtype_size,
        training=training,
        landmark_max=runtime.landmark_max,
        text_count_estimate=cond_count_estimate,
    )
    if training:
        scale = (runtime.feature_dim / 256.0) * (head_dim / 128.0) * max(1.0, (batch * heads) / 24.0)
        mem_bytes = max(mem_bytes, int(_EMPIRICAL_TRAINING_FLOOR_BYTES * scale))

    mem_bytes = int(mem_bytes * _MEMORY_RESERVE_FACTOR)
    if mem_bytes <= 0:
        return

    if transformer_options is not None:
        key = (
            "train" if training else "infer",
            layer_key,
            batch,
            heads,
            n_query,
            n_key,
            head_dim,
            runtime.feature_dim,
            runtime.query_chunk_size,
            runtime.key_chunk_size,
            runtime.landmark_max,
            cond_count_estimate,
            dtype_size,
            _MEMORY_RESERVE_FACTOR,
        )
        if transformer_options.get("flux2_ttr_memory_reserved") == key:
            return
        transformer_options["flux2_ttr_memory_reserved"] = key

    try:
        model_management.free_memory(mem_bytes, q.device)
        mode = "training" if training else "inference"
        reserve_signature = (
            int(mem_bytes),
            int(runtime.query_chunk_size),
            int(runtime.key_chunk_size),
        )
        reserve_log_state = getattr(runtime, "_memory_reserve_last_logged", None)
        if not isinstance(reserve_log_state, dict):
            reserve_log_state = {}
            runtime._memory_reserve_last_logged = reserve_log_state
        if reserve_log_state.get(mode) != reserve_signature:
            logger.info(
                "Flux2TTR reserved ~%.2f MB for %s (q_chunk=%d k_chunk=%d)",
                mem_bytes / (1024 * 1024),
                mode,
                runtime.query_chunk_size,
                runtime.key_chunk_size,
            )
            reserve_log_state[mode] = reserve_signature
    except Exception as exc:
        logger.warning("Flux2TTR reserve memory failed: %s", exc)


def infer_flux_single_layer_specs(model: Any) -> list[FluxLayerSpec]:
    root = getattr(model, "model", model)
    diffusion_model = getattr(root, "diffusion_model", None)
    if diffusion_model is None:
        diffusion_model = getattr(getattr(root, "model", None), "diffusion_model", None)
    single_blocks = getattr(diffusion_model, "single_blocks", None)
    if single_blocks is None:
        return []

    specs: list[FluxLayerSpec] = []
    for idx, block in enumerate(single_blocks):
        num_heads = int(getattr(block, "num_heads", 0))
        hidden_size = int(getattr(block, "hidden_size", getattr(block, "hidden_dim", 0)))
        if num_heads <= 0 or hidden_size <= 0 or hidden_size % num_heads != 0:
            logger.warning("Flux2TTR: skipping single block %d due to invalid head metadata.", idx)
            continue
        specs.append(
            FluxLayerSpec(
                layer_key=f"single:{idx}",
                num_heads=num_heads,
                head_dim=hidden_size // num_heads,
            )
        )
    return specs


class KernelRegressorAttention(nn.Module):
    def __init__(
        self,
        head_dim: int,
        feature_dim: int,
        *,
        eps: float = 1e-6,
        split_qk: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.head_dim = int(head_dim)
        self.feature_dim = validate_feature_dim(feature_dim)
        self.eps = float(eps)
        self.split_qk = bool(split_qk)
        self.qk_norm = bool(qk_norm)

        hidden = max(self.head_dim, 2 * self.feature_dim)
        def _build_phi_net() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.head_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.feature_dim),
            )

        self.phi_net_q = _build_phi_net()
        if self.split_qk:
            self.phi_net_k = _build_phi_net()
        else:
            self.phi_net_k = self.phi_net_q

        self.last_den_min = float("nan")

    def _phi(self, x: torch.Tensor, net: nn.Module) -> torch.Tensor:
        phi = F.elu(net(x)) + 1.0
        if self.qk_norm:
            phi = phi / (phi.norm(dim=-1, keepdim=True) + self.eps)
        return phi

    def _phi_q(self, q: torch.Tensor) -> torch.Tensor:
        return self._phi(q, self.phi_net_q)

    def _phi_k(self, k: torch.Tensor) -> torch.Tensor:
        return self._phi(k, self.phi_net_k)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        q_chunk: int = _DEFAULT_Q_CHUNK,
        k_chunk: int = _DEFAULT_K_CHUNK,
    ) -> torch.Tensor:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("KernelRegressorAttention expects q/k/v with shape [B,H,N,D].")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0] or q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
            raise ValueError("KernelRegressorAttention expects matching batch/head dimensions.")
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError(f"KernelRegressorAttention expects q/k head_dim={self.head_dim}.")

        batch, heads, n_query, _ = q.shape
        n_key = int(k.shape[2])
        value_dim = int(v.shape[-1])
        if v.shape[2] != n_key:
            raise ValueError("KernelRegressorAttention expects k/v sequence lengths to match.")

        if attn_mask is not None:
            raise NotImplementedError("KernelRegressorAttention only supports key-padding masks for now.")

        if key_mask is not None:
            if key_mask.dtype != torch.bool:
                key_mask = key_mask != 0
            if key_mask.ndim != 2 or key_mask.shape != (batch, n_key):
                raise ValueError(f"KernelRegressorAttention expected key_mask [B,Nk], got {tuple(key_mask.shape)}.")

        q_chunk = max(1, int(q_chunk))
        k_chunk = max(1, int(k_chunk))
        out_dtype = v.dtype
        bh = batch * heads

        kv = q.new_zeros((batch, heads, self.feature_dim, value_dim), dtype=torch.float32)
        ksum = q.new_zeros((batch, heads, self.feature_dim), dtype=torch.float32)

        for k0 in range(0, n_key, k_chunk):
            k1 = min(k0 + k_chunk, n_key)
            k_phi = self._phi_k(k[:, :, k0:k1, :])
            v_chunk = v[:, :, k0:k1, :]

            if key_mask is not None:
                m = key_mask[:, None, k0:k1, None].to(device=k_phi.device, dtype=k_phi.dtype)
                k_phi = k_phi * m
                v_chunk = v_chunk * m.to(dtype=v_chunk.dtype)

            k_phi_f = k_phi.float().reshape(bh, k1 - k0, self.feature_dim)
            v_f = v_chunk.float().reshape(bh, k1 - k0, value_dim)
            kv += (k_phi_f.transpose(1, 2) @ v_f).reshape(batch, heads, self.feature_dim, value_dim)
            ksum += k_phi_f.sum(dim=1).reshape(batch, heads, self.feature_dim)

        out_chunks = []
        den_min = float("inf")
        kv_bh = kv.reshape(bh, self.feature_dim, value_dim)
        ksum_bh = ksum.reshape(bh, self.feature_dim, 1)

        for q0 in range(0, n_query, q_chunk):
            q1 = min(q0 + q_chunk, n_query)
            q_phi = self._phi_q(q[:, :, q0:q1, :]).float().reshape(bh, q1 - q0, self.feature_dim)
            num = (q_phi @ kv_bh).reshape(batch, heads, q1 - q0, value_dim)
            den = (q_phi @ ksum_bh).reshape(batch, heads, q1 - q0, 1)
            den = den.clamp_min(self.eps)
            den_min = min(den_min, float(den.min().item()))
            out_chunks.append((num / den).to(dtype=out_dtype))

        if not out_chunks:
            return v.new_zeros((batch, heads, 0, value_dim))

        self.last_den_min = den_min
        return torch.cat(out_chunks, dim=2)


class SigmaCFGConditioner(nn.Module):
    def __init__(self, embed_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.sigma_embed = ScalarSinusoidalEmbedding(embed_dim)
        self.cfg_embed = ScalarSinusoidalEmbedding(embed_dim)
        mlp_in = self.sigma_embed.embed_dim + self.cfg_embed.embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        # Start as identity modulation so sigma/cfg wiring is backward-compatible.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        sigma: float,
        cfg_scale: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        param = next(self.mlp.parameters())
        work_dtype = param.dtype if torch.is_floating_point(param) else torch.float32
        sigma_t = torch.tensor([float(sigma)], device=device, dtype=torch.float32)
        cfg_t = torch.tensor([float(cfg_scale)], device=device, dtype=torch.float32)
        emb = torch.cat([self.sigma_embed(sigma_t), self.cfg_embed(cfg_t)], dim=-1).to(dtype=work_dtype)
        mod = self.mlp(emb).reshape(-1).to(device=device, dtype=dtype)
        return mod[0], mod[1], mod[2], mod[3]


class Flux2HKRAttnLayer(nn.Module):
    def __init__(
        self,
        head_dim: int,
        feature_dim: int = 256,
        *,
        eps: float = 1e-6,
        query_chunk_size: int = _DEFAULT_Q_CHUNK,
        key_chunk_size: int = _DEFAULT_K_CHUNK,
        split_qk: bool = True,
        qk_norm: bool = True,
        landmark_fraction: float = _DEFAULT_LANDMARK_FRACTION,
        landmark_min: int = _DEFAULT_LANDMARK_MIN,
        landmark_max: int = _DEFAULT_LANDMARK_MAX,
        text_tokens_guess: int = _DEFAULT_TEXT_TOKENS_GUESS,
        landmark_qk_norm: bool = False,
        alpha_init: float = 0.1,
        alpha_max: float = 0.2,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.head_dim = int(head_dim)
        self.feature_dim = validate_feature_dim(feature_dim)
        self.eps = float(eps)
        self.split_qk = bool(split_qk)
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.key_chunk_size = max(1, int(key_chunk_size))
        self.landmark_fraction = max(0.0, min(1.0, float(landmark_fraction)))
        self.landmark_min = _normalize_landmark_min(landmark_min)
        self.landmark_max = _normalize_landmark_max(landmark_max)
        self.text_tokens_guess = max(0, int(text_tokens_guess))
        self.landmark_qk_norm = bool(landmark_qk_norm)
        self.adaptive_alpha = True
        self.alpha_max = max(0.0, min(1.0, float(alpha_max)))
        self.gamma = float(gamma)

        self.kernel = KernelRegressorAttention(
            head_dim=self.head_dim,
            feature_dim=self.feature_dim,
            eps=self.eps,
            split_qk=self.split_qk,
            qk_norm=qk_norm,
        )
        # Store alpha in logit-space: sigmoid(self.alpha) gives base blending weight.
        # logit(0.1) ≈ -2.197 for backward-compatible default.
        _alpha_raw = float(alpha_init)
        if _alpha_raw <= 0.0:
            _alpha_logit = -5.0
        elif _alpha_raw >= 1.0:
            _alpha_logit = 5.0
        else:
            _alpha_logit = math.log(_alpha_raw / (1.0 - _alpha_raw))
        self.alpha = nn.Parameter(torch.tensor(_alpha_logit, dtype=torch.float32))
        self.conditioner = SigmaCFGConditioner(embed_dim=32, hidden_dim=128)

        self.last_landmark_count = 0
        self.last_den_min = float("nan")

    @staticmethod
    def _even_indices(start: int, end: int, count: int, device: torch.device) -> torch.Tensor:
        length = max(0, end - start)
        if length <= 0 or count <= 0:
            return torch.empty((0,), device=device, dtype=torch.long)
        if count >= length:
            return torch.arange(start, end, device=device, dtype=torch.long)
        return torch.linspace(start, end - 1, steps=count, device=device, dtype=torch.float32).round().to(dtype=torch.long)

    def _effective_landmark_count(self, n_image_tokens: int) -> int:
        """Compute landmark count dynamically from the number of image tokens."""
        raw = round(max(0, int(n_image_tokens)) * self.landmark_fraction)
        return _bounded_landmark_count(raw, self.landmark_min, self.landmark_max)

    def _resolve_conditioning_token_count(
        self,
        num_keys: int,
        text_token_count: Optional[int],
        conditioning_token_count: Optional[int],
    ) -> int:
        cond_count = conditioning_token_count
        if cond_count is None:
            cond_count = self.text_tokens_guess if text_token_count is None else max(0, int(text_token_count))
        return min(num_keys, max(0, int(cond_count)))

    def _select_landmarks(
        self,
        num_keys: int,
        device: torch.device,
        key_mask: Optional[torch.Tensor],
        text_token_count: Optional[int],
        conditioning_token_count: Optional[int] = None,
    ) -> torch.Tensor:
        if num_keys <= 0:
            return torch.empty((0,), device=device, dtype=torch.long)

        cond_count = self._resolve_conditioning_token_count(num_keys, text_token_count, conditioning_token_count)
        cond_idx = torch.arange(0, cond_count, device=device, dtype=torch.long)

        image_start = cond_count
        image_length = num_keys - image_start
        image_budget = min(self._effective_landmark_count(image_length), image_length)
        image_idx = self._even_indices(image_start, num_keys, image_budget, device)
        target = cond_count + image_budget

        if cond_idx.numel() == 0 and image_idx.numel() == 0:
            idx = torch.tensor([0], device=device, dtype=torch.long)
        else:
            idx = torch.unique(torch.cat([cond_idx, image_idx], dim=0), sorted=True)

        if key_mask is not None:
            valid = key_mask.any(dim=0)
            idx = idx[valid[idx]]
            if idx.numel() == 0:
                fallback = torch.where(valid)[0]
                if fallback.numel() == 0:
                    fallback = torch.tensor([0], device=device, dtype=torch.long)
                if fallback.numel() > target:
                    even = self._even_indices(0, fallback.numel(), target, device)
                    fallback = fallback[even]
                idx = fallback

        if idx.numel() > target:
            even = self._even_indices(0, idx.numel(), target, device)
            idx = idx[even]

        return idx

    def _landmark_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: Optional[torch.Tensor],
        text_token_count: Optional[int],
        conditioning_token_count: Optional[int] = None,
    ) -> torch.Tensor:
        batch, heads, _, _ = q.shape
        n_keys = int(k.shape[2])
        idx = self._select_landmarks(
            n_keys,
            q.device,
            key_mask,
            text_token_count,
            conditioning_token_count=conditioning_token_count,
        )
        self.last_landmark_count = int(idx.numel())
        if idx.numel() == 0:
            return v.new_zeros((batch, heads, q.shape[2], v.shape[-1]))

        k_l = torch.index_select(k, dim=2, index=idx)
        v_l = torch.index_select(v, dim=2, index=idx)

        q_f = q.float()
        k_f = k_l.float()
        if self.landmark_qk_norm:
            q_f = q_f / (q_f.norm(dim=-1, keepdim=True) + self.eps)
            k_f = k_f / (k_f.norm(dim=-1, keepdim=True) + self.eps)

        scores = torch.einsum("b h i d, b h j d -> b h i j", q_f, k_f) * (self.head_dim ** -0.5)
        if key_mask is not None:
            key_mask_l = _safe_key_mask(key_mask[:, idx])
            scores = scores.masked_fill(~key_mask_l[:, None, None, :].to(device=scores.device), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v_l.float())
        return out.to(dtype=v.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        q_chunk: Optional[int] = None,
        k_chunk: Optional[int] = None,
        text_token_count: Optional[int] = None,
        conditioning_token_count: Optional[int] = None,
        sigma: Optional[float] = None,
        cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("Flux2HKRAttnLayer expects q/k/v with shape [B,H,N,D].")
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError(f"Flux2HKRAttnLayer expected head_dim={self.head_dim} for q/k.")
        if v.shape[:2] != k.shape[:2] or v.shape[2] != k.shape[2]:
            raise ValueError("Flux2HKRAttnLayer expects k/v to share [B,H,Nk].")

        q_chunk = self.query_chunk_size if q_chunk is None else max(1, int(q_chunk))
        k_chunk = self.key_chunk_size if k_chunk is None else max(1, int(k_chunk))

        out_kernel = self.kernel(
            q=q,
            k=k,
            v=v,
            key_mask=key_mask,
            attn_mask=attn_mask,
            q_chunk=q_chunk,
            k_chunk=k_chunk,
        )
        out_land = self._landmark_attention(
            q,
            k,
            v,
            key_mask=key_mask,
            text_token_count=text_token_count,
            conditioning_token_count=conditioning_token_count,
        )

        sigma_value = None
        if sigma is not None:
            try:
                sigma_value = float(sigma)
            except Exception:
                sigma_value = None
        if sigma_value is not None and math.isfinite(sigma_value):
            cfg_value = _DEFAULT_CFG_SCALE if cfg_scale is None else float(cfg_scale)
            if not math.isfinite(cfg_value):
                cfg_value = _DEFAULT_CFG_SCALE
            scale_k, bias_k, scale_l, bias_l = self.conditioner(
                sigma=sigma_value,
                cfg_scale=cfg_value,
                device=out_kernel.device,
                dtype=out_kernel.dtype,
            )
            out_kernel = out_kernel * (1.0 + scale_k.view(1, 1, 1, 1)) + bias_k.view(1, 1, 1, 1)
            out_land = out_land * (1.0 + scale_l.view(1, 1, 1, 1)) + bias_l.view(1, 1, 1, 1)

        self.last_den_min = float(self.kernel.last_den_min)

        base_logit = self.alpha  # raw logit, not sigmoided yet

        if self.adaptive_alpha:
            with torch.no_grad():
                diff = out_kernel.float() - out_land.float()
                # Use cosine disagreement instead of L2 norm to avoid magnitude bias
                disagreement = 1.0 - F.cosine_similarity(out_kernel.float(), out_land.float(), dim=-1)  # [B, H, N]
                d_mean = disagreement.mean(dim=-1, keepdim=True).clamp(min=1e-6)
                d_norm = (disagreement / d_mean).clamp(max=3.0) / 3.0    # [B, H, N] in [0, 1]

            # Modulate the logit directly so alpha_per_token stays in (0, alpha_max) by construction
            alpha_per_token = self.alpha_max * torch.sigmoid(base_logit + self.gamma * (d_norm - 0.5))
            alpha_per_token = alpha_per_token.unsqueeze(-1).to(dtype=v.dtype)  # [B, H, N, 1]
            return out_kernel + alpha_per_token * (out_land - out_kernel)
        else:
            alpha = (self.alpha_max * torch.sigmoid(base_logit)).to(dtype=v.dtype)
            return out_kernel + alpha.view(1, 1, 1, 1) * (out_land - out_kernel)


# Backward-compat alias for downstream imports.
TTRFluxLayer = Flux2HKRAttnLayer


class Flux2TTRRuntime:
    def __init__(
        self,
        feature_dim: int,
        learning_rate: float,
        training: bool,
        steps: int,
        scan_chunk_size: int = _DEFAULT_Q_CHUNK,
        key_chunk_size: int = _DEFAULT_K_CHUNK,
        landmark_fraction: float = _DEFAULT_LANDMARK_FRACTION,
        landmark_min: int = _DEFAULT_LANDMARK_MIN,
        landmark_max: int = _DEFAULT_LANDMARK_MAX,
        text_tokens_guess: int = _DEFAULT_TEXT_TOKENS_GUESS,
        alpha_init: float = 0.1,
        alpha_max: float = 0.2,
        gamma: float = 2.0,
        alpha_lr_multiplier: float = _DEFAULT_ALPHA_LR_MUL,
        phi_lr_multiplier: float = _DEFAULT_PHI_LR_MUL,
        training_query_token_cap: int = _DEFAULT_TRAIN_QUERY_CAP,
        replay_buffer_size: int = _DEFAULT_REPLAY_BUFFER,
        train_steps_per_call: int = _DEFAULT_TRAIN_STEPS_PER_CALL,
        huber_beta: float = _DEFAULT_HUBER_BETA,
        grad_clip_norm: float = _DEFAULT_GRAD_CLIP,
        readiness_threshold: float = _DEFAULT_READY_THRESHOLD,
        readiness_min_updates: int = _DEFAULT_READY_MIN_UPDATES,
        replay_offload_cpu: bool = _DEFAULT_REPLAY_OFFLOAD_CPU,
        replay_storage_dtype: str = _DEFAULT_REPLAY_STORAGE_DTYPE,
        replay_max_bytes: int = _DEFAULT_REPLAY_MAX_BYTES,
        enable_memory_reserve: bool = False,
        layer_start: int = -1,
        layer_end: int = -1,
        inference_mixed_precision: bool = True,
        training_preview_ttr: bool = True,
        cfg_scale: float = _DEFAULT_CFG_SCALE,
        min_swap_layers: int = _DEFAULT_MIN_SWAP_LAYERS,
        max_swap_layers: int = _DEFAULT_MAX_SWAP_LAYERS,
        comet_enabled: bool = False,
        comet_api_key: str = "",
        comet_project_name: str = "ttr-distillation",
        comet_workspace: str = "comet-workspace",
        comet_experiment: str = "",
        comet_persist_experiment: bool = True,
        comet_log_every: int = _DEFAULT_COMET_LOG_EVERY,
    ):
        self.feature_dim = validate_feature_dim(feature_dim)
        self.learning_rate = float(learning_rate)

        self.training_mode = bool(training)
        self.training_enabled = bool(training)
        self.steps_remaining = max(0, int(steps))
        self.training_steps_total = max(0, int(steps))
        self.training_updates_done = 0
        self.training_log_every = 10

        self.query_chunk_size = max(1, int(scan_chunk_size))
        self.key_chunk_size = max(1, int(key_chunk_size))
        self.landmark_fraction = max(0.0, min(1.0, float(landmark_fraction)))
        self.landmark_min = _normalize_landmark_min(landmark_min)
        self.landmark_max = _normalize_landmark_max(landmark_max)
        self.text_tokens_guess = max(0, int(text_tokens_guess))
        self.alpha_init = float(alpha_init)
        self.alpha_max = max(0.0, min(1.0, float(alpha_max)))
        self.gamma = float(gamma)
        self.alpha_lr_multiplier = max(0.0, float(alpha_lr_multiplier))
        self.phi_lr_multiplier = max(0.0, float(phi_lr_multiplier))

        self.training_query_token_cap = max(1, int(training_query_token_cap))
        self.replay_buffer_size = max(1, int(replay_buffer_size))
        self.train_steps_per_call = max(1, int(train_steps_per_call))
        self.huber_beta = max(1e-6, float(huber_beta))
        self.grad_clip_norm = max(0.0, float(grad_clip_norm))
        self.readiness_threshold = float(readiness_threshold)
        self.readiness_min_updates = max(0, int(readiness_min_updates))
        self.replay_offload_cpu = bool(replay_offload_cpu)
        self.replay_storage_dtype = str(replay_storage_dtype or _DEFAULT_REPLAY_STORAGE_DTYPE).strip().lower()
        if self.replay_storage_dtype not in ("float32", "float16", "bfloat16"):
            logger.warning(
                "Flux2TTR: unsupported replay_storage_dtype=%r; using %r.",
                self.replay_storage_dtype,
                _DEFAULT_REPLAY_STORAGE_DTYPE,
            )
            self.replay_storage_dtype = _DEFAULT_REPLAY_STORAGE_DTYPE
        self.replay_max_bytes = max(0, int(replay_max_bytes))
        self.replay_total_bytes = 0
        self._replay_created_counter = 0
        self.enable_memory_reserve = bool(enable_memory_reserve)

        self.layer_start = int(layer_start)
        self.layer_end = int(layer_end)
        self.inference_mixed_precision = bool(inference_mixed_precision)
        self.training_preview_ttr = bool(training_preview_ttr)
        cfg_value = float(cfg_scale)
        self.cfg_scale = cfg_value if math.isfinite(cfg_value) else float(_DEFAULT_CFG_SCALE)
        self.min_swap_layers = max(0, int(min_swap_layers))
        self.max_swap_layers = int(max_swap_layers)

        self.comet_enabled = bool(comet_enabled)
        self.comet_api_key = str(comet_api_key or "")
        self.comet_project_name = str(comet_project_name or "ttr-distillation")
        self.comet_workspace = str(comet_workspace or "comet-workspace")
        self.comet_experiment = str(comet_experiment or "").strip()
        self.comet_persist_experiment = bool(comet_persist_experiment) and bool(self.comet_experiment)
        self.comet_display_name = _generate_experiment_display_name()
        # Auto-generate a unique experiment key if none was provided.
        if not self.comet_experiment:
            self.comet_experiment = _generate_experiment_key()
            self.comet_persist_experiment = True
        self.comet_log_every = max(1, int(comet_log_every))

        self.max_safe_inference_loss = float(_MAX_SAFE_INFERENCE_LOSS)
        self.last_loss = float("nan")
        with torch.inference_mode(False):
            self.log_var_huber = nn.Parameter(
                torch.full((1,), float(_DEFAULT_LOSS_LOG_VAR_HUBER), dtype=torch.float32)
            )
            self.log_var_cosine = nn.Parameter(
                torch.full((1,), float(_DEFAULT_LOSS_LOG_VAR_COSINE), dtype=torch.float32)
            )
        self.loss_weight_optimizer: Optional[torch.optim.Optimizer] = None
        self.pending_loss_weight_optimizer_state: Optional[Dict[str, Any]] = None

        self.layers: Dict[str, Flux2HKRAttnLayer] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.pending_state: Dict[str, Dict[str, torch.Tensor]] = {}
        self.pending_optimizer_states: Dict[str, Dict[str, Any]] = {}
        self.layer_specs: Dict[str, FluxLayerSpec] = {}
        self.replay_buffers: Dict[str, Deque[ReplaySample]] = {}
        self.layer_ema_loss: Dict[str, float] = {}
        self.layer_ema_cosine_dist: Dict[str, float] = {}
        self.layer_update_count: Dict[str, int] = {}
        self.layer_ready: Dict[str, bool] = {}
        self.layer_last_loss: Dict[str, float] = {}
        self.layer_readiness_threshold: Dict[str, float] = {}
        self.layer_readiness_min_updates: Dict[str, int] = {}
        self._run_ema_accum_loss: Dict[str, list[float]] = {}
        self._run_ema_accum_cosine_dist: Dict[str, list[float]] = {}
        self._run_last_sigma: Optional[float] = None

        self.capture_remaining = 0

        self._layer_metric_latest: Dict[str, Dict[str, float]] = {}
        self._layer_metric_running: Dict[str, Dict[str, float]] = {}
        self._layer_metric_count: Dict[str, int] = {}
        self._comet_experiment = None
        self._comet_disabled = False
        self._warned_high_loss = False
        self._current_step_swap_set: Optional[set[str]] = None
        self._current_step_id: Optional[Any] = None
        self._current_step_eligible_count = 0
        self._controller_cache_key: Optional[Any] = None
        self._controller_cache_mask: Optional[torch.Tensor] = None
        self._controller_debug_last_step_key: Optional[Any] = None
        self._memory_reserve_last_logged: Dict[str, tuple[int, int, int]] = {}

    @staticmethod
    def _layer_sort_key(layer_key: str) -> tuple[str, int]:
        if ":" not in layer_key:
            return (layer_key, -1)
        prefix, idx = layer_key.split(":", 1)
        try:
            return (prefix, int(idx))
        except Exception:
            return (prefix, -1)

    @staticmethod
    def _quartile_range(values: list[float]) -> tuple[float, float]:
        finite = [float(v) for v in values if math.isfinite(float(v))]
        if not finite:
            return (float("nan"), float("nan"))
        t = torch.tensor(finite, dtype=torch.float32)
        q = torch.quantile(t, torch.tensor([0.25, 0.75], dtype=torch.float32))
        return (float(q[0].item()), float(q[1].item()))

    @staticmethod
    def _quantile_summary(values: list[float]) -> dict[str, float]:
        finite = [float(v) for v in values if math.isfinite(float(v))]
        if not finite:
            nan = float("nan")
            return {
                "min": nan,
                "p25": nan,
                "p50": nan,
                "p75": nan,
                "max": nan,
            }
        t = torch.tensor(finite, dtype=torch.float32)
        q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32))
        return {
            "min": float(t.min().item()),
            "p25": float(q[0].item()),
            "p50": float(q[1].item()),
            "p75": float(q[2].item()),
            "max": float(t.max().item()),
        }

    @staticmethod
    def _cosine_distance_from_similarity(cosine_similarity: float) -> float:
        cosine_dist = 1.0 - float(cosine_similarity)
        if not math.isfinite(cosine_dist):
            return float("nan")
        return max(0.0, min(1.0, cosine_dist))

    def _pareto_frontier_score(self, tracked_layers: list[str]) -> tuple[float, int, float]:
        ready_layers = [lk for lk in tracked_layers if bool(self.layer_ready.get(lk, False))]
        ready_count = len(ready_layers)
        if ready_count <= 0:
            return 0.0, 0, float("nan")

        ema_dists: list[float] = []
        for layer_key in ready_layers:
            dist = self.layer_ema_cosine_dist.get(layer_key)
            if dist is None:
                cosine_similarity = self._layer_metric_latest.get(layer_key, {}).get("cosine_similarity")
                if cosine_similarity is not None and math.isfinite(float(cosine_similarity)):
                    dist = self._cosine_distance_from_similarity(float(cosine_similarity))
            if dist is None or not math.isfinite(float(dist)):
                dist = 1.0
            ema_dists.append(max(0.0, min(1.0, float(dist))))

        worst = max(ema_dists) if ema_dists else 1.0
        score = float(ready_count) * max(0.0, 1.0 - float(worst))
        return float(score), int(ready_count), float(worst)

    @staticmethod
    def _format_layer_list(layer_keys: list[str], limit: int = 12) -> str:
        if not layer_keys:
            return "-"
        if len(layer_keys) <= limit:
            return ",".join(layer_keys)
        head = max(1, limit // 2)
        tail = max(1, limit - head)
        return ",".join(layer_keys[:head] + ["..."] + layer_keys[-tail:])

    def _log_training_snapshot(self) -> None:
        latest = self._layer_metric_latest
        log_var_h = float(self.log_var_huber.detach().cpu().item())
        log_var_c = float(self.log_var_cosine.detach().cpu().item())
        if not latest:
            logger.info(
                (
                    "Flux2TTR distill snapshot: updates=%d/%d remaining=%d tracked_layers=0 "
                    "ready_layers=0 replay=%.1f/%.1fMB log_vars=(h=%.6g c=%.6g)"
                ),
                self.training_updates_done,
                max(self.training_steps_total, self.training_updates_done),
                self.steps_remaining,
                self.replay_total_bytes / (1024.0 * 1024.0),
                self.replay_max_bytes / (1024.0 * 1024.0),
                log_var_h,
                log_var_c,
            )
            return

        tracked_layers = sorted(latest.keys(), key=self._layer_sort_key)
        ready_layers = sorted(
            [layer_key for layer_key in tracked_layers if bool(self.layer_ready.get(layer_key, False))],
            key=self._layer_sort_key,
        )

        loss_q25, loss_q75 = self._quartile_range([latest[k].get("loss", float("nan")) for k in tracked_layers])
        ema_q25, ema_q75 = self._quartile_range([latest[k].get("ema_loss", float("nan")) for k in tracked_layers])
        cos_q25, cos_q75 = self._quartile_range([latest[k].get("cosine_similarity", float("nan")) for k in tracked_layers])
        nmse_q25, nmse_q75 = self._quartile_range([latest[k].get("nmse", float("nan")) for k in tracked_layers])

        logger.info(
            (
                "Flux2TTR distill snapshot: updates=%d/%d remaining=%d tracked_layers=%d "
                "ready_layers=%d ready=[%s] replay=%.1f/%.1fMB "
                "log_vars=(h=%.6g c=%.6g) "
                "q25-q75 loss=%.6g..%.6g ema=%.6g..%.6g cosine=%.6g..%.6g nmse=%.6g..%.6g"
            ),
            self.training_updates_done,
            max(self.training_steps_total, self.training_updates_done),
            self.steps_remaining,
            len(tracked_layers),
            len(ready_layers),
            self._format_layer_list(ready_layers),
            self.replay_total_bytes / (1024.0 * 1024.0),
            self.replay_max_bytes / (1024.0 * 1024.0),
            log_var_h,
            log_var_c,
            loss_q25,
            loss_q75,
            ema_q25,
            ema_q75,
            cos_q25,
            cos_q75,
            nmse_q25,
            nmse_q75,
        )

    def release_resources(self) -> None:
        if self._comet_experiment is not None:
            flux2_comet_logging.release_experiment(
                namespace=_TTR_COMET_NAMESPACE,
                logger=logger,
                component_name="Flux2TTR",
                experiment=self._comet_experiment,
                enabled=bool(self.comet_enabled),
                persist_experiment=bool(self.comet_persist_experiment),
                experiment_key=str(self.comet_experiment),
            )
            self._comet_experiment = None

        # Avoid explicit CPU transfers here; cleanup can be called near the end of
        # a run where outstanding CUDA work may still reference these modules.
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        self.layers.clear()

        for optimizer in self.optimizers.values():
            try:
                optimizer.state.clear()
            except Exception:
                pass
        self.optimizers.clear()
        if self.loss_weight_optimizer is not None:
            try:
                self.loss_weight_optimizer.state.clear()
            except Exception:
                pass
            self.loss_weight_optimizer = None
        self.pending_optimizer_states.clear()
        self.pending_loss_weight_optimizer_state = None
        for param in (self.log_var_huber, self.log_var_cosine):
            if param.device.type != "cpu":
                param.data = param.data.to(device="cpu", dtype=torch.float32)
            if param.grad is not None and param.grad.device.type != "cpu":
                param.grad.data = param.grad.data.to(device="cpu", dtype=torch.float32)

        self.replay_buffers.clear()
        self.replay_total_bytes = 0
        self._replay_created_counter = 0
        self._layer_metric_latest.clear()
        self._layer_metric_running.clear()
        self._layer_metric_count.clear()
        self._run_ema_accum_loss.clear()
        self._run_ema_accum_cosine_dist.clear()
        self._run_last_sigma = None
        self._current_step_swap_set = None
        self._current_step_id = None
        self._current_step_eligible_count = 0
        self._controller_cache_key = None
        self._controller_cache_mask = None
        self._controller_debug_last_step_key = None
        self._memory_reserve_last_logged.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def register_layer_specs(self, specs: Iterable[FluxLayerSpec]) -> None:
        for spec in specs:
            self.layer_specs[spec.layer_key] = spec

    def _layer_key_from_options(self, transformer_options: Optional[dict]) -> str:
        if transformer_options is None:
            return "single:0"
        block_type = transformer_options.get("block_type", "single")
        block_index = transformer_options.get("block_index", 0)
        if isinstance(block_index, int):
            return f"{block_type}:{block_index}"
        return f"{block_type}:0"

    def _is_single_block_selected(self, transformer_options: Optional[dict]) -> bool:
        if transformer_options is None:
            return True
        block_type = transformer_options.get("block_type", "single")
        if block_type != "single":
            return False
        block_index = transformer_options.get("block_index")
        if isinstance(block_index, int):
            if self.layer_start >= 0 and block_index < self.layer_start:
                return False
            if self.layer_end >= 0 and block_index > self.layer_end:
                return False
        return True

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            x = float(value)
            return x if math.isfinite(x) else None
        if torch.is_tensor(value):
            if value.numel() <= 0:
                return None
            return Flux2TTRRuntime._as_float(value.reshape(-1)[0].item())
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return Flux2TTRRuntime._as_float(value[0])
        return None

    @staticmethod
    def _clone_state_to_cpu(value: Any) -> Any:
        if torch.is_tensor(value):
            # clone() strips inference-tensor metadata so restored states are
            # safe for optimizer in-place updates outside inference mode.
            return value.detach().clone().cpu()
        if isinstance(value, dict):
            return {k: Flux2TTRRuntime._clone_state_to_cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [Flux2TTRRuntime._clone_state_to_cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(Flux2TTRRuntime._clone_state_to_cpu(v) for v in value)
        return value

    @staticmethod
    def _is_inference_tensor(value: Any) -> bool:
        if not torch.is_tensor(value):
            return False
        try:
            return bool(value.is_inference())
        except Exception:
            return False

    @staticmethod
    def _extract_layer_index(layer_key: str) -> Optional[int]:
        if not isinstance(layer_key, str) or ":" not in layer_key:
            return None
        prefix, suffix = layer_key.split(":", 1)
        if prefix != "single":
            return None
        try:
            return int(suffix)
        except Exception:
            return None

    def _layer_key_within_range(self, layer_key: str) -> bool:
        idx = self._extract_layer_index(layer_key)
        if idx is None:
            return False
        if self.layer_start >= 0 and idx < self.layer_start:
            return False
        if self.layer_end >= 0 and idx > self.layer_end:
            return False
        return True

    def _single_layer_keys_from_options(self, transformer_options: Optional[dict]) -> list[str]:
        if not isinstance(transformer_options, dict):
            return []
        keys: set[str] = set()
        inner = transformer_options.get("flux2_ttr") if isinstance(transformer_options.get("flux2_ttr"), dict) else {}

        for key_name in ("all_single_layer_keys", "single_layer_keys"):
            value = transformer_options.get(key_name)
            if isinstance(value, (list, tuple)):
                keys.update(str(v) for v in value if isinstance(v, str) and v.startswith("single:"))
            value = inner.get(key_name) if isinstance(inner, dict) else None
            if isinstance(value, (list, tuple)):
                keys.update(str(v) for v in value if isinstance(v, str) and v.startswith("single:"))

        for count_key in ("num_single_blocks", "single_block_count", "total_single_blocks"):
            count = transformer_options.get(count_key)
            if not isinstance(count, int):
                count = inner.get(count_key) if isinstance(inner, dict) else None
            if isinstance(count, int) and count > 0:
                keys.update(f"single:{idx}" for idx in range(count))
                break

        resolved = [key for key in keys if self._layer_key_within_range(key)]
        resolved.sort(key=self._layer_sort_key)
        return resolved

    def _eligible_single_layer_keys(
        self,
        current_layer_key: Optional[str] = None,
        transformer_options: Optional[dict] = None,
    ) -> list[str]:
        keys: set[str] = set()
        keys.update(key for key in self.layer_specs.keys() if self._layer_key_within_range(key))
        keys.update(key for key in self.layers.keys() if self._layer_key_within_range(key))
        keys.update(key for key in self.replay_buffers.keys() if self._layer_key_within_range(key))
        keys.update(self._single_layer_keys_from_options(transformer_options))
        if current_layer_key is not None and self._layer_key_within_range(current_layer_key):
            keys.add(current_layer_key)
        ordered = list(keys)
        ordered.sort(key=self._layer_sort_key)
        return ordered

    def _extract_sigma(self, transformer_options: Optional[dict]) -> Optional[float]:
        if not isinstance(transformer_options, dict):
            return None

        for key in ("sigma", "current_sigma"):
            sigma = self._as_float(transformer_options.get(key))
            if sigma is not None:
                return sigma

        sigmas = transformer_options.get("sigmas")
        sigma = self._as_float(sigmas)
        if sigma is not None:
            return sigma

        inner = transformer_options.get("flux2_ttr")
        if isinstance(inner, dict):
            for key in ("sigma", "current_sigma"):
                sigma = self._as_float(inner.get(key))
                if sigma is not None:
                    return sigma
            sigma = self._as_float(inner.get("sigmas"))
            if sigma is not None:
                return sigma
        return None

    def _extract_cfg_scale(self, transformer_options: Optional[dict]) -> float:
        if isinstance(transformer_options, dict):
            for key in ("cfg_scale", "cond_scale", "guidance_scale", "scale"):
                cfg_scale = self._as_float(transformer_options.get(key))
                if cfg_scale is not None:
                    return cfg_scale
            inner = transformer_options.get("flux2_ttr")
            if isinstance(inner, dict):
                for key in ("cfg_scale", "cond_scale", "guidance_scale"):
                    cfg_scale = self._as_float(inner.get(key))
                    if cfg_scale is not None:
                        return cfg_scale
        return float(self.cfg_scale)

    @staticmethod
    def _normalize_controller_policy(value: Any) -> str:
        policy = str(value or _DEFAULT_CONTROLLER_POLICY).strip().lower()
        if policy in {"sample", "sampling", "stochastic", "bernoulli", "gumbel"}:
            return _CONTROLLER_POLICY_STOCHASTIC
        return _CONTROLLER_POLICY_THRESHOLD

    @staticmethod
    def _sanitize_controller_threshold(value: Any, default: float = 0.5) -> float:
        try:
            threshold = float(value)
        except Exception:
            threshold = float(default)
        if not math.isfinite(threshold):
            threshold = float(default)
        return max(1e-4, min(1.0 - 1e-4, threshold))

    @staticmethod
    def _sanitize_controller_temperature(value: Any, default: float = _DEFAULT_CONTROLLER_TEMPERATURE) -> float:
        try:
            temperature = float(value)
        except Exception:
            temperature = float(default)
        if not math.isfinite(temperature):
            temperature = float(default)
        return max(1e-3, temperature)

    def _extract_step_id(self, transformer_options: Optional[dict]) -> Any:
        if isinstance(transformer_options, dict):
            for key in ("step", "current_step", "sample_step", "timestep", "sigma_idx", "denoise_step"):
                value = transformer_options.get(key)
                if isinstance(value, (int, str)):
                    return (key, value)
                if torch.is_tensor(value) and value.numel() > 0:
                    scalar = self._as_float(value)
                    if scalar is not None:
                        return (key, scalar)

        sigma = self._extract_sigma(transformer_options)
        if sigma is not None:
            return ("sigma", round(float(sigma), 10))
        return None

    def _extract_resolution(
        self,
        transformer_options: Optional[dict],
        n_key: Optional[int] = None,
        text_token_count: Optional[int] = None,
    ) -> tuple[int, int]:
        if isinstance(transformer_options, dict):
            candidates = (
                ("width", "height"),
                ("latent_width", "latent_height"),
                ("image_width", "image_height"),
                ("w", "h"),
            )
            for w_key, h_key in candidates:
                w = transformer_options.get(w_key)
                h = transformer_options.get(h_key)
                if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                    return int(w), int(h)
            inner = transformer_options.get("flux2_ttr")
            if isinstance(inner, dict):
                for w_key, h_key in candidates:
                    w = inner.get(w_key)
                    h = inner.get(h_key)
                    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                        return int(w), int(h)

        if isinstance(n_key, int) and n_key > 0:
            txt = max(0, int(text_token_count or 0))
            image_tokens = max(1, n_key - txt)
            side = int(round(math.sqrt(image_tokens)))
            if side > 0 and side * side == image_tokens:
                return side, side
            return image_tokens, 1
        return 1, 1

    def _resolve_swap_set_for_step(
        self,
        transformer_options: Optional[dict],
        all_eligible_layers: list[str],
    ) -> set[str]:
        step_id = self._extract_step_id(transformer_options)
        if step_id == self._current_step_id and self._current_step_swap_set is not None:
            return self._current_step_swap_set

        n_eligible = len(all_eligible_layers)
        if n_eligible <= 0:
            swap_set: set[str] = set()
        else:
            min_n = max(0, min(self.min_swap_layers, n_eligible))
            max_n = n_eligible if self.max_swap_layers < 0 else max(0, min(self.max_swap_layers, n_eligible))
            if max_n < min_n:
                max_n = min_n
            n_swap = 0 if max_n <= 0 else random.randint(min_n, max_n)
            swap_set = set(random.sample(all_eligible_layers, n_swap)) if n_swap > 0 else set()

        self._current_step_id = step_id
        self._current_step_swap_set = swap_set
        self._current_step_eligible_count = int(n_eligible)
        return swap_set

    def _get_controller_mask(
        self,
        controller: nn.Module,
        sigma: Optional[float],
        cfg_scale: float,
        width: int,
        height: int,
        *,
        controller_threshold: float,
        controller_policy: str,
        controller_temperature: float,
    ) -> torch.Tensor:
        sigma_key = None if sigma is None else round(float(sigma), 10)
        policy = self._normalize_controller_policy(controller_policy)
        threshold = self._sanitize_controller_threshold(controller_threshold, default=0.5)
        temperature = self._sanitize_controller_temperature(
            controller_temperature,
            default=_DEFAULT_CONTROLLER_TEMPERATURE,
        )
        step_id = (
            sigma_key,
            round(float(cfg_scale), 6),
            int(width),
            int(height),
            policy,
            round(float(threshold), 6),
            round(float(temperature), 6),
        )
        if step_id == self._controller_cache_key and self._controller_cache_mask is not None:
            return self._controller_cache_mask

        with torch.no_grad():
            logits = controller(sigma if sigma is not None else 0.0, cfg_scale, width, height)
            if logits.ndim > 1:
                logits = logits.reshape(-1)
            probs = torch.sigmoid(logits.float())
            if policy == _CONTROLLER_POLICY_STOCHASTIC:
                probs = probs.clamp(min=1e-6, max=1.0 - 1e-6)
                threshold_logit = math.log(threshold / (1.0 - threshold))
                adjusted = torch.sigmoid((torch.logit(probs) - threshold_logit) / temperature)
                draws = torch.rand_like(adjusted)
                mask = (draws < adjusted).to(dtype=torch.float32)
            else:
                mask = probs
        self._controller_cache_key = step_id
        self._controller_cache_mask = mask
        return mask

    def _controller_student_layer_set(
        self,
        controller_mask: torch.Tensor,
        controller_threshold: float,
        transformer_options: Optional[dict],
        current_layer_key: str,
    ) -> tuple[list[str], int]:
        eligible_layers = self._eligible_single_layer_keys(
            current_layer_key=current_layer_key,
            transformer_options=transformer_options,
        )
        student_layers: list[str] = []
        mask_len = int(controller_mask.numel())
        for layer_key in eligible_layers:
            idx = self._extract_layer_index(layer_key)
            if idx is None or idx < 0 or idx >= mask_len:
                continue
            if float(controller_mask[idx].item()) <= float(controller_threshold):
                student_layers.append(layer_key)
        student_layers.sort(key=self._layer_sort_key)
        return student_layers, len(eligible_layers)

    def _maybe_log_controller_step_routing(
        self,
        *,
        transformer_options: Optional[dict],
        controller_mask: torch.Tensor,
        controller_threshold: float,
        controller_policy: str,
        controller_temperature: float,
        decision_threshold: float,
        sigma: Optional[float],
        cfg_scale: float,
        width: int,
        height: int,
        current_layer_key: str,
    ) -> None:
        step_id = self._extract_step_id(transformer_options)
        sigma_value = float(sigma) if sigma is not None else float("nan")
        sigma_key = round(sigma_value, 10) if math.isfinite(sigma_value) else None
        key = (
            step_id,
            sigma_key,
            round(float(cfg_scale), 6),
            round(float(controller_threshold), 6),
            str(controller_policy),
            round(float(controller_temperature), 6),
            round(float(decision_threshold), 6),
            int(width),
            int(height),
        )
        if key == self._controller_debug_last_step_key:
            return
        self._controller_debug_last_step_key = key

        student_layers, total_layers = self._controller_student_layer_set(
            controller_mask=controller_mask,
            controller_threshold=decision_threshold,
            transformer_options=transformer_options,
            current_layer_key=current_layer_key,
        )
        sigma_text = f"{sigma_value:.6g}" if math.isfinite(sigma_value) else "nan"
        logger.info(
            (
                "Flux2TTR controller step routing: step_id=%r extracted_sigma=%s "
                "controller_policy=%s controller_threshold=%.4g decision_threshold=%.4g "
                "controller_temperature=%.4g cfg_scale=%.4g student_layers=%d/%d [%s]"
            ),
            step_id,
            sigma_text,
            str(controller_policy),
            float(controller_threshold),
            float(decision_threshold),
            float(controller_temperature),
            float(cfg_scale),
            len(student_layers),
            int(total_layers),
            self._format_layer_list(student_layers),
        )

    def _ensure_optimizer(self, layer: Flux2HKRAttnLayer) -> torch.optim.Optimizer:
        alpha_params = [layer.alpha]
        alpha_ids = {id(p) for p in alpha_params}
        non_alpha_params = [p for p in layer.parameters() if id(p) not in alpha_ids]

        groups = []
        if non_alpha_params:
            groups.append({"params": non_alpha_params, "lr": self.learning_rate * self.phi_lr_multiplier})
        if alpha_params:
            groups.append({"params": alpha_params, "lr": self.learning_rate * self.alpha_lr_multiplier})
        if not groups:
            raise RuntimeError("Flux2TTR: no parameters available for optimizer.")
        return torch.optim.AdamW(groups)

    def _ensure_loss_weight_optimizer(self, device: torch.device) -> torch.optim.Optimizer:
        rebuilt = False
        for attr_name in ("log_var_huber", "log_var_cosine"):
            param = getattr(self, attr_name)
            is_inference = False
            try:
                is_inference = bool(param.is_inference())
            except Exception:
                is_inference = False
            if is_inference:
                with torch.inference_mode(False):
                    new_param = nn.Parameter(param.detach().clone().to(device=device, dtype=torch.float32))
                if param.grad is not None:
                    new_param.grad = param.grad.detach().clone().to(device=device, dtype=torch.float32)
                setattr(self, attr_name, new_param)
                rebuilt = True
                continue
            if param.device != device or param.dtype != torch.float32:
                param.data = param.data.to(device=device, dtype=torch.float32)
            if param.grad is not None and (param.grad.device != device or param.grad.dtype != torch.float32):
                param.grad.data = param.grad.data.to(device=device, dtype=torch.float32)

        if rebuilt and self.loss_weight_optimizer is not None:
            try:
                self.loss_weight_optimizer.state.clear()
            except Exception:
                pass
            self.loss_weight_optimizer = None

        if self.loss_weight_optimizer is None:
            self.loss_weight_optimizer = torch.optim.AdamW(
                [self.log_var_huber, self.log_var_cosine],
                lr=self.learning_rate,
            )
        pending_state = self.pending_loss_weight_optimizer_state
        if isinstance(pending_state, dict):
            try:
                self.loss_weight_optimizer.load_state_dict(pending_state)
            except Exception as exc:
                logger.warning("Flux2TTR: failed to restore loss-weight optimizer state (%s).", exc)
            self.pending_loss_weight_optimizer_state = None

        for group in self.loss_weight_optimizer.param_groups:
            group["lr"] = float(self.learning_rate)
        for state in self.loss_weight_optimizer.state.values():
            for key, value in state.items():
                if not torch.is_tensor(value):
                    continue
                if self._is_inference_tensor(value):
                    value = value.detach().clone()
                if torch.is_floating_point(value):
                    state[key] = value.to(device=device, dtype=torch.float32)
                else:
                    state[key] = value.to(device=device)
        return self.loss_weight_optimizer

    def _sync_layer_landmark_config(self, layer: Flux2HKRAttnLayer) -> None:
        layer.landmark_fraction = max(0.0, min(1.0, float(self.landmark_fraction)))
        layer.landmark_min = _normalize_landmark_min(self.landmark_min)
        layer.landmark_max = _normalize_landmark_max(self.landmark_max)
        layer.text_tokens_guess = max(0, int(self.text_tokens_guess))

    def _ensure_layer(self, layer_key: str, head_dim: int, device: torch.device) -> Flux2HKRAttnLayer:
        layer = self.layers.get(layer_key)
        if layer is not None:
            has_inference_params = any(self._is_inference_tensor(p) for p in layer.parameters())
            if has_inference_params:
                logger.debug(
                    "Flux2TTR: detected inference tensors in layer %s; rebuilding trainable layer/optimizer.",
                    layer_key,
                )
                try:
                    self.pending_state[layer_key] = {
                        k: v.detach().clone().cpu()
                        for k, v in layer.state_dict().items()
                    }
                except Exception as exc:
                    logger.warning("Flux2TTR: failed to snapshot layer %s before rebuild (%s).", layer_key, exc)

                stale_optimizer = self.optimizers.get(layer_key)
                if stale_optimizer is not None:
                    try:
                        self.pending_optimizer_states[layer_key] = self._clone_state_to_cpu(stale_optimizer.state_dict())
                    except Exception as exc:
                        logger.warning("Flux2TTR: failed to snapshot optimizer state for %s (%s).", layer_key, exc)
                    try:
                        stale_optimizer.state.clear()
                    except Exception:
                        pass

                self.layers.pop(layer_key, None)
                self.optimizers.pop(layer_key, None)
                layer = None

        if layer is None:
            with torch.inference_mode(False):
                layer = Flux2HKRAttnLayer(
                    head_dim=head_dim,
                    feature_dim=self.feature_dim,
                    query_chunk_size=self.query_chunk_size,
                    key_chunk_size=self.key_chunk_size,
                    landmark_fraction=self.landmark_fraction,
                    landmark_min=self.landmark_min,
                    landmark_max=self.landmark_max,
                    text_tokens_guess=self.text_tokens_guess,
                    alpha_init=self.alpha_init,
                    alpha_max=self.alpha_max,
                    gamma=self.gamma,
                ).to(device=device, dtype=torch.float32)
            self._sync_layer_landmark_config(layer)
            optimizer = self._ensure_optimizer(layer)
            pending = self.pending_state.get(layer_key)
            if pending:
                missing, unexpected = layer.load_state_dict(pending, strict=False)
                if missing or unexpected:
                    logger.warning(
                        "Flux2TTR: checkpoint load mismatch for %s (missing=%s unexpected=%s).",
                        layer_key,
                        missing,
                        unexpected,
                    )
            pending_optimizer_state = self.pending_optimizer_states.pop(layer_key, None)
            if isinstance(pending_optimizer_state, dict):
                try:
                    optimizer.load_state_dict(pending_optimizer_state)
                except Exception as exc:
                    logger.warning("Flux2TTR: failed to restore optimizer state for %s (%s).", layer_key, exc)
            for state in optimizer.state.values():
                for key, value in state.items():
                    if not torch.is_tensor(value):
                        continue
                    if self._is_inference_tensor(value):
                        value = value.detach().clone()
                    if torch.is_floating_point(value):
                        state[key] = value.to(device=device, dtype=torch.float32)
                    else:
                        state[key] = value.to(device=device)
            self.layers[layer_key] = layer
            self.optimizers[layer_key] = optimizer
            self.replay_buffers.setdefault(layer_key, deque())
            logger.debug(
                (
                    "Flux2TTR: created HKR layer %s (head_dim=%d feature_dim=%d "
                    "landmarks=fraction=%.4g min=%d max=%d)."
                ),
                layer_key,
                head_dim,
                self.feature_dim,
                self.landmark_fraction,
                self.landmark_min,
                self.landmark_max,
            )
            return layer

        self._sync_layer_landmark_config(layer)
        layer_device = next(layer.parameters()).device
        if layer_device != device:
            layer.to(device=device)
            optimizer = self.optimizers[layer_key]
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        if self._is_inference_tensor(value):
                            value = value.detach().clone()
                        state[key] = value.to(device=device)
        return layer

    def _set_layer_dtype(self, layer_key: str, layer: Flux2HKRAttnLayer, target_dtype: torch.dtype) -> None:
        current_dtype = next(layer.parameters()).dtype
        if current_dtype == target_dtype:
            return
        layer.to(dtype=target_dtype)
        optimizer = self.optimizers.get(layer_key)
        if optimizer is None:
            return
        device = next(layer.parameters()).device
        for state in optimizer.state.values():
            for key, value in state.items():
                if not torch.is_tensor(value):
                    continue
                if self._is_inference_tensor(value):
                    value = value.detach().clone()
                if torch.is_floating_point(value):
                    state[key] = value.to(device=device, dtype=target_dtype)
                else:
                    state[key] = value.to(device=device)

    def _replay_dtype(self) -> torch.dtype:
        if self.replay_storage_dtype == "float32":
            return torch.float32
        if self.replay_storage_dtype == "bfloat16":
            return torch.bfloat16
        return torch.float16

    @staticmethod
    def _ensure_layer_device(layer: Flux2HKRAttnLayer, device: torch.device) -> None:
        needs_move = False
        for p in layer.parameters():
            if p.device != device:
                needs_move = True
                break
        if not needs_move:
            for b in layer.buffers():
                if b.device != device:
                    needs_move = True
                    break
        if needs_move:
            layer.to(device=device)

    def _resolve_inference_dtype(self, q: torch.Tensor) -> torch.dtype:
        if not self.inference_mixed_precision:
            return torch.float32
        if q.device.type == "cuda" and q.dtype in (torch.float16, torch.bfloat16):
            return q.dtype
        return torch.float32

    def _ensure_comet_experiment(self):
        if not self.comet_enabled or self._comet_disabled:
            return None
        if self._comet_experiment is not None:
            return self._comet_experiment

        raw_experiment_key = self.comet_experiment.strip()
        experiment_key = flux2_comet_logging.normalize_experiment_key(raw_experiment_key, allow_empty=True)
        if raw_experiment_key != experiment_key:
            logger.info(
                "Flux2TTR: normalized comet_experiment from %r to %r.",
                raw_experiment_key,
                experiment_key,
            )
            self.comet_experiment = experiment_key
        display_name = str(self.comet_display_name or "").strip()
        if not display_name:
            display_name = _generate_experiment_display_name()
            self.comet_display_name = display_name
        comet_params = {
            "learning_rate": float(self.learning_rate),
            "feature_dim": int(self.feature_dim),
            "query_chunk_size": int(self.query_chunk_size),
            "key_chunk_size": int(self.key_chunk_size),
            "landmark_fraction": float(self.landmark_fraction),
            "landmark_min": int(self.landmark_min),
            "landmark_max": int(self.landmark_max),
            "training_query_token_cap": int(self.training_query_token_cap),
            "replay_buffer_size": int(self.replay_buffer_size),
            "train_steps_per_call": int(self.train_steps_per_call),
            "huber_beta": float(self.huber_beta),
            "readiness_threshold": float(self.readiness_threshold),
            "readiness_min_updates": int(self.readiness_min_updates),
            "cfg_scale": float(self.cfg_scale),
            "min_swap_layers": int(self.min_swap_layers),
            "max_swap_layers": int(self.max_swap_layers),
            "comet_log_every": int(self.comet_log_every),
            "comet_experiment": experiment_key,
        }
        experiment, disabled = flux2_comet_logging.ensure_experiment(
            namespace=_TTR_COMET_NAMESPACE,
            logger=logger,
            component_name="Flux2TTR",
            enabled=bool(self.comet_enabled),
            disabled=bool(self._comet_disabled),
            existing_experiment=self._comet_experiment,
            api_key=str(self.comet_api_key),
            project_name=str(self.comet_project_name),
            workspace=str(self.comet_workspace),
            experiment_key=experiment_key,
            display_name=display_name,
            persist_experiment=bool(self.comet_persist_experiment),
            params=comet_params,
        )
        self._comet_disabled = bool(disabled)
        self._comet_experiment = experiment
        return experiment

    def flush_run_emas(self) -> None:
        """Flush accumulated per-run EMA samples into single EMA updates.

        Called at run boundaries (when sigma jumps up, indicating a new image).
        Each layer gets at most one EMA update per sampling run, preventing
        within-run sample correlation from destabilizing readiness flags.
        """
        refreshed_layers: list[str] = []

        for layer_key, samples in self._run_ema_accum_loss.items():
            if not samples:
                continue
            mean_loss = sum(samples) / len(samples)
            prev = self.layer_ema_loss.get(layer_key, mean_loss)
            self.layer_ema_loss[layer_key] = _EMA_DECAY * prev + (1.0 - _EMA_DECAY) * mean_loss
            if layer_key not in refreshed_layers:
                refreshed_layers.append(layer_key)
        self._run_ema_accum_loss.clear()

        for layer_key, samples in self._run_ema_accum_cosine_dist.items():
            if not samples:
                continue
            mean_dist = sum(samples) / len(samples)
            prev = self.layer_ema_cosine_dist.get(layer_key, mean_dist)
            self.layer_ema_cosine_dist[layer_key] = _EMA_DECAY * prev + (1.0 - _EMA_DECAY) * mean_dist
            if layer_key not in refreshed_layers:
                refreshed_layers.append(layer_key)
        self._run_ema_accum_cosine_dist.clear()

        for layer_key in refreshed_layers:
            self._refresh_layer_ready(layer_key)
            logger.info(
                "Flux2TTR: flushed run EMAs layer=%s ema_loss=%.6g ema_cosine_dist=%.6g",
                layer_key,
                float(self.layer_ema_loss.get(layer_key, float("nan"))),
                float(self.layer_ema_cosine_dist.get(layer_key, float("nan"))),
            )

    def _detect_run_boundary(self, sigma: Optional[float]) -> None:
        """Detect when a new sampling run starts and flush accumulated EMAs.

        Within a diffusion sampling run, sigma monotonically decreases.
        When sigma jumps significantly higher than the last seen value,
        a new run (new image/prompt) has started.
        """
        if sigma is None or not math.isfinite(sigma):
            return
        prev = self._run_last_sigma
        self._run_last_sigma = sigma
        if prev is not None and sigma > prev * 1.5:
            self.flush_run_emas()

    def _record_training_metrics(self, layer_key: str, metrics: Dict[str, float]) -> None:
        payload_metrics = dict(metrics)
        layer = self.layers.get(layer_key)
        if layer is not None and hasattr(layer, "alpha"):
            try:
                payload_metrics["alpha_sigmoid"] = float(torch.sigmoid(layer.alpha.detach()).item())
            except Exception:
                pass

        cosine_similarity = payload_metrics.get("cosine_similarity")
        if cosine_similarity is not None and math.isfinite(float(cosine_similarity)):
            cosine_dist = self._cosine_distance_from_similarity(float(cosine_similarity))
            self._run_ema_accum_cosine_dist.setdefault(layer_key, []).append(cosine_dist)
            # Report last-flushed EMA value (stable within run); falls back to
            # instantaneous distance if no flush has happened yet.
            ema_cosine_dist = self.layer_ema_cosine_dist.get(layer_key, cosine_dist)
            payload_metrics["ema_cosine_dist"] = float(ema_cosine_dist)

        self._layer_metric_latest[layer_key] = payload_metrics

        count = int(self._layer_metric_count.get(layer_key, 0)) + 1
        self._layer_metric_count[layer_key] = count
        running = self._layer_metric_running.setdefault(layer_key, {})
        for key, value in payload_metrics.items():
            if not math.isfinite(float(value)):
                continue
            prev = running.get(key, float(value))
            running[key] = float(prev + (float(value) - prev) / count)

        experiment = self._ensure_comet_experiment()
        if experiment is None:
            return

        should_log = (self.training_updates_done % self.comet_log_every == 0) or (self.steps_remaining <= 0)
        if not should_log:
            return

        tracked_layers = list(self._layer_metric_latest.keys())
        payload = {}
        for tracked_layer in tracked_layers:
            latest = self._layer_metric_latest.get(tracked_layer, {})
            for key, value in latest.items():
                if not math.isfinite(float(value)):
                    continue
                payload[f"flux2ttr/{tracked_layer}/{key}"] = float(value)
        for key, value in running.items():
            payload[f"flux2ttr/{layer_key}/avg_{key}"] = float(value)
        payload["flux2ttr/global/steps_remaining"] = float(self.steps_remaining)
        payload["flux2ttr/global/updates_done"] = float(self.training_updates_done)
        payload["flux2ttr/global/log_var_huber"] = float(self.log_var_huber.detach().cpu().item())
        payload["flux2ttr/global/log_var_cosine"] = float(self.log_var_cosine.detach().cpu().item())

        # Cross-layer aggregate distributions (latest values per layer).
        if tracked_layers:
            ready_count = sum(1 for lk in tracked_layers if bool(self.layer_ready.get(lk, False)))
            payload["flux2ttr/global/layers_tracked"] = float(len(tracked_layers))
            payload["flux2ttr/global/layers_ready"] = float(ready_count)
            payload["flux2ttr/global/layers_ready_ratio"] = float(ready_count / max(1, len(tracked_layers)))
            for metric_name in _COMET_AGG_METRICS:
                values = [self._layer_metric_latest[lk].get(metric_name, float("nan")) for lk in tracked_layers]
                summary = self._quantile_summary(values)
                for stat_name, stat_value in summary.items():
                    payload[f"flux2ttr/global/{metric_name}_{stat_name}"] = float(stat_value)
            pareto_frontier, _, worst_ema_cosine_dist = self._pareto_frontier_score(tracked_layers)
            payload["flux2ttr/global/pareto_frontier"] = float(pareto_frontier)
            payload["flux2ttr/global/pareto_worst_ema_cosine_dist"] = float(worst_ema_cosine_dist)
        if not flux2_comet_logging.safe_log_metrics(
            experiment=experiment,
            payload=payload,
            step=int(self.training_updates_done),
            logger=logger,
            component_name="Flux2TTR",
        ):
            self._comet_disabled = True

    def _select_training_query_indices(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        cap = max(0, int(self.training_query_token_cap))
        if cap <= 0 or seq_len <= cap:
            return None
        return torch.linspace(0, seq_len - 1, steps=cap, device=device, dtype=torch.float32).round().to(dtype=torch.long)

    def _infer_text_token_count(self, transformer_options: Optional[dict], n_key: int) -> int:
        if isinstance(transformer_options, dict):
            for key in ("text_token_count", "txt_token_count", "prefix_tokens"):
                value = transformer_options.get(key)
                if isinstance(value, int) and value >= 0:
                    return min(n_key, value)
            inner = transformer_options.get("flux2_ttr")
            if isinstance(inner, dict):
                value = inner.get("text_tokens_guess")
                if isinstance(value, int) and value >= 0:
                    return min(n_key, value)
        return min(n_key, self.text_tokens_guess)

    def _infer_conditioning_token_count(self, transformer_options: Optional[dict], n_key: int) -> Optional[int]:
        """Infer total conditioning tokens (text + references + controls), if available."""
        if not isinstance(transformer_options, dict):
            return None
        for key in ("conditioning_token_count", "cond_token_count", "prefix_token_count"):
            value = transformer_options.get(key)
            if isinstance(value, int) and value >= 0:
                return min(n_key, value)
        inner = transformer_options.get("flux2_ttr")
        if isinstance(inner, dict):
            for key in ("conditioning_token_count", "cond_token_count", "prefix_token_count"):
                value = inner.get(key)
                if isinstance(value, int) and value >= 0:
                    return min(n_key, value)
        return None

    def _compute_distill_metrics(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        loss_value: float,
        layer_key: str,
        sigma: Optional[float] = None,
        cfg_scale: Optional[float] = None,
        layers_swapped: Optional[float] = None,
        layers_total: Optional[float] = None,
    ) -> Dict[str, float]:
        diff = (student - teacher).float()
        teacher_f = teacher.float()
        student_f = student.float()

        mse = float(torch.mean(diff.square()).item())
        teacher_power = float(torch.mean(teacher_f.square()).item())
        nmse = mse / (teacher_power + _DISTILL_METRIC_EPS)

        cosine = torch.nn.functional.cosine_similarity(student_f, teacher_f, dim=-1, eps=_DISTILL_METRIC_EPS).mean()

        ema = self.layer_ema_loss.get(layer_key, float(loss_value))
        updates = self.layer_update_count.get(layer_key, 0)
        ready = self.layer_ready.get(layer_key, False)

        return {
            "loss": float(loss_value),
            "mse": mse,
            "nmse": float(nmse),
            "cosine_similarity": float(cosine.item()),
            "ema_loss": float(ema),
            "updates": float(updates),
            "ready": 1.0 if ready else 0.0,
            "sigma": float(sigma) if sigma is not None else float("nan"),
            "cfg_scale": float(cfg_scale) if cfg_scale is not None else float("nan"),
            "layers_swapped": float(layers_swapped) if layers_swapped is not None else float("nan"),
            "layers_total": float(layers_total) if layers_total is not None else float("nan"),
        }

    def _refresh_layer_ready(self, layer_key: str) -> bool:
        updates = int(self.layer_update_count.get(layer_key, 0))
        ema = float(self.layer_ema_cosine_dist.get(layer_key, float("inf")))
        threshold = float(self.layer_readiness_threshold.get(layer_key, self.readiness_threshold))
        min_updates = int(self.layer_readiness_min_updates.get(layer_key, self.readiness_min_updates))
        if layer_key not in self.layer_readiness_threshold:
            self.layer_readiness_threshold[layer_key] = threshold
        if layer_key not in self.layer_readiness_min_updates:
            self.layer_readiness_min_updates[layer_key] = min_updates
        prev_ready = bool(self.layer_ready.get(layer_key, False))
        if updates < min_updates:
            ready = False
        elif prev_ready:
            ready = bool(ema <= threshold * _READINESS_HYSTERESIS)
        else:
            ready = bool(ema <= threshold)
        prev = self.layer_ready.get(layer_key)
        self.layer_ready[layer_key] = bool(ready)
        if prev is None or bool(prev) != bool(ready):
            logger.info(
                (
                    "Flux2TTR: layer readiness changed layer=%s ready=%s updates=%d "
                    "ema=%.6g threshold=%.6g hysteresis=%.3g"
                ),
                layer_key,
                ready,
                updates,
                ema,
                threshold,
                _READINESS_HYSTERESIS,
            )
        return bool(ready)

    def _teacher_from_fallback(
        self,
        fallback_attention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pe: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        transformer_options: Optional[dict],
    ) -> torch.Tensor:
        if fallback_attention is None:
            if pe is not None:
                try:
                    from comfy.ldm.flux.math import apply_rope

                    q, k = apply_rope(q, k, pe)
                except Exception:
                    logger.warning("Flux2TTR: could not import apply_rope for teacher fallback; using un-rope q/k.")
            return _flatten_heads(_softmax_attention(q.float(), k.float(), v.float(), mask=mask).to(dtype=v.dtype))
        return fallback_attention(q, k, v, pe, mask=mask, transformer_options=transformer_options)

    def _student_from_runtime(
        self,
        q_eff: torch.Tensor,
        k_eff: torch.Tensor,
        v: torch.Tensor,
        layer_key: str,
        head_dim: int,
        key_mask: Optional[torch.Tensor],
        text_token_count: Optional[int],
        conditioning_token_count: Optional[int],
        sigma: Optional[float],
        cfg_scale: Optional[float],
        transformer_options: Optional[dict],
        reserve_memory: bool = True,
    ) -> torch.Tensor:
        layer = self._ensure_layer(layer_key, head_dim, q_eff.device)
        self._ensure_layer_device(layer, q_eff.device)
        layer.eval()

        inference_dtype = self._resolve_inference_dtype(q_eff)
        if reserve_memory and self.enable_memory_reserve:
            _maybe_reserve_memory(
                self,
                q_eff,
                k_eff,
                transformer_options,
                training=False,
                dtype_accum=inference_dtype,
                layer_key=layer_key,
            )

        self._set_layer_dtype(layer_key, layer, inference_dtype)
        self._ensure_layer_device(layer, q_eff.device)
        with torch.no_grad():
            student = layer(
                q=q_eff.to(dtype=inference_dtype),
                k=k_eff.to(dtype=inference_dtype),
                v=v.to(dtype=inference_dtype),
                key_mask=key_mask,
                q_chunk=self.query_chunk_size,
                k_chunk=self.key_chunk_size,
                text_token_count=text_token_count,
                conditioning_token_count=conditioning_token_count,
                sigma=sigma,
                cfg_scale=cfg_scale,
            )

        if not torch.isfinite(student).all():
            raise RuntimeError(f"Flux2TTR: non-finite student output on {layer_key}.")
        den_min = float(getattr(layer, "last_den_min", float("nan")))
        if math.isfinite(den_min) and den_min < layer.eps:
            logger.warning("Flux2TTR: denominator floor hit on %s (den_min=%.6g).", layer_key, den_min)

        return _flatten_heads(student).to(dtype=v.dtype)

    @staticmethod
    def _sample_tensor_nbytes(x: Optional[torch.Tensor]) -> int:
        if x is None:
            return 0
        return int(x.numel() * x.element_size())

    def _evict_one_oldest_replay_sample(self) -> bool:
        oldest_layer = None
        oldest_step = None
        for layer_key, layer_buf in self.replay_buffers.items():
            if not layer_buf:
                continue
            created = int(layer_buf[0].created_step)
            if oldest_step is None or created < oldest_step:
                oldest_step = created
                oldest_layer = layer_key
        if oldest_layer is None:
            return False

        popped = self.replay_buffers[oldest_layer].popleft()
        self.replay_total_bytes = max(0, self.replay_total_bytes - int(popped.nbytes))
        return True

    def _evict_replay_until_within_budget(self, incoming_bytes: int = 0) -> None:
        if self.replay_max_bytes <= 0:
            return
        target = self.replay_max_bytes - max(0, int(incoming_bytes))
        while self.replay_total_bytes > max(0, target):
            if not self._evict_one_oldest_replay_sample():
                break

    def _push_replay_sample(
        self,
        layer_key: str,
        q_sub: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        teacher_sub: torch.Tensor,
        key_mask: Optional[torch.Tensor],
        text_token_count: Optional[int],
        sigma: Optional[float],
        cfg_scale: Optional[float],
        conditioning_token_count: Optional[int] = None,
    ) -> None:
        buf = self.replay_buffers.setdefault(layer_key, deque())
        store_device = torch.device("cpu") if self.replay_offload_cpu else q_sub.device
        replay_dtype = self._replay_dtype()

        def _pack(x: torch.Tensor) -> torch.Tensor:
            y = x.detach()
            if torch.is_floating_point(y):
                y = y.to(dtype=replay_dtype)
            return y.to(device=store_device)

        q_store = _pack(q_sub)
        k_store = _pack(k_full)
        v_store = _pack(v_full)
        teacher_store = _pack(teacher_sub)
        key_mask_store = key_mask.detach().to(device=store_device) if key_mask is not None else None
        nbytes = (
            self._sample_tensor_nbytes(q_store)
            + self._sample_tensor_nbytes(k_store)
            + self._sample_tensor_nbytes(v_store)
            + self._sample_tensor_nbytes(teacher_store)
            + self._sample_tensor_nbytes(key_mask_store)
        )

        self._evict_replay_until_within_budget(incoming_bytes=nbytes)

        self._replay_created_counter += 1
        sample = ReplaySample(
            q_sub=q_store,
            k_full=k_store,
            v_full=v_store,
            teacher_sub=teacher_store,
            key_mask=key_mask_store,
            text_token_count=text_token_count,
            conditioning_token_count=conditioning_token_count,
            sigma=self._as_float(sigma),
            cfg_scale=self._as_float(cfg_scale),
            nbytes=int(nbytes),
            created_step=int(self._replay_created_counter),
        )
        buf.append(sample)
        self.replay_total_bytes += int(nbytes)

        while len(buf) > self.replay_buffer_size:
            popped = buf.popleft()
            self.replay_total_bytes = max(0, self.replay_total_bytes - int(popped.nbytes))

        self._evict_replay_until_within_budget(incoming_bytes=0)

    def _handle_training_oom(self, layer_key: str, device: torch.device) -> bool:
        old_q_cap = self.training_query_token_cap
        old_q_chunk = self.query_chunk_size
        old_k_chunk = self.key_chunk_size
        old_landmarks = self.landmark_max

        changed = False
        if self.training_query_token_cap > 32:
            self.training_query_token_cap = max(32, self.training_query_token_cap // 2)
            changed = True
        if self.query_chunk_size > 64:
            self.query_chunk_size = max(64, self.query_chunk_size // 2)
            changed = True
        if self.key_chunk_size > 256:
            self.key_chunk_size = max(256, self.key_chunk_size // 2)
            changed = True
        if self.landmark_max <= 0:
            # Unlimited landmark mode can be too expensive under pressure;
            # cap first, then normal halving can continue on subsequent OOMs.
            self.landmark_max = max(self.landmark_min, int(_OOM_RECOVERY_LANDMARK_MAX))
            changed = True
        elif self.landmark_max > self.landmark_min:
            self.landmark_max = max(self.landmark_min, self.landmark_max // 2)
            changed = True

        layer_buf = self.replay_buffers.get(layer_key)
        if layer_buf is not None and len(layer_buf) > 0:
            released = sum(int(s.nbytes) for s in layer_buf)
            layer_buf.clear()
            self.replay_total_bytes = max(0, self.replay_total_bytes - released)
            changed = True

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if changed:
            logger.warning(
                (
                    "Flux2TTR OOM recovery on %s: training_query_token_cap %d->%d, "
                    "q_chunk %d->%d, k_chunk %d->%d, landmark_max %d->%d; cleared layer replay buffer."
                ),
                layer_key,
                old_q_cap,
                self.training_query_token_cap,
                old_q_chunk,
                self.query_chunk_size,
                old_k_chunk,
                self.key_chunk_size,
                old_landmarks,
                self.landmark_max,
            )
        return changed

    def _train_from_replay(self, layer_key: str, head_dim: int, device: torch.device) -> None:
        if not self.training_enabled or self.steps_remaining <= 0:
            return
        buf = self.replay_buffers.get(layer_key)
        if not buf:
            return

        layer = self._ensure_layer(layer_key, head_dim, device)
        self._ensure_layer_device(layer, device)
        optimizer = self.optimizers[layer_key]
        loss_weight_optimizer = self._ensure_loss_weight_optimizer(device)
        self._set_layer_dtype(layer_key, layer, torch.float32)
        self._ensure_layer_device(layer, device)
        layer.train()

        steps = min(self.train_steps_per_call, self.steps_remaining)
        for _ in range(steps):
            sample = random.choice(tuple(buf))
            q_sub = sample.q_sub.to(device=device, dtype=torch.float32).clone()
            k_full = sample.k_full.to(device=device, dtype=torch.float32).clone()
            v_full = sample.v_full.to(device=device, dtype=torch.float32).clone()
            teacher_sub = sample.teacher_sub.to(device=device, dtype=torch.float32).clone()
            key_mask = sample.key_mask.to(device=device) if sample.key_mask is not None else None

            student_sub = layer(
                q=q_sub,
                k=k_full,
                v=v_full,
                key_mask=key_mask,
                q_chunk=self.query_chunk_size,
                k_chunk=self.key_chunk_size,
                text_token_count=sample.text_token_count,
                conditioning_token_count=sample.conditioning_token_count,
                sigma=sample.sigma,
                cfg_scale=sample.cfg_scale,
            )
            huber = F.smooth_l1_loss(student_sub, teacher_sub, beta=self.huber_beta)
            cosine_sim = F.cosine_similarity(student_sub, teacher_sub, dim=-1, eps=1e-8).mean()
            cosine_term = 1.0 - cosine_sim
            loss = (
                huber / (2.0 * self.log_var_huber.exp())
                + self.log_var_huber / 2.0
                + cosine_term / (2.0 * self.log_var_cosine.exp())
                + self.log_var_cosine / 2.0
            )

            optimizer.zero_grad(set_to_none=True)
            loss_weight_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(layer.parameters(), self.grad_clip_norm)
            optimizer.step()
            loss_weight_optimizer.step()

            loss_value = float(loss.item())
            self.last_loss = loss_value
            self.layer_last_loss[layer_key] = loss_value

            self._run_ema_accum_loss.setdefault(layer_key, []).append(loss_value)
            self.layer_update_count[layer_key] = int(self.layer_update_count.get(layer_key, 0)) + 1
            self._refresh_layer_ready(layer_key)

            self.steps_remaining -= 1
            self.training_updates_done += 1

            metrics = self._compute_distill_metrics(
                student=student_sub.detach(),
                teacher=teacher_sub.detach(),
                loss_value=loss_value,
                layer_key=layer_key,
                sigma=sample.sigma,
                cfg_scale=sample.cfg_scale,
                layers_swapped=float(len(self._current_step_swap_set or set())),
                layers_total=float(self._current_step_eligible_count),
            )
            self._record_training_metrics(layer_key, metrics)
            total_accum = sum(len(v) for v in self._run_ema_accum_loss.values())
            if total_accum >= _RUN_EMA_PERIODIC_FLUSH_UPDATES:
                self.flush_run_emas()

            if (
                self.training_log_every > 0
                and (
                    self.training_updates_done % self.training_log_every == 0
                    or self.steps_remaining <= 0
                )
            ):
                self._log_training_snapshot()

            if self.steps_remaining <= 0:
                self.flush_run_emas()
                self.training_enabled = False
                logger.info("Flux2TTR: online distillation reached configured steps.")
                break

    def run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pe: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        transformer_options: Optional[dict],
        fallback_attention,
    ) -> torch.Tensor:
        layer_key = self._layer_key_from_options(transformer_options)
        if not self._is_single_block_selected(transformer_options):
            return self._teacher_from_fallback(fallback_attention, q, k, v, pe, mask, transformer_options)
        if not layer_key.startswith("single:"):
            return self._teacher_from_fallback(fallback_attention, q, k, v, pe, mask, transformer_options)

        if not _supports_key_padding_mask(mask, q.shape[0], q.shape[2], k.shape[2]):
            logger.warning(
                "Flux2TTR: unsupported mask shape=%s on %s; falling back to teacher attention.",
                tuple(mask.shape) if torch.is_tensor(mask) else None,
                layer_key,
            )
            return self._teacher_from_fallback(fallback_attention, q, k, v, pe, mask, transformer_options)

        sigma = self._extract_sigma(transformer_options)
        cfg_scale = self._extract_cfg_scale(transformer_options)

        if pe is not None:
            try:
                from comfy.ldm.flux.math import apply_rope

                q_eff, k_eff = apply_rope(q, k, pe)
            except Exception:
                logger.warning("Flux2TTR: apply_rope unavailable; running without RoPE.")
                q_eff, k_eff = q, k
        else:
            q_eff, k_eff = q, k

        spec = self.layer_specs.get(layer_key)
        head_dim = spec.head_dim if spec else int(q_eff.shape[-1])
        key_mask = _safe_key_mask(_key_mask_from_mask(mask, q.shape[0], k.shape[2]))
        text_token_count = self._infer_text_token_count(transformer_options, int(k.shape[2]))
        conditioning_token_count = self._infer_conditioning_token_count(transformer_options, int(k.shape[2]))

        teacher_out: Optional[torch.Tensor] = None
        if self.training_mode or not self.layer_ready.get(layer_key, False):
            teacher_out = self._teacher_from_fallback(fallback_attention, q, k, v, pe, mask, transformer_options)

        if self.training_mode:
            self._detect_run_boundary(sigma)
            eligible_layers: list[str] = []
            swap_set: set[str] = set()
            if self.training_enabled and self.steps_remaining > 0:
                eligible_layers = self._eligible_single_layer_keys(
                    current_layer_key=layer_key,
                    transformer_options=transformer_options,
                )
                swap_set = self._resolve_swap_set_for_step(transformer_options, eligible_layers)
                if layer_key not in swap_set:
                    if isinstance(transformer_options, dict):
                        cfg = transformer_options.get("flux2_ttr")
                        if isinstance(cfg, dict):
                            cfg["training_steps_remaining"] = int(self.steps_remaining)
                            cfg["training_updates_done"] = int(self.training_updates_done)
                            cfg["layers_swapped"] = int(len(swap_set))
                            cfg["layers_total"] = int(len(eligible_layers))
                    return teacher_out
                try:
                    if self.enable_memory_reserve:
                        _maybe_reserve_memory(
                            self,
                            q_eff,
                            k_eff,
                            transformer_options,
                            training=True,
                            dtype_accum=torch.float32,
                            layer_key=layer_key,
                        )

                    teacher_bhnd = _unflatten_heads(teacher_out, q.shape[1], q.shape[3]).float().clone()
                    q_train = q_eff.float().clone()
                    k_train = k_eff.float().clone()
                    v_train = v.float().clone()

                    idx_q = self._select_training_query_indices(q_train.shape[2], q_train.device)
                    if idx_q is not None:
                        q_sub = q_train[:, :, idx_q, :]
                        teacher_sub = teacher_bhnd[:, :, idx_q, :]
                    else:
                        q_sub = q_train
                        teacher_sub = teacher_bhnd

                    self._push_replay_sample(
                        layer_key=layer_key,
                        q_sub=q_sub,
                        k_full=k_train,
                        v_full=v_train,
                        teacher_sub=teacher_sub,
                        key_mask=key_mask,
                        text_token_count=text_token_count,
                        conditioning_token_count=conditioning_token_count,
                        sigma=sigma,
                        cfg_scale=cfg_scale,
                    )

                    with torch.inference_mode(False):
                        with torch.enable_grad():
                            self._train_from_replay(layer_key, head_dim, q_eff.device)
                except torch.OutOfMemoryError:
                    recovered = self._handle_training_oom(layer_key, q_eff.device)
                    if not recovered:
                        self.training_enabled = False
                        logger.warning("Flux2TTR training OOM on layer %s; disabling training for this run.", layer_key)

                if isinstance(transformer_options, dict):
                    cfg = transformer_options.get("flux2_ttr")
                    if isinstance(cfg, dict):
                        cfg["training_steps_remaining"] = int(self.steps_remaining)
                        cfg["training_updates_done"] = int(self.training_updates_done)
                        cfg["layers_swapped"] = int(len(swap_set))
                        cfg["layers_total"] = int(len(eligible_layers))

            if not self.training_preview_ttr:
                return teacher_out

            if not self._refresh_layer_ready(layer_key):
                return teacher_out

            try:
                return self._student_from_runtime(
                    q_eff=q_eff,
                    k_eff=k_eff,
                    v=v,
                    layer_key=layer_key,
                    head_dim=head_dim,
                    key_mask=key_mask,
                    text_token_count=text_token_count,
                    conditioning_token_count=conditioning_token_count,
                    sigma=sigma,
                    cfg_scale=cfg_scale,
                    transformer_options=transformer_options,
                    reserve_memory=False,
                )
            except Exception as exc:
                logger.warning("Flux2TTR preview fallback on %s: %s", layer_key, exc)
                return teacher_out

        controller = None
        controller_mask_override = None
        controller_threshold = 0.5
        controller_policy = _DEFAULT_CONTROLLER_POLICY
        controller_temperature = _DEFAULT_CONTROLLER_TEMPERATURE
        if isinstance(transformer_options, dict):
            cfg = transformer_options.get("flux2_ttr")
            if isinstance(cfg, dict):
                controller = cfg.get("controller")
                controller_mask_override = cfg.get("controller_mask_override")
                try:
                    threshold_value = float(cfg.get("controller_threshold", 0.5))
                    if math.isfinite(threshold_value):
                        controller_threshold = threshold_value
                except Exception:
                    controller_threshold = 0.5
                controller_policy = self._normalize_controller_policy(cfg.get("controller_policy", _DEFAULT_CONTROLLER_POLICY))
                controller_temperature = self._sanitize_controller_temperature(
                    cfg.get("controller_temperature", _DEFAULT_CONTROLLER_TEMPERATURE),
                    default=_DEFAULT_CONTROLLER_TEMPERATURE,
                )
        if controller_mask_override is not None or controller is not None:
            width, height = self._extract_resolution(
                transformer_options,
                n_key=int(k.shape[2]),
                text_token_count=conditioning_token_count if conditioning_token_count is not None else text_token_count,
            )
            decision_threshold = (
                0.5
                if controller_mask_override is None and controller_policy == _CONTROLLER_POLICY_STOCHASTIC
                else float(controller_threshold)
            )
            try:
                if controller_mask_override is not None:
                    controller_mask = torch.as_tensor(
                        controller_mask_override,
                        device=q.device,
                        dtype=torch.float32,
                    ).reshape(-1)
                else:
                    controller_mask = self._get_controller_mask(
                        controller,
                        sigma,
                        cfg_scale,
                        width,
                        height,
                        controller_threshold=controller_threshold,
                        controller_policy=controller_policy,
                        controller_temperature=controller_temperature,
                    )
                self._maybe_log_controller_step_routing(
                    transformer_options=transformer_options,
                    controller_mask=controller_mask,
                    controller_threshold=controller_threshold,
                    controller_policy=controller_policy,
                    controller_temperature=controller_temperature,
                    decision_threshold=decision_threshold,
                    sigma=sigma,
                    cfg_scale=cfg_scale,
                    width=width,
                    height=height,
                    current_layer_key=layer_key,
                )
                layer_idx = self._extract_layer_index(layer_key)
                if layer_idx is None or layer_idx < 0 or layer_idx >= int(controller_mask.numel()):
                    logger.warning(
                        "Flux2TTR: controller mask index out of range for %s (idx=%s mask_len=%d); using teacher fallback.",
                        layer_key,
                        layer_idx,
                        int(controller_mask.numel()),
                    )
                    return teacher_out if teacher_out is not None else self._teacher_from_fallback(
                        fallback_attention,
                        q,
                        k,
                        v,
                        pe,
                        mask,
                        transformer_options,
                    )
                use_full_attention = bool(float(controller_mask[layer_idx].item()) > float(decision_threshold))
            except Exception as exc:
                logger.warning("Flux2TTR: controller evaluation failed on %s (%s); using teacher fallback.", layer_key, exc)
                return teacher_out if teacher_out is not None else self._teacher_from_fallback(
                    fallback_attention,
                    q,
                    k,
                    v,
                    pe,
                    mask,
                    transformer_options,
                )
            if use_full_attention:
                return teacher_out if teacher_out is not None else self._teacher_from_fallback(
                    fallback_attention,
                    q,
                    k,
                    v,
                    pe,
                    mask,
                    transformer_options,
                )

        # Inference mode: fail closed when not ready.
        if not self._refresh_layer_ready(layer_key):
            return teacher_out if teacher_out is not None else self._teacher_from_fallback(
                fallback_attention,
                q,
                k,
                v,
                pe,
                mask,
                transformer_options,
            )

        if (
            math.isfinite(self.last_loss)
            and self.max_safe_inference_loss > 0
            and self.last_loss > self.max_safe_inference_loss
        ):
            if not self._warned_high_loss:
                logger.warning(
                    "Flux2TTR: checkpoint loss %.6g exceeds safe inference threshold %.6g; using native attention fallback.",
                    self.last_loss,
                    self.max_safe_inference_loss,
                )
                self._warned_high_loss = True
            return teacher_out if teacher_out is not None else self._teacher_from_fallback(
                fallback_attention,
                q,
                k,
                v,
                pe,
                mask,
                transformer_options,
            )

        try:
            return self._student_from_runtime(
                q_eff=q_eff,
                k_eff=k_eff,
                v=v,
                layer_key=layer_key,
                head_dim=head_dim,
                key_mask=key_mask,
                text_token_count=text_token_count,
                conditioning_token_count=conditioning_token_count,
                sigma=sigma,
                cfg_scale=cfg_scale,
                transformer_options=transformer_options,
            )
        except Exception as exc:
            logger.warning("Flux2TTR: student inference failed on %s (%s); using teacher fallback.", layer_key, exc)
            return teacher_out if teacher_out is not None else self._teacher_from_fallback(
                fallback_attention,
                q,
                k,
                v,
                pe,
                mask,
                transformer_options,
            )

    def calibrate_from_inputs(
        self,
        model: Any,
        latents: Any,
        conditioning: Any,
        steps: int,
        max_tokens: int = 256,
    ) -> float:
        del latents, conditioning, max_tokens
        specs = infer_flux_single_layer_specs(model)
        if specs:
            self.register_layer_specs(specs)

        capture_steps = max(0, int(steps))
        self.training_mode = True
        self.training_enabled = capture_steps > 0
        self.training_preview_ttr = False
        self.training_steps_total = capture_steps
        self.steps_remaining = capture_steps
        self.training_updates_done = 0
        self.replay_buffers.clear()

        logger.info(
            "Flux2TTR calibration mode updated: real-sample capture enabled for %d attention calls. "
            "Run the sampler to collect q/k/v and train from replay.",
            capture_steps,
        )
        if math.isnan(self.last_loss):
            return 0.0
        return float(self.last_loss)

    def checkpoint_state(self) -> Dict[str, Any]:
        self.flush_run_emas()
        experiment = self._ensure_comet_experiment()
        if experiment is not None:
            last_sigma = float("nan")
            if isinstance(self._current_step_id, tuple) and len(self._current_step_id) == 2 and self._current_step_id[0] == "sigma":
                maybe_sigma = self._as_float(self._current_step_id[1])
                if maybe_sigma is not None:
                    last_sigma = maybe_sigma
            if not flux2_comet_logging.safe_log_parameters(
                experiment=experiment,
                payload={
                    "cfg_scale": float(self.cfg_scale),
                    "last_sigma": float(last_sigma),
                    "min_swap_layers": int(self.min_swap_layers),
                    "max_swap_layers": int(self.max_swap_layers),
                },
                logger=logger,
                component_name="Flux2TTR",
                failure_message="failed to log checkpoint hyperparameters to Comet",
            ):
                self._comet_disabled = True

        layer_states = {}
        for layer_key, layer in self.layers.items():
            layer_states[layer_key] = {k: v.detach().cpu() for k, v in layer.state_dict().items()}
        for layer_key, state in self.pending_state.items():
            if layer_key not in layer_states:
                layer_states[layer_key] = state

        optimizer_states: Dict[str, Dict[str, Any]] = {}
        for layer_key, state in self.pending_optimizer_states.items():
            if isinstance(state, dict):
                optimizer_states[layer_key] = self._clone_state_to_cpu(state)
        for layer_key, optimizer in self.optimizers.items():
            try:
                optimizer_states[layer_key] = self._clone_state_to_cpu(optimizer.state_dict())
            except Exception as exc:
                logger.warning("Flux2TTR: failed to serialize optimizer state for %s (%s).", layer_key, exc)

        loss_weight_optimizer_state: Optional[Dict[str, Any]] = None
        if isinstance(self.pending_loss_weight_optimizer_state, dict):
            loss_weight_optimizer_state = self._clone_state_to_cpu(self.pending_loss_weight_optimizer_state)
        if self.loss_weight_optimizer is not None:
            try:
                loss_weight_optimizer_state = self._clone_state_to_cpu(self.loss_weight_optimizer.state_dict())
            except Exception as exc:
                logger.warning("Flux2TTR: failed to serialize loss-weight optimizer state (%s).", exc)

        readiness_keys = set(layer_states.keys())
        readiness_keys.update(self.layer_specs.keys())
        readiness_keys.update(self.layer_ema_loss.keys())
        readiness_keys.update(self.layer_ema_cosine_dist.keys())
        readiness_keys.update(self.layer_update_count.keys())
        readiness_keys.update(self.layer_ready.keys())
        layer_readiness = {}
        for layer_key in sorted(readiness_keys):
            layer_readiness[layer_key] = {
                "readiness_threshold": float(self.layer_readiness_threshold.get(layer_key, self.readiness_threshold)),
                "readiness_min_updates": int(self.layer_readiness_min_updates.get(layer_key, self.readiness_min_updates)),
                "ema_loss": float(self.layer_ema_loss.get(layer_key, float("inf"))),
                "ema_cosine_dist": float(self.layer_ema_cosine_dist.get(layer_key, float("nan"))),
                "update_count": int(self.layer_update_count.get(layer_key, 0)),
                "ready": bool(self.layer_ready.get(layer_key, False)),
            }

        return {
            "format": "flux2_ttr_v2",
            "feature_dim": self.feature_dim,
            "alpha_format": "logit",
            "learning_rate": self.learning_rate,
            "training_mode": self.training_mode,
            "training_preview_ttr": self.training_preview_ttr,
            "comet_enabled": self.comet_enabled,
            "comet_project_name": self.comet_project_name,
            "comet_workspace": self.comet_workspace,
            "comet_experiment": self.comet_experiment,
            "comet_display_name": self.comet_display_name,
            "comet_persist_experiment": self.comet_persist_experiment,
            "last_loss": self.last_loss,
            "loss_log_var_huber": float(self.log_var_huber.detach().cpu().item()),
            "loss_log_var_cosine": float(self.log_var_cosine.detach().cpu().item()),
            "query_chunk_size": self.query_chunk_size,
            "key_chunk_size": self.key_chunk_size,
            "landmark_fraction": self.landmark_fraction,
            "landmark_min": self.landmark_min,
            "landmark_max": self.landmark_max,
            "text_tokens_guess": self.text_tokens_guess,
            "alpha_init": self.alpha_init,
            "alpha_max": self.alpha_max,
            "gamma": self.gamma,
            "alpha_lr_multiplier": self.alpha_lr_multiplier,
            "phi_lr_multiplier": self.phi_lr_multiplier,
            "training_query_token_cap": self.training_query_token_cap,
            "replay_buffer_size": self.replay_buffer_size,
            "replay_offload_cpu": self.replay_offload_cpu,
            "replay_storage_dtype": self.replay_storage_dtype,
            "replay_max_bytes": self.replay_max_bytes,
            "train_steps_per_call": self.train_steps_per_call,
            "huber_beta": self.huber_beta,
            "grad_clip_norm": self.grad_clip_norm,
            "readiness_threshold": self.readiness_threshold,
            "readiness_min_updates": self.readiness_min_updates,
            "enable_memory_reserve": self.enable_memory_reserve,
            "max_safe_inference_loss": self.max_safe_inference_loss,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "inference_mixed_precision": self.inference_mixed_precision,
            "cfg_scale": self.cfg_scale,
            "min_swap_layers": self.min_swap_layers,
            "max_swap_layers": self.max_swap_layers,
            "comet_log_every": self.comet_log_every,
            "layer_specs": {
                key: {"num_heads": spec.num_heads, "head_dim": spec.head_dim}
                for key, spec in self.layer_specs.items()
            },
            "layer_ema_loss": dict(self.layer_ema_loss),
            "layer_ema_cosine_dist": dict(self.layer_ema_cosine_dist),
            "layer_update_count": dict(self.layer_update_count),
            "layer_ready": dict(self.layer_ready),
            "layer_last_loss": dict(self.layer_last_loss),
            "layer_readiness": layer_readiness,
            "layers": layer_states,
            "optimizer_states": optimizer_states,
            "loss_weight_optimizer_state": loss_weight_optimizer_state,
        }

    def save_checkpoint(self, checkpoint_path: str) -> None:
        path = checkpoint_path.strip()
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.checkpoint_state(), path)
        logger.info("Flux2TTR: saved checkpoint to %s", path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        path = checkpoint_path.strip()
        if not path:
            raise ValueError("Flux2TTR: checkpoint_path must be set when loading.")

        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Flux2TTR: unsupported checkpoint payload in {path}.")
        ckpt_feature_dim = _checkpoint_feature_dim_from_payload(payload, path)
        if ckpt_feature_dim != self.feature_dim:
            raise ValueError(
                f"Flux2TTR: checkpoint feature_dim={ckpt_feature_dim} does not match requested feature_dim={self.feature_dim}."
            )

        self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        self.training_mode = bool(payload.get("training_mode", self.training_mode))
        self.training_preview_ttr = bool(payload.get("training_preview_ttr", self.training_preview_ttr))
        self.comet_enabled = bool(payload.get("comet_enabled", self.comet_enabled))
        self.comet_project_name = str(payload.get("comet_project_name", self.comet_project_name))
        self.comet_workspace = str(payload.get("comet_workspace", self.comet_workspace))
        self.comet_experiment = str(payload.get("comet_experiment", self.comet_experiment)).strip()
        self.comet_display_name = str(payload.get("comet_display_name", self.comet_display_name)).strip()
        if not self.comet_display_name:
            self.comet_display_name = _generate_experiment_display_name()
        self.comet_persist_experiment = bool(payload.get("comet_persist_experiment", self.comet_persist_experiment))
        self.comet_persist_experiment = bool(self.comet_persist_experiment and self.comet_experiment)
        self.comet_log_every = max(1, int(payload.get("comet_log_every", self.comet_log_every)))
        self.last_loss = float(payload.get("last_loss", self.last_loss))
        log_var_huber = float(payload.get("loss_log_var_huber", _DEFAULT_LOSS_LOG_VAR_HUBER))
        if not math.isfinite(log_var_huber):
            log_var_huber = _DEFAULT_LOSS_LOG_VAR_HUBER
        self.log_var_huber.data.fill_(float(log_var_huber))
        if self.log_var_huber.grad is not None:
            self.log_var_huber.grad.zero_()
        log_var_cosine = float(payload.get("loss_log_var_cosine", _DEFAULT_LOSS_LOG_VAR_COSINE))
        if not math.isfinite(log_var_cosine):
            log_var_cosine = _DEFAULT_LOSS_LOG_VAR_COSINE
        self.log_var_cosine.data.fill_(float(log_var_cosine))
        if self.log_var_cosine.grad is not None:
            self.log_var_cosine.grad.zero_()
        if self.loss_weight_optimizer is not None:
            try:
                self.loss_weight_optimizer.state.clear()
            except Exception:
                pass
            self.loss_weight_optimizer = None
        pending_loss_weight_optimizer_state = payload.get("loss_weight_optimizer_state")
        if isinstance(pending_loss_weight_optimizer_state, dict):
            self.pending_loss_weight_optimizer_state = pending_loss_weight_optimizer_state
        else:
            self.pending_loss_weight_optimizer_state = None

        self.query_chunk_size = max(1, int(payload.get("query_chunk_size", self.query_chunk_size)))
        self.key_chunk_size = max(1, int(payload.get("key_chunk_size", self.key_chunk_size)))
        # Backward compat: old checkpoints only stored landmark_count, which
        # cannot be converted into a resolution-aware fraction. Fall back to
        # module defaults unless explicit fraction/min/max are present.
        if "landmark_fraction" in payload:
            self.landmark_fraction = max(0.0, min(1.0, float(payload["landmark_fraction"])))
        else:
            self.landmark_fraction = float(_DEFAULT_LANDMARK_FRACTION)
        if "landmark_min" in payload:
            self.landmark_min = _normalize_landmark_min(payload["landmark_min"])
        else:
            self.landmark_min = int(_DEFAULT_LANDMARK_MIN)
        if "landmark_max" in payload:
            self.landmark_max = _normalize_landmark_max(payload["landmark_max"])
        else:
            self.landmark_max = int(_DEFAULT_LANDMARK_MAX)
        self.landmark_fraction = max(0.0, min(1.0, float(self.landmark_fraction)))
        self.landmark_min = _normalize_landmark_min(self.landmark_min)
        self.landmark_max = _normalize_landmark_max(self.landmark_max)
        self.text_tokens_guess = max(0, int(payload.get("text_tokens_guess", self.text_tokens_guess)))
        self.alpha_init = float(payload.get("alpha_init", self.alpha_init))
        self.alpha_max = max(0.0, min(1.0, float(payload.get("alpha_max", 0.2))))
        self.gamma = float(payload.get("gamma", 2.0))
        self.alpha_lr_multiplier = float(payload.get("alpha_lr_multiplier", self.alpha_lr_multiplier))
        self.phi_lr_multiplier = float(payload.get("phi_lr_multiplier", self.phi_lr_multiplier))

        self.training_query_token_cap = max(1, int(payload.get("training_query_token_cap", self.training_query_token_cap)))
        self.replay_buffer_size = max(1, int(payload.get("replay_buffer_size", self.replay_buffer_size)))
        self.replay_offload_cpu = bool(payload.get("replay_offload_cpu", self.replay_offload_cpu))
        self.replay_storage_dtype = str(payload.get("replay_storage_dtype", self.replay_storage_dtype)).strip().lower()
        if self.replay_storage_dtype not in ("float32", "float16", "bfloat16"):
            self.replay_storage_dtype = _DEFAULT_REPLAY_STORAGE_DTYPE
        self.replay_max_bytes = max(0, int(payload.get("replay_max_bytes", self.replay_max_bytes)))
        self.train_steps_per_call = max(1, int(payload.get("train_steps_per_call", self.train_steps_per_call)))
        self.huber_beta = max(1e-6, float(payload.get("huber_beta", self.huber_beta)))
        self.grad_clip_norm = max(0.0, float(payload.get("grad_clip_norm", self.grad_clip_norm)))
        self.readiness_threshold = float(payload.get("readiness_threshold", self.readiness_threshold))
        self.readiness_min_updates = max(0, int(payload.get("readiness_min_updates", self.readiness_min_updates)))
        self.enable_memory_reserve = bool(payload.get("enable_memory_reserve", self.enable_memory_reserve))

        self.max_safe_inference_loss = float(payload.get("max_safe_inference_loss", self.max_safe_inference_loss))
        self.layer_start = int(payload.get("layer_start", self.layer_start))
        self.layer_end = int(payload.get("layer_end", self.layer_end))
        self.inference_mixed_precision = bool(payload.get("inference_mixed_precision", self.inference_mixed_precision))
        self.cfg_scale = float(payload.get("cfg_scale", self.cfg_scale))
        self.min_swap_layers = max(0, int(payload.get("min_swap_layers", self.min_swap_layers)))
        self.max_swap_layers = int(payload.get("max_swap_layers", self.max_swap_layers))

        specs = payload.get("layer_specs", {})
        self.layer_specs.clear()
        for layer_key, meta in specs.items():
            try:
                self.layer_specs[layer_key] = FluxLayerSpec(
                    layer_key=layer_key,
                    num_heads=int(meta["num_heads"]),
                    head_dim=int(meta["head_dim"]),
                )
            except Exception:
                logger.warning("Flux2TTR: invalid layer spec in checkpoint for %s; skipping.", layer_key)

        self.layer_ema_loss = {str(k): float(v) for k, v in payload.get("layer_ema_loss", {}).items()}
        self.layer_ema_cosine_dist = {str(k): float(v) for k, v in payload.get("layer_ema_cosine_dist", {}).items()}
        self.layer_update_count = {str(k): int(v) for k, v in payload.get("layer_update_count", {}).items()}
        self.layer_ready = {str(k): bool(v) for k, v in payload.get("layer_ready", {}).items()}
        self.layer_last_loss = {str(k): float(v) for k, v in payload.get("layer_last_loss", {}).items()}
        self.pending_optimizer_states = {}
        raw_optimizer_states = payload.get("optimizer_states", {})
        if isinstance(raw_optimizer_states, dict):
            for layer_key, state in raw_optimizer_states.items():
                if isinstance(state, dict):
                    self.pending_optimizer_states[str(layer_key)] = state
        self.layer_readiness_threshold = {}
        self.layer_readiness_min_updates = {}
        for layer_key, meta in payload.get("layer_readiness", {}).items():
            if not isinstance(meta, dict):
                continue
            key = str(layer_key)
            self.layer_readiness_threshold[key] = float(meta.get("readiness_threshold", self.readiness_threshold))
            self.layer_readiness_min_updates[key] = int(meta.get("readiness_min_updates", self.readiness_min_updates))
            if key not in self.layer_ema_loss and "ema_loss" in meta:
                self.layer_ema_loss[key] = float(meta.get("ema_loss", float("inf")))
            if key not in self.layer_ema_cosine_dist and "ema_cosine_dist" in meta:
                self.layer_ema_cosine_dist[key] = float(meta.get("ema_cosine_dist", float("nan")))
            if key not in self.layer_update_count and "update_count" in meta:
                self.layer_update_count[key] = int(meta.get("update_count", 0))
            if key not in self.layer_ready and "ready" in meta:
                self.layer_ready[key] = bool(meta.get("ready", False))

        # Migrate old raw-space alpha to logit-space for sigmoid interpretation.
        _alpha_format = payload.get("alpha_format", "raw")
        _raw_layers = payload.get("layers", {})
        if not isinstance(_raw_layers, dict):
            _raw_layers = {}
        if _alpha_format != "logit":
            for _lk, _lstate in _raw_layers.items():
                if not isinstance(_lstate, dict):
                    continue
                if "alpha" not in _lstate:
                    continue
                _alpha_tensor = _lstate["alpha"]
                if not torch.is_tensor(_alpha_tensor):
                    continue
                _alpha_val = float(_alpha_tensor.reshape(-1)[0].item())
                # Old checkpoints stored raw alpha (typically 0.01–0.5).
                # New code interprets self.alpha through sigmoid, so convert:
                # if the stored value looks like raw-space (0 < val < 1 and small),
                # convert to logit. Values outside (0,1) or already large-magnitude
                # are assumed to already be in logit-space.
                if 0.0 < _alpha_val < 1.0 and abs(_alpha_val) < 2.0:
                    _logit_val = math.log(_alpha_val / (1.0 - _alpha_val))
                    _lstate["alpha"] = torch.tensor(_logit_val, dtype=torch.float32)
                    logger.info(
                        "Flux2TTR: migrated alpha for %s from raw %.6g to logit %.6g",
                        _lk, _alpha_val, _logit_val,
                    )

        self.pending_state = _raw_layers
        for layer_key, layer in self.layers.items():
            pending = self.pending_state.get(layer_key)
            if pending:
                layer.load_state_dict(pending, strict=False)
            pending_optimizer_state = self.pending_optimizer_states.pop(layer_key, None)
            if isinstance(pending_optimizer_state, dict):
                optimizer = self.optimizers.get(layer_key)
                if optimizer is not None:
                    try:
                        optimizer.load_state_dict(pending_optimizer_state)
                    except Exception as exc:
                        logger.warning("Flux2TTR: failed to restore optimizer state for %s (%s).", layer_key, exc)

        for layer_key in set(list(self.layer_update_count.keys()) + list(self.layer_ema_loss.keys())):
            self._refresh_layer_ready(layer_key)

        self._current_step_swap_set = None
        self._current_step_id = None
        self._current_step_eligible_count = 0
        self._controller_cache_key = None
        self._controller_cache_mask = None
        self._controller_debug_last_step_key = None

        logger.info("Flux2TTR: loaded checkpoint from %s (%d layers).", path, len(self.pending_state))


def register_runtime(runtime: Flux2TTRRuntime) -> str:
    runtime_id = uuid.uuid4().hex
    _RUNTIME_REGISTRY[runtime_id] = runtime
    return runtime_id


def get_runtime(runtime_id: str) -> Optional[Flux2TTRRuntime]:
    return _RUNTIME_REGISTRY.get(runtime_id)


def unregister_runtime(runtime_id: str) -> None:
    _RUNTIME_REGISTRY.pop(runtime_id, None)


def _recover_runtime_from_config(cfg: dict) -> Optional[Flux2TTRRuntime]:
    if not isinstance(cfg, dict):
        return None

    feature_dim = int(cfg.get("feature_dim", 256))
    training_mode = bool(cfg.get("training_mode", False))
    training_total = max(0, int(cfg.get("training_steps_total", 0)))
    training_remaining = max(0, int(cfg.get("training_steps_remaining", training_total)))

    runtime = Flux2TTRRuntime(
        feature_dim=feature_dim,
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        training=training_mode,
        steps=training_total,
        scan_chunk_size=int(cfg.get("query_chunk_size", cfg.get("scan_chunk_size", _DEFAULT_Q_CHUNK))),
        key_chunk_size=int(cfg.get("key_chunk_size", _DEFAULT_K_CHUNK)),
        landmark_fraction=float(cfg.get("landmark_fraction", _DEFAULT_LANDMARK_FRACTION)),
        landmark_min=int(cfg.get("landmark_min", _DEFAULT_LANDMARK_MIN)),
        landmark_max=int(cfg.get("landmark_max", _DEFAULT_LANDMARK_MAX)),
        text_tokens_guess=int(cfg.get("text_tokens_guess", _DEFAULT_TEXT_TOKENS_GUESS)),
        alpha_init=float(cfg.get("alpha_init", 0.1)),
        alpha_max=float(cfg.get("alpha_max", 0.2)),
        gamma=float(cfg.get("gamma", 2.0)),
        alpha_lr_multiplier=float(cfg.get("alpha_lr_multiplier", _DEFAULT_ALPHA_LR_MUL)),
        phi_lr_multiplier=float(cfg.get("phi_lr_multiplier", _DEFAULT_PHI_LR_MUL)),
        training_query_token_cap=int(cfg.get("training_query_token_cap", _DEFAULT_TRAIN_QUERY_CAP)),
        replay_buffer_size=int(cfg.get("replay_buffer_size", _DEFAULT_REPLAY_BUFFER)),
        replay_offload_cpu=bool(cfg.get("replay_offload_cpu", _DEFAULT_REPLAY_OFFLOAD_CPU)),
        replay_storage_dtype=str(cfg.get("replay_storage_dtype", _DEFAULT_REPLAY_STORAGE_DTYPE)),
        replay_max_bytes=int(cfg.get("replay_max_bytes", _DEFAULT_REPLAY_MAX_BYTES)),
        train_steps_per_call=int(cfg.get("train_steps_per_call", _DEFAULT_TRAIN_STEPS_PER_CALL)),
        huber_beta=float(cfg.get("huber_beta", _DEFAULT_HUBER_BETA)),
        grad_clip_norm=float(cfg.get("grad_clip_norm", _DEFAULT_GRAD_CLIP)),
        readiness_threshold=float(cfg.get("readiness_threshold", _DEFAULT_READY_THRESHOLD)),
        readiness_min_updates=int(cfg.get("readiness_min_updates", _DEFAULT_READY_MIN_UPDATES)),
        enable_memory_reserve=bool(cfg.get("enable_memory_reserve", False)),
        layer_start=int(cfg.get("layer_start", -1)),
        layer_end=int(cfg.get("layer_end", -1)),
        inference_mixed_precision=bool(cfg.get("inference_mixed_precision", True)),
        training_preview_ttr=bool(cfg.get("training_preview_ttr", True)),
        cfg_scale=float(cfg.get("cfg_scale", _DEFAULT_CFG_SCALE)),
        min_swap_layers=int(cfg.get("min_swap_layers", _DEFAULT_MIN_SWAP_LAYERS)),
        max_swap_layers=int(cfg.get("max_swap_layers", _DEFAULT_MAX_SWAP_LAYERS)),
        comet_enabled=bool(cfg.get("comet_enabled", False)),
        comet_project_name=str(cfg.get("comet_project_name", "ttr-distillation")),
        comet_workspace=str(cfg.get("comet_workspace", "comet-workspace")),
        comet_experiment=str(cfg.get("comet_experiment", "")),
        comet_persist_experiment=bool(cfg.get("comet_persist_experiment", True)),
        comet_log_every=int(cfg.get("comet_log_every", _DEFAULT_COMET_LOG_EVERY)),
    )

    runtime.training_mode = training_mode
    runtime.training_preview_ttr = bool(cfg.get("training_preview_ttr", runtime.training_preview_ttr))
    runtime.training_steps_total = training_total
    runtime.steps_remaining = training_remaining
    runtime.training_enabled = bool(cfg.get("training", False)) and training_remaining > 0
    runtime.max_safe_inference_loss = float(cfg.get("max_safe_inference_loss", runtime.max_safe_inference_loss))
    runtime.cfg_scale = float(cfg.get("cfg_scale", runtime.cfg_scale))
    runtime.min_swap_layers = max(0, int(cfg.get("min_swap_layers", runtime.min_swap_layers)))
    runtime.max_swap_layers = int(cfg.get("max_swap_layers", runtime.max_swap_layers))
    runtime.comet_experiment = str(cfg.get("comet_experiment", runtime.comet_experiment)).strip()
    runtime.comet_persist_experiment = bool(cfg.get("comet_persist_experiment", runtime.comet_persist_experiment))
    runtime.comet_persist_experiment = bool(runtime.comet_persist_experiment and runtime.comet_experiment)
    runtime.comet_log_every = max(1, int(cfg.get("comet_log_every", runtime.comet_log_every)))

    checkpoint_path = (cfg.get("checkpoint_path") or "").strip()
    if checkpoint_path and os.path.isfile(checkpoint_path):
        runtime.load_checkpoint(checkpoint_path)
        runtime.training_mode = training_mode
        runtime.training_preview_ttr = bool(cfg.get("training_preview_ttr", runtime.training_preview_ttr))
        runtime.comet_enabled = bool(cfg.get("comet_enabled", runtime.comet_enabled))
        runtime.comet_project_name = str(cfg.get("comet_project_name", runtime.comet_project_name))
        runtime.comet_workspace = str(cfg.get("comet_workspace", runtime.comet_workspace))
        runtime.comet_experiment = str(cfg.get("comet_experiment", runtime.comet_experiment)).strip()
        runtime.comet_persist_experiment = bool(cfg.get("comet_persist_experiment", runtime.comet_persist_experiment))
        runtime.comet_persist_experiment = bool(runtime.comet_persist_experiment and runtime.comet_experiment)
        runtime.cfg_scale = float(cfg.get("cfg_scale", runtime.cfg_scale))
        runtime.min_swap_layers = max(0, int(cfg.get("min_swap_layers", runtime.min_swap_layers)))
        runtime.max_swap_layers = int(cfg.get("max_swap_layers", runtime.max_swap_layers))
        runtime.comet_log_every = max(1, int(cfg.get("comet_log_every", runtime.comet_log_every)))
    elif not training_mode:
        logger.warning(
            "Flux2TTR: cannot recover inference runtime without a valid checkpoint_path (got %r).",
            checkpoint_path,
        )
        return None

    controller_path = (cfg.get("controller_checkpoint_path") or "").strip()
    if controller_path and os.path.isfile(controller_path):
        if flux2_ttr_controller is None:
            logger.warning(
                "Flux2TTR: controller_checkpoint_path provided but controller module is unavailable: %s",
                controller_path,
            )
        else:
            try:
                controller = flux2_ttr_controller.load_controller_checkpoint(controller_path, map_location="cpu")
                cfg["controller"] = controller
                logger.info("Flux2TTR: loaded controller checkpoint: %s", controller_path)
            except Exception as exc:
                logger.warning("Flux2TTR: failed to load controller checkpoint %s (%s).", controller_path, exc)

    if not training_mode:
        runtime.training_enabled = False
        runtime.steps_remaining = 0
        runtime.training_updates_done = 0

    return runtime


def flux2_ttr_attention(q, k, v, pe, mask=None, transformer_options=None):
    cfg = transformer_options.get("flux2_ttr") if transformer_options else None
    original = _ORIGINAL_FLUX_ATTENTION.get("math")
    if not cfg or not cfg.get("enabled", False):
        if original is None:
            raise RuntimeError("Flux2TTR: original Flux attention is not available.")
        return original(q, k, v, pe, mask=mask, transformer_options=transformer_options)

    runtime_id = cfg.get("runtime_id")
    runtime = get_runtime(runtime_id)
    if runtime is None:
        recovered = _recover_runtime_from_config(cfg)
        if recovered is None:
            logger.warning("Flux2TTR: runtime_id=%s not found and recovery failed; falling back to original attention.", runtime_id)
            return original(q, k, v, pe, mask=mask, transformer_options=transformer_options)
        if isinstance(runtime_id, str) and runtime_id:
            _RUNTIME_REGISTRY[runtime_id] = recovered
        runtime = recovered
        logger.info("Flux2TTR: recovered runtime_id=%s from config/checkpoint.", runtime_id)

    return runtime.run_attention(
        q=q,
        k=k,
        v=v,
        pe=pe,
        mask=mask,
        transformer_options=transformer_options,
        fallback_attention=original,
    )


def patch_flux_attention() -> None:
    global _PATCH_DEPTH, _ORIGINAL_FLUX_ATTENTION
    if _PATCH_DEPTH == 0:
        import comfy.ldm.flux.math as flux_math
        import comfy.ldm.flux.layers as flux_layers

        _ORIGINAL_FLUX_ATTENTION["math"] = flux_math.attention
        _ORIGINAL_FLUX_ATTENTION["layers"] = flux_layers.attention
        flux_math.attention = flux2_ttr_attention
        flux_layers.attention = flux2_ttr_attention
        logger.info("Flux2TTR: Flux attention patched.")
    _PATCH_DEPTH += 1


def restore_flux_attention() -> None:
    global _PATCH_DEPTH, _ORIGINAL_FLUX_ATTENTION
    if _PATCH_DEPTH <= 0:
        return
    _PATCH_DEPTH -= 1
    if _PATCH_DEPTH > 0:
        return
    if not _ORIGINAL_FLUX_ATTENTION:
        return

    import comfy.ldm.flux.math as flux_math
    import comfy.ldm.flux.layers as flux_layers

    flux_math.attention = _ORIGINAL_FLUX_ATTENTION.get("math", flux_math.attention)
    flux_layers.attention = _ORIGINAL_FLUX_ATTENTION.get("layers", flux_layers.attention)
    _ORIGINAL_FLUX_ATTENTION = {}
    logger.info("Flux2TTR: Flux attention restored.")


def pre_run_callback(patcher) -> None:
    transformer_options = getattr(patcher, "model_options", {}).get("transformer_options", {})
    if not transformer_options:
        transformer_options = getattr(getattr(patcher, "model", None), "model_options", {}).get("transformer_options", {})
    cfg = transformer_options.get("flux2_ttr")
    if not cfg or not cfg.get("enabled", False):
        return
    patch_flux_attention()


def cleanup_callback(patcher) -> None:
    transformer_options = getattr(patcher, "model_options", {}).get("transformer_options", {})
    if not transformer_options:
        transformer_options = getattr(getattr(patcher, "model", None), "model_options", {}).get("transformer_options", {})
    cfg = transformer_options.get("flux2_ttr")
    if not cfg or not cfg.get("enabled", False):
        return

    runtime_id = cfg.get("runtime_id", "")
    runtime = get_runtime(runtime_id)
    if runtime is not None and cfg.get("training_mode", False):
        checkpoint_path = (cfg.get("checkpoint_path") or "").strip()
        if checkpoint_path:
            runtime.save_checkpoint(checkpoint_path)
    if runtime is not None:
        runtime.release_resources()
    if isinstance(runtime_id, str) and runtime_id:
        unregister_runtime(runtime_id)
    restore_flux_attention()
