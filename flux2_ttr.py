from __future__ import annotations

import logging
import math
import os
import random
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

_ORIGINAL_FLUX_ATTENTION: Dict[str, Any] = {}
_PATCH_DEPTH = 0
_RUNTIME_REGISTRY: Dict[str, "Flux2TTRRuntime"] = {}
_MEMORY_RESERVE_FACTOR = 1.1
_EMPIRICAL_TRAINING_FLOOR_BYTES = 3 * 1024 * 1024 * 1024
_DISTILL_METRIC_EPS = 1e-8
_EMA_DECAY = 0.9
_MAX_SAFE_INFERENCE_LOSS = 0.5

_DEFAULT_Q_CHUNK = 256
_DEFAULT_K_CHUNK = 1024
_DEFAULT_TRAIN_QUERY_CAP = 256
_DEFAULT_REPLAY_BUFFER = 64
_DEFAULT_TRAIN_STEPS_PER_CALL = 1
_DEFAULT_HUBER_BETA = 0.05
_DEFAULT_READY_THRESHOLD = 0.12
_DEFAULT_READY_MIN_UPDATES = 24
_DEFAULT_ALPHA_LR_MUL = 5.0
_DEFAULT_PHI_LR_MUL = 1.0
_DEFAULT_GRAD_CLIP = 1.0
_DEFAULT_LANDMARK_COUNT = 128
_DEFAULT_TEXT_TOKENS_GUESS = 77

try:
    from comfy import model_management
except Exception:
    model_management = None


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


def validate_feature_dim(feature_dim: int) -> int:
    dim = int(feature_dim)
    if dim < 128:
        raise ValueError(f"Flux2TTR: feature_dim must be >= 128 (got {dim}).")
    if dim % 256 != 0:
        raise ValueError(f"Flux2TTR: feature_dim must be a multiple of 256 (got {dim}).")
    return dim


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
    landmarks = min(nk, _DEFAULT_LANDMARK_COUNT)
    landmark_elems = bh * (q_chunk * landmarks + landmarks * head_dim + q_chunk * head_dim)

    total = kv_elems + ksum_elems + chunk_elems + landmark_elems
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
            dtype_size,
            _MEMORY_RESERVE_FACTOR,
        )
        if transformer_options.get("flux2_ttr_memory_reserved") == key:
            return
        transformer_options["flux2_ttr_memory_reserved"] = key

    try:
        model_management.free_memory(mem_bytes, q.device)
        logger.info(
            "Flux2TTR reserved ~%.2f MB for %s (q_chunk=%d k_chunk=%d)",
            mem_bytes / (1024 * 1024),
            "training" if training else "inference",
            runtime.query_chunk_size,
            runtime.key_chunk_size,
        )
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
        split_qk: bool = False,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.head_dim = int(head_dim)
        self.feature_dim = validate_feature_dim(feature_dim)
        self.eps = float(eps)
        self.qk_norm = bool(qk_norm)

        hidden = max(self.feature_dim, self.head_dim)
        self.phi_net_q = nn.Sequential(
            nn.Linear(self.head_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.feature_dim),
        )
        if split_qk:
            self.phi_net_k = nn.Sequential(
                nn.Linear(self.head_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, self.feature_dim),
            )
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


class Flux2HKRAttnLayer(nn.Module):
    def __init__(
        self,
        head_dim: int,
        feature_dim: int = 256,
        *,
        eps: float = 1e-6,
        query_chunk_size: int = _DEFAULT_Q_CHUNK,
        key_chunk_size: int = _DEFAULT_K_CHUNK,
        split_qk: bool = False,
        qk_norm: bool = True,
        landmark_count: int = _DEFAULT_LANDMARK_COUNT,
        text_tokens_guess: int = _DEFAULT_TEXT_TOKENS_GUESS,
        landmark_qk_norm: bool = False,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.head_dim = int(head_dim)
        self.feature_dim = validate_feature_dim(feature_dim)
        self.eps = float(eps)
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.key_chunk_size = max(1, int(key_chunk_size))
        self.landmark_count = max(1, int(landmark_count))
        self.text_tokens_guess = max(0, int(text_tokens_guess))
        self.landmark_qk_norm = bool(landmark_qk_norm)

        self.kernel = KernelRegressorAttention(
            head_dim=self.head_dim,
            feature_dim=self.feature_dim,
            eps=self.eps,
            split_qk=split_qk,
            qk_norm=qk_norm,
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

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

    def _select_landmarks(
        self,
        num_keys: int,
        device: torch.device,
        key_mask: Optional[torch.Tensor],
        text_token_count: Optional[int],
    ) -> torch.Tensor:
        if num_keys <= 0:
            return torch.empty((0,), device=device, dtype=torch.long)

        target = min(self.landmark_count, num_keys)
        text_count = self.text_tokens_guess if text_token_count is None else max(0, int(text_token_count))
        text_count = min(text_count, num_keys)

        text_idx = self._even_indices(0, text_count, min(text_count, target), device)
        remaining = max(0, target - text_idx.numel())
        image_idx = self._even_indices(text_count, num_keys, remaining, device)

        if text_idx.numel() == 0 and image_idx.numel() == 0:
            idx = torch.tensor([0], device=device, dtype=torch.long)
        else:
            idx = torch.unique(torch.cat([text_idx, image_idx], dim=0), sorted=True)

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
    ) -> torch.Tensor:
        batch, heads, _, _ = q.shape
        idx = self._select_landmarks(k.shape[2], q.device, key_mask, text_token_count)
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
        out_land = self._landmark_attention(q, k, v, key_mask=key_mask, text_token_count=text_token_count)

        self.last_den_min = float(self.kernel.last_den_min)
        alpha = self.alpha.to(dtype=v.dtype)
        return out_kernel + alpha.view(1, 1, 1, 1) * out_land


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
        landmark_count: int = _DEFAULT_LANDMARK_COUNT,
        text_tokens_guess: int = _DEFAULT_TEXT_TOKENS_GUESS,
        alpha_init: float = 0.1,
        alpha_lr_multiplier: float = _DEFAULT_ALPHA_LR_MUL,
        phi_lr_multiplier: float = _DEFAULT_PHI_LR_MUL,
        training_query_token_cap: int = _DEFAULT_TRAIN_QUERY_CAP,
        replay_buffer_size: int = _DEFAULT_REPLAY_BUFFER,
        train_steps_per_call: int = _DEFAULT_TRAIN_STEPS_PER_CALL,
        huber_beta: float = _DEFAULT_HUBER_BETA,
        grad_clip_norm: float = _DEFAULT_GRAD_CLIP,
        readiness_threshold: float = _DEFAULT_READY_THRESHOLD,
        readiness_min_updates: int = _DEFAULT_READY_MIN_UPDATES,
        enable_memory_reserve: bool = False,
        layer_start: int = -1,
        layer_end: int = -1,
        inference_mixed_precision: bool = True,
        training_preview_ttr: bool = True,
        comet_enabled: bool = False,
        comet_api_key: str = "",
        comet_project_name: str = "ttr-distillation",
        comet_workspace: str = "ken-simpson",
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
        self.landmark_count = max(1, int(landmark_count))
        self.text_tokens_guess = max(0, int(text_tokens_guess))
        self.alpha_init = float(alpha_init)
        self.alpha_lr_multiplier = max(0.0, float(alpha_lr_multiplier))
        self.phi_lr_multiplier = max(0.0, float(phi_lr_multiplier))

        self.training_query_token_cap = max(1, int(training_query_token_cap))
        self.replay_buffer_size = max(1, int(replay_buffer_size))
        self.train_steps_per_call = max(1, int(train_steps_per_call))
        self.huber_beta = max(1e-6, float(huber_beta))
        self.grad_clip_norm = max(0.0, float(grad_clip_norm))
        self.readiness_threshold = float(readiness_threshold)
        self.readiness_min_updates = max(0, int(readiness_min_updates))
        self.enable_memory_reserve = bool(enable_memory_reserve)

        self.layer_start = int(layer_start)
        self.layer_end = int(layer_end)
        self.inference_mixed_precision = bool(inference_mixed_precision)
        self.training_preview_ttr = bool(training_preview_ttr)

        self.comet_enabled = bool(comet_enabled)
        self.comet_api_key = str(comet_api_key or "")
        self.comet_project_name = str(comet_project_name or "ttr-distillation")
        self.comet_workspace = str(comet_workspace or "ken-simpson")

        self.max_safe_inference_loss = float(_MAX_SAFE_INFERENCE_LOSS)
        self.last_loss = float("nan")

        self.layers: Dict[str, Flux2HKRAttnLayer] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.pending_state: Dict[str, Dict[str, torch.Tensor]] = {}
        self.layer_specs: Dict[str, FluxLayerSpec] = {}
        self.replay_buffers: Dict[str, Deque[ReplaySample]] = {}
        self.layer_ema_loss: Dict[str, float] = {}
        self.layer_update_count: Dict[str, int] = {}
        self.layer_ready: Dict[str, bool] = {}
        self.layer_last_loss: Dict[str, float] = {}

        self.capture_remaining = 0

        self._layer_metric_latest: Dict[str, Dict[str, float]] = {}
        self._layer_metric_running: Dict[str, Dict[str, float]] = {}
        self._layer_metric_count: Dict[str, int] = {}
        self._comet_experiment = None
        self._comet_disabled = False
        self._warned_high_loss = False

    def release_resources(self) -> None:
        if self._comet_experiment is not None:
            try:
                self._comet_experiment.end()
            except Exception as exc:
                logger.warning("Flux2TTR: failed to end Comet experiment cleanly: %s", exc)
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

        self.replay_buffers.clear()
        self._layer_metric_latest.clear()
        self._layer_metric_running.clear()
        self._layer_metric_count.clear()
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

    def _ensure_layer(self, layer_key: str, head_dim: int, device: torch.device) -> Flux2HKRAttnLayer:
        layer = self.layers.get(layer_key)
        if layer is None:
            layer = Flux2HKRAttnLayer(
                head_dim=head_dim,
                feature_dim=self.feature_dim,
                query_chunk_size=self.query_chunk_size,
                key_chunk_size=self.key_chunk_size,
                landmark_count=self.landmark_count,
                text_tokens_guess=self.text_tokens_guess,
                alpha_init=self.alpha_init,
            ).to(device=device, dtype=torch.float32)
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
            self.layers[layer_key] = layer
            self.optimizers[layer_key] = optimizer
            self.replay_buffers.setdefault(layer_key, deque(maxlen=self.replay_buffer_size))
            logger.info(
                "Flux2TTR: created HKR layer %s (head_dim=%d feature_dim=%d landmarks=%d).",
                layer_key,
                head_dim,
                self.feature_dim,
                self.landmark_count,
            )
            return layer

        layer_device = next(layer.parameters()).device
        if layer_device != device:
            layer.to(device=device)
            optimizer = self.optimizers[layer_key]
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
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
                if torch.is_floating_point(value):
                    state[key] = value.to(device=device, dtype=target_dtype)
                else:
                    state[key] = value.to(device=device)

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

        api_key = self.comet_api_key.strip() or os.getenv("COMET_API_KEY", "").strip()
        if not api_key:
            logger.warning("Flux2TTR: Comet logging enabled but no API key configured; disabling Comet logging.")
            self._comet_disabled = True
            return None
        try:
            from comet_ml import start
        except Exception as exc:
            logger.warning("Flux2TTR: could not import comet_ml; disabling Comet logging (%s).", exc)
            self._comet_disabled = True
            return None

        try:
            experiment = start(
                api_key=api_key,
                project_name=self.comet_project_name,
                workspace=self.comet_workspace,
            )
            experiment.log_parameters(
                {
                    "learning_rate": float(self.learning_rate),
                    "feature_dim": int(self.feature_dim),
                    "query_chunk_size": int(self.query_chunk_size),
                    "key_chunk_size": int(self.key_chunk_size),
                    "landmark_count": int(self.landmark_count),
                    "training_query_token_cap": int(self.training_query_token_cap),
                    "replay_buffer_size": int(self.replay_buffer_size),
                    "train_steps_per_call": int(self.train_steps_per_call),
                    "huber_beta": float(self.huber_beta),
                    "readiness_threshold": float(self.readiness_threshold),
                    "readiness_min_updates": int(self.readiness_min_updates),
                }
            )
            self._comet_experiment = experiment
            logger.info(
                "Flux2TTR: Comet logging enabled (project=%s workspace=%s).",
                self.comet_project_name,
                self.comet_workspace,
            )
            return experiment
        except Exception as exc:
            logger.warning("Flux2TTR: failed to start Comet experiment; disabling Comet logging (%s).", exc)
            self._comet_disabled = True
            return None

    def _record_training_metrics(self, layer_key: str, metrics: Dict[str, float]) -> None:
        self._layer_metric_latest[layer_key] = dict(metrics)

        count = int(self._layer_metric_count.get(layer_key, 0)) + 1
        self._layer_metric_count[layer_key] = count
        running = self._layer_metric_running.setdefault(layer_key, {})
        for key, value in metrics.items():
            prev = running.get(key, float(value))
            running[key] = float(prev + (float(value) - prev) / count)

        experiment = self._ensure_comet_experiment()
        if experiment is None:
            return

        payload = {}
        for key, value in metrics.items():
            payload[f"flux2ttr/{layer_key}/{key}"] = float(value)
        for key, value in running.items():
            payload[f"flux2ttr/{layer_key}/avg_{key}"] = float(value)
        payload["flux2ttr/global/steps_remaining"] = float(self.steps_remaining)
        payload["flux2ttr/global/updates_done"] = float(self.training_updates_done)
        try:
            experiment.log_metrics(payload, step=int(self.training_updates_done))
        except Exception as exc:
            logger.warning("Flux2TTR: Comet metric logging failed; disabling Comet logging (%s).", exc)
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

    def _compute_distill_metrics(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        loss_value: float,
        layer_key: str,
    ) -> Dict[str, float]:
        diff = (student - teacher).float()
        teacher_f = teacher.float()
        student_f = student.float()

        mse = float(torch.mean(diff.square()).item())
        teacher_power = float(torch.mean(teacher_f.square()).item())
        nmse = mse / (teacher_power + _DISTILL_METRIC_EPS)

        student_flat = student_f.reshape(student_f.shape[0], -1)
        teacher_flat = teacher_f.reshape(teacher_f.shape[0], -1)
        cosine = torch.nn.functional.cosine_similarity(student_flat, teacher_flat, dim=1, eps=_DISTILL_METRIC_EPS).mean()

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
        }

    def _refresh_layer_ready(self, layer_key: str) -> bool:
        updates = int(self.layer_update_count.get(layer_key, 0))
        ema = float(self.layer_ema_loss.get(layer_key, float("inf")))
        ready = updates >= self.readiness_min_updates and ema <= self.readiness_threshold
        prev = self.layer_ready.get(layer_key)
        self.layer_ready[layer_key] = bool(ready)
        if prev is None or bool(prev) != bool(ready):
            logger.info(
                "Flux2TTR: layer readiness changed layer=%s ready=%s updates=%d ema=%.6g threshold=%.6g",
                layer_key,
                ready,
                updates,
                ema,
                self.readiness_threshold,
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
            )

        if not torch.isfinite(student).all():
            raise RuntimeError(f"Flux2TTR: non-finite student output on {layer_key}.")
        den_min = float(getattr(layer, "last_den_min", float("nan")))
        if math.isfinite(den_min) and den_min < layer.eps:
            logger.warning("Flux2TTR: denominator floor hit on %s (den_min=%.6g).", layer_key, den_min)

        return _flatten_heads(student).to(dtype=v.dtype)

    def _push_replay_sample(
        self,
        layer_key: str,
        q_sub: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        teacher_sub: torch.Tensor,
        key_mask: Optional[torch.Tensor],
        text_token_count: Optional[int],
    ) -> None:
        buf = self.replay_buffers.setdefault(layer_key, deque(maxlen=self.replay_buffer_size))
        sample = ReplaySample(
            q_sub=q_sub.detach(),
            k_full=k_full.detach(),
            v_full=v_full.detach(),
            teacher_sub=teacher_sub.detach(),
            key_mask=key_mask.detach() if key_mask is not None else None,
            text_token_count=text_token_count,
        )
        buf.append(sample)

    def _train_from_replay(self, layer_key: str, head_dim: int, device: torch.device) -> None:
        if not self.training_enabled or self.steps_remaining <= 0:
            return
        buf = self.replay_buffers.get(layer_key)
        if not buf:
            return

        layer = self._ensure_layer(layer_key, head_dim, device)
        self._ensure_layer_device(layer, device)
        optimizer = self.optimizers[layer_key]
        self._set_layer_dtype(layer_key, layer, torch.float32)
        self._ensure_layer_device(layer, device)
        layer.train()

        steps = min(self.train_steps_per_call, self.steps_remaining)
        for _ in range(steps):
            sample = random.choice(tuple(buf))
            q_sub = sample.q_sub.float().clone()
            k_full = sample.k_full.float().clone()
            v_full = sample.v_full.float().clone()
            teacher_sub = sample.teacher_sub.float().clone()
            key_mask = sample.key_mask

            student_sub = layer(
                q=q_sub,
                k=k_full,
                v=v_full,
                key_mask=key_mask,
                q_chunk=self.query_chunk_size,
                k_chunk=self.key_chunk_size,
                text_token_count=sample.text_token_count,
            )
            loss = F.smooth_l1_loss(student_sub, teacher_sub, beta=self.huber_beta)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(layer.parameters(), self.grad_clip_norm)
            optimizer.step()

            loss_value = float(loss.item())
            self.last_loss = loss_value
            self.layer_last_loss[layer_key] = loss_value

            prev_ema = self.layer_ema_loss.get(layer_key, loss_value)
            ema = _EMA_DECAY * prev_ema + (1.0 - _EMA_DECAY) * loss_value
            self.layer_ema_loss[layer_key] = float(ema)
            self.layer_update_count[layer_key] = int(self.layer_update_count.get(layer_key, 0)) + 1
            self._refresh_layer_ready(layer_key)

            self.steps_remaining -= 1
            self.training_updates_done += 1

            metrics = self._compute_distill_metrics(
                student=student_sub.detach(),
                teacher=teacher_sub.detach(),
                loss_value=loss_value,
                layer_key=layer_key,
            )
            self._record_training_metrics(layer_key, metrics)

            if (
                self.training_log_every > 0
                and (
                    self.training_updates_done % self.training_log_every == 0
                    or self.steps_remaining <= 0
                )
            ):
                logger.info(
                    (
                        "Flux2TTR distill progress: updates=%d/%d layer=%s "
                        "loss=%.6g ema=%.6g ready=%s remaining=%d"
                    ),
                    self.training_updates_done,
                    max(self.training_steps_total, self.training_updates_done),
                    layer_key,
                    loss_value,
                    float(self.layer_ema_loss.get(layer_key, float("nan"))),
                    bool(self.layer_ready.get(layer_key, False)),
                    self.steps_remaining,
                )

            if self.steps_remaining <= 0:
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

        teacher_out: Optional[torch.Tensor] = None
        if self.training_mode or not self.layer_ready.get(layer_key, False):
            teacher_out = self._teacher_from_fallback(fallback_attention, q, k, v, pe, mask, transformer_options)

        if self.training_mode:
            if self.training_enabled and self.steps_remaining > 0:
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
                )

                try:
                    with torch.inference_mode(False):
                        with torch.enable_grad():
                            self._train_from_replay(layer_key, head_dim, q_eff.device)
                except torch.OutOfMemoryError:
                    self.training_enabled = False
                    logger.warning("Flux2TTR training OOM on layer %s; disabling training for this run.", layer_key)
                    if q_eff.device.type == "cuda":
                        torch.cuda.empty_cache()

                if isinstance(transformer_options, dict):
                    cfg = transformer_options.get("flux2_ttr")
                    if isinstance(cfg, dict):
                        cfg["training_steps_remaining"] = int(self.steps_remaining)
                        cfg["training_updates_done"] = int(self.training_updates_done)

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
                    transformer_options=transformer_options,
                    reserve_memory=False,
                )
            except Exception as exc:
                logger.warning("Flux2TTR preview fallback on %s: %s", layer_key, exc)
                return teacher_out

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
        layer_states = {}
        for layer_key, layer in self.layers.items():
            layer_states[layer_key] = {k: v.detach().cpu() for k, v in layer.state_dict().items()}
        for layer_key, state in self.pending_state.items():
            if layer_key not in layer_states:
                layer_states[layer_key] = state

        return {
            "format": "flux2_ttr_v2",
            "feature_dim": self.feature_dim,
            "learning_rate": self.learning_rate,
            "training_mode": self.training_mode,
            "training_preview_ttr": self.training_preview_ttr,
            "comet_enabled": self.comet_enabled,
            "comet_project_name": self.comet_project_name,
            "comet_workspace": self.comet_workspace,
            "last_loss": self.last_loss,
            "query_chunk_size": self.query_chunk_size,
            "key_chunk_size": self.key_chunk_size,
            "landmark_count": self.landmark_count,
            "text_tokens_guess": self.text_tokens_guess,
            "alpha_init": self.alpha_init,
            "alpha_lr_multiplier": self.alpha_lr_multiplier,
            "phi_lr_multiplier": self.phi_lr_multiplier,
            "training_query_token_cap": self.training_query_token_cap,
            "replay_buffer_size": self.replay_buffer_size,
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
            "layer_specs": {
                key: {"num_heads": spec.num_heads, "head_dim": spec.head_dim}
                for key, spec in self.layer_specs.items()
            },
            "layer_ema_loss": dict(self.layer_ema_loss),
            "layer_update_count": dict(self.layer_update_count),
            "layer_ready": dict(self.layer_ready),
            "layer_last_loss": dict(self.layer_last_loss),
            "layers": layer_states,
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
        fmt = payload.get("format")
        if fmt != "flux2_ttr_v2":
            raise ValueError(f"Flux2TTR: unsupported checkpoint format in {path}: {fmt!r}")

        ckpt_feature_dim = int(payload.get("feature_dim", 0))
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
        self.last_loss = float(payload.get("last_loss", self.last_loss))

        self.query_chunk_size = max(1, int(payload.get("query_chunk_size", self.query_chunk_size)))
        self.key_chunk_size = max(1, int(payload.get("key_chunk_size", self.key_chunk_size)))
        self.landmark_count = max(1, int(payload.get("landmark_count", self.landmark_count)))
        self.text_tokens_guess = max(0, int(payload.get("text_tokens_guess", self.text_tokens_guess)))
        self.alpha_init = float(payload.get("alpha_init", self.alpha_init))
        self.alpha_lr_multiplier = float(payload.get("alpha_lr_multiplier", self.alpha_lr_multiplier))
        self.phi_lr_multiplier = float(payload.get("phi_lr_multiplier", self.phi_lr_multiplier))

        self.training_query_token_cap = max(1, int(payload.get("training_query_token_cap", self.training_query_token_cap)))
        self.replay_buffer_size = max(1, int(payload.get("replay_buffer_size", self.replay_buffer_size)))
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
        self.layer_update_count = {str(k): int(v) for k, v in payload.get("layer_update_count", {}).items()}
        self.layer_ready = {str(k): bool(v) for k, v in payload.get("layer_ready", {}).items()}
        self.layer_last_loss = {str(k): float(v) for k, v in payload.get("layer_last_loss", {}).items()}

        self.pending_state = payload.get("layers", {})
        for layer_key, layer in self.layers.items():
            pending = self.pending_state.get(layer_key)
            if pending:
                layer.load_state_dict(pending, strict=False)

        for layer_key in set(list(self.layer_update_count.keys()) + list(self.layer_ema_loss.keys())):
            self._refresh_layer_ready(layer_key)

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
        landmark_count=int(cfg.get("landmark_count", _DEFAULT_LANDMARK_COUNT)),
        text_tokens_guess=int(cfg.get("text_tokens_guess", _DEFAULT_TEXT_TOKENS_GUESS)),
        alpha_init=float(cfg.get("alpha_init", 0.1)),
        alpha_lr_multiplier=float(cfg.get("alpha_lr_multiplier", _DEFAULT_ALPHA_LR_MUL)),
        phi_lr_multiplier=float(cfg.get("phi_lr_multiplier", _DEFAULT_PHI_LR_MUL)),
        training_query_token_cap=int(cfg.get("training_query_token_cap", _DEFAULT_TRAIN_QUERY_CAP)),
        replay_buffer_size=int(cfg.get("replay_buffer_size", _DEFAULT_REPLAY_BUFFER)),
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
        comet_enabled=bool(cfg.get("comet_enabled", False)),
        comet_project_name=str(cfg.get("comet_project_name", "ttr-distillation")),
        comet_workspace=str(cfg.get("comet_workspace", "ken-simpson")),
    )

    runtime.training_mode = training_mode
    runtime.training_preview_ttr = bool(cfg.get("training_preview_ttr", runtime.training_preview_ttr))
    runtime.training_steps_total = training_total
    runtime.steps_remaining = training_remaining
    runtime.training_enabled = bool(cfg.get("training", False)) and training_remaining > 0
    runtime.max_safe_inference_loss = float(cfg.get("max_safe_inference_loss", runtime.max_safe_inference_loss))

    checkpoint_path = (cfg.get("checkpoint_path") or "").strip()
    if checkpoint_path and os.path.isfile(checkpoint_path):
        runtime.load_checkpoint(checkpoint_path)
        runtime.training_mode = training_mode
        runtime.training_preview_ttr = bool(cfg.get("training_preview_ttr", runtime.training_preview_ttr))
        runtime.comet_enabled = bool(cfg.get("comet_enabled", runtime.comet_enabled))
        runtime.comet_project_name = str(cfg.get("comet_project_name", runtime.comet_project_name))
        runtime.comet_workspace = str(cfg.get("comet_workspace", runtime.comet_workspace))
    elif not training_mode:
        logger.warning(
            "Flux2TTR: cannot recover inference runtime without a valid checkpoint_path (got %r).",
            checkpoint_path,
        )
        return None

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
