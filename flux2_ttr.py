from __future__ import annotations

import logging
import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch
from einops import rearrange
from torch import nn

logger = logging.getLogger(__name__)

_ORIGINAL_FLUX_ATTENTION: Dict[str, Any] = {}
_PATCH_DEPTH = 0
_RUNTIME_REGISTRY: Dict[str, "Flux2TTRRuntime"] = {}
_MEMORY_RESERVE_FACTOR = 1.1
_TRAINING_SCAN_CHUNK_CAP = 64
_EMPIRICAL_TRAINING_FLOOR_BYTES = 3 * 1024 * 1024 * 1024
_TRAINING_TOKEN_CAP = 128
_MAX_SAFE_INFERENCE_LOSS = 0.5
_DISTILL_METRIC_EPS = 1e-8
_ATTN_METRIC_QUERY_SAMPLES = 16
_ATTN_METRIC_KEY_SAMPLES = 64
_ATTN_METRIC_TOPK = 8

try:
    from comfy import model_management
except Exception:
    model_management = None


def validate_feature_dim(feature_dim: int) -> int:
    dim = int(feature_dim)
    if dim < 128:
        raise ValueError(f"Flux2TTR: feature_dim must be >= 128 (got {dim}).")
    if dim % 256 != 0:
        raise ValueError(f"Flux2TTR: feature_dim must be a multiple of 256 (got {dim}).")
    return dim


@dataclass
class FluxLayerSpec:
    layer_key: str
    num_heads: int
    head_dim: int


class TTRCell(nn.Module):
    def __init__(self, feature_dim: int, value_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.value_dim = int(value_dim)
        self.last_used_chunk = 1

    def forward_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_assoc = torch.einsum("bf,bv->bfv", k, v)
        w_new = w_prev + kv_assoc
        output = torch.einsum("bf,bfv->bv", q, w_new)
        return output, w_new

    def _scan_impl(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: int) -> torch.Tensor:
        batch_heads, seq_len, _ = q.shape
        if seq_len == 0:
            return v.new_zeros((batch_heads, 0, self.value_dim))

        chunk = max(1, int(chunk_size))
        w_state = q.new_zeros((batch_heads, self.feature_dim, self.value_dim))
        inv_counts = torch.arange(1, seq_len + 1, device=q.device, dtype=q.dtype).view(1, seq_len, 1)
        output_chunks = []
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            q_chunk = q[:, start:end, :]
            k_chunk = k[:, start:end, :]
            v_chunk = v[:, start:end, :]

            # Prefix update of W per token inside the chunk.
            kv_assoc = torch.einsum("bnf,bnv->bnfv", k_chunk, v_chunk)
            w_prefix = kv_assoc.cumsum(dim=1) + w_state.unsqueeze(1)
            out_chunk = torch.einsum("bnf,bnfv->bnv", q_chunk, w_prefix)
            out_chunk = out_chunk * inv_counts[:, start:end, :].reciprocal()
            output_chunks.append(out_chunk)
            w_state = w_prefix[:, -1, :, :]
        self.last_used_chunk = int(chunk)
        return torch.cat(output_chunks, dim=1)

    def scan(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
        chunk = max(1, int(chunk_size))
        while True:
            try:
                return self._scan_impl(q, k, v, chunk)
            except torch.OutOfMemoryError:
                if q.device.type != "cuda" or chunk <= 1:
                    raise
                next_chunk = max(1, chunk // 2)
                if next_chunk == chunk:
                    raise
                logger.warning("Flux2TTR scan OOM at chunk_size=%d; retrying with chunk_size=%d", chunk, next_chunk)
                chunk = next_chunk
                torch.cuda.empty_cache()


class TTRFluxLayer(nn.Module):
    def __init__(self, head_dim: int, feature_dim: int = 256, scan_chunk_tokens: int = 128):
        super().__init__()
        self.head_dim = int(head_dim)
        self.feature_dim = validate_feature_dim(feature_dim)
        self.scan_chunk_tokens = max(1, int(scan_chunk_tokens))
        self.last_used_scan_chunk = self.scan_chunk_tokens
        self.phi_net = nn.Sequential(
            nn.Linear(self.head_dim, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.regressor = TTRCell(self.feature_dim, self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: Optional[int] = None) -> torch.Tensor:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("TTRFluxLayer expects q/k/v with shape [B, H, N, D].")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("TTRFluxLayer expects q/k/v shapes to match.")
        if q.shape[-1] != self.head_dim:
            raise ValueError(f"TTRFluxLayer expected head_dim={self.head_dim}, got {q.shape[-1]}.")

        batch, heads, _, _ = q.shape
        q_flat = rearrange(q, "b h n d -> (b h) n d")
        k_flat = rearrange(k, "b h n d -> (b h) n d")
        v_flat = rearrange(v, "b h n d -> (b h) n d")

        # Training path keeps q/k projections separate to reduce peak activation memory.
        if self.training:
            q_phi = self.phi_net(q_flat)
            k_phi = self.phi_net(k_flat)
        else:
            # One shared pass through phi_net reduces kernel launch overhead.
            qk_phi = self.phi_net(torch.cat([q_flat, k_flat], dim=0))
            q_phi, k_phi = torch.split(qk_phi, q_flat.shape[0], dim=0)

        effective_chunk = self.scan_chunk_tokens if chunk_size is None else max(1, int(chunk_size))

        out_fwd = self.regressor.scan(q_phi, k_phi, v_flat, chunk_size=effective_chunk)
        used_fwd = int(getattr(self.regressor, "last_used_chunk", effective_chunk))
        out_rev = self.regressor.scan(
            torch.flip(q_phi, dims=(1,)),
            torch.flip(k_phi, dims=(1,)),
            torch.flip(v_flat, dims=(1,)),
            chunk_size=effective_chunk,
        )
        out_rev = torch.flip(out_rev, dims=(1,))
        used_rev = int(getattr(self.regressor, "last_used_chunk", effective_chunk))
        self.last_used_scan_chunk = max(1, min(used_fwd, used_rev))

        out = out_fwd + out_rev
        return rearrange(out, "(b h) n d -> b h n d", b=batch, h=heads)


def _key_mask_from_mask(mask: Optional[torch.Tensor], batch: int, keys: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        mask = mask != 0
    if mask.ndim == 2:
        if mask.shape[0] == batch and mask.shape[1] == keys:
            return mask
        return None
    if mask.ndim == 3:
        # [B, Nq, Nk] -> key valid if any query can attend to key
        if mask.shape[0] == batch and mask.shape[2] == keys:
            return mask.any(dim=1)
        return None
    if mask.ndim == 4:
        # [B, H, Nq, Nk] -> key valid if any head/query can attend to key
        if mask.shape[0] == batch and mask.shape[3] == keys:
            return mask.any(dim=1).any(dim=1)
        return None
    return None


def _softmax_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
    key_mask = _key_mask_from_mask(mask, q.shape[0], k.shape[2])
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask[:, None, None, :].to(device=scores.device), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("b h i j, b h j d -> b h i d", attn, v)


def _flatten_heads(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], x.shape[1] * x.shape[3])


def _extract_latent_samples(latents: Any) -> torch.Tensor:
    if torch.is_tensor(latents):
        return latents
    if isinstance(latents, dict):
        for key in ("samples", "latent", "latents"):
            value = latents.get(key)
            if torch.is_tensor(value):
                return value
    raise ValueError("Flux2TTR: LATENT input must contain a tensor under 'samples'.")


def _find_first_tensor(obj: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            tensor = _find_first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(obj, dict):
        for value in obj.values():
            tensor = _find_first_tensor(value)
            if tensor is not None:
                return tensor
        return None
    return None


def _conditioning_vector(conditioning: Any, batch: int, device: torch.device) -> Optional[torch.Tensor]:
    tensor = _find_first_tensor(conditioning)
    if tensor is None:
        return None
    vec = tensor.float()
    while vec.ndim > 2:
        vec = vec.mean(dim=1)
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    if vec.shape[0] != batch:
        if vec.shape[0] == 1:
            vec = vec.expand(batch, vec.shape[1])
        else:
            vec = vec[:batch]
            if vec.shape[0] < batch:
                pad = vec[-1:].expand(batch - vec.shape[0], vec.shape[1])
                vec = torch.cat([vec, pad], dim=0)
    return vec.to(device=device)


def _fit_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    if x.shape[-1] == target_dim:
        return x
    if x.shape[-1] > target_dim:
        return x[..., :target_dim]
    repeat = math.ceil(target_dim / x.shape[-1])
    return x.repeat_interleave(repeat, dim=-1)[..., :target_dim]


def _projection_matrix(in_dim: int, out_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn((in_dim, out_dim), generator=generator, device=device, dtype=torch.float32) / math.sqrt(max(1, in_dim))


def _estimate_flux2_ttr_memory_bytes(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    feature_dim: int,
    chunk_size: int,
    dtype_size: int,
    training: bool,
) -> int:
    bh = batch * heads
    seq = max(1, int(seq_len))
    chunk = min(seq, max(1, int(chunk_size)))

    # Core forward tensors for phi projection + bidirectional scan.
    cat_qk_elems = 2 * bh * seq * head_dim
    phi_elems = 2 * bh * seq * feature_dim
    state_elems = bh * feature_dim * head_dim
    chunk_scan_elems = bh * chunk * feature_dim * head_dim * 2 + bh * chunk * head_dim
    output_elems = 2 * bh * seq * head_dim

    total_elems = cat_qk_elems + phi_elems + state_elems + chunk_scan_elems + output_elems

    if training:
        # Training keeps additional buffers for q/k/v clones and autograd.
        train_clones = 3 * bh * seq * head_dim
        total_elems += train_clones
        total_elems = int(total_elems * 1.4)

    return int(total_elems * dtype_size)


def _maybe_reserve_memory(
    runtime: "Flux2TTRRuntime",
    q: torch.Tensor,
    transformer_options: Optional[dict],
    training: bool,
    dtype_accum: torch.dtype,
    layer_key: Optional[str] = None,
) -> None:
    if model_management is None:
        return
    if q.device.type == "cpu":
        return

    batch, heads, seq_len, head_dim = q.shape
    if training:
        seq_len = min(seq_len, max(1, int(runtime.training_token_cap)))
    dtype_size = torch.tensor([], dtype=dtype_accum).element_size()
    effective_chunk = runtime._effective_scan_chunk(training, layer_key=layer_key)
    mem_bytes = _estimate_flux2_ttr_memory_bytes(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        feature_dim=runtime.feature_dim,
        chunk_size=effective_chunk,
        dtype_size=dtype_size,
        training=training,
    )
    if training:
        # Empirical floor: feature_dim=256 typically needs around 3GB during online distillation.
        scale = (runtime.feature_dim / 256.0) * (head_dim / 128.0) * max(1.0, (batch * heads) / 24.0)
        mem_bytes = max(mem_bytes, int(_EMPIRICAL_TRAINING_FLOOR_BYTES * scale))
    mem_bytes = int(mem_bytes * _MEMORY_RESERVE_FACTOR)
    if mem_bytes <= 0:
        return

    if transformer_options is not None:
        key = (
            "train" if training else "infer",
            batch,
            heads,
            seq_len,
            head_dim,
            runtime.feature_dim,
            effective_chunk,
            dtype_size,
            _MEMORY_RESERVE_FACTOR,
        )
        if transformer_options.get("flux2_ttr_memory_reserved") == key:
            return
        transformer_options["flux2_ttr_memory_reserved"] = key

    try:
        model_management.free_memory(mem_bytes, q.device)
        logger.info(
            "Flux2TTR reserved ~%.2f MB for %s (chunk=%d)",
            mem_bytes / (1024 * 1024),
            "training" if training else "inference",
            effective_chunk,
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


class Flux2TTRRuntime:
    def __init__(
        self,
        feature_dim: int,
        learning_rate: float,
        training: bool,
        steps: int,
        scan_chunk_size: int = 128,
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
        self.scan_chunk_size = max(1, int(scan_chunk_size))
        self.training_scan_chunk_size = max(1, min(self.scan_chunk_size, _TRAINING_SCAN_CHUNK_CAP))
        self.training_token_cap = int(_TRAINING_TOKEN_CAP)
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
        self.layers: Dict[str, TTRFluxLayer] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.pending_state: Dict[str, Dict[str, torch.Tensor]] = {}
        self.layer_specs: Dict[str, FluxLayerSpec] = {}
        self._projection_cache: Dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._scan_chunk_override_train: Dict[str, int] = {}
        self._scan_chunk_override_infer: Dict[str, int] = {}
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

        for layer in self.layers.values():
            try:
                layer.to(device="cpu")
            except Exception:
                pass
        self.layers.clear()

        for optimizer in self.optimizers.values():
            try:
                optimizer.state.clear()
            except Exception:
                pass
        self.optimizers.clear()

        self._projection_cache.clear()
        self._scan_chunk_override_train.clear()
        self._scan_chunk_override_infer.clear()
        self._layer_metric_latest.clear()
        self._layer_metric_running.clear()
        self._layer_metric_count.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def register_layer_specs(self, specs: Iterable[FluxLayerSpec]) -> None:
        for spec in specs:
            self.layer_specs[spec.layer_key] = spec

    def _ensure_layer(self, layer_key: str, head_dim: int, device: torch.device) -> TTRFluxLayer:
        layer = self.layers.get(layer_key)
        if layer is None:
            layer = TTRFluxLayer(
                head_dim=head_dim,
                feature_dim=self.feature_dim,
                scan_chunk_tokens=self.scan_chunk_size,
            ).to(device=device, dtype=torch.float32)
            optimizer = torch.optim.AdamW(layer.parameters(), lr=self.learning_rate)
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
            logger.info("Flux2TTR: created TTR layer %s (head_dim=%d feature_dim=%d).", layer_key, head_dim, self.feature_dim)
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

    def _set_layer_dtype(self, layer_key: str, layer: TTRFluxLayer, target_dtype: torch.dtype) -> None:
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

    def _resolve_inference_dtype(self, q: torch.Tensor) -> torch.dtype:
        if not self.inference_mixed_precision:
            return torch.float32
        if q.device.type == "cuda" and q.dtype in (torch.float16, torch.bfloat16):
            return q.dtype
        return torch.float32

    def _effective_scan_chunk(self, training: bool, layer_key: Optional[str] = None) -> int:
        base = int(self.training_scan_chunk_size) if training else int(self.scan_chunk_size)
        if not layer_key:
            return base
        override_map = self._scan_chunk_override_train if training else self._scan_chunk_override_infer
        override = override_map.get(layer_key)
        if override is None:
            return base
        return max(1, min(base, int(override)))

    def _update_scan_chunk_override(self, layer_key: str, training: bool, used_chunk: int) -> None:
        used = max(1, int(used_chunk))
        override_map = self._scan_chunk_override_train if training else self._scan_chunk_override_infer
        prev = override_map.get(layer_key)
        if prev is None or used < prev:
            override_map[layer_key] = used
            logger.info(
                "Flux2TTR: adaptive scan chunk for %s (%s) -> %d",
                layer_key,
                "training" if training else "inference",
                used,
            )

    def _select_training_token_indices(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        cap = max(0, int(self.training_token_cap))
        if cap <= 0 or seq_len <= cap:
            return None
        return (
            torch.linspace(0, seq_len - 1, steps=cap, device=device, dtype=torch.float32)
            .round()
            .to(dtype=torch.long)
        )

    @staticmethod
    def _sample_even_indices(length: int, max_count: int, device: torch.device) -> torch.Tensor:
        size = max(1, min(int(max_count), int(length)))
        if size >= length:
            return torch.arange(length, device=device, dtype=torch.long)
        return (
            torch.linspace(0, length - 1, steps=size, device=device, dtype=torch.float32)
            .round()
            .to(dtype=torch.long)
        )

    def _compute_attention_alignment_metrics(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        student: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        q_idx = self._sample_even_indices(q.shape[2], _ATTN_METRIC_QUERY_SAMPLES, q.device)
        k_idx = self._sample_even_indices(k.shape[2], _ATTN_METRIC_KEY_SAMPLES, k.device)

        q_sub = q[:, :, q_idx, :].float()
        k_sub = k[:, :, k_idx, :].float()
        v_sub = v[:, :, k_idx, :].float()
        student_sub = student[:, :, q_idx, :].float()

        key_mask = _key_mask_from_mask(mask, q.shape[0], k.shape[2])
        if key_mask is not None:
            key_mask = key_mask[:, k_idx]
            # Avoid all-masked rows causing all -inf softmax.
            if not bool(key_mask.any(dim=-1).all()):
                key_mask = None

        scale = q_sub.shape[-1] ** -0.5
        teacher_scores = torch.einsum("b h i d, b h j d -> b h i j", q_sub, k_sub) * scale
        if key_mask is not None:
            teacher_scores = teacher_scores.masked_fill(~key_mask[:, None, None, :].to(device=teacher_scores.device), float("-inf"))
        teacher_probs = torch.softmax(teacher_scores, dim=-1)

        # Student does not expose explicit attention maps; induce weights by
        # scoring student outputs against value vectors on sampled keys.
        student_scores = torch.einsum("b h i d, b h j d -> b h i j", student_sub, v_sub)
        if key_mask is not None:
            student_scores = student_scores.masked_fill(~key_mask[:, None, None, :].to(device=student_scores.device), float("-inf"))
        student_probs = torch.softmax(student_scores, dim=-1)

        kl = (
            teacher_probs
            * ((teacher_probs + _DISTILL_METRIC_EPS).log() - (student_probs + _DISTILL_METRIC_EPS).log())
        ).sum(dim=-1).mean()

        topk = max(1, min(_ATTN_METRIC_TOPK, teacher_probs.shape[-1]))
        teacher_topk = torch.topk(teacher_probs, k=topk, dim=-1).indices
        student_topk = torch.topk(student_probs, k=topk, dim=-1).indices
        overlap = (teacher_topk[..., :, None] == student_topk[..., None, :]).any(dim=-1).float().mean()

        return {
            "attn_kl_div": float(kl.item()),
            "attn_topk_overlap": float(overlap.item()),
        }

    def _compute_distill_metrics(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        loss_value: float,
    ) -> Dict[str, float]:
        diff = (student - teacher).float()
        teacher_f = teacher.float()
        student_f = student.float()
        mse = float(loss_value)
        teacher_power = float(torch.mean(teacher_f.square()).item())
        nmse = mse / (teacher_power + _DISTILL_METRIC_EPS)

        student_flat = student_f.reshape(student_f.shape[0], -1)
        teacher_flat = teacher_f.reshape(teacher_f.shape[0], -1)
        cosine = torch.nn.functional.cosine_similarity(
            student_flat,
            teacher_flat,
            dim=1,
            eps=_DISTILL_METRIC_EPS,
        ).mean()
        student_norm = torch.linalg.vector_norm(student_flat, dim=1).mean()
        teacher_norm = torch.linalg.vector_norm(teacher_flat, dim=1).mean()
        norm_ratio = student_norm / (teacher_norm + _DISTILL_METRIC_EPS)
        mean_ratio = student_f.mean() / (teacher_f.mean() + _DISTILL_METRIC_EPS)
        std_ratio = student_f.std(unbiased=False) / (teacher_f.std(unbiased=False) + _DISTILL_METRIC_EPS)

        abs_err = diff.abs().reshape(-1)
        p95_abs_err = torch.quantile(abs_err, 0.95)
        p99_abs_err = torch.quantile(abs_err, 0.99)

        metrics = {
            "loss": mse,
            "nmse": float(nmse),
            "cosine_similarity": float(cosine.item()),
            "norm_ratio": float(norm_ratio.item()),
            "mean_ratio": float(mean_ratio.item()),
            "std_ratio": float(std_ratio.item()),
            "p95_abs_error": float(p95_abs_err.item()),
            "p99_abs_error": float(p99_abs_err.item()),
        }
        metrics.update(self._compute_attention_alignment_metrics(q=q, k=k, v=v, student=student, mask=mask))
        return metrics

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
                    "steps": int(self.training_steps_total),
                    "feature_dim": int(self.feature_dim),
                    "scan_chunk_size": int(self.scan_chunk_size),
                    "training_scan_chunk_size": int(self.training_scan_chunk_size),
                    "training_token_cap": int(self.training_token_cap),
                    "layer_start": int(self.layer_start),
                    "layer_end": int(self.layer_end),
                    "inference_mixed_precision": bool(self.inference_mixed_precision),
                    "training_preview_ttr": bool(self.training_preview_ttr),
                    "max_safe_inference_loss": float(self.max_safe_inference_loss),
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

    def _ensure_projection(
        self,
        layer_key: str,
        in_dim: int,
        out_dim: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = (layer_key, in_dim)
        cached = self._projection_cache.get(cache_key)
        if cached is not None and cached[0].shape[1] == out_dim and cached[0].device == device:
            return cached

        base_seed = abs(hash((layer_key, in_dim, out_dim))) % (2**31 - 1)
        q_proj = _projection_matrix(in_dim, out_dim, base_seed + 11, device)
        k_proj = _projection_matrix(in_dim, out_dim, base_seed + 23, device)
        v_proj = _projection_matrix(in_dim, out_dim, base_seed + 37, device)
        self._projection_cache[cache_key] = (q_proj, k_proj, v_proj)
        return q_proj, k_proj, v_proj

    def _layer_key_from_options(self, transformer_options: Optional[dict]) -> str:
        if transformer_options is None:
            return "single:0"
        block_type = transformer_options.get("block_type", "single")
        block_index = transformer_options.get("block_index", 0)
        if isinstance(block_index, int):
            return f"{block_type}:{block_index}"
        return f"{block_type}:0"

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
        transformer_options: Optional[dict],
        reserve_memory: bool = True,
    ) -> torch.Tensor:
        layer = self._ensure_layer(layer_key, head_dim, q_eff.device)
        layer.eval()
        inference_dtype = self._resolve_inference_dtype(q_eff)
        if reserve_memory:
            _maybe_reserve_memory(
                self,
                q_eff,
                transformer_options,
                training=False,
                dtype_accum=inference_dtype,
                layer_key=layer_key,
            )
        self._set_layer_dtype(layer_key, layer, inference_dtype)
        q_in = q_eff.to(dtype=inference_dtype)
        k_in = k_eff.to(dtype=inference_dtype)
        v_in = v.to(dtype=inference_dtype)
        with torch.no_grad():
            student = layer(
                q_in,
                k_in,
                v_in,
                chunk_size=self._effective_scan_chunk(training=False, layer_key=layer_key),
            )
        self._update_scan_chunk_override(
            layer_key,
            training=False,
            used_chunk=getattr(layer, "last_used_scan_chunk", self._effective_scan_chunk(False, layer_key)),
        )
        return _flatten_heads(student).to(dtype=v.dtype)

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

        if self.training_mode:
            with torch.no_grad():
                teacher_out = self._teacher_from_fallback(fallback_attention, q, k, v, pe, mask, transformer_options)
            if self.training_enabled and self.steps_remaining > 0:
                _maybe_reserve_memory(
                    self,
                    q_eff,
                    transformer_options,
                    training=True,
                    dtype_accum=torch.float32,
                    layer_key=layer_key,
                )
                try:
                    with torch.inference_mode(False):
                        with torch.enable_grad():
                            layer = self._ensure_layer(layer_key, head_dim, q_eff.device)
                            self._set_layer_dtype(layer_key, layer, torch.float32)
                            layer.train()
                            # ComfyUI may call us under torch.inference_mode(); clone converts
                            # inference tensors to normal tensors that autograd can save.
                            q_in = q_eff.float().clone()
                            k_in = k_eff.float().clone()
                            v_in = v.float().clone()
                            teacher = (
                                teacher_out.view(q.shape[0], q.shape[2], q.shape[1], q.shape[3])
                                .permute(0, 2, 1, 3)
                                .float()
                                .clone()
                            )
                            idx = self._select_training_token_indices(q_in.shape[2], q_in.device)
                            if idx is not None:
                                q_in = q_in[:, :, idx, :]
                                k_in = k_in[:, :, idx, :]
                                v_in = v_in[:, :, idx, :]
                                teacher = teacher[:, :, idx, :]

                            student = layer(
                                q_in,
                                k_in,
                                v_in,
                                chunk_size=self._effective_scan_chunk(training=True, layer_key=layer_key),
                            )
                            self._update_scan_chunk_override(
                                layer_key,
                                training=True,
                                used_chunk=getattr(layer, "last_used_scan_chunk", self._effective_scan_chunk(True, layer_key)),
                            )
                            loss = torch.nn.functional.mse_loss(student, teacher)
                            optimizer = self.optimizers[layer_key]
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()
                            with torch.no_grad():
                                try:
                                    metrics = self._compute_distill_metrics(
                                        student=student.detach(),
                                        teacher=teacher.detach(),
                                        q=q_in.detach(),
                                        k=k_in.detach(),
                                        v=v_in.detach(),
                                        mask=mask,
                                        loss_value=float(loss.item()),
                                    )
                                except Exception as metric_exc:
                                    logger.warning("Flux2TTR: failed to compute distill metrics for %s: %s", layer_key, metric_exc)
                                    metrics = {"loss": float(loss.item())}
                    self.last_loss = float(loss.item())
                    self.steps_remaining -= 1
                    self.training_updates_done += 1
                    self._record_training_metrics(layer_key, metrics)
                    if (
                        self.training_log_every > 0
                        and (
                            self.training_updates_done % self.training_log_every == 0
                            or self.steps_remaining <= 0
                        )
                    ):
                        current = self._layer_metric_latest.get(layer_key, {})
                        logger.info(
                            (
                                "Flux2TTR distill progress: updates=%d/%d loss=%.6g nmse=%.6g "
                                "cos=%.6g norm_ratio=%.6g std_ratio=%.6g p95=%.6g p99=%.6g "
                                "attn_kl=%.6g topk=%.6g layer=%s remaining=%d"
                            ),
                            self.training_updates_done,
                            max(self.training_steps_total, self.training_updates_done),
                            self.last_loss,
                            float(current.get("nmse", float("nan"))),
                            float(current.get("cosine_similarity", float("nan"))),
                            float(current.get("norm_ratio", float("nan"))),
                            float(current.get("std_ratio", float("nan"))),
                            float(current.get("p95_abs_error", float("nan"))),
                            float(current.get("p99_abs_error", float("nan"))),
                            float(current.get("attn_kl_div", float("nan"))),
                            float(current.get("attn_topk_overlap", float("nan"))),
                            layer_key,
                            self.steps_remaining,
                        )
                    if isinstance(transformer_options, dict):
                        cfg = transformer_options.get("flux2_ttr")
                        if isinstance(cfg, dict):
                            cfg["training_steps_remaining"] = int(self.steps_remaining)
                            cfg["training_updates_done"] = int(self.training_updates_done)
                    if self.steps_remaining <= 0:
                        self.training_enabled = False
                        logger.info(
                            "Flux2TTR: online distillation reached configured steps; continuing %s for this run.",
                            "TTR preview output" if self.training_preview_ttr else "teacher passthrough",
                        )
                except torch.OutOfMemoryError:
                    self.training_enabled = False
                    logger.warning(
                        "Flux2TTR training OOM on layer %s; disabling training and continuing %s.",
                        layer_key,
                        "TTR preview output" if self.training_preview_ttr else "teacher passthrough",
                    )
                    if q_eff.device.type == "cuda":
                        torch.cuda.empty_cache()
            if not self.training_preview_ttr:
                return teacher_out
            try:
                return self._student_from_runtime(
                    q_eff=q_eff,
                    k_eff=k_eff,
                    v=v,
                    layer_key=layer_key,
                    head_dim=head_dim,
                    transformer_options=transformer_options,
                    reserve_memory=False,
                )
            except torch.OutOfMemoryError:
                logger.warning("Flux2TTR preview OOM on layer %s; falling back to teacher output for this call.", layer_key)
                if q_eff.device.type == "cuda":
                    torch.cuda.empty_cache()
                return teacher_out

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
            return self._teacher_from_fallback(fallback_attention, q, k, v, pe, mask, transformer_options)

        return self._student_from_runtime(
            q_eff=q_eff,
            k_eff=k_eff,
            v=v,
            layer_key=layer_key,
            head_dim=head_dim,
            transformer_options=transformer_options,
        )

    def calibrate_from_inputs(
        self,
        model: Any,
        latents: Any,
        conditioning: Any,
        steps: int,
        max_tokens: int = 256,
    ) -> float:
        train_steps = max(0, int(steps))
        if train_steps == 0:
            self.training_enabled = False
            self.steps_remaining = 0
            return 0.0 if math.isnan(self.last_loss) else float(self.last_loss)

        specs = infer_flux_single_layer_specs(model)
        if not specs:
            specs = [FluxLayerSpec(layer_key="single:0", num_heads=1, head_dim=128)]
            logger.warning("Flux2TTR: could not infer Flux single block specs, using fallback single:0 spec.")

        self.register_layer_specs(specs)

        samples = _extract_latent_samples(latents).float()
        if samples.ndim != 4:
            raise ValueError(f"Flux2TTR: expected latent tensor [B, C, H, W], got shape={tuple(samples.shape)}.")

        batch = samples.shape[0]
        token_features = samples.permute(0, 2, 3, 1).reshape(batch, -1, samples.shape[1])
        if token_features.shape[1] > max_tokens:
            idx = torch.linspace(
                0,
                token_features.shape[1] - 1,
                steps=max_tokens,
                device=token_features.device,
                dtype=torch.float32,
            ).round().long()
            token_features = token_features[:, idx, :]

        cond_vec = _conditioning_vector(conditioning, batch, token_features.device)
        if cond_vec is not None:
            cond_vec = _fit_last_dim(cond_vec, token_features.shape[-1])
            token_features = token_features + 0.01 * cond_vec[:, None, :]

        token_features = token_features.to(dtype=torch.float32)
        layer_order = [spec.layer_key for spec in specs]
        if not layer_order:
            return 0.0

        logger.info(
            "Flux2TTR: starting calibration (%d steps, %d layer specs, %d tokens).",
            train_steps,
            len(layer_order),
            token_features.shape[1],
        )

        with torch.inference_mode(False):
            with torch.enable_grad():
                for step_idx in range(train_steps):
                    layer_key = layer_order[step_idx % len(layer_order)]
                    spec = self.layer_specs[layer_key]
                    out_dim = spec.num_heads * spec.head_dim
                    q_proj, k_proj, v_proj = self._ensure_projection(layer_key, token_features.shape[-1], out_dim, token_features.device)

                    q_flat = torch.tanh(token_features @ q_proj)
                    k_flat = torch.tanh(token_features @ k_proj)
                    v_flat = token_features @ v_proj

                    q_seq = q_flat.view(batch, q_flat.shape[1], spec.num_heads, spec.head_dim).permute(0, 2, 1, 3)
                    k_seq = k_flat.view(batch, k_flat.shape[1], spec.num_heads, spec.head_dim).permute(0, 2, 1, 3)
                    v_seq = v_flat.view(batch, v_flat.shape[1], spec.num_heads, spec.head_dim).permute(0, 2, 1, 3)

                    q_seq = q_seq / (torch.norm(q_seq, dim=-1, keepdim=True) + 1e-6)
                    k_seq = k_seq / (torch.norm(k_seq, dim=-1, keepdim=True) + 1e-6)
                    q_seq = q_seq.float().clone()
                    k_seq = k_seq.float().clone()
                    v_seq = v_seq.float().clone()

                    layer = self._ensure_layer(layer_key, spec.head_dim, token_features.device)
                    layer.train()
                    with torch.no_grad():
                        teacher = _softmax_attention(q_seq, k_seq, v_seq, mask=None)
                    student = layer(q_seq, k_seq, v_seq)
                    loss = torch.nn.functional.mse_loss(student, teacher)

                    optimizer = self.optimizers[layer_key]
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    self.last_loss = float(loss.item())
                    if step_idx == 0 or (step_idx + 1) % max(1, train_steps // 5) == 0 or step_idx + 1 == train_steps:
                        logger.info(
                            "Flux2TTR calibration step %d/%d layer=%s loss=%.6g",
                            step_idx + 1,
                            train_steps,
                            layer_key,
                            self.last_loss,
                        )

        self.training_enabled = False
        self.steps_remaining = 0
        return float(self.last_loss)

    def checkpoint_state(self) -> Dict[str, Any]:
        layer_states = {}
        for layer_key, layer in self.layers.items():
            layer_states[layer_key] = {k: v.detach().cpu() for k, v in layer.state_dict().items()}
        for layer_key, state in self.pending_state.items():
            if layer_key not in layer_states:
                layer_states[layer_key] = state

        return {
            "format": "flux2_ttr_v1",
            "feature_dim": self.feature_dim,
            "learning_rate": self.learning_rate,
            "training_mode": self.training_mode,
            "training_preview_ttr": self.training_preview_ttr,
            "comet_enabled": self.comet_enabled,
            "comet_project_name": self.comet_project_name,
            "comet_workspace": self.comet_workspace,
            "last_loss": self.last_loss,
            "scan_chunk_size": self.scan_chunk_size,
            "training_scan_chunk_size": self.training_scan_chunk_size,
            "training_token_cap": self.training_token_cap,
            "max_safe_inference_loss": self.max_safe_inference_loss,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "inference_mixed_precision": self.inference_mixed_precision,
            "layer_specs": {
                key: {"num_heads": spec.num_heads, "head_dim": spec.head_dim}
                for key, spec in self.layer_specs.items()
            },
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
        if payload.get("format") != "flux2_ttr_v1":
            raise ValueError(f"Flux2TTR: unsupported checkpoint format in {path}.")
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
        self.scan_chunk_size = max(1, int(payload.get("scan_chunk_size", self.scan_chunk_size)))
        self.training_scan_chunk_size = max(
            1,
            min(
                self.scan_chunk_size,
                int(payload.get("training_scan_chunk_size", self.training_scan_chunk_size)),
            ),
        )
        self.training_token_cap = max(1, int(payload.get("training_token_cap", self.training_token_cap)))
        self.max_safe_inference_loss = float(payload.get("max_safe_inference_loss", self.max_safe_inference_loss))
        self.layer_start = int(payload.get("layer_start", self.layer_start))
        self.layer_end = int(payload.get("layer_end", self.layer_end))
        self.inference_mixed_precision = bool(payload.get("inference_mixed_precision", self.inference_mixed_precision))
        specs = payload.get("layer_specs", {})
        for layer_key, meta in specs.items():
            try:
                self.layer_specs[layer_key] = FluxLayerSpec(
                    layer_key=layer_key,
                    num_heads=int(meta["num_heads"]),
                    head_dim=int(meta["head_dim"]),
                )
            except Exception:
                logger.warning("Flux2TTR: invalid layer spec in checkpoint for %s; skipping.", layer_key)

        self.pending_state = payload.get("layers", {})
        for layer_key, layer in self.layers.items():
            pending = self.pending_state.get(layer_key)
            if pending:
                layer.load_state_dict(pending, strict=False)
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
    scan_chunk_size = int(cfg.get("scan_chunk_size", 128))
    layer_start = int(cfg.get("layer_start", -1))
    layer_end = int(cfg.get("layer_end", -1))
    inference_mixed_precision = bool(cfg.get("inference_mixed_precision", True))
    training_preview_ttr = bool(cfg.get("training_preview_ttr", True))
    comet_enabled = bool(cfg.get("comet_enabled", False))
    comet_project_name = str(cfg.get("comet_project_name", "ttr-distillation"))
    comet_workspace = str(cfg.get("comet_workspace", "ken-simpson"))
    training_mode = bool(cfg.get("training_mode", False))
    training_total = max(0, int(cfg.get("training_steps_total", 0)))
    training_remaining = max(0, int(cfg.get("training_steps_remaining", training_total)))
    learning_rate = float(cfg.get("learning_rate", 1e-4))

    runtime = Flux2TTRRuntime(
        feature_dim=feature_dim,
        learning_rate=learning_rate,
        training=training_mode,
        steps=training_total,
        scan_chunk_size=scan_chunk_size,
        layer_start=layer_start,
        layer_end=layer_end,
        inference_mixed_precision=inference_mixed_precision,
        training_preview_ttr=training_preview_ttr,
        comet_enabled=comet_enabled,
        comet_project_name=comet_project_name,
        comet_workspace=comet_workspace,
    )
    runtime.training_mode = training_mode
    runtime.training_preview_ttr = training_preview_ttr
    runtime.training_steps_total = training_total
    runtime.steps_remaining = training_remaining
    runtime.training_enabled = bool(cfg.get("training", False)) and training_remaining > 0
    runtime.max_safe_inference_loss = float(cfg.get("max_safe_inference_loss", runtime.max_safe_inference_loss))

    checkpoint_path = (cfg.get("checkpoint_path") or "").strip()
    if checkpoint_path and os.path.isfile(checkpoint_path):
        runtime.load_checkpoint(checkpoint_path)
        runtime.training_mode = training_mode
        runtime.training_preview_ttr = training_preview_ttr
        runtime.comet_enabled = comet_enabled
        runtime.comet_project_name = comet_project_name
        runtime.comet_workspace = comet_workspace
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
