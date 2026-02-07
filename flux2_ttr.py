from __future__ import annotations

import logging
import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

_ORIGINAL_FLUX_ATTENTION: Dict[str, Any] = {}
_PATCH_DEPTH = 0
_RUNTIME_REGISTRY: Dict[str, "Flux2TTRRuntime"] = {}


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

    def scan(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_heads, seq_len, _ = q.shape
        w_state = q.new_zeros((batch_heads, self.feature_dim, self.value_dim))
        outputs = []
        for token_idx in range(seq_len):
            out_t, w_state = self.forward_chunk(q[:, token_idx, :], k[:, token_idx, :], v[:, token_idx, :], w_state)
            # Average by seen token count to keep the unrolled update numerically bounded.
            outputs.append(out_t / float(token_idx + 1))
        return torch.stack(outputs, dim=1)


class TTRFluxLayer(nn.Module):
    def __init__(self, head_dim: int, feature_dim: int = 256):
        super().__init__()
        self.head_dim = int(head_dim)
        self.feature_dim = validate_feature_dim(feature_dim)
        self.phi_net = nn.Sequential(
            nn.Linear(self.head_dim, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.regressor = TTRCell(self.feature_dim, self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("TTRFluxLayer expects q/k/v with shape [B, H, N, D].")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("TTRFluxLayer expects q/k/v shapes to match.")
        if q.shape[-1] != self.head_dim:
            raise ValueError(f"TTRFluxLayer expected head_dim={self.head_dim}, got {q.shape[-1]}.")

        batch, heads, seq_len, dim = q.shape
        q_flat = q.reshape(batch * heads, seq_len, dim)
        k_flat = k.reshape(batch * heads, seq_len, dim)
        v_flat = v.reshape(batch * heads, seq_len, dim)

        q_phi = self.phi_net(q_flat)
        k_phi = self.phi_net(k_flat)

        out_fwd = self.regressor.scan(q_phi, k_phi, v_flat)
        out_rev = self.regressor.scan(torch.flip(q_phi, dims=(1,)), torch.flip(k_phi, dims=(1,)), torch.flip(v_flat, dims=(1,)))
        out_rev = torch.flip(out_rev, dims=(1,))

        out = out_fwd + out_rev
        return out.reshape(batch, heads, seq_len, dim)


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
    ):
        self.feature_dim = validate_feature_dim(feature_dim)
        self.learning_rate = float(learning_rate)
        self.training_enabled = bool(training)
        self.steps_remaining = max(0, int(steps))
        self.last_loss = float("nan")
        self.layers: Dict[str, TTRFluxLayer] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.pending_state: Dict[str, Dict[str, torch.Tensor]] = {}
        self.layer_specs: Dict[str, FluxLayerSpec] = {}
        self._projection_cache: Dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def register_layer_specs(self, specs: Iterable[FluxLayerSpec]) -> None:
        for spec in specs:
            self.layer_specs[spec.layer_key] = spec

    def _ensure_layer(self, layer_key: str, head_dim: int, device: torch.device) -> TTRFluxLayer:
        layer = self.layers.get(layer_key)
        if layer is None:
            layer = TTRFluxLayer(head_dim=head_dim, feature_dim=self.feature_dim).to(device=device, dtype=torch.float32)
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
        layer = self._ensure_layer(layer_key, head_dim, q_eff.device)

        if self.training_enabled and self.steps_remaining > 0:
            layer.train()
            with torch.inference_mode(False):
                with torch.enable_grad():
                    q_in = q_eff.float()
                    k_in = k_eff.float()
                    v_in = v.float()
                    student = layer(q_in, k_in, v_in)
                    with torch.no_grad():
                        teacher_out = self._teacher_from_fallback(fallback_attention, q, k, v, pe, mask, transformer_options)
                        teacher = (
                            teacher_out.view(q.shape[0], q.shape[2], q.shape[1], q.shape[3])
                            .permute(0, 2, 1, 3)
                            .float()
                        )
                    loss = torch.nn.functional.mse_loss(student, teacher)
                    optimizer = self.optimizers[layer_key]
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            self.last_loss = float(loss.item())
            self.steps_remaining -= 1
            if self.steps_remaining <= 0:
                self.training_enabled = False
                logger.info("Flux2TTR: online distillation complete; inference mode enabled.")
            return teacher_out

        layer.eval()
        q_in = q_eff.float()
        k_in = k_eff.float()
        v_in = v.float()
        student = layer(q_in, k_in, v_in)
        student_out = _flatten_heads(student).to(dtype=v.dtype)
        return student_out

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
            "last_loss": self.last_loss,
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
        self.last_loss = float(payload.get("last_loss", self.last_loss))
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
        logger.warning("Flux2TTR: runtime_id=%s not found; falling back to original attention.", runtime_id)
        return original(q, k, v, pe, mask=mask, transformer_options=transformer_options)

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
    restore_flux_attention()
