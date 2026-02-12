from __future__ import annotations

import io
import importlib.util
import json
import logging
import os
import sys
import types
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

from flux2_ttr_embeddings import ScalarSinusoidalEmbedding

logger = logging.getLogger(__name__)

_CONTROLLER_CHECKPOINT_FORMAT = "flux2_ttr_controller_v1"
_LPIPS_EPS = 1e-8
DEFAULT_CONTROLLER_CHECKPOINT_EVERY = 10
_REWARD_BASELINE_QUALITY_FLOOR_DEFAULT = -0.3
_DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
_DEFAULT_GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
_DEFAULT_GEMINI_PROMPT = (
    "On a scale of 1 to 10, how similar is image 2 to image 1? "
    "Respond using the `similarity` value in your output.\n"
    "On a scale of 1 to 10, rank the relative quality of image 2 in comparison to image 1. "
    "Respond using the `quality` value in your output.\n"
    "Return JSON only, for example: {\"similarity\":8,\"quality\":2}."
)
_GEMINI_SCORE_MIN = 1.0
_GEMINI_SCORE_MAX = 10.0


def should_save_controller_checkpoint_step(step: int, checkpoint_every: int = DEFAULT_CONTROLLER_CHECKPOINT_EVERY) -> bool:
    return int(step) > 0 and int(checkpoint_every) > 0 and (int(step) % int(checkpoint_every) == 0)

class _ResolutionEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.width_embed = ScalarSinusoidalEmbedding(embed_dim)
        self.height_embed = ScalarSinusoidalEmbedding(embed_dim)
        in_dim = self.width_embed.embed_dim + self.height_embed.embed_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, width: torch.Tensor, height: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        emb = torch.cat([self.width_embed(width), self.height_embed(height)], dim=-1)
        return self.proj(emb.to(dtype=dtype))


class TTRController(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        if int(num_layers) <= 0:
            raise ValueError("TTRController: num_layers must be positive.")

        self.num_layers = int(num_layers)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)

        self.sigma_embed = ScalarSinusoidalEmbedding(self.embed_dim)
        self.cfg_embed = ScalarSinusoidalEmbedding(self.embed_dim)
        self.resolution_embed = _ResolutionEmbedding(self.embed_dim)

        mlp_in = self.sigma_embed.embed_dim + self.cfg_embed.embed_dim + self.embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.num_layers),
        )

    def _input_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        param = next(self.parameters())
        return param.device, param.dtype

    def forward(
        self,
        sigma: float | torch.Tensor,
        cfg_scale: float | torch.Tensor,
        width: int | float | torch.Tensor,
        height: int | float | torch.Tensor,
    ) -> torch.Tensor:
        device, dtype = self._input_device_dtype()
        sigma_t = torch.as_tensor(sigma, device=device, dtype=torch.float32).reshape(1)
        cfg_t = torch.as_tensor(cfg_scale, device=device, dtype=torch.float32).reshape(1)
        width_t = torch.as_tensor(width, device=device, dtype=torch.float32).reshape(1)
        height_t = torch.as_tensor(height, device=device, dtype=torch.float32).reshape(1)

        sigma_e = self.sigma_embed(sigma_t).to(dtype=dtype)
        cfg_e = self.cfg_embed(cfg_t).to(dtype=dtype)
        res_e = self.resolution_embed(width_t, height_t, dtype=dtype)
        hidden = torch.cat([sigma_e, cfg_e, res_e], dim=-1)
        logits = self.mlp(hidden).reshape(-1)
        return logits.float()

    @staticmethod
    def sample_training_mask(logits: torch.Tensor, temperature: float = 1.0, hard: bool = True) -> torch.Tensor:
        t = max(1e-4, float(temperature))
        u = torch.rand_like(logits).clamp(min=1e-6, max=1.0 - 1e-6)
        g = -torch.log(-torch.log(u))
        probs = torch.sigmoid((logits + g) / t)
        if not hard:
            return probs
        hard_mask = (probs > 0.5).to(dtype=probs.dtype)
        return hard_mask.detach() - probs.detach() + probs

    @staticmethod
    def inference_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (torch.sigmoid(logits.float()) > float(threshold)).to(dtype=torch.float32)


class TrainingControllerWrapper(nn.Module):
    """Controller wrapper for sigma-aware, per-step REINFORCE collection."""

    _LOGIT_CLAMP = 20.0

    def __init__(
        self,
        controller: TTRController,
        *,
        temperature: float = 1.0,
        ready_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if not isinstance(controller, nn.Module):
            raise TypeError("TrainingControllerWrapper: controller must be an nn.Module.")
        self.controller = controller
        self.temperature = float(temperature)
        with torch.inference_mode(False):
            self.ready_mask = (
                ready_mask.detach().clone().reshape(-1).to(dtype=torch.float32)
                if torch.is_tensor(ready_mask)
                else None
            )
        self.step_records: list[dict[str, Any]] = []
        self.last_recompute_debug: dict[str, Any] = {}

    @staticmethod
    def _scalar(value: float | torch.Tensor) -> float:
        if torch.is_tensor(value):
            if value.numel() <= 0:
                return 0.0
            return float(value.reshape(-1)[0].detach().cpu().item())
        return float(value)

    def _ready_mask_for(self, ref: torch.Tensor) -> Optional[torch.Tensor]:
        if self.ready_mask is None:
            return None
        ready = self.ready_mask.to(device=ref.device, dtype=ref.dtype).reshape(-1)
        if int(ready.numel()) != int(ref.numel()):
            raise ValueError(
                "TrainingControllerWrapper: ready_mask length "
                f"{int(ready.numel())} does not match controller layer count {int(ref.numel())}."
            )
        return ready

    def reset(self) -> None:
        self.step_records.clear()

    def forward(
        self,
        sigma: float | torch.Tensor,
        cfg_scale: float | torch.Tensor,
        width: int | float | torch.Tensor,
        height: int | float | torch.Tensor,
    ) -> torch.Tensor:
        # Flux2TTRRuntime calls controllers under no_grad; force grad-enabled
        # context here so policy log-probs remain differentiable.
        with torch.inference_mode(False):
            with torch.enable_grad():
                logits = self.controller(sigma=sigma, cfg_scale=cfg_scale, width=width, height=height)
                if logits.ndim > 1:
                    logits = logits.reshape(-1)
                logits = logits.float()

                sampled_mask = TTRController.sample_training_mask(
                    logits,
                    temperature=float(self.temperature),
                    hard=True,
                )
                mask = (sampled_mask > 0.5).to(dtype=logits.dtype)
                ready = self._ready_mask_for(mask)
                if ready is not None:
                    # Non-ready layers are forced to full attention.
                    mask = mask * ready + (1.0 - ready)

                probs = torch.sigmoid(logits).clamp(min=1e-7, max=1.0 - 1e-7)
                mask_det = mask.detach()
                log_probs = mask_det * torch.log(probs) + (1.0 - mask_det) * torch.log(1.0 - probs)

                self.step_records.append(
                    {
                        "sigma": self._scalar(sigma),
                        "log_probs": log_probs,
                        "mask": mask_det.detach().clone(),
                        "probs": probs.detach().clone(),
                    }
                )

                synthetic_logits = (mask_det * 2.0 - 1.0) * float(self._LOGIT_CLAMP)
                return synthetic_logits.detach()

    def total_log_prob(self) -> torch.Tensor:
        if not self.step_records:
            raise RuntimeError("TrainingControllerWrapper.total_log_prob: no step records collected.")
        return torch.stack([rec["log_probs"].sum() for rec in self.step_records]).sum()

    def sigma_weighted_log_prob(self, sigma_max: float = 1.0) -> torch.Tensor:
        if not self.step_records:
            raise RuntimeError("TrainingControllerWrapper.sigma_weighted_log_prob: no step records collected.")
        sigma_norm = max(float(sigma_max), 1e-8)
        ref = self.step_records[0]["log_probs"]
        result = torch.zeros((), device=ref.device, dtype=ref.dtype)
        for rec in self.step_records:
            sigma_value = max(0.0, min(1.0, float(rec["sigma"]) / sigma_norm))
            weight = max(0.1, 1.0 - sigma_value)
            result = result + float(weight) * rec["log_probs"].sum()
        return result

    def sigma_weighted_log_prob_recompute(
        self,
        *,
        cfg_scale: float,
        width: int,
        height: int,
        sigma_max: Optional[float] = None,
        eligible_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Recompute sigma-weighted trajectory log-prob from stored actions.

        This path is robust even if step_records were collected in a context
        where rec["log_probs"] became detached, because it re-evaluates
        controller logits under grad-enabled mode using stored (sigma, mask).
        """
        if not self.step_records:
            raise RuntimeError("TrainingControllerWrapper.sigma_weighted_log_prob_recompute: no step records collected.")
        sigma_norm = max(
            float(sigma_max) if sigma_max is not None else max(float(rec["sigma"]) for rec in self.step_records),
            1e-8,
        )
        debug: dict[str, Any] = {
            "steps": int(len(self.step_records)),
            "any_logits_requires_grad": False,
            "any_log_probs_requires_grad": False,
            "any_weighted_requires_grad": False,
            "result_requires_grad": False,
            "eligible_counts": [],
            "used_bool_indexing": False,
        }

        with torch.inference_mode(False):
            with torch.enable_grad():
                result: Optional[torch.Tensor] = None
                for rec in self.step_records:
                    logits = self.controller(
                        sigma=float(rec["sigma"]),
                        cfg_scale=float(cfg_scale),
                        width=int(width),
                        height=int(height),
                    )
                    if logits.ndim > 1:
                        logits = logits.reshape(-1)
                    probs = torch.sigmoid(logits.float()).clamp(min=1e-7, max=1.0 - 1e-7)
                    debug["any_logits_requires_grad"] = bool(debug["any_logits_requires_grad"] or bool(logits.requires_grad))
                    mask = rec["mask"].to(device=probs.device, dtype=probs.dtype).reshape(-1).detach()
                    if int(mask.numel()) != int(probs.numel()):
                        raise ValueError(
                            "TrainingControllerWrapper.sigma_weighted_log_prob_recompute: stored mask length "
                            f"{int(mask.numel())} does not match controller layer count {int(probs.numel())}."
                        )
                    if eligible_mask is not None:
                        eligible = (
                            eligible_mask.to(device=probs.device, dtype=torch.bool)
                            .reshape(-1)
                            .detach()
                            .clone()
                        )
                        if int(eligible.numel()) != int(probs.numel()):
                            raise ValueError(
                                "TrainingControllerWrapper.sigma_weighted_log_prob_recompute: eligible_mask length "
                                f"{int(eligible.numel())} does not match controller layer count {int(probs.numel())}."
                            )
                    else:
                        eligible = torch.ones_like(probs, dtype=torch.bool)

                    log_probs = mask * torch.log(probs) + (1.0 - mask) * torch.log(1.0 - probs)
                    eligible_f = eligible.to(dtype=log_probs.dtype).detach().clone()
                    debug["eligible_counts"].append(int(eligible.sum().item()))
                    log_prob_sum = (log_probs * eligible_f).sum()
                    debug["any_log_probs_requires_grad"] = bool(
                        debug["any_log_probs_requires_grad"] or bool(log_prob_sum.requires_grad)
                    )
                    sigma_value = max(0.0, min(1.0, float(rec["sigma"]) / sigma_norm))
                    weight = max(0.1, 1.0 - sigma_value)
                    weighted = float(weight) * log_prob_sum
                    debug["any_weighted_requires_grad"] = bool(
                        debug["any_weighted_requires_grad"] or bool(weighted.requires_grad)
                    )
                    result = weighted if result is None else (result + weighted)
        if result is None:
            raise RuntimeError("TrainingControllerWrapper.sigma_weighted_log_prob_recompute: no terms accumulated.")
        debug["result_requires_grad"] = bool(result.requires_grad)
        self.last_recompute_debug = debug
        return result

    def per_step_ttr_ratio(self) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for rec in self.step_records:
            mask = rec["mask"]
            ready = self._ready_mask_for(mask)
            if ready is not None:
                eligible = float(ready.sum().item())
                if eligible > 0:
                    ttr_fraction = float((((1.0 - mask) * ready).sum() / ready.sum()).item())
                else:
                    ttr_fraction = 0.0
            else:
                ttr_fraction = float((1.0 - mask).mean().item())
            out.append((float(rec["sigma"]), ttr_fraction))
        return out

    def mean_entropy(self) -> float:
        if not self.step_records:
            return 0.0
        total = 0.0
        count = 0
        for rec in self.step_records:
            probs = rec["probs"]
            entropy = -(
                probs * torch.log(probs + 1e-8)
                + (1.0 - probs) * torch.log(1.0 - probs + 1e-8)
            )
            total += float(entropy.sum().item())
            count += int(entropy.numel())
        return total / float(max(1, count))

    def mean_full_attn_eligible(self) -> float:
        if not self.step_records:
            return 0.0
        total = 0.0
        for rec in self.step_records:
            mask = rec["mask"]
            ready = self._ready_mask_for(mask)
            if ready is not None:
                eligible = float(ready.sum().item())
                if eligible > 0:
                    total += float(((mask * ready).sum() / ready.sum()).item())
                else:
                    total += 1.0
            else:
                total += float(mask.mean().item())
        return total / float(max(1, len(self.step_records)))

    def mean_full_attn_overall(self) -> float:
        if not self.step_records:
            return 0.0
        total = 0.0
        for rec in self.step_records:
            mask = rec["mask"]
            ready = self._ready_mask_for(mask)
            if ready is not None:
                forced_full = float((1.0 - ready).sum().item())
                full_ready = float((mask * ready).sum().item())
                denom = max(1.0, float(mask.numel()))
                total += (full_ready + forced_full) / denom
            else:
                total += float(mask.mean().item())
        return total / float(max(1, len(self.step_records)))

    def mean_expected_full_attn_eligible(self) -> float:
        if not self.step_records:
            return 0.0
        total = 0.0
        for rec in self.step_records:
            probs = rec["probs"]
            ready = self._ready_mask_for(probs)
            if ready is not None:
                eligible = float(ready.sum().item())
                if eligible > 0:
                    total += float(((probs * ready).sum() / ready.sum()).item())
                else:
                    total += 1.0
            else:
                total += float(probs.mean().item())
        return total / float(max(1, len(self.step_records)))

    def mean_expected_full_attn_overall(self) -> float:
        if not self.step_records:
            return 0.0
        total = 0.0
        for rec in self.step_records:
            probs = rec["probs"]
            ready = self._ready_mask_for(probs)
            if ready is not None:
                forced_full = float((1.0 - ready).sum().item())
                full_ready = float((probs * ready).sum().item())
                denom = max(1.0, float(probs.numel()))
                total += (full_ready + forced_full) / denom
            else:
                total += float(probs.mean().item())
        return total / float(max(1, len(self.step_records)))

def controller_checkpoint_state(
    controller: TTRController,
    trainer: Optional["ControllerTrainer"] = None,
) -> Dict[str, Any]:
    if not isinstance(controller, TTRController):
        raise TypeError("controller_checkpoint_state expects a TTRController instance.")
    return {
        "format": _CONTROLLER_CHECKPOINT_FORMAT,
        "num_layers": int(controller.num_layers),
        "embed_dim": int(controller.embed_dim),
        "hidden_dim": int(controller.hidden_dim),
        "state_dict": {k: v.detach().cpu() for k, v in controller.state_dict().items()},
        "reward_baseline": float(trainer.reward_baseline) if trainer is not None else None,
        "reward_count": int(trainer.reward_count) if trainer is not None else None,
        "reward_baseline_quality_floor": (
            float(trainer.reward_baseline_quality_floor)
            if trainer is not None
            else None
        ),
        "lambda_entropy": float(trainer.lambda_entropy) if trainer is not None else None,
        "optimizer_state_dict": trainer.optimizer.state_dict() if trainer is not None else None,
    }


def save_controller_checkpoint(
    controller: TTRController,
    checkpoint_path: str,
    trainer: Optional["ControllerTrainer"] = None,
) -> None:
    path = (checkpoint_path or "").strip()
    if not path:
        raise ValueError("save_controller_checkpoint: checkpoint_path is required.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(controller_checkpoint_state(controller, trainer=trainer), path)
    logger.info("Flux2TTR controller: saved checkpoint to %s", path)


def load_controller_training_state(
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    path = (checkpoint_path or "").strip()
    if not path:
        raise ValueError("load_controller_training_state: checkpoint_path is required.")
    payload = torch.load(path, map_location=map_location)
    fmt = payload.get("format")
    if fmt != _CONTROLLER_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format: {fmt!r}")
    return payload


def load_controller_checkpoint(checkpoint_path: str, map_location: str | torch.device = "cpu") -> TTRController:
    payload = load_controller_training_state(checkpoint_path, map_location=map_location)
    controller = TTRController(
        num_layers=int(payload["num_layers"]),
        embed_dim=int(payload.get("embed_dim", 64)),
        hidden_dim=int(payload.get("hidden_dim", 256)),
    )
    controller.load_state_dict(payload["state_dict"], strict=True)
    controller.eval()
    return controller


class ControllerTrainer:
    def __init__(
        self,
        controller: TTRController,
        *,
        training_config: Optional[Dict[str, Any]] = None,
        learning_rate: float = 1e-3,
        rmse_weight: float = 1.0,
        cosine_weight: float = 1.0,
        lpips_weight: float = 0.0,
        dreamsim_weight: float = 0.0,
        hps_weight: float = 0.0,
        biqa_quality_weight: float = 0.0,
        biqa_aesthetic_weight: float = 0.0,
        gemini_similarity_weight: float = 0.0,
        gemini_quality_weight: float = 0.0,
        gemini_model: str = _DEFAULT_GEMINI_MODEL,
        gemini_api_key: str = "",
        gemini_api_key_env: str = _DEFAULT_GEMINI_API_KEY_ENV,
        gemini_prompt: str = _DEFAULT_GEMINI_PROMPT,
        hps_prompt: str = "",
        reward_baseline_quality_floor: float = _REWARD_BASELINE_QUALITY_FLOOR_DEFAULT,
        target_ttr_ratio: float = 0.7,
        lambda_eff: float = 1.0,
        lambda_entropy: float = 0.1,
        grad_clip_norm: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        if not isinstance(controller, TTRController):
            raise TypeError("ControllerTrainer expects a TTRController instance.")

        if training_config is not None:
            loss_cfg = training_config.get("loss_config", {}) if isinstance(training_config, dict) else {}
            opt_cfg = training_config.get("optimizer_config", {}) if isinstance(training_config, dict) else {}
            sched_cfg = training_config.get("schedule_config", {}) if isinstance(training_config, dict) else {}

            rmse_weight = float(loss_cfg.get("rmse_weight", rmse_weight))
            cosine_weight = float(loss_cfg.get("cosine_weight", cosine_weight))
            lpips_weight = float(loss_cfg.get("lpips_weight", lpips_weight))
            dreamsim_weight = float(loss_cfg.get("dreamsim_weight", dreamsim_weight))
            hps_weight = float(loss_cfg.get("hps_weight", hps_weight))
            biqa_quality_weight = float(loss_cfg.get("biqa_quality_weight", biqa_quality_weight))
            biqa_aesthetic_weight = float(loss_cfg.get("biqa_aesthetic_weight", biqa_aesthetic_weight))
            gemini_similarity_weight = float(loss_cfg.get("gemini_similarity_weight", gemini_similarity_weight))
            gemini_quality_weight = float(loss_cfg.get("gemini_quality_weight", gemini_quality_weight))
            gemini_model = str(loss_cfg.get("gemini_model", gemini_model))
            gemini_api_key = str(loss_cfg.get("gemini_api_key", gemini_api_key))
            gemini_api_key_env = str(loss_cfg.get("gemini_api_key_env", gemini_api_key_env))
            gemini_prompt = str(loss_cfg.get("gemini_prompt", gemini_prompt))
            hps_prompt = str(loss_cfg.get("hps_prompt", hps_prompt))
            reward_baseline_quality_floor = float(
                loss_cfg.get(
                    "reward_baseline_quality_floor",
                    reward_baseline_quality_floor,
                )
            )
            target_ttr_ratio = float(sched_cfg.get("target_ttr_ratio", target_ttr_ratio))
            lambda_eff = float(sched_cfg.get("lambda_eff", lambda_eff))
            lambda_entropy = float(sched_cfg.get("lambda_entropy", lambda_entropy))
            learning_rate = float(opt_cfg.get("learning_rate", learning_rate))
            grad_clip_norm = float(opt_cfg.get("grad_clip_norm", grad_clip_norm))

        if self._module_contains_inference_tensors(controller):
            logger.warning(
                "ControllerTrainer: controller parameters were created under inference mode; rebuilding trainable copy."
            )
            controller = self._rebuild_controller_trainable_copy(controller)

        self.controller = controller
        if device is not None:
            with torch.inference_mode(False):
                self.controller.to(device=device)

        self.optimizer = torch.optim.AdamW(self.controller.parameters(), lr=float(learning_rate))
        self.rmse_weight = float(rmse_weight)
        self.cosine_weight = float(cosine_weight)
        self.lpips_weight = float(lpips_weight)
        self.dreamsim_weight = max(0.0, float(dreamsim_weight))
        self.hps_weight = max(0.0, float(hps_weight))
        self.biqa_quality_weight = max(0.0, float(biqa_quality_weight))
        self.biqa_aesthetic_weight = max(0.0, float(biqa_aesthetic_weight))
        self.gemini_similarity_weight = max(0.0, float(gemini_similarity_weight))
        self.gemini_quality_weight = max(0.0, float(gemini_quality_weight))
        self.gemini_model = str(gemini_model or _DEFAULT_GEMINI_MODEL)
        self.gemini_api_key = str(gemini_api_key or "").strip()
        self.gemini_api_key_env = str(gemini_api_key_env or _DEFAULT_GEMINI_API_KEY_ENV)
        self.gemini_prompt = str(gemini_prompt or _DEFAULT_GEMINI_PROMPT)
        self.hps_prompt = str(hps_prompt or "")
        self.reward_baseline_quality_floor = float(reward_baseline_quality_floor)
        self.target_ttr_ratio = float(target_ttr_ratio)
        self.lambda_eff = max(0.0, float(lambda_eff))
        self.lambda_entropy = max(0.0, float(lambda_entropy))
        self.grad_clip_norm = max(0.0, float(grad_clip_norm))
        self._reward_baseline = 0.0
        self._reward_count = 0
        self._train_step_warned = False

        self.lpips_model = None
        if self.lpips_weight > 0:
            try:
                import lpips  # type: ignore

                self.lpips_model = lpips.LPIPS(net="alex")
                self.lpips_model.eval()
                controller_device, _ = self.controller._input_device_dtype()
                self.lpips_model.to(device=controller_device)
            except Exception as exc:
                raise RuntimeError(
                    "ControllerTrainer: lpips_weight > 0 requires the 'lpips' package."
                ) from exc

        self.dreamsim_model = None
        self.dreamsim_preprocess = None
        self.hps_backend = ""
        self.hps_module = None
        self.hps_model = None
        self.hps_version = "v2.1"
        self.biqa_backend = ""
        self.biqa_quality_model = None
        self.biqa_aesthetic_model = None
        self.gemini_client = None
        self.gemini_types = None
        if self.dreamsim_weight > 0:
            self._init_dreamsim_model()
        if self.hps_weight > 0:
            self._init_hps_model()
        if self.biqa_quality_weight > 0 or self.biqa_aesthetic_weight > 0:
            self._init_biqa_models()
        if self.gemini_similarity_weight > 0 or self.gemini_quality_weight > 0:
            self._init_gemini_client()
        logger.info(
            (
                "ControllerTrainer initialized: lr=%.6g rmse=%.4g cosine=%.4g lpips=%.4g dreamsim=%.4g hps=%.4g biqa_q=%.4g biqa_a=%.4g "
                "gemini_similarity=%.4g gemini_quality=%.4g gemini_model=%s gemini_api_key=%s gemini_api_key_env=%s "
                "target_ttr_ratio=%.4g target_full_attn_ratio=%.4g "
                "lambda_eff=%.4g lambda_entropy=%.4g grad_clip=%.4g baseline_quality_floor=%.4g"
            ),
            float(learning_rate),
            self.rmse_weight,
            self.cosine_weight,
            self.lpips_weight,
            self.dreamsim_weight,
            self.hps_weight,
            self.biqa_quality_weight,
            self.biqa_aesthetic_weight,
            self.gemini_similarity_weight,
            self.gemini_quality_weight,
            self.gemini_model,
            "<set>" if self.gemini_api_key else "<empty>",
            self.gemini_api_key_env,
            self.target_ttr_ratio,
            self._target_full_attn_ratio_from_ttr_ratio(self.target_ttr_ratio),
            self.lambda_eff,
            self.lambda_entropy,
            self.grad_clip_norm,
            self.reward_baseline_quality_floor,
        )

    @property
    def reward_baseline(self) -> float:
        return float(self._reward_baseline)

    @property
    def reward_count(self) -> int:
        return int(self._reward_count)

    def restore_training_state(self, payload: Dict[str, Any]) -> None:
        """Restore reward baseline and optimizer state from a checkpoint payload."""
        if not isinstance(payload, dict):
            return
        if "reward_baseline" in payload and payload["reward_baseline"] is not None:
            self._reward_baseline = float(payload["reward_baseline"])
        if "reward_count" in payload and payload["reward_count"] is not None:
            self._reward_count = int(payload["reward_count"])
        if "reward_baseline_quality_floor" in payload and payload["reward_baseline_quality_floor"] is not None:
            self.reward_baseline_quality_floor = float(payload["reward_baseline_quality_floor"])
        if "lambda_entropy" in payload and payload["lambda_entropy"] is not None:
            self.lambda_entropy = max(0.0, float(payload["lambda_entropy"]))
        if "optimizer_state_dict" in payload and payload["optimizer_state_dict"] is not None:
            try:
                with torch.inference_mode(False):
                    optimizer_state = self._clone_state_tensors(payload["optimizer_state_dict"])
                    self.optimizer.load_state_dict(optimizer_state)
                target_device = self.controller._input_device_dtype()[0]
                for state in self.optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            with torch.inference_mode(False):
                                state[key] = value.detach().clone().to(device=target_device)
            except Exception as exc:
                logger.warning(
                    "ControllerTrainer: failed to restore optimizer state (%s). "
                    "Continuing with fresh optimizer.",
                    exc,
                )

    @staticmethod
    def _is_inference_tensor(t: torch.Tensor) -> bool:
        checker = getattr(t, "is_inference", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        return False

    @classmethod
    def _module_contains_inference_tensors(cls, module: nn.Module) -> bool:
        for tensor in list(module.parameters()) + list(module.buffers()):
            if cls._is_inference_tensor(tensor):
                return True
        return False

    @staticmethod
    def _clone_state_dict_tensors(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in state_dict.items()}

    @classmethod
    def _clone_state_tensors(cls, obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.detach().clone()
        if isinstance(obj, dict):
            return {k: cls._clone_state_tensors(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._clone_state_tensors(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(cls._clone_state_tensors(v) for v in obj)
        return obj

    @classmethod
    def _rebuild_controller_trainable_copy(cls, controller: TTRController) -> TTRController:
        with torch.inference_mode(False):
            rebuilt = TTRController(
                num_layers=int(controller.num_layers),
                embed_dim=int(controller.embed_dim),
                hidden_dim=int(controller.hidden_dim),
            )
            state = cls._clone_state_dict_tensors(controller.state_dict())
            rebuilt.load_state_dict(state, strict=True)
        return rebuilt

    @staticmethod
    def _cosine_distance(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        s = student.reshape(student.shape[0], -1)
        t = teacher.reshape(teacher.shape[0], -1)
        cos = F.cosine_similarity(s, t, dim=1, eps=_LPIPS_EPS)
        return (1.0 - cos).mean()

    @staticmethod
    def _prep_lpips_rgb(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("LPIPS tensors must be [B,C,H,W].")
        x = x.float()
        if x.shape[1] != 3:
            raise ValueError("LPIPS tensors must have 3 channels.")
        return x.clamp(-1.0, 1.0)

    @staticmethod
    def _module_primary_device(module: nn.Module) -> Optional[torch.device]:
        for tensor in list(module.parameters()) + list(module.buffers()):
            return tensor.device
        return None

    def _ensure_lpips_model_device(self, target_device: torch.device) -> None:
        if self.lpips_model is None:
            return
        current_device = self._module_primary_device(self.lpips_model)
        if current_device != target_device:
            logger.debug(
                "ControllerTrainer: moving LPIPS model from %s to %s.",
                str(current_device),
                str(target_device),
            )
            with torch.inference_mode(False):
                self.lpips_model.to(device=target_device)

    def _ensure_metric_model_device(
        self,
        module: Any,
        *,
        target_device: torch.device,
        component_name: str,
    ) -> None:
        if module is None or not hasattr(module, "to"):
            return
        current_device = self._module_primary_device(module) if isinstance(module, nn.Module) else None
        if current_device is not None and current_device == target_device:
            return
        logger.debug("ControllerTrainer: moving %s from %s to %s.", component_name, str(current_device), str(target_device))
        with torch.inference_mode(False):
            module.to(device=target_device)

    @staticmethod
    def _prep_metric_rgb(x: torch.Tensor) -> torch.Tensor:
        return ((ControllerTrainer._prep_lpips_rgb(x) + 1.0) * 0.5).clamp(0.0, 1.0)

    @staticmethod
    def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
        arr = img.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        arr_u8 = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(arr_u8, mode="RGB")

    @classmethod
    def _pil_batch_from_rgb(cls, rgb: torch.Tensor) -> list[Image.Image]:
        return [cls._tensor_to_pil(rgb[idx]) for idx in range(int(rgb.shape[0]))]

    @staticmethod
    def _to_scalar_tensor(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(value):
            t = value.to(device=device, dtype=dtype)
            return t.mean() if t.ndim > 0 else t
        return torch.tensor(float(value), device=device, dtype=dtype)

    @staticmethod
    def _scores_to_tensor(scores: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(scores):
            return scores.to(device=device, dtype=dtype).reshape(-1)
        if isinstance(scores, np.ndarray):
            return torch.as_tensor(scores, device=device, dtype=dtype).reshape(-1)
        if isinstance(scores, (list, tuple)):
            if scores and any(torch.is_tensor(item) for item in scores):
                flat: list[float] = []
                for item in scores:
                    if torch.is_tensor(item):
                        flat.extend(item.detach().cpu().reshape(-1).tolist())
                    else:
                        flat.append(float(item))
                return torch.tensor(flat, device=device, dtype=dtype).reshape(-1)
            return torch.as_tensor(scores, device=device, dtype=dtype).reshape(-1)
        return torch.tensor([float(scores)], device=device, dtype=dtype)

    @staticmethod
    def _is_dreamsim_utils_shadow_error(exc: Exception) -> bool:
        message = str(exc)
        return "trunc_normal_" in message and "from 'utils'" in message

    @staticmethod
    def _find_dreamsim_utils_file() -> Optional[str]:
        try:
            spec = importlib.util.find_spec("dreamsim")
        except Exception:
            return None
        if spec is None:
            return None

        candidate_dirs: list[str] = []
        if spec.submodule_search_locations:
            candidate_dirs.extend(str(location) for location in spec.submodule_search_locations)
        if spec.origin:
            candidate_dirs.append(os.path.dirname(str(spec.origin)))

        seen: set[str] = set()
        for location in candidate_dirs:
            if not location or location in seen:
                continue
            seen.add(location)
            for rel_path in ("utils.py", os.path.join("utils", "__init__.py")):
                candidate = os.path.join(location, rel_path)
                if os.path.isfile(candidate):
                    return candidate
        return None

    @classmethod
    def _import_dreamsim_with_utils_shim(cls) -> Optional[Callable[..., Any]]:
        utils_file = cls._find_dreamsim_utils_file()
        if not utils_file:
            logger.debug("ControllerTrainer: dreamsim retry skipped; dreamsim/utils.py was not found.")
            return None

        utils_spec = importlib.util.spec_from_file_location("_dreamsim_utils_shim", utils_file)
        if utils_spec is None or utils_spec.loader is None:
            logger.debug("ControllerTrainer: dreamsim retry skipped; unable to load spec for %s.", utils_file)
            return None

        dreamsim_utils = importlib.util.module_from_spec(utils_spec)
        try:
            utils_spec.loader.exec_module(dreamsim_utils)
        except Exception:
            logger.debug(
                "ControllerTrainer: dreamsim retry skipped; failed to execute utils shim from %s.",
                utils_file,
                exc_info=True,
            )
            return None

        sentinel = object()
        previous_utils = sys.modules.get("utils", sentinel)
        sys.modules["utils"] = dreamsim_utils
        for name in [k for k in list(sys.modules.keys()) if k == "dreamsim" or k.startswith("dreamsim.")]:
            sys.modules.pop(name, None)

        try:
            from dreamsim import dreamsim as dreamsim_factory  # type: ignore

            def _wrapped_factory(*args: Any, **kwargs: Any) -> Any:
                call_sentinel = object()
                previous_call_utils = sys.modules.get("utils", call_sentinel)
                sys.modules["utils"] = dreamsim_utils
                try:
                    return dreamsim_factory(*args, **kwargs)
                finally:
                    if previous_call_utils is call_sentinel:
                        sys.modules.pop("utils", None)
                    else:
                        sys.modules["utils"] = previous_call_utils

            logger.warning(
                "ControllerTrainer: recovered dreamsim import by retrying with dreamsim's bundled utils shim."
            )
            return _wrapped_factory
        except Exception:
            logger.debug("ControllerTrainer: dreamsim retry with bundled utils shim failed.", exc_info=True)
            return None
        finally:
            if previous_utils is sentinel:
                sys.modules.pop("utils", None)
            else:
                sys.modules["utils"] = previous_utils

    @classmethod
    def _import_dreamsim_with_trunc_normal_patch(cls) -> Optional[Callable[..., Any]]:
        sentinel = object()
        previous_utils = sys.modules.get("utils", sentinel)
        utils_module: Any = previous_utils

        if previous_utils is sentinel:
            try:
                import utils as imported_utils  # type: ignore

                utils_module = imported_utils
            except Exception:
                utils_module = types.ModuleType("utils")

        injected = False
        if not hasattr(utils_module, "trunc_normal_"):
            setattr(utils_module, "trunc_normal_", torch.nn.init.trunc_normal_)
            injected = True

        sys.modules["utils"] = utils_module
        for name in [k for k in list(sys.modules.keys()) if k == "dreamsim" or k.startswith("dreamsim.")]:
            sys.modules.pop(name, None)

        try:
            from dreamsim import dreamsim as dreamsim_factory  # type: ignore

            def _wrapped_factory(*args: Any, **kwargs: Any) -> Any:
                call_sentinel = object()
                previous_call_utils = sys.modules.get("utils", call_sentinel)
                call_utils: Any = previous_call_utils
                if previous_call_utils is call_sentinel:
                    try:
                        import utils as imported_utils  # type: ignore

                        call_utils = imported_utils
                    except Exception:
                        call_utils = types.ModuleType("utils")

                injected_call = False
                if not hasattr(call_utils, "trunc_normal_"):
                    setattr(call_utils, "trunc_normal_", torch.nn.init.trunc_normal_)
                    injected_call = True

                sys.modules["utils"] = call_utils
                try:
                    return dreamsim_factory(*args, **kwargs)
                finally:
                    if previous_call_utils is call_sentinel:
                        sys.modules.pop("utils", None)
                    else:
                        if injected_call:
                            try:
                                delattr(previous_call_utils, "trunc_normal_")
                            except Exception:
                                pass
                        sys.modules["utils"] = previous_call_utils

            logger.warning(
                "ControllerTrainer: recovered dreamsim import by patching missing utils.trunc_normal_."
            )
            return _wrapped_factory
        except Exception:
            logger.debug("ControllerTrainer: dreamsim retry with utils.trunc_normal_ patch failed.", exc_info=True)
            return None
        finally:
            if previous_utils is sentinel:
                sys.modules.pop("utils", None)
            elif injected:
                try:
                    delattr(previous_utils, "trunc_normal_")
                except Exception:
                    pass
                sys.modules["utils"] = previous_utils
            else:
                sys.modules["utils"] = previous_utils

    @classmethod
    def _import_dreamsim_factory(cls) -> Callable[..., Any]:
        try:
            from dreamsim import dreamsim as dreamsim_factory  # type: ignore

            return dreamsim_factory
        except Exception as exc:
            if not cls._is_dreamsim_utils_shadow_error(exc):
                raise
            recovered = cls._import_dreamsim_with_utils_shim()
            if recovered is not None:
                return recovered
            recovered = cls._import_dreamsim_with_trunc_normal_patch()
            if recovered is not None:
                return recovered
            raise

    def _init_dreamsim_model(self) -> None:
        try:
            dreamsim = self._import_dreamsim_factory()

            controller_device, _ = self.controller._input_device_dtype()
            try:
                loaded = dreamsim(pretrained=True, device=str(controller_device))
            except Exception as exc:
                if not self._is_dreamsim_utils_shadow_error(exc):
                    raise
                logger.warning(
                    "ControllerTrainer: dreamsim() failed with utils.trunc_normal_ collision; retrying import recovery path."
                )
                recovered = self._import_dreamsim_with_utils_shim()
                if recovered is None:
                    recovered = self._import_dreamsim_with_trunc_normal_patch()
                if recovered is None:
                    raise
                loaded = recovered(pretrained=True, device=str(controller_device))
            if not isinstance(loaded, tuple) or len(loaded) < 2:
                raise RuntimeError("unexpected dreamsim() return signature")
            self.dreamsim_model, self.dreamsim_preprocess = loaded[0], loaded[1]
            if hasattr(self.dreamsim_model, "eval"):
                self.dreamsim_model.eval()
        except Exception as exc:
            raise RuntimeError(
                    "ControllerTrainer: dreamsim_weight > 0 requires the 'dreamsim' package: %s." % exc
            ) from exc

    def _init_hps_model(self) -> None:
        try:
            import hpsv2  # type: ignore

            self.hps_backend = "hpsv2"
            self.hps_module = hpsv2
            return
        except Exception:
            pass
        try:
            import ImageReward as RM  # type: ignore

            self.hps_backend = "imagereward"
            self.hps_model = RM.load("ImageReward-v1.0")
            return
        except Exception as exc:
            raise RuntimeError(
                "ControllerTrainer: hps_weight > 0 requires either 'hpsv2' or 'image-reward'."
            ) from exc

    def _init_biqa_models(self) -> None:
        try:
            import pyiqa  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "ControllerTrainer: BIQA weights > 0 require the 'pyiqa' package."
            ) from exc
        controller_device, _ = self.controller._input_device_dtype()
        try:
            self.biqa_backend = "qalign"
            self.biqa_quality_model = pyiqa.create_metric("qalign", device=controller_device)
            self.biqa_aesthetic_model = self.biqa_quality_model
            return
        except Exception:
            pass
        try:
            self.biqa_backend = "liqe"
            self.biqa_quality_model = pyiqa.create_metric("liqe_mix", device=controller_device)
            self.biqa_aesthetic_model = pyiqa.create_metric("nima", device=controller_device)
        except Exception as exc:
            raise RuntimeError(
                "ControllerTrainer: BIQA weights > 0 require pyiqa metrics (qalign or liqe_mix+nima)."
            ) from exc

    def _dreamsim_batch_from_rgb(self, rgb: torch.Tensor, *, target_device: torch.device) -> torch.Tensor:
        if self.dreamsim_preprocess is None:
            return rgb.to(device=target_device)
        processed: list[torch.Tensor] = []
        for pil_img in self._pil_batch_from_rgb(rgb):
            out = self.dreamsim_preprocess(pil_img)
            if not torch.is_tensor(out):
                out = torch.as_tensor(out)
            if out.ndim == 3:
                out = out.unsqueeze(0)
            processed.append(out.float())
        return torch.cat(processed, dim=0).to(device=target_device)

    def _score_dreamsim(
        self,
        *,
        teacher_rgb_01: torch.Tensor,
        student_rgb_01: torch.Tensor,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        if self.dreamsim_model is None:
            raise RuntimeError("DreamSim requested but dreamsim model is not initialized.")
        target_device = self._module_primary_device(self.dreamsim_model) or teacher_rgb_01.device
        self._ensure_metric_model_device(self.dreamsim_model, target_device=target_device, component_name="DreamSim model")
        teacher_batch = self._dreamsim_batch_from_rgb(teacher_rgb_01, target_device=target_device)
        student_batch = self._dreamsim_batch_from_rgb(student_rgb_01, target_device=target_device)
        score = self.dreamsim_model(student_batch, teacher_batch)
        return self._to_scalar_tensor(score, device=loss.device, dtype=loss.dtype)

    def _score_hps_batch(
        self,
        *,
        rgb_01: torch.Tensor,
        prompt: str,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        pil_images = self._pil_batch_from_rgb(rgb_01)
        if self.hps_backend == "hpsv2":
            if self.hps_module is None:
                raise RuntimeError("HPS requested but hpsv2 is not initialized.")
            try:
                scores = self.hps_module.score(pil_images, prompt, hps_version=self.hps_version)
            except Exception:
                try:
                    scores = [
                        self.hps_module.score(pil_img, prompt, hps_version=self.hps_version)
                        for pil_img in pil_images
                    ]
                except FileNotFoundError as exc:
                    message = str(exc)
                    if "bpe_simple_vocab_16e6.txt.gz" in message:
                        raise RuntimeError(
                            "ControllerTrainer: hpsv2 is missing tokenizer asset "
                            "'bpe_simple_vocab_16e6.txt.gz'. Re-run install_requirements.sh for this venv."
                        ) from exc
                    raise
            return self._scores_to_tensor(scores, device=loss.device, dtype=loss.dtype).mean()
        if self.hps_backend != "imagereward" or self.hps_model is None:
            raise RuntimeError("HPS requested but no HPS backend is initialized.")
        scores = [self.hps_model.score(prompt, img) for img in pil_images]
        return self._scores_to_tensor(scores, device=loss.device, dtype=loss.dtype).mean()

    def _score_biqa_quality(self, *, rgb_01: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        if self.biqa_quality_model is None:
            raise RuntimeError("BIQA quality requested but quality model is not initialized.")
        target_device = self._module_primary_device(self.biqa_quality_model) or rgb_01.device
        self._ensure_metric_model_device(self.biqa_quality_model, target_device=target_device, component_name="BIQA quality model")
        inp = rgb_01.to(device=target_device)
        out = self.biqa_quality_model(inp, task_="quality") if self.biqa_backend == "qalign" else self.biqa_quality_model(inp)
        return self._to_scalar_tensor(out, device=loss.device, dtype=loss.dtype)

    def _score_biqa_aesthetic(self, *, rgb_01: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        if self.biqa_aesthetic_model is None:
            raise RuntimeError("BIQA aesthetic requested but aesthetic model is not initialized.")
        target_device = self._module_primary_device(self.biqa_aesthetic_model) or rgb_01.device
        self._ensure_metric_model_device(
            self.biqa_aesthetic_model,
            target_device=target_device,
            component_name="BIQA aesthetic model",
        )
        inp = rgb_01.to(device=target_device)
        out = self.biqa_aesthetic_model(inp, task_="aesthetic") if self.biqa_backend == "qalign" else self.biqa_aesthetic_model(inp)
        return self._to_scalar_tensor(out, device=loss.device, dtype=loss.dtype)

    def _init_gemini_client(self) -> None:
        api_key = str(self.gemini_api_key or "").strip()
        api_key_env = str(self.gemini_api_key_env or "").strip()
        if not api_key and not api_key_env:
            raise RuntimeError(
                "ControllerTrainer: Gemini scoring requires gemini_api_key or a non-empty gemini_api_key_env."
            )
        if not api_key:
            api_key = str(os.environ.get(api_key_env, "")).strip()
        if not api_key:
            raise RuntimeError(
                "ControllerTrainer: Gemini scoring enabled but no API key was provided via gemini_api_key "
                f"and environment variable {api_key_env!r} is not set."
            )
        try:
            from google import genai  # type: ignore
            from google.genai import types as genai_types  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ControllerTrainer: Gemini weights > 0 require the 'google-genai' package."
            ) from exc

        self.gemini_client = genai.Client(api_key=api_key)
        self.gemini_types = genai_types

    @staticmethod
    def _parse_gemini_json_scores(response_text: str) -> tuple[float, float]:
        payload = str(response_text or "").strip()
        if not payload:
            raise ValueError("Gemini response is empty.")

        parsed: Any
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            start = payload.find("{")
            end = payload.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"Gemini response is not valid JSON: {payload!r}") from None
            snippet = payload[start : end + 1]
            try:
                parsed = json.loads(snippet)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Gemini response is not valid JSON: {payload!r}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"Gemini response JSON must be an object: {payload!r}")
        if "similarity" not in parsed or "quality" not in parsed:
            raise ValueError(
                "Gemini response JSON must include numeric keys 'similarity' and 'quality'."
            )

        try:
            similarity = float(parsed["similarity"])
            quality = float(parsed["quality"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Gemini response JSON keys 'similarity' and 'quality' must be numeric."
            ) from exc
        if not (np.isfinite(similarity) and np.isfinite(quality)):
            raise ValueError("Gemini response values must be finite numbers.")

        similarity = float(np.clip(similarity, _GEMINI_SCORE_MIN, _GEMINI_SCORE_MAX))
        quality = float(np.clip(quality, _GEMINI_SCORE_MIN, _GEMINI_SCORE_MAX))
        return similarity, quality

    @classmethod
    def _png_bytes_from_rgb01(cls, image_rgb_01: torch.Tensor) -> bytes:
        pil_image = cls._tensor_to_pil(image_rgb_01)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _query_gemini_pair(
        self,
        *,
        teacher_png: bytes,
        student_png: bytes,
    ) -> tuple[float, float]:
        if self.gemini_client is None or self.gemini_types is None:
            raise RuntimeError("Gemini scoring requested but Gemini client is not initialized.")
        if not self.gemini_model:
            raise RuntimeError("Gemini scoring requested but gemini_model is empty.")
        if not self.gemini_prompt:
            raise RuntimeError("Gemini scoring requested but gemini_prompt is empty.")

        types_mod = self.gemini_types
        contents = [
            types_mod.Content(
                role="user",
                parts=[
                    types_mod.Part.from_bytes(mime_type="image/png", data=teacher_png),
                    types_mod.Part.from_bytes(mime_type="image/png", data=student_png),
                    types_mod.Part.from_text(text=self.gemini_prompt),
                ],
            ),
        ]
        config = types_mod.GenerateContentConfig(response_mime_type="application/json")

        chunks: list[str] = []
        try:
            for chunk in self.gemini_client.models.generate_content_stream(
                model=self.gemini_model,
                contents=contents,
                config=config,
            ):
                text = getattr(chunk, "text", "")
                if text:
                    chunks.append(str(text))
        except Exception as exc:
            logger.error(
                "ControllerTrainer: Gemini API request failed (%s: %s).",
                type(exc).__name__,
                exc,
            )
            raise RuntimeError("ControllerTrainer: Gemini API request failed.") from exc
        response_text = "".join(chunks).strip()
        if not response_text:
            raise RuntimeError("ControllerTrainer: Gemini API returned an empty response.")
        return self._parse_gemini_json_scores(response_text)

    def _score_gemini_batch(
        self,
        *,
        teacher_rgb_01: torch.Tensor,
        student_rgb_01: torch.Tensor,
        loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if teacher_rgb_01.shape != student_rgb_01.shape:
            raise ValueError(
                "ControllerTrainer._score_gemini_batch expects teacher and student RGB tensors with matching shape."
            )
        batch_size = int(teacher_rgb_01.shape[0])
        if batch_size <= 0:
            raise ValueError("ControllerTrainer._score_gemini_batch requires a non-empty batch.")
        if batch_size > 1:
            logger.debug("ControllerTrainer: Gemini scoring %d image pairs (one API call per pair).", batch_size)

        similarity_scores: list[float] = []
        quality_scores: list[float] = []
        for idx in range(batch_size):
            teacher_png = self._png_bytes_from_rgb01(teacher_rgb_01[idx])
            student_png = self._png_bytes_from_rgb01(student_rgb_01[idx])
            similarity, quality = self._query_gemini_pair(teacher_png=teacher_png, student_png=student_png)
            similarity_scores.append(float(similarity))
            quality_scores.append(float(quality))

        similarity_tensor = torch.tensor(similarity_scores, device=loss.device, dtype=loss.dtype).mean()
        quality_tensor = torch.tensor(quality_scores, device=loss.device, dtype=loss.dtype).mean()
        return similarity_tensor, quality_tensor

    @staticmethod
    def _gemini_score_to_penalty(score: torch.Tensor) -> torch.Tensor:
        score_min = torch.tensor(_GEMINI_SCORE_MIN, device=score.device, dtype=score.dtype)
        score_max = torch.tensor(_GEMINI_SCORE_MAX, device=score.device, dtype=score.dtype)
        denom = torch.clamp(score_max - score_min, min=1e-6)
        return torch.relu((score_max - score) / denom)

    @staticmethod
    def _ratio_tensor(
        actual_full_attn_ratio: float | torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if torch.is_tensor(actual_full_attn_ratio):
            ratio = actual_full_attn_ratio.to(device=device, dtype=dtype)
            if ratio.ndim > 0:
                ratio = ratio.mean()
            return ratio
        return torch.tensor(float(actual_full_attn_ratio), device=device, dtype=dtype)

    @staticmethod
    def _target_full_attn_ratio_from_ttr_ratio(target_ttr_ratio: float) -> float:
        # target_ttr_ratio is semantically "desired TTR usage fraction", so
        # full-attention usage should be constrained by (1 - target_ttr_ratio).
        target_ttr = max(0.0, min(1.0, float(target_ttr_ratio)))
        return 1.0 - target_ttr

    def compute_loss(
        self,
        *,
        teacher_latent: torch.Tensor,
        student_latent: torch.Tensor,
        actual_full_attn_ratio: float | torch.Tensor,
        teacher_rgb: Optional[torch.Tensor] = None,
        student_rgb: Optional[torch.Tensor] = None,
        include_efficiency_penalty: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        teacher = teacher_latent.float()
        student = student_latent.float()

        rmse = torch.sqrt(torch.mean((student - teacher).square()) + _LPIPS_EPS)
        cosine = self._cosine_distance(student, teacher)
        loss = self.rmse_weight * rmse + self.cosine_weight * cosine

        lpips_term = torch.tensor(0.0, device=loss.device)
        dreamsim_term = torch.tensor(0.0, device=loss.device)
        hps_teacher = torch.tensor(0.0, device=loss.device)
        hps_student = torch.tensor(0.0, device=loss.device)
        hps_penalty = torch.tensor(0.0, device=loss.device)
        biqa_quality_teacher = torch.tensor(0.0, device=loss.device)
        biqa_quality_student = torch.tensor(0.0, device=loss.device)
        biqa_quality_penalty = torch.tensor(0.0, device=loss.device)
        biqa_aesthetic_teacher = torch.tensor(0.0, device=loss.device)
        biqa_aesthetic_student = torch.tensor(0.0, device=loss.device)
        biqa_aesthetic_penalty = torch.tensor(0.0, device=loss.device)
        gemini_similarity = torch.tensor(0.0, device=loss.device)
        gemini_quality = torch.tensor(0.0, device=loss.device)
        gemini_similarity_penalty = torch.tensor(0.0, device=loss.device)
        gemini_quality_penalty = torch.tensor(0.0, device=loss.device)

        needs_rgb = (
            self.lpips_weight > 0
            or self.dreamsim_weight > 0
            or self.hps_weight > 0
            or self.biqa_quality_weight > 0
            or self.biqa_aesthetic_weight > 0
            or self.gemini_similarity_weight > 0
            or self.gemini_quality_weight > 0
        )
        if needs_rgb and (teacher_rgb is None or student_rgb is None):
            raise ValueError("teacher_rgb and student_rgb are required when any RGB quality weights are enabled.")

        if self.lpips_weight > 0:
            if self.lpips_model is None:
                raise RuntimeError("LPIPS requested but lpips_model is not initialized.")
            teacher_rgb = self._prep_lpips_rgb(teacher_rgb)
            student_rgb = self._prep_lpips_rgb(student_rgb)

            lpips_device = teacher_rgb.device
            if student_rgb.device != lpips_device:
                student_rgb = student_rgb.to(device=lpips_device)
            self._ensure_lpips_model_device(lpips_device)

            lpips_term = self.lpips_model(student_rgb, teacher_rgb).mean().to(device=loss.device)
            loss = loss + self.lpips_weight * lpips_term

        if (
            self.dreamsim_weight > 0
            or self.hps_weight > 0
            or self.biqa_quality_weight > 0
            or self.biqa_aesthetic_weight > 0
            or self.gemini_similarity_weight > 0
            or self.gemini_quality_weight > 0
        ):
            teacher_rgb_01 = self._prep_metric_rgb(teacher_rgb)
            student_rgb_01 = self._prep_metric_rgb(student_rgb)
            if self.dreamsim_weight > 0:
                dreamsim_term = self._score_dreamsim(
                    teacher_rgb_01=teacher_rgb_01,
                    student_rgb_01=student_rgb_01,
                    loss=loss,
                )
                loss = loss + self.dreamsim_weight * dreamsim_term
            if self.hps_weight > 0:
                if not self.hps_prompt:
                    logger.warning("ControllerTrainer: hps_weight > 0 but hps_prompt is empty; HPS scores may be less meaningful.")
                hps_teacher = self._score_hps_batch(rgb_01=teacher_rgb_01, prompt=self.hps_prompt, loss=loss)
                hps_student = self._score_hps_batch(rgb_01=student_rgb_01, prompt=self.hps_prompt, loss=loss)
                hps_penalty = torch.relu(hps_teacher - hps_student)
                loss = loss + self.hps_weight * hps_penalty
            if self.biqa_quality_weight > 0:
                biqa_quality_teacher = self._score_biqa_quality(rgb_01=teacher_rgb_01, loss=loss)
                biqa_quality_student = self._score_biqa_quality(rgb_01=student_rgb_01, loss=loss)
                biqa_quality_penalty = torch.relu(biqa_quality_teacher - biqa_quality_student)
                loss = loss + self.biqa_quality_weight * biqa_quality_penalty
            if self.biqa_aesthetic_weight > 0:
                biqa_aesthetic_teacher = self._score_biqa_aesthetic(rgb_01=teacher_rgb_01, loss=loss)
                biqa_aesthetic_student = self._score_biqa_aesthetic(rgb_01=student_rgb_01, loss=loss)
                biqa_aesthetic_penalty = torch.relu(biqa_aesthetic_teacher - biqa_aesthetic_student)
                loss = loss + self.biqa_aesthetic_weight * biqa_aesthetic_penalty
            if self.gemini_similarity_weight > 0 or self.gemini_quality_weight > 0:
                gemini_similarity, gemini_quality = self._score_gemini_batch(
                    teacher_rgb_01=teacher_rgb_01,
                    student_rgb_01=student_rgb_01,
                    loss=loss,
                )
                gemini_similarity_penalty = self._gemini_score_to_penalty(gemini_similarity)
                gemini_quality_penalty = self._gemini_score_to_penalty(gemini_quality)
                loss = loss + self.gemini_similarity_weight * gemini_similarity_penalty
                loss = loss + self.gemini_quality_weight * gemini_quality_penalty

        ratio = self._ratio_tensor(
            actual_full_attn_ratio,
            device=loss.device,
            dtype=loss.dtype,
        )
        target_full_attn_ratio = self._target_full_attn_ratio_from_ttr_ratio(self.target_ttr_ratio)
        target_full_ratio = torch.tensor(target_full_attn_ratio, device=loss.device, dtype=loss.dtype)
        efficiency_penalty = torch.relu(ratio - target_full_ratio)
        if include_efficiency_penalty:
            loss = loss + efficiency_penalty

        actual_full_attn_ratio_value = float(ratio.detach().item())
        metrics = {
            "loss": float(loss.detach().item()),
            "rmse": float(rmse.detach().item()),
            "cosine_distance": float(cosine.detach().item()),
            "lpips": float(lpips_term.detach().item()),
            "dreamsim": float(dreamsim_term.detach().item()),
            "hps_teacher": float(hps_teacher.detach().item()),
            "hps_student": float(hps_student.detach().item()),
            "hps_penalty": float(hps_penalty.detach().item()),
            "biqa_quality_teacher": float(biqa_quality_teacher.detach().item()),
            "biqa_quality_student": float(biqa_quality_student.detach().item()),
            "biqa_quality_penalty": float(biqa_quality_penalty.detach().item()),
            "biqa_aesthetic_teacher": float(biqa_aesthetic_teacher.detach().item()),
            "biqa_aesthetic_student": float(biqa_aesthetic_student.detach().item()),
            "biqa_aesthetic_penalty": float(biqa_aesthetic_penalty.detach().item()),
            "gemini_similarity": float(gemini_similarity.detach().item()),
            "gemini_quality": float(gemini_quality.detach().item()),
            "gemini_similarity_penalty": float(gemini_similarity_penalty.detach().item()),
            "gemini_quality_penalty": float(gemini_quality_penalty.detach().item()),
            "efficiency_penalty": float(efficiency_penalty.detach().item()),
            "actual_full_attn_ratio": actual_full_attn_ratio_value,
            "actual_ttr_ratio": float(1.0 - actual_full_attn_ratio_value),
            "target_ttr_ratio": float(self.target_ttr_ratio),
            "target_full_attn_ratio": float(target_full_attn_ratio),
        }
        return loss, metrics

    def _update_reward_baseline(self, reward: float, decay: float = 0.95) -> None:
        self._reward_count += 1
        reward_f = float(reward)
        # Keep baseline adaptation from accepting sustained low-quality rewards.
        # reward is -quality_loss, so more negative values correspond to worse quality.
        clamped_reward = max(reward_f, float(self.reward_baseline_quality_floor))
        if self._reward_count == 1:
            self._reward_baseline = clamped_reward
        else:
            self._reward_baseline = decay * self._reward_baseline + (1.0 - decay) * clamped_reward

    def reinforce_step(
        self,
        *,
        sigma: float,
        cfg_scale: float,
        width: int,
        height: int,
        sampled_mask: torch.Tensor,
        reward: float,
        actual_full_attn_ratio: float,
        eligible_layer_mask: Optional[torch.Tensor] = None,
        actual_full_attn_ratio_overall: Optional[float] = None,
    ) -> Dict[str, float]:
        self.controller.train()
        reward_quality = float(reward)

        # ComfyUI often executes nodes in inference_mode; force grad-enabled training here.
        with torch.inference_mode(False):
            with torch.enable_grad():
                logits = self.controller(sigma=sigma, cfg_scale=cfg_scale, width=width, height=height)
                probs = torch.sigmoid(logits)
                probs_clamped = probs.clamp(min=1e-6, max=1.0 - 1e-6)
                per_layer_entropy = -(
                    probs_clamped * torch.log(probs_clamped)
                    + (1.0 - probs_clamped) * torch.log(1.0 - probs_clamped)
                )

                mask = sampled_mask.to(device=logits.device, dtype=logits.dtype).reshape(-1).detach().clone()
                if mask.numel() != probs.numel():
                    raise ValueError(
                        "ControllerTrainer.reinforce_step: sampled_mask length "
                        f"{int(mask.numel())} does not match controller layer count {int(probs.numel())}."
                    )

                if eligible_layer_mask is not None:
                    # ComfyUI can pass inference-mode tensors into nodes; clone here so
                    # boolean indexing works in autograd-tracked ops below.
                    eligible = (
                        eligible_layer_mask.to(device=logits.device, dtype=torch.bool)
                        .reshape(-1)
                        .detach()
                        .clone()
                    )
                    if eligible.numel() != probs.numel():
                        raise ValueError(
                            "ControllerTrainer.reinforce_step: eligible_layer_mask length "
                            f"{int(eligible.numel())} does not match controller layer count {int(probs.numel())}."
                        )
                else:
                    eligible = torch.ones_like(probs, dtype=torch.bool)

                eligible_count = int(eligible.sum().item())
                total_layer_count = int(probs.numel())
                if eligible_count <= 0:
                    raise ValueError("ControllerTrainer.reinforce_step: eligible_layer_mask must include at least one layer.")
                forced_full_layer_count = int(total_layer_count - eligible_count)
                entropy = per_layer_entropy[eligible].mean()
                entropy_bonus = float(self.lambda_entropy * entropy.detach().item())

                log_probs = mask * torch.log(probs + 1e-8) + (1.0 - mask) * torch.log(1.0 - probs + 1e-8)
                # Restrict policy gradient to controllable layers only.
                total_log_prob = log_probs[eligible].sum()

                target_full_attn_ratio = self._target_full_attn_ratio_from_ttr_ratio(self.target_ttr_ratio)
                actual_full_attn_eligible = float(mask[eligible].mean().item())
                efficiency_penalty_value = abs(actual_full_attn_eligible - float(target_full_attn_ratio))
                efficiency_penalty_weighted = float(self.lambda_eff * efficiency_penalty_value)
                reward_value = reward_quality - efficiency_penalty_weighted + entropy_bonus
                baselined_reward = reward_value - self._reward_baseline
                self._update_reward_baseline(reward_value)

                policy_loss = -baselined_reward * total_log_prob
                probs_mean_eligible = probs[eligible].mean()
                loss = policy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                mask_mean_overall = float(mask.mean().item())
                mask_mean_eligible = float(mask[eligible].mean().item())
                probs_mean_overall = float(probs.detach().mean().item())
                probs_mean_eligible_value = float(probs_mean_eligible.detach().item())
                expected_full_attn_ratio_overall = float(
                    (probs_mean_eligible_value * float(eligible_count) + float(forced_full_layer_count))
                    / float(max(1, total_layer_count))
                )
                actual_full_attn_ratio_value = float(actual_full_attn_ratio)
                actual_full_attn_ratio_overall_value = (
                    float(actual_full_attn_ratio_overall)
                    if actual_full_attn_ratio_overall is not None
                    else mask_mean_overall
                )
                metrics = {
                    "policy_loss": float(policy_loss.detach().item()),
                    "efficiency_penalty": float(efficiency_penalty_value),
                    "efficiency_penalty_weighted": float(efficiency_penalty_weighted),
                    "total_loss": float(loss.detach().item()),
                    "reward": reward_value,
                    "reward_quality": reward_quality,
                    "reward_baseline": float(self._reward_baseline),
                    "baselined_reward": float(baselined_reward),
                    "entropy": float(entropy.detach().item()),
                    "entropy_bonus": float(entropy_bonus),
                    "mask_mean": mask_mean_eligible,
                    "mask_mean_overall": mask_mean_overall,
                    "full_attn_mask_mean": mask_mean_eligible,
                    "full_attn_mask_mean_overall": mask_mean_overall,
                    "ttr_mask_mean": float(1.0 - mask_mean_eligible),
                    "ttr_mask_mean_overall": float(1.0 - mask_mean_overall),
                    "probs_mean": probs_mean_eligible_value,
                    "probs_mean_overall": probs_mean_overall,
                    "expected_full_attn_ratio": probs_mean_eligible_value,
                    "expected_full_attn_ratio_overall": expected_full_attn_ratio_overall,
                    "expected_ttr_ratio": float(1.0 - probs_mean_eligible_value),
                    "expected_ttr_ratio_overall": float(1.0 - expected_full_attn_ratio_overall),
                    "actual_full_attn_ratio": actual_full_attn_ratio_value,
                    "actual_full_attn_ratio_overall": actual_full_attn_ratio_overall_value,
                    "actual_ttr_ratio": float(1.0 - actual_full_attn_ratio_value),
                    "actual_ttr_ratio_overall": float(1.0 - actual_full_attn_ratio_overall_value),
                    "target_ttr_ratio": float(self.target_ttr_ratio),
                    "target_full_attn_ratio": float(target_full_attn_ratio),
                    "lambda_eff": float(self.lambda_eff),
                    "lambda_entropy": float(self.lambda_entropy),
                    "eligible_layer_count": float(eligible_count),
                    "total_layer_count": float(total_layer_count),
                    "forced_full_layer_count": float(forced_full_layer_count),
                }
        return metrics

    def train_step(
        self,
        *,
        sigma: float,
        cfg_scale: float,
        width: int,
        height: int,
        teacher_latent: torch.Tensor,
        student_forward_fn: Callable[[torch.Tensor, torch.Tensor], Dict[str, Any]],
        teacher_rgb: Optional[torch.Tensor] = None,
        gumbel_temperature: float = 1.0,
        hard_mask: bool = True,
    ) -> Dict[str, float]:
        if not self._train_step_warned:
            logger.warning(
                "ControllerTrainer.train_step is deprecated; prefer reinforce_step for policy-gradient training."
            )
            self._train_step_warned = True
        if not callable(student_forward_fn):
            raise TypeError("ControllerTrainer.train_step requires a callable student_forward_fn(mask, logits).")
        self.controller.train()
        with torch.inference_mode(False):
            with torch.enable_grad():
                logits = self.controller(sigma=sigma, cfg_scale=cfg_scale, width=width, height=height)
                mask = self.controller.sample_training_mask(logits, temperature=gumbel_temperature, hard=hard_mask)

                forward_out = student_forward_fn(mask, logits)
                if not isinstance(forward_out, dict):
                    raise TypeError("student_forward_fn must return a dict with at least 'student_latent'.")
                student_latent = forward_out.get("student_latent")
                if not torch.is_tensor(student_latent):
                    raise ValueError("student_forward_fn output must include tensor 'student_latent'.")
                actual_full_attn_ratio = forward_out.get("actual_full_attn_ratio", mask.mean())
                student_rgb = forward_out.get("student_rgb")

                loss, metrics = self.compute_loss(
                    teacher_latent=teacher_latent,
                    student_latent=student_latent,
                    actual_full_attn_ratio=actual_full_attn_ratio,
                    teacher_rgb=teacher_rgb,
                    student_rgb=student_rgb,
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                metrics["mask_mean"] = float(mask.detach().mean().item())
                metrics["layers_full_ratio"] = float(metrics.get("actual_full_attn_ratio", metrics["mask_mean"]))
        return metrics
