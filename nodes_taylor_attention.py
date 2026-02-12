from __future__ import annotations

import logging
import math
import os
from typing import Dict, Any, Optional

import torch
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import comfy.patcher_extension as patcher_extension

import flux2_comet_logging
import flux2_ttr
import flux2_ttr_controller
import sweep_utils
from flux2_ttr_controller_trainer_node import (
    ControllerTrainerNodeDependencies,
    ControllerTrainerNodeInputs,
    run_controller_trainer_node,
)

try:
    import comfy.sample as comfy_sample
    import comfy.samplers as comfy_samplers
    import comfy.model_management as comfy_model_management
except Exception:
    comfy_sample = None
    comfy_samplers = None
    comfy_model_management = None

logger = logging.getLogger(__name__)

_TRAINING_CONFIG_KEYS = (
    "loss_config",
    "optimizer_config",
    "schedule_config",
    "logging_config",
)

_TRAINING_CONFIG_DEFAULTS: dict[str, dict[str, Any]] = {
    "loss_config": {
        "rmse_weight": 1.0,
        "cosine_weight": 1.0,
        "lpips_weight": 0.0,
        "dreamsim_weight": 0.0,
        "hps_weight": 0.0,
        "biqa_quality_weight": 0.0,
        "biqa_aesthetic_weight": 0.0,
        "gemini_similarity_weight": 0.0,
        "gemini_quality_weight": 0.0,
        "gemini_model": "gemini-3-flash-preview",
        "gemini_api_key_env": "GEMINI_API_KEY",
        "reward_baseline_quality_floor": -0.3,
        "huber_beta": 0.05,
    },
    "optimizer_config": {
        "learning_rate": 1e-4,
        "grad_clip_norm": 1.0,
        "alpha_lr_multiplier": 5.0,
        "phi_lr_multiplier": 1.0,
    },
    "schedule_config": {
        "target_ttr_ratio": 0.7,
        "lambda_eff": 1.0,
        "lambda_entropy": 0.1,
        "gumbel_temperature_start": 1.0,
        "gumbel_temperature_end": 0.5,
        "warmup_steps": 0,
    },
    "logging_config": {
        "comet_enabled": False,
        "comet_api_key": "",
        "comet_project_name": "ttr-distillation",
        "comet_workspace": "comet-workspace",
        "comet_experiment": "",
        "log_every": 10,
    },
}

_LEGACY_TRAINER_DEFAULTS = {
    "learning_rate": 1e-4,
    "grad_clip_norm": 1.0,
    "alpha_lr_multiplier": 5.0,
    "phi_lr_multiplier": 1.0,
    "huber_beta": 0.05,
    "comet_enabled": True,
    "comet_api_key": "",
    "comet_project_name": "ttr-distillation",
    "comet_workspace": "comet-workspace",
    "comet_experiment": "",
    "log_every": 50,
}

_CONTROLLER_COMET_NAMESPACE = "flux2_ttr_controller"
_DEFAULT_TTR_CHECKPOINT_FILENAME = "flux2_ttr.pt"
_DEFAULT_CONTROLLER_CHECKPOINT_FILENAME = "flux2_ttr_controller.pt"


@io.comfytype(io_type="TTR_TRAINING_CONFIG")
class TTRTrainingConfig(io.ComfyTypeIO):
    Type = dict


@io.comfytype(io_type="TTR_CONTROLLER")
class TTRControllerType(io.ComfyTypeIO):
    Type = flux2_ttr_controller.TTRController


def _normalize_training_config(training_config: Optional[dict]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {key: {} for key in _TRAINING_CONFIG_KEYS}
    if not isinstance(training_config, dict):
        return normalized
    for key in _TRAINING_CONFIG_KEYS:
        section = training_config.get(key)
        if isinstance(section, dict):
            normalized[key] = dict(section)
    return normalized


def _float_or(value: Any, default: float) -> float:
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _bool_or(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _string_or(value: Any, default: str) -> str:
    if value is None:
        return str(default)
    return str(value)


def _default_comet_experiment_name() -> str:
    return flux2_comet_logging.generate_experiment_key(__file__)


def _default_ttr_checkpoint_path() -> str:
    return flux2_ttr.resolve_default_checkpoint_path(
        "",
        default_filename=_DEFAULT_TTR_CHECKPOINT_FILENAME,
        anchor_file=__file__,
        ensure_dir=False,
    )


def _default_controller_checkpoint_path() -> str:
    return flux2_ttr.resolve_default_checkpoint_path(
        "",
        default_filename=_DEFAULT_CONTROLLER_CHECKPOINT_FILENAME,
        anchor_file=__file__,
        ensure_dir=False,
    )


def _resolve_checkpoint_path_or_default(
    raw_path: str,
    *,
    default_filename: str,
    component_name: str,
    field_name: str,
) -> str:
    provided = str(raw_path or "").strip()
    resolved = flux2_ttr.resolve_default_checkpoint_path(
        provided,
        default_filename=default_filename,
        anchor_file=__file__,
        ensure_dir=True,
    )
    if not provided:
        logger.info("%s: using default %s=%s", component_name, field_name, resolved)
    return resolved


def _extract_layer_idx(layer_key: str) -> Optional[int]:
    if ":" not in layer_key:
        return None
    _, idx_text = layer_key.split(":", 1)
    try:
        return int(idx_text)
    except Exception:
        return None


def _vae_input(name: str, *, tooltip: Optional[str] = None):
    # Comfy API variants use different capitalization for this type.
    for attr_name in ("Vae", "VAE"):
        vae_type = getattr(io, attr_name, None)
        if vae_type is not None and hasattr(vae_type, "Input"):
            return vae_type.Input(name, tooltip=tooltip)
    logger.warning("Flux2TTR: io.VAE is unavailable in this ComfyUI API build; using AnyType input for '%s'.", name)
    return io.AnyType.Input(name, tooltip=tooltip)


def _resolve_ttr_runtime_from_model(model_ttr) -> tuple[flux2_ttr.Flux2TTRRuntime, dict]:
    transformer_options = model_ttr.model_options.setdefault("transformer_options", {})
    cfg = transformer_options.get("flux2_ttr")
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        raise ValueError(
            "Flux2TTRControllerTrainer requires a model with trained TTR modules. Run Flux2TTRTrainer first."
        )

    runtime_id = cfg.get("runtime_id")
    runtime = flux2_ttr.get_runtime(runtime_id) if isinstance(runtime_id, str) else None
    if runtime is None:
        runtime = flux2_ttr._recover_runtime_from_config(cfg)
        if runtime is not None:
            new_runtime_id = flux2_ttr.register_runtime(runtime)
            cfg["runtime_id"] = new_runtime_id
    if runtime is None:
        raise ValueError(
            "Flux2TTRControllerTrainer requires a model with trained TTR modules. Run Flux2TTRTrainer first."
        )

    ready_layers = [layer_key for layer_key, ready in runtime.layer_ready.items() if ready and layer_key.startswith("single:")]
    if not ready_layers:
        raise ValueError(
            "Flux2TTRControllerTrainer requires a model with trained TTR modules. Run Flux2TTRTrainer first."
        )
    return runtime, cfg


def _build_training_config_payload(
    *,
    rmse_weight: float,
    cosine_weight: float,
    lpips_weight: float,
    dreamsim_weight: float,
    hps_weight: float,
    biqa_quality_weight: float,
    biqa_aesthetic_weight: float,
    gemini_similarity_weight: float,
    gemini_quality_weight: float,
    gemini_model: str,
    gemini_api_key_env: str,
    reward_baseline_quality_floor: float,
    huber_beta: float,
    learning_rate: float,
    grad_clip_norm: float,
    alpha_lr_multiplier: float,
    phi_lr_multiplier: float,
    target_ttr_ratio: float,
    lambda_eff: float,
    lambda_entropy: float,
    gumbel_temperature_start: float,
    gumbel_temperature_end: float,
    warmup_steps: int,
    comet_enabled: bool,
    comet_api_key: str,
    comet_project_name: str,
    comet_workspace: str,
    comet_experiment: str,
    log_every: int,
) -> dict[str, dict[str, Any]]:
    return {
        "loss_config": {
            "rmse_weight": float(rmse_weight),
            "cosine_weight": float(cosine_weight),
            "lpips_weight": float(lpips_weight),
            "dreamsim_weight": float(dreamsim_weight),
            "hps_weight": float(hps_weight),
            "biqa_quality_weight": float(biqa_quality_weight),
            "biqa_aesthetic_weight": float(biqa_aesthetic_weight),
            "gemini_similarity_weight": float(gemini_similarity_weight),
            "gemini_quality_weight": float(gemini_quality_weight),
            "gemini_model": str(gemini_model or "gemini-3-flash-preview"),
            "gemini_api_key_env": str(gemini_api_key_env or "GEMINI_API_KEY"),
            "reward_baseline_quality_floor": float(reward_baseline_quality_floor),
            "huber_beta": float(huber_beta),
        },
        "optimizer_config": {
            "learning_rate": float(learning_rate),
            "grad_clip_norm": float(grad_clip_norm),
            "alpha_lr_multiplier": float(alpha_lr_multiplier),
            "phi_lr_multiplier": float(phi_lr_multiplier),
        },
        "schedule_config": {
            "target_ttr_ratio": float(target_ttr_ratio),
            "lambda_eff": max(0.0, float(lambda_eff)),
            "lambda_entropy": max(0.0, float(lambda_entropy)),
            "gumbel_temperature_start": float(gumbel_temperature_start),
            "gumbel_temperature_end": float(gumbel_temperature_end),
            "warmup_steps": int(warmup_steps),
        },
        "logging_config": {
            "comet_enabled": bool(comet_enabled),
            "comet_api_key": str(comet_api_key or ""),
            "comet_project_name": str(comet_project_name or "ttr-distillation"),
            "comet_workspace": str(comet_workspace or "comet-workspace"),
            "comet_experiment": flux2_comet_logging.normalize_experiment_key(
                str(comet_experiment or "").strip(),
                allow_empty=True,
            ),
            "log_every": max(1, int(log_every)),
        },
    }


class Flux2TTRTrainingParameters(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Flux2TTRTrainingParameters",
            display_name="Flux2TTRTrainingParameters",
            category="advanced/attention",
            description="Shared Flux2TTR training hyperparameters for Phase-1 and Phase-2 nodes.",
            inputs=[
                io.Float.Input("rmse_weight", default=1.0, min=0.0, max=1000.0, step=1e-3),
                io.Float.Input("cosine_weight", default=1.0, min=0.0, max=1000.0, step=1e-3),
                io.Float.Input("lpips_weight", default=0.0, min=0.0, max=1000.0, step=1e-3),
                io.Float.Input("dreamsim_weight", default=0.0, min=0.0, max=1000.0, step=1e-3),
                io.Float.Input("hps_weight", default=0.0, min=0.0, max=1000.0, step=1e-3),
                io.Float.Input("biqa_quality_weight", default=0.0, min=0.0, max=1000.0, step=1e-3),
                io.Float.Input("biqa_aesthetic_weight", default=0.0, min=0.0, max=1000.0, step=1e-3),
                io.Float.Input("gemini_similarity_weight", default=0.0, min=0.0, max=1000.0, step=1e-3),
                io.Float.Input("gemini_quality_weight", default=0.0, min=0.0, max=1000.0, step=1e-3),
                io.String.Input(
                    "gemini_model",
                    default="gemini-3-flash-preview",
                    multiline=False,
                    tooltip="Gemini model used for image-to-image quality scoring when Gemini weights are enabled.",
                ),
                io.String.Input(
                    "gemini_api_key_env",
                    default="GEMINI_API_KEY",
                    multiline=False,
                    tooltip="Environment variable name that holds the Gemini API key.",
                ),
                io.Float.Input(
                    "reward_baseline_quality_floor",
                    default=-0.3,
                    min=-10.0,
                    max=1.0,
                    step=1e-3,
                    tooltip="Clamp floor for reward-baseline EMA updates in controller training.",
                ),
                io.Float.Input("huber_beta", default=0.05, min=1e-6, max=10.0, step=1e-4),
                io.Float.Input("learning_rate", default=1e-4, min=1e-7, max=1.0, step=1e-7),
                io.Float.Input("grad_clip_norm", default=1.0, min=0.0, max=100.0, step=1e-3),
                io.Float.Input("alpha_lr_multiplier", default=5.0, min=0.0, max=100.0, step=1e-3),
                io.Float.Input("phi_lr_multiplier", default=1.0, min=0.0, max=100.0, step=1e-3),
                io.Float.Input("target_ttr_ratio", default=0.7, min=0.0, max=1.0, step=1e-3),
                io.Float.Input("lambda_eff", default=1.0, min=0.0, max=100.0, step=1e-3),
                io.Float.Input("lambda_entropy", default=0.1, min=0.0, max=1.0, step=1e-3),
                io.Float.Input("gumbel_temperature_start", default=1.0, min=1e-4, max=10.0, step=1e-3),
                io.Float.Input("gumbel_temperature_end", default=0.5, min=1e-4, max=10.0, step=1e-3),
                io.Int.Input("warmup_steps", default=0, min=0, max=1000000, step=1),
                io.Boolean.Input("comet_enabled", default=False),
                io.String.Input("comet_api_key", default="", multiline=False),
                io.String.Input("comet_project_name", default="ttr-distillation", multiline=False),
                io.String.Input("comet_workspace", default="", multiline=False),
                io.String.Input(
                    "comet_experiment",
                    default=_default_comet_experiment_name(),
                    multiline=False,
                    tooltip="Comet experiment key. Use a stable value to keep logging to one persistent experiment.",
                ),
                io.Int.Input("log_every", default=10, min=1, max=1000000, step=1),
            ],
            outputs=[TTRTrainingConfig.Output("training_config")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        rmse_weight: float,
        cosine_weight: float,
        lpips_weight: float,
        dreamsim_weight: float,
        hps_weight: float,
        biqa_quality_weight: float,
        biqa_aesthetic_weight: float,
        gemini_similarity_weight: float,
        gemini_quality_weight: float,
        gemini_model: str,
        gemini_api_key_env: str,
        reward_baseline_quality_floor: float,
        huber_beta: float,
        learning_rate: float,
        grad_clip_norm: float,
        alpha_lr_multiplier: float,
        phi_lr_multiplier: float,
        target_ttr_ratio: float,
        lambda_eff: float,
        lambda_entropy: float,
        gumbel_temperature_start: float,
        gumbel_temperature_end: float,
        warmup_steps: int,
        comet_enabled: bool,
        comet_api_key: str,
        comet_project_name: str,
        comet_workspace: str,
        comet_experiment: str,
        log_every: int,
    ) -> io.NodeOutput:
        training_config = _build_training_config_payload(
            rmse_weight=rmse_weight,
            cosine_weight=cosine_weight,
            lpips_weight=lpips_weight,
            dreamsim_weight=dreamsim_weight,
            hps_weight=hps_weight,
            biqa_quality_weight=biqa_quality_weight,
            biqa_aesthetic_weight=biqa_aesthetic_weight,
            gemini_similarity_weight=gemini_similarity_weight,
            gemini_quality_weight=gemini_quality_weight,
            gemini_model=gemini_model,
            gemini_api_key_env=gemini_api_key_env,
            reward_baseline_quality_floor=reward_baseline_quality_floor,
            huber_beta=huber_beta,
            learning_rate=learning_rate,
            grad_clip_norm=grad_clip_norm,
            alpha_lr_multiplier=alpha_lr_multiplier,
            phi_lr_multiplier=phi_lr_multiplier,
            target_ttr_ratio=target_ttr_ratio,
            lambda_eff=lambda_eff,
            lambda_entropy=lambda_entropy,
            gumbel_temperature_start=gumbel_temperature_start,
            gumbel_temperature_end=gumbel_temperature_end,
            warmup_steps=warmup_steps,
            comet_enabled=comet_enabled,
            comet_api_key=comet_api_key,
            comet_project_name=comet_project_name,
            comet_workspace=comet_workspace,
            comet_experiment=comet_experiment,
            log_every=log_every,
        )
        return io.NodeOutput(training_config)


class Flux2TTRTrainer(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Flux2TTRTrainer",
            display_name="Flux2TTRTrainer",
            category="advanced/attention",
            description="Phase-1: distill Flux TTR linear attention modules from native attention.",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("latents"),
                io.Conditioning.Input("conditioning"),
                TTRTrainingConfig.Input("training_config", optional=True),
                io.Int.Input("steps", default=512, min=0, max=200000, step=1),
                io.Boolean.Input("training", default=True, tooltip="Train TTR layers by distillation when enabled."),
                io.Boolean.Input(
                    "training_preview_ttr",
                    default=True,
                    tooltip="When training, output TTR student attention for visual preview instead of teacher passthrough.",
                ),
                io.String.Input(
                    "checkpoint_path",
                    default=_default_ttr_checkpoint_path(),
                    multiline=False,
                    tooltip="Checkpoint file to load/save TTR layer weights (defaults to ComfyUI/models/approximate_attention/flux2_ttr.pt).",
                ),
                io.Int.Input(
                    "feature_dim",
                    default=256,
                    min=128,
                    max=8192,
                    step=256,
                    tooltip="Kernel feature dimension (must be a multiple of 256).",
                ),
                io.Int.Input(
                    "query_chunk_size",
                    default=256,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Query chunk size for kernel attention evaluation.",
                ),
                io.Int.Input(
                    "key_chunk_size",
                    default=1024,
                    min=1,
                    max=8192,
                    step=1,
                    tooltip="Key chunk size for kernel KV/Ksum accumulation.",
                ),
                io.Float.Input(
                    "landmark_fraction",
                    default=0.08,
                    min=0.01,
                    max=0.5,
                    step=1e-3,
                    tooltip="Fraction of image tokens used as landmarks for exact softmax residual.",
                ),
                io.Int.Input(
                    "landmark_min",
                    default=64,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip="Minimum landmark count used at low resolution.",
                ),
                io.Int.Input(
                    "landmark_max",
                    default=0,
                    min=0,
                    max=2048,
                    step=1,
                    tooltip="Maximum landmark count used at high resolution (0 means unlimited).",
                ),
                io.Int.Input(
                    "text_tokens_guess",
                    default=77,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip="Assumed number of text tokens at the start of sequence for landmark selection.",
                ),
                io.Float.Input(
                    "alpha_init",
                    default=0.1,
                    min=0.0,
                    max=10.0,
                    step=1e-3,
                    tooltip="Initial residual gate for landmark softmax branch.",
                ),
                io.Int.Input(
                    "training_query_token_cap",
                    default=128,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Max number of query tokens per replay sample; keys/values always stay full length.",
                ),
                io.Int.Input(
                    "replay_buffer_size",
                    default=8,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Replay buffer capacity per layer for distillation samples.",
                ),
                io.Boolean.Input(
                    "replay_offload_cpu",
                    default=True,
                    tooltip="Store replay samples on CPU (reduced precision) to reduce VRAM pressure.",
                ),
                io.Int.Input(
                    "replay_max_mb",
                    default=768,
                    min=64,
                    max=65536,
                    step=1,
                    tooltip="Global replay memory budget in MB across all layers.",
                ),
                io.Int.Input(
                    "train_steps_per_call",
                    default=1,
                    min=1,
                    max=32,
                    step=1,
                    tooltip="Number of replay optimization steps run per attention call.",
                ),
                io.Float.Input(
                    "readiness_threshold",
                    default=0.12,
                    min=0.0,
                    max=10.0,
                    step=1e-4,
                    tooltip="Enable student inference for a layer only when EMA loss is below this threshold.",
                ),
                io.Int.Input(
                    "readiness_min_updates",
                    default=24,
                    min=0,
                    max=100000,
                    step=1,
                    tooltip="Minimum replay updates before a layer can be marked ready.",
                ),
                io.Boolean.Input(
                    "enable_memory_reserve",
                    default=False,
                    tooltip="Call ComfyUI free_memory before HKR attention allocations (can offload aggressively).",
                ),
                io.Int.Input(
                    "layer_start",
                    default=-1,
                    min=-1,
                    max=512,
                    step=1,
                    tooltip="Only apply TTR to single blocks with index >= layer_start (-1 disables).",
                ),
                io.Int.Input(
                    "layer_end",
                    default=-1,
                    min=-1,
                    max=512,
                    step=1,
                    tooltip="Only apply TTR to single blocks with index <= layer_end (-1 disables).",
                ),
                io.Float.Input(
                    "cfg_scale",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=1e-3,
                    tooltip="Default CFG scale used when transformer_options does not provide guidance scale.",
                ),
                io.Int.Input(
                    "min_swap_layers",
                    default=1,
                    min=0,
                    max=512,
                    step=1,
                    tooltip="Training mode: minimum number of eligible single layers to swap to TTR per diffusion step.",
                ),
                io.Int.Input(
                    "max_swap_layers",
                    default=-1,
                    min=-1,
                    max=512,
                    step=1,
                    tooltip="Training mode: maximum swapped layers per diffusion step (-1 means all eligible layers).",
                ),
                io.Boolean.Input(
                    "inference_mixed_precision",
                    default=True,
                    tooltip="Use input dtype (bf16/fp16) for TTR inference on CUDA for speed.",
                ),
                io.String.Input(
                    "controller_checkpoint_path",
                    default="",
                    multiline=False,
                    tooltip="Optional Phase-2 controller checkpoint used for inference-time layer routing.",
                ),
            ],
            outputs=[io.Model.Output(), io.Float.Output("loss_value")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        latents,
        conditioning,
        training_config: Optional[dict],
        steps: int,
        training: bool,
        training_preview_ttr: bool,
        checkpoint_path: str,
        feature_dim: int,
        query_chunk_size: int,
        key_chunk_size: int,
        landmark_fraction: float,
        landmark_min: int,
        landmark_max: int,
        text_tokens_guess: int,
        alpha_init: float,
        training_query_token_cap: int,
        replay_buffer_size: int,
        replay_offload_cpu: bool,
        replay_max_mb: int,
        train_steps_per_call: int,
        readiness_threshold: float,
        readiness_min_updates: int,
        enable_memory_reserve: bool,
        layer_start: int,
        layer_end: int,
        cfg_scale: float,
        min_swap_layers: int,
        max_swap_layers: int,
        inference_mixed_precision: bool,
        controller_checkpoint_path: str,
    ) -> io.NodeOutput:
        del latents, conditioning
        feature_dim = flux2_ttr.validate_feature_dim(feature_dim)
        checkpoint_path = _resolve_checkpoint_path_or_default(
            checkpoint_path,
            default_filename=_DEFAULT_TTR_CHECKPOINT_FILENAME,
            component_name="Flux2TTRTrainer",
            field_name="checkpoint_path",
        )
        controller_checkpoint_path = (controller_checkpoint_path or "").strip()
        train_steps = int(steps)

        normalized_cfg = _normalize_training_config(training_config)
        loss_cfg = normalized_cfg["loss_config"]
        opt_cfg = normalized_cfg["optimizer_config"]
        schedule_cfg = normalized_cfg["schedule_config"]
        logging_cfg = normalized_cfg["logging_config"]
        has_training_config = isinstance(training_config, dict)

        if has_training_config:
            learning_rate = _float_or(opt_cfg.get("learning_rate"), _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["learning_rate"])
            grad_clip_norm = _float_or(opt_cfg.get("grad_clip_norm"), _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["grad_clip_norm"])
            alpha_lr_multiplier = _float_or(opt_cfg.get("alpha_lr_multiplier"), _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["alpha_lr_multiplier"])
            phi_lr_multiplier = _float_or(opt_cfg.get("phi_lr_multiplier"), _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["phi_lr_multiplier"])
            huber_beta = _float_or(loss_cfg.get("huber_beta"), _TRAINING_CONFIG_DEFAULTS["loss_config"]["huber_beta"])
            comet_enabled = _bool_or(logging_cfg.get("comet_enabled"), _TRAINING_CONFIG_DEFAULTS["logging_config"]["comet_enabled"])
            comet_api_key = _string_or(logging_cfg.get("comet_api_key"), _TRAINING_CONFIG_DEFAULTS["logging_config"]["comet_api_key"])
            comet_project_name = _string_or(logging_cfg.get("comet_project_name"), _TRAINING_CONFIG_DEFAULTS["logging_config"]["comet_project_name"])
            comet_workspace = _string_or(logging_cfg.get("comet_workspace"), _TRAINING_CONFIG_DEFAULTS["logging_config"]["comet_workspace"])
            comet_experiment = _string_or(logging_cfg.get("comet_experiment"), _TRAINING_CONFIG_DEFAULTS["logging_config"]["comet_experiment"])
            comet_log_every = max(1, _int_or(logging_cfg.get("log_every"), _TRAINING_CONFIG_DEFAULTS["logging_config"]["log_every"]))
        else:
            learning_rate = float(_LEGACY_TRAINER_DEFAULTS["learning_rate"])
            grad_clip_norm = float(_LEGACY_TRAINER_DEFAULTS["grad_clip_norm"])
            alpha_lr_multiplier = float(_LEGACY_TRAINER_DEFAULTS["alpha_lr_multiplier"])
            phi_lr_multiplier = float(_LEGACY_TRAINER_DEFAULTS["phi_lr_multiplier"])
            huber_beta = float(_LEGACY_TRAINER_DEFAULTS["huber_beta"])
            comet_enabled = bool(_LEGACY_TRAINER_DEFAULTS["comet_enabled"])
            comet_api_key = str(_LEGACY_TRAINER_DEFAULTS["comet_api_key"])
            comet_project_name = str(_LEGACY_TRAINER_DEFAULTS["comet_project_name"])
            comet_workspace = str(_LEGACY_TRAINER_DEFAULTS["comet_workspace"])
            comet_experiment = str(_LEGACY_TRAINER_DEFAULTS["comet_experiment"])
            comet_log_every = int(_LEGACY_TRAINER_DEFAULTS["log_every"])

        target_ttr_ratio = _float_or(
            schedule_cfg.get("target_ttr_ratio"),
            _TRAINING_CONFIG_DEFAULTS["schedule_config"]["target_ttr_ratio"],
        )
        lambda_eff = _float_or(
            schedule_cfg.get("lambda_eff"),
            _TRAINING_CONFIG_DEFAULTS["schedule_config"]["lambda_eff"],
        )
        lambda_entropy = _float_or(
            schedule_cfg.get("lambda_entropy"),
            _TRAINING_CONFIG_DEFAULTS["schedule_config"]["lambda_entropy"],
        )
        gumbel_temperature_start = _float_or(
            schedule_cfg.get("gumbel_temperature_start"),
            _TRAINING_CONFIG_DEFAULTS["schedule_config"]["gumbel_temperature_start"],
        )
        gumbel_temperature_end = _float_or(
            schedule_cfg.get("gumbel_temperature_end"),
            _TRAINING_CONFIG_DEFAULTS["schedule_config"]["gumbel_temperature_end"],
        )
        warmup_steps = max(0, _int_or(schedule_cfg.get("warmup_steps"), _TRAINING_CONFIG_DEFAULTS["schedule_config"]["warmup_steps"]))
        rmse_weight = _float_or(loss_cfg.get("rmse_weight"), _TRAINING_CONFIG_DEFAULTS["loss_config"]["rmse_weight"])
        cosine_weight = _float_or(loss_cfg.get("cosine_weight"), _TRAINING_CONFIG_DEFAULTS["loss_config"]["cosine_weight"])
        lpips_weight = _float_or(loss_cfg.get("lpips_weight"), _TRAINING_CONFIG_DEFAULTS["loss_config"]["lpips_weight"])
        dreamsim_weight = _float_or(loss_cfg.get("dreamsim_weight"), _TRAINING_CONFIG_DEFAULTS["loss_config"]["dreamsim_weight"])
        hps_weight = _float_or(loss_cfg.get("hps_weight"), _TRAINING_CONFIG_DEFAULTS["loss_config"]["hps_weight"])
        biqa_quality_weight = _float_or(
            loss_cfg.get("biqa_quality_weight"),
            _TRAINING_CONFIG_DEFAULTS["loss_config"]["biqa_quality_weight"],
        )
        biqa_aesthetic_weight = _float_or(
            loss_cfg.get("biqa_aesthetic_weight"),
            _TRAINING_CONFIG_DEFAULTS["loss_config"]["biqa_aesthetic_weight"],
        )
        gemini_similarity_weight = _float_or(
            loss_cfg.get("gemini_similarity_weight"),
            _TRAINING_CONFIG_DEFAULTS["loss_config"]["gemini_similarity_weight"],
        )
        gemini_quality_weight = _float_or(
            loss_cfg.get("gemini_quality_weight"),
            _TRAINING_CONFIG_DEFAULTS["loss_config"]["gemini_quality_weight"],
        )
        gemini_model = _string_or(
            loss_cfg.get("gemini_model"),
            _TRAINING_CONFIG_DEFAULTS["loss_config"]["gemini_model"],
        )
        gemini_api_key_env = _string_or(
            loss_cfg.get("gemini_api_key_env"),
            _TRAINING_CONFIG_DEFAULTS["loss_config"]["gemini_api_key_env"],
        )
        reward_baseline_quality_floor = _float_or(
            loss_cfg.get("reward_baseline_quality_floor"),
            _TRAINING_CONFIG_DEFAULTS["loss_config"]["reward_baseline_quality_floor"],
        )

        resolved_training_config = _build_training_config_payload(
            rmse_weight=rmse_weight,
            cosine_weight=cosine_weight,
            lpips_weight=lpips_weight,
            dreamsim_weight=dreamsim_weight,
            hps_weight=hps_weight,
            biqa_quality_weight=biqa_quality_weight,
            biqa_aesthetic_weight=biqa_aesthetic_weight,
            gemini_similarity_weight=gemini_similarity_weight,
            gemini_quality_weight=gemini_quality_weight,
            gemini_model=gemini_model,
            gemini_api_key_env=gemini_api_key_env,
            reward_baseline_quality_floor=reward_baseline_quality_floor,
            huber_beta=huber_beta,
            learning_rate=learning_rate,
            grad_clip_norm=grad_clip_norm,
            alpha_lr_multiplier=alpha_lr_multiplier,
            phi_lr_multiplier=phi_lr_multiplier,
            target_ttr_ratio=target_ttr_ratio,
            lambda_eff=lambda_eff,
            lambda_entropy=lambda_entropy,
            gumbel_temperature_start=gumbel_temperature_start,
            gumbel_temperature_end=gumbel_temperature_end,
            warmup_steps=warmup_steps,
            comet_enabled=comet_enabled,
            comet_api_key=comet_api_key,
            comet_project_name=comet_project_name,
            comet_workspace=comet_workspace,
            comet_experiment=comet_experiment,
            log_every=comet_log_every,
        )
        ui_comet_enabled = bool(comet_enabled)
        ui_comet_api_key = str(comet_api_key or "")
        ui_comet_project_name = str(comet_project_name or "ttr-distillation")
        ui_comet_workspace = str(comet_workspace or "comet-workspace")
        ui_comet_experiment = str(comet_experiment or "").strip()
        ui_comet_persist_experiment = bool(ui_comet_experiment)
        ui_comet_log_every = max(1, int(comet_log_every))
        checkpoint_exists = bool(checkpoint_path and os.path.isfile(checkpoint_path))
        runtime = None
        runtime_cache_hit = False
        if checkpoint_exists:
            runtime = flux2_ttr.get_cached_runtime_for_checkpoint(
                checkpoint_path,
                feature_dim=feature_dim,
                training=bool(training),
            )
            runtime_cache_hit = runtime is not None
            if runtime_cache_hit:
                logger.info("Flux2TTRTrainer: reusing cached runtime for checkpoint: %s", checkpoint_path)

        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})

        prev_cfg = transformer_options.get("flux2_ttr")
        if isinstance(prev_cfg, dict):
            prev_runtime = prev_cfg.get("runtime_id")
            if isinstance(prev_runtime, str):
                flux2_ttr.unregister_runtime(prev_runtime)

        if runtime is None:
            runtime = flux2_ttr.Flux2TTRRuntime(
                feature_dim=feature_dim,
                learning_rate=float(learning_rate),
                training=bool(training),
                steps=train_steps,
                scan_chunk_size=int(query_chunk_size),
                key_chunk_size=int(key_chunk_size),
                landmark_fraction=float(landmark_fraction),
                landmark_min=int(landmark_min),
                landmark_max=int(landmark_max),
                text_tokens_guess=int(text_tokens_guess),
                alpha_init=float(alpha_init),
                alpha_lr_multiplier=float(alpha_lr_multiplier),
                phi_lr_multiplier=float(phi_lr_multiplier),
                training_query_token_cap=int(training_query_token_cap),
                replay_buffer_size=int(replay_buffer_size),
                replay_offload_cpu=bool(replay_offload_cpu),
                replay_max_bytes=int(replay_max_mb) * 1024 * 1024,
                train_steps_per_call=int(train_steps_per_call),
                huber_beta=float(huber_beta),
                grad_clip_norm=float(grad_clip_norm),
                readiness_threshold=float(readiness_threshold),
                readiness_min_updates=int(readiness_min_updates),
                enable_memory_reserve=bool(enable_memory_reserve),
                layer_start=int(layer_start),
                layer_end=int(layer_end),
                cfg_scale=float(cfg_scale),
                min_swap_layers=int(min_swap_layers),
                max_swap_layers=int(max_swap_layers),
                inference_mixed_precision=bool(inference_mixed_precision),
                training_preview_ttr=bool(training_preview_ttr),
                comet_enabled=bool(comet_enabled),
                comet_project_name=str(comet_project_name or "ttr-distillation"),
                comet_workspace=str(comet_workspace or "comet-workspace"),
                comet_experiment=str(comet_experiment or ""),
                comet_persist_experiment=bool(comet_experiment),
                comet_api_key=str(comet_api_key or ""),
                comet_log_every=int(comet_log_every),
            )
        runtime.register_layer_specs(flux2_ttr.infer_flux_single_layer_specs(m))

        if training:
            if checkpoint_exists and not runtime_cache_hit:
                logger.info("Flux2TTRTrainer: loading existing checkpoint before online distillation: %s", checkpoint_path)
                runtime.load_checkpoint(checkpoint_path)
                flux2_ttr.cache_runtime_for_checkpoint(
                    checkpoint_path,
                    feature_dim=feature_dim,
                    training=True,
                    runtime=runtime,
                )
            runtime.training_mode = True
            runtime.training_enabled = train_steps > 0
            runtime.training_steps_total = max(0, train_steps)
            runtime.steps_remaining = max(0, train_steps)
            runtime.training_updates_done = 0
            loss_value = float(runtime.last_loss) if not math.isnan(runtime.last_loss) else 0.0
        else:
            if not checkpoint_path:
                raise ValueError("Flux2TTRTrainer: checkpoint_path is required when training is disabled.")
            if not checkpoint_exists:
                raise FileNotFoundError(f"Flux2TTRTrainer: checkpoint not found: {checkpoint_path}")
            if not runtime_cache_hit:
                runtime.load_checkpoint(checkpoint_path)
                flux2_ttr.cache_runtime_for_checkpoint(
                    checkpoint_path,
                    feature_dim=feature_dim,
                    training=False,
                    runtime=runtime,
                )
            runtime.training_mode = False
            runtime.training_enabled = False
            runtime.training_steps_total = 0
            runtime.steps_remaining = 0
            runtime.training_updates_done = 0
            loss_value = float(runtime.last_loss) if not math.isnan(runtime.last_loss) else 0.0

        # UI training_config should override checkpoint metadata for Comet fields.
        if has_training_config:
            runtime.comet_enabled = ui_comet_enabled
            runtime.comet_api_key = ui_comet_api_key
            runtime.comet_project_name = ui_comet_project_name
            runtime.comet_workspace = ui_comet_workspace
            runtime.comet_experiment = ui_comet_experiment
            runtime.comet_persist_experiment = bool(ui_comet_persist_experiment and ui_comet_experiment)
            runtime.comet_log_every = ui_comet_log_every

        runtime.cfg_scale = float(cfg_scale)
        runtime.min_swap_layers = max(0, int(min_swap_layers))
        runtime.max_swap_layers = int(max_swap_layers)

        controller = None
        if controller_checkpoint_path:
            if not os.path.isfile(controller_checkpoint_path):
                raise FileNotFoundError(f"Flux2TTRTrainer: controller checkpoint not found: {controller_checkpoint_path}")
            controller = flux2_ttr_controller.load_controller_checkpoint(controller_checkpoint_path, map_location="cpu")

        runtime_id = flux2_ttr.register_runtime(runtime)
        transformer_options["flux2_ttr"] = {
            "enabled": True,
            "runtime_id": runtime_id,
            "training": runtime.training_enabled,
            "training_mode": runtime.training_mode,
            "training_preview_ttr": runtime.training_preview_ttr,
            "comet_enabled": runtime.comet_enabled,
            "comet_project_name": runtime.comet_project_name,
            "comet_workspace": runtime.comet_workspace,
            "comet_experiment": runtime.comet_experiment,
            "comet_persist_experiment": runtime.comet_persist_experiment,
            "comet_log_every": int(runtime.comet_log_every),
            "training_steps_total": int(runtime.training_steps_total),
            "training_steps_remaining": int(runtime.steps_remaining),
            "learning_rate": float(learning_rate),
            "feature_dim": feature_dim,
            "query_chunk_size": int(query_chunk_size),
            "scan_chunk_size": int(query_chunk_size),
            "key_chunk_size": int(key_chunk_size),
            "landmark_fraction": float(landmark_fraction),
            "landmark_min": int(landmark_min),
            "landmark_max": int(landmark_max),
            "text_tokens_guess": int(text_tokens_guess),
            "alpha_init": float(alpha_init),
            "alpha_lr_multiplier": float(alpha_lr_multiplier),
            "phi_lr_multiplier": float(phi_lr_multiplier),
            "training_query_token_cap": int(training_query_token_cap),
            "replay_buffer_size": int(replay_buffer_size),
            "replay_offload_cpu": bool(replay_offload_cpu),
            "replay_max_bytes": int(replay_max_mb) * 1024 * 1024,
            "train_steps_per_call": int(train_steps_per_call),
            "huber_beta": float(huber_beta),
            "grad_clip_norm": float(grad_clip_norm),
            "readiness_threshold": float(readiness_threshold),
            "readiness_min_updates": int(readiness_min_updates),
            "enable_memory_reserve": bool(enable_memory_reserve),
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "cfg_scale": float(cfg_scale),
            "min_swap_layers": int(min_swap_layers),
            "max_swap_layers": int(max_swap_layers),
            "inference_mixed_precision": bool(inference_mixed_precision),
            "max_safe_inference_loss": float(runtime.max_safe_inference_loss),
            "checkpoint_path": checkpoint_path,
            "controller_checkpoint_path": controller_checkpoint_path,
            "training_config": resolved_training_config,
        }
        if controller is not None:
            transformer_options["flux2_ttr"]["controller"] = controller

        callback_key = "flux2_ttr"
        m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_PRE_RUN, callback_key)
        m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_CLEANUP, callback_key)
        m.add_callback_with_key(
            patcher_extension.CallbacksMP.ON_PRE_RUN,
            callback_key,
            flux2_ttr.pre_run_callback,
        )
        m.add_callback_with_key(
            patcher_extension.CallbacksMP.ON_CLEANUP,
            callback_key,
            flux2_ttr.cleanup_callback,
        )

        logger.info(
            (
                "Flux2TTRTrainer configured: training_mode=%s training_preview_ttr=%s comet_enabled=%s "
                "training_steps=%d feature_dim=%d q_chunk=%d k_chunk=%d landmarks=(fraction=%.4g,min=%d,max=%d) "
                "lr=%.6g alpha_lr_mul=%.4g phi_lr_mul=%.4g huber_beta=%.6g grad_clip=%.4g "
                "replay=%d replay_offload_cpu=%s replay_max_mb=%d train_steps_per_call=%d readiness=(%.6g,%d) reserve=%s layer_range=[%d,%d] "
                "cfg_scale=%.4g swap_layers=[%d,%d] mixed_precision=%s checkpoint=%s controller=%s loss=%.6g"
            ),
            training,
            bool(training_preview_ttr),
            bool(comet_enabled),
            train_steps,
            feature_dim,
            int(query_chunk_size),
            int(key_chunk_size),
            float(landmark_fraction),
            int(landmark_min),
            int(landmark_max),
            float(learning_rate),
            float(alpha_lr_multiplier),
            float(phi_lr_multiplier),
            float(huber_beta),
            float(grad_clip_norm),
            int(replay_buffer_size),
            bool(replay_offload_cpu),
            int(replay_max_mb),
            int(train_steps_per_call),
            float(readiness_threshold),
            int(readiness_min_updates),
            bool(enable_memory_reserve),
            int(layer_start),
            int(layer_end),
            float(cfg_scale),
            int(min_swap_layers),
            int(max_swap_layers),
            bool(inference_mixed_precision),
            checkpoint_path if checkpoint_path else "<none>",
            controller_checkpoint_path if controller_checkpoint_path else "<none>",
            float(loss_value),
        )
        return io.NodeOutput(m, float(loss_value))


class Flux2TTRControllerTrainer(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        sampler_options = list(comfy_samplers.KSampler.SAMPLERS) if comfy_samplers is not None else ["euler"]
        scheduler_options = list(comfy_samplers.KSampler.SCHEDULERS) if comfy_samplers is not None else ["normal"]
        return io.Schema(
            node_id="Flux2TTRControllerTrainer",
            display_name="Flux2TTRControllerTrainer",
            category="advanced/attention",
            description="Phase-2: train a TTR controller using dual-path sampling and REINFORCE updates.",
            inputs=[
                io.Model.Input("model_original"),
                io.Model.Input("model_ttr"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
                _vae_input("vae"),
                TTRTrainingConfig.Input("training_config"),
                io.Int.Input("steps", default=20, min=1, max=10000, step=1),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, step=1),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=1e-3),
                io.Combo.Input("sampler_name", options=sampler_options, default="euler"),
                io.Combo.Input("scheduler", options=scheduler_options, default="normal"),
                io.String.Input(
                    "checkpoint_path",
                    default=_default_controller_checkpoint_path(),
                    multiline=False,
                    tooltip=(
                        "Controller checkpoint path (defaults to "
                        "ComfyUI/models/approximate_attention/flux2_ttr_controller.pt)."
                    ),
                ),
                io.Int.Input("training_iterations", default=100, min=1, max=100000, step=1),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=1e-3),
                io.Boolean.Input("sigma_aware_training", default=True),
            ],
            outputs=[
                io.Image.Output("IMAGE_ORIGINAL"),
                io.Image.Output("IMAGE_PATCHED"),
                TTRControllerType.Output("CONTROLLER"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model_original,
        model_ttr,
        positive,
        negative,
        latent,
        vae,
        training_config: dict,
        steps: int,
        seed: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        checkpoint_path: str,
        training_iterations: int,
        denoise: float,
        sigma_aware_training: bool,
    ) -> io.NodeOutput:
        original_images, patched_images, controller = run_controller_trainer_node(
            inputs=ControllerTrainerNodeInputs(
                model_original=model_original,
                model_ttr=model_ttr,
                positive=positive,
                negative=negative,
                latent=latent,
                vae=vae,
                training_config=training_config,
                steps=steps,
                seed=seed,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                training_iterations=training_iterations,
                denoise=denoise,
                sigma_aware_training=sigma_aware_training,
            ),
            deps=ControllerTrainerNodeDependencies(
                logger=logger,
                training_defaults=_TRAINING_CONFIG_DEFAULTS,
                default_checkpoint_filename=_DEFAULT_CONTROLLER_CHECKPOINT_FILENAME,
                comet_namespace=_CONTROLLER_COMET_NAMESPACE,
                normalize_training_config=_normalize_training_config,
                float_or=_float_or,
                int_or=_int_or,
                string_or=_string_or,
                bool_or=_bool_or,
                resolve_ttr_runtime_from_model=_resolve_ttr_runtime_from_model,
                extract_layer_idx=_extract_layer_idx,
                resolve_checkpoint_path_or_default=_resolve_checkpoint_path_or_default,
            ),
        )
        return io.NodeOutput(original_images, patched_images, controller)


class Flux2TTRController(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Flux2TTRController",
            display_name="Flux2TTRController",
            category="advanced/attention",
            description="Inference node: load TTR + controller checkpoints and patch the model for dynamic layer routing.",
            inputs=[
                io.Model.Input("model"),
                io.String.Input(
                    "ttr_checkpoint_path",
                    default="",
                    multiline=False,
                    tooltip="Path to the Phase-1 TTR checkpoint (distilled attention layers).",
                ),
                io.String.Input(
                    "controller_checkpoint_path",
                    default="",
                    multiline=False,
                    tooltip="Path to the Phase-2 controller checkpoint used for per-step routing decisions.",
                ),
                io.Float.Input(
                    "quality_speed",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=1e-3,
                    tooltip=(
                        "Quality/speed tradeoff. Internally mapped to controller threshold "
                        "(0.1 + 0.8 * quality_speed). Lower values keep more layers on full attention "
                        "(usually higher quality, lower speed). Higher values route more layers to TTR "
                        "(usually faster, potentially lower quality)."
                    ),
                ),
                io.Combo.Input(
                    "policy_mode",
                    options=["stochastic", "threshold"],
                    default="stochastic",
                    tooltip=(
                        "Inference routing policy. 'stochastic' samples one controller mask per diffusion step "
                        "(matching sigma-aware training behavior). 'threshold' uses deterministic thresholding "
                        "from controller probabilities."
                    ),
                ),
                io.Float.Input(
                    "policy_temperature",
                    default=1.0,
                    min=1e-3,
                    max=10.0,
                    step=1e-3,
                    tooltip=(
                        "Sampling temperature used only when policy_mode='stochastic'. "
                        "Lower values make decisions harder/more deterministic near the threshold; "
                        "higher values make decisions softer/more random."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        ttr_checkpoint_path: str,
        controller_checkpoint_path: str,
        quality_speed: float,
        policy_mode: str,
        policy_temperature: float,
    ) -> io.NodeOutput:
        ttr_checkpoint_path = (ttr_checkpoint_path or "").strip()
        controller_checkpoint_path = (controller_checkpoint_path or "").strip()
        if not ttr_checkpoint_path:
            raise ValueError("Flux2TTRController: ttr_checkpoint_path is required.")
        if not os.path.isfile(ttr_checkpoint_path):
            raise FileNotFoundError(f"Flux2TTRController: TTR checkpoint not found: {ttr_checkpoint_path}")
        if not controller_checkpoint_path:
            raise ValueError("Flux2TTRController: controller_checkpoint_path is required.")
        if not os.path.isfile(controller_checkpoint_path):
            raise FileNotFoundError(
                f"Flux2TTRController: controller checkpoint not found: {controller_checkpoint_path}"
            )

        m = model.clone()
        transformer_options = m.model_options.setdefault("transformer_options", {})
        prev_cfg = transformer_options.get("flux2_ttr")
        if isinstance(prev_cfg, dict):
            prev_runtime = prev_cfg.get("runtime_id")
            if isinstance(prev_runtime, str):
                flux2_ttr.unregister_runtime(prev_runtime)

        feature_dim = flux2_ttr.load_checkpoint_feature_dim(ttr_checkpoint_path)
        if isinstance(prev_cfg, dict):
            prev_feature_dim_raw = prev_cfg.get("feature_dim")
            if prev_feature_dim_raw is not None:
                try:
                    prev_feature_dim = flux2_ttr.validate_feature_dim(int(prev_feature_dim_raw))
                    if prev_feature_dim != feature_dim:
                        logger.info(
                            "Flux2TTRController: using feature_dim=%d from checkpoint metadata (ignoring previous config value %d).",
                            feature_dim,
                            prev_feature_dim,
                        )
                except Exception:
                    logger.debug(
                        "Flux2TTRController: ignored invalid previous feature_dim value %r when loading checkpoint metadata.",
                        prev_feature_dim_raw,
                    )
        learning_rate = _float_or(
            prev_cfg.get("learning_rate", _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["learning_rate"]) if isinstance(prev_cfg, dict) else _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["learning_rate"],
            _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["learning_rate"],
        )
        query_chunk_size = _int_or(prev_cfg.get("query_chunk_size", 256) if isinstance(prev_cfg, dict) else 256, 256)
        key_chunk_size = _int_or(prev_cfg.get("key_chunk_size", 1024) if isinstance(prev_cfg, dict) else 1024, 1024)
        landmark_fraction = _float_or(
            prev_cfg.get("landmark_fraction", 0.08) if isinstance(prev_cfg, dict) else 0.08,
            0.08,
        )
        landmark_min = _int_or(prev_cfg.get("landmark_min", 64) if isinstance(prev_cfg, dict) else 64, 64)
        landmark_max = _int_or(prev_cfg.get("landmark_max", 0) if isinstance(prev_cfg, dict) else 0, 0)
        text_tokens_guess = _int_or(prev_cfg.get("text_tokens_guess", 77) if isinstance(prev_cfg, dict) else 77, 77)
        alpha_init = _float_or(prev_cfg.get("alpha_init", 0.1) if isinstance(prev_cfg, dict) else 0.1, 0.1)
        alpha_lr_multiplier = _float_or(
            prev_cfg.get("alpha_lr_multiplier", _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["alpha_lr_multiplier"]) if isinstance(prev_cfg, dict) else _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["alpha_lr_multiplier"],
            _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["alpha_lr_multiplier"],
        )
        phi_lr_multiplier = _float_or(
            prev_cfg.get("phi_lr_multiplier", _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["phi_lr_multiplier"]) if isinstance(prev_cfg, dict) else _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["phi_lr_multiplier"],
            _TRAINING_CONFIG_DEFAULTS["optimizer_config"]["phi_lr_multiplier"],
        )
        training_query_token_cap = _int_or(prev_cfg.get("training_query_token_cap", 128) if isinstance(prev_cfg, dict) else 128, 128)
        replay_buffer_size = _int_or(prev_cfg.get("replay_buffer_size", 8) if isinstance(prev_cfg, dict) else 8, 8)
        replay_offload_cpu = _bool_or(prev_cfg.get("replay_offload_cpu", True) if isinstance(prev_cfg, dict) else True, True)
        replay_max_bytes = _int_or(prev_cfg.get("replay_max_bytes", 768 * 1024 * 1024) if isinstance(prev_cfg, dict) else 768 * 1024 * 1024, 768 * 1024 * 1024)
        train_steps_per_call = _int_or(prev_cfg.get("train_steps_per_call", 1) if isinstance(prev_cfg, dict) else 1, 1)
        huber_beta = _float_or(prev_cfg.get("huber_beta", 0.05) if isinstance(prev_cfg, dict) else 0.05, 0.05)
        grad_clip_norm = _float_or(prev_cfg.get("grad_clip_norm", 1.0) if isinstance(prev_cfg, dict) else 1.0, 1.0)
        readiness_threshold = _float_or(prev_cfg.get("readiness_threshold", 0.12) if isinstance(prev_cfg, dict) else 0.12, 0.12)
        readiness_min_updates = _int_or(prev_cfg.get("readiness_min_updates", 24) if isinstance(prev_cfg, dict) else 24, 24)
        enable_memory_reserve = _bool_or(prev_cfg.get("enable_memory_reserve", False) if isinstance(prev_cfg, dict) else False, False)
        layer_start = _int_or(prev_cfg.get("layer_start", -1) if isinstance(prev_cfg, dict) else -1, -1)
        layer_end = _int_or(prev_cfg.get("layer_end", -1) if isinstance(prev_cfg, dict) else -1, -1)
        cfg_scale = _float_or(prev_cfg.get("cfg_scale", 1.0) if isinstance(prev_cfg, dict) else 1.0, 1.0)
        min_swap_layers = _int_or(prev_cfg.get("min_swap_layers", 1) if isinstance(prev_cfg, dict) else 1, 1)
        max_swap_layers = _int_or(prev_cfg.get("max_swap_layers", -1) if isinstance(prev_cfg, dict) else -1, -1)
        inference_mixed_precision = _bool_or(
            prev_cfg.get("inference_mixed_precision", True) if isinstance(prev_cfg, dict) else True,
            True,
        )

        runtime = flux2_ttr.Flux2TTRRuntime(
            feature_dim=feature_dim,
            learning_rate=float(learning_rate),
            training=False,
            steps=0,
            scan_chunk_size=int(query_chunk_size),
            key_chunk_size=int(key_chunk_size),
            landmark_fraction=float(landmark_fraction),
            landmark_min=int(landmark_min),
            landmark_max=int(landmark_max),
            text_tokens_guess=int(text_tokens_guess),
            alpha_init=float(alpha_init),
            alpha_lr_multiplier=float(alpha_lr_multiplier),
            phi_lr_multiplier=float(phi_lr_multiplier),
            training_query_token_cap=int(training_query_token_cap),
            replay_buffer_size=int(replay_buffer_size),
            replay_offload_cpu=bool(replay_offload_cpu),
            replay_max_bytes=int(replay_max_bytes),
            train_steps_per_call=int(train_steps_per_call),
            huber_beta=float(huber_beta),
            grad_clip_norm=float(grad_clip_norm),
            readiness_threshold=float(readiness_threshold),
            readiness_min_updates=int(readiness_min_updates),
            enable_memory_reserve=bool(enable_memory_reserve),
            layer_start=int(layer_start),
            layer_end=int(layer_end),
            cfg_scale=float(cfg_scale),
            min_swap_layers=int(min_swap_layers),
            max_swap_layers=int(max_swap_layers),
            inference_mixed_precision=bool(inference_mixed_precision),
            training_preview_ttr=False,
        )
        runtime.register_layer_specs(flux2_ttr.infer_flux_single_layer_specs(m))
        runtime.load_checkpoint(ttr_checkpoint_path)
        runtime.training_mode = False
        runtime.training_enabled = False
        runtime.steps_remaining = 0
        runtime.training_updates_done = 0

        controller = flux2_ttr_controller.load_controller_checkpoint(controller_checkpoint_path, map_location="cpu")
        quality_speed = min(1.0, max(0.0, float(quality_speed)))
        threshold = 0.1 + 0.8 * quality_speed
        policy_mode_norm = str(policy_mode or "stochastic").strip().lower()
        if policy_mode_norm not in {"stochastic", "threshold"}:
            logger.warning("Flux2TTRController: unknown policy_mode=%r; defaulting to 'stochastic'.", policy_mode)
            policy_mode_norm = "stochastic"
        try:
            policy_temperature_value = float(policy_temperature)
        except Exception:
            policy_temperature_value = 1.0
        if not math.isfinite(policy_temperature_value) or policy_temperature_value <= 0:
            logger.warning(
                "Flux2TTRController: invalid policy_temperature=%r; defaulting to 1.0.",
                policy_temperature,
            )
            policy_temperature_value = 1.0

        runtime_id = flux2_ttr.register_runtime(runtime)
        transformer_options["flux2_ttr"] = {
            "enabled": True,
            "training": False,
            "training_mode": False,
            "runtime_id": runtime_id,
            "controller": controller,
            "controller_threshold": float(threshold),
            "controller_policy": str(policy_mode_norm),
            "controller_temperature": float(policy_temperature_value),
            "checkpoint_path": ttr_checkpoint_path,
            "controller_checkpoint_path": controller_checkpoint_path,
            "feature_dim": int(feature_dim),
            "query_chunk_size": int(query_chunk_size),
            "scan_chunk_size": int(query_chunk_size),
            "key_chunk_size": int(key_chunk_size),
            "landmark_fraction": float(landmark_fraction),
            "landmark_min": int(landmark_min),
            "landmark_max": int(landmark_max),
            "text_tokens_guess": int(text_tokens_guess),
            "alpha_init": float(alpha_init),
            "alpha_lr_multiplier": float(alpha_lr_multiplier),
            "phi_lr_multiplier": float(phi_lr_multiplier),
            "training_query_token_cap": int(training_query_token_cap),
            "replay_buffer_size": int(replay_buffer_size),
            "replay_offload_cpu": bool(replay_offload_cpu),
            "replay_max_bytes": int(replay_max_bytes),
            "train_steps_per_call": int(train_steps_per_call),
            "huber_beta": float(huber_beta),
            "grad_clip_norm": float(grad_clip_norm),
            "readiness_threshold": float(readiness_threshold),
            "readiness_min_updates": int(readiness_min_updates),
            "enable_memory_reserve": bool(enable_memory_reserve),
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "cfg_scale": float(cfg_scale),
            "min_swap_layers": int(min_swap_layers),
            "max_swap_layers": int(max_swap_layers),
            "inference_mixed_precision": bool(inference_mixed_precision),
        }

        callback_key = "flux2_ttr"
        m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_PRE_RUN, callback_key)
        m.remove_callbacks_with_key(patcher_extension.CallbacksMP.ON_CLEANUP, callback_key)
        m.add_callback_with_key(
            patcher_extension.CallbacksMP.ON_PRE_RUN,
            callback_key,
            flux2_ttr.pre_run_callback,
        )
        m.add_callback_with_key(
            patcher_extension.CallbacksMP.ON_CLEANUP,
            callback_key,
            flux2_ttr.cleanup_callback,
        )

        logger.info(
            (
                "Flux2TTRController configured: ttr_ckpt=%s controller_ckpt=%s "
                "quality_speed=%.4g threshold=%.4g policy_mode=%s policy_temperature=%.4g"
            ),
            ttr_checkpoint_path,
            controller_checkpoint_path,
            quality_speed,
            threshold,
            policy_mode_norm,
            policy_temperature_value,
        )
        return io.NodeOutput(m)


class RandomSeedBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RandomSeedBatch",
            display_name="Random Seed Batch",
            category="advanced/scheduling",
            description="Generate a deterministic list of random seeds from a base seed.",
            inputs=[
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    tooltip="Base seed used to deterministically generate the output seed list.",
                ),
                io.Int.Input(
                    "count",
                    default=8,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Number of random seeds to generate.",
                ),
            ],
            outputs=[io.Int.Output("seeds", is_output_list=True)],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, seed: int, count: int) -> io.NodeOutput:
        seeds = sweep_utils.generate_seed_batch(int(seed), int(count))
        logger.info(
            "RandomSeedBatch generated %d seeds from base seed %d (first=%d).",
            len(seeds),
            int(seed),
            int(seeds[0]) if seeds else -1,
        )
        return io.NodeOutput(seeds)


class LoadPromptListFromJSON(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LoadPromptListFromJSON",
            display_name="LoadPromptListFromJSON",
            category="advanced/scheduling",
            description="Load a JSON file containing an array of prompt strings.",
            inputs=[
                io.String.Input(
                    "json_path",
                    default="",
                    multiline=False,
                    tooltip="Path to JSON file containing a simple array of strings.",
                ),
            ],
            outputs=[io.String.Output("prompts", is_output_list=True)],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, json_path: str) -> io.NodeOutput:
        prompts = sweep_utils.load_prompt_list_from_json(json_path)
        logger.info(
            "LoadPromptListFromJSON loaded %d prompts from %s.",
            len(prompts),
            str(json_path or "").strip(),
        )
        return io.NodeOutput(prompts)


class ClockedSweepValues(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ClockedSweepValues",
            display_name="Clocked Sweep Values",
            category="advanced/scheduling",
            description="Map a clock list to evenly distributed sweep values.",
            inputs=[
                io.MultiType.Input(
                    io.String.Input(
                        "clock",
                        multiline=True,
                        placeholder="0, 1, 2 or 30",
                        tooltip="Clock list (length defines output length). Accepts JSON list, comma/space-separated values, a list input, or a single integer string to create 1..N.",
                    ),
                    [io.AnyType],
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "values",
                        multiline=True,
                        placeholder="0.1, 0.2, 0.3",
                        tooltip="Values to sweep across the clock (JSON list, comma/space-separated, or list input). If clock is blank, its length is inferred from values.",
                    ),
                    [io.AnyType],
                ),
            ],
            outputs=[io.Float.Output("values", is_output_list=True)],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clock, values) -> io.NodeOutput:
        clock_list, values_list = sweep_utils.parse_clock_and_values(clock, values)
        output = sweep_utils.build_clocked_sweep(clock_list, values_list)
        logger.info(
            "Clocked sweep built: clock=%d values=%d output=%d",
            len(clock_list),
            len(values_list),
            len(output),
        )
        return io.NodeOutput(output)


class Combinations(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Combinations",
            display_name="Combinations",
            category="advanced/scheduling",
            description="Generate repeated lists that cover all combinations of provided values.",
            inputs=[
                io.MultiType.Input(
                    io.String.Input(
                        "a",
                        multiline=True,
                        placeholder="1, 2, 3",
                        tooltip="Values for A (JSON list, comma/space-separated, or list input).",
                    ),
                    [io.AnyType],
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "b",
                        multiline=True,
                        placeholder="4, 5",
                        tooltip="Values for B (optional).",
                    ),
                    [io.AnyType],
                    optional=True,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "c",
                        multiline=True,
                        placeholder="",
                        tooltip="Values for C (optional).",
                    ),
                    [io.AnyType],
                    optional=True,
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "d",
                        multiline=True,
                        placeholder="",
                        tooltip="Values for D (optional).",
                    ),
                    [io.AnyType],
                    optional=True,
                ),
            ],
            outputs=[
                io.Float.Output("a_out", is_output_list=True),
                io.Float.Output("b_out", is_output_list=True),
                io.Float.Output("c_out", is_output_list=True),
                io.Float.Output("d_out", is_output_list=True),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, a, b=None, c=None, d=None) -> io.NodeOutput:
        values = []
        names = []
        a_list = sweep_utils.to_float_list(a, "a")
        if not a_list:
            raise ValueError("Combinations: 'a' must contain at least one value.")
        values.append(a_list)
        names.append("a")
        for label, item in (("b", b), ("c", c), ("d", d)):
            if item is None or item == "":
                continue
            parsed = sweep_utils.to_float_list(item, label)
            if parsed:
                values.append(parsed)
                names.append(label)

        outputs = sweep_utils.build_combinations(values)
        out_map = dict(zip(names, outputs))
        return io.NodeOutput(
            out_map.get("a", []),
            out_map.get("b", []),
            out_map.get("c", []),
            out_map.get("d", []),
        )


class TaylorAttentionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Flux2TTRTrainingParameters,
            Flux2TTRTrainer,
            Flux2TTRControllerTrainer,
            Flux2TTRController,
            RandomSeedBatch,
            LoadPromptListFromJSON,
            ClockedSweepValues,
            Combinations,
        ]


async def comfy_entrypoint() -> TaylorAttentionExtension:
    return TaylorAttentionExtension()
