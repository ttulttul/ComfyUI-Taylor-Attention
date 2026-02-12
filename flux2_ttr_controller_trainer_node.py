from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

import flux2_comet_logging
import flux2_ttr_controller

try:
    import comfy.sample as comfy_sample
    import comfy.samplers as comfy_samplers
    import comfy.model_management as comfy_model_management
except Exception:
    comfy_sample = None
    comfy_samplers = None
    comfy_model_management = None


@dataclass(frozen=True)
class ControllerTrainerNodeDependencies:
    logger: logging.Logger
    training_defaults: dict[str, dict[str, Any]]
    default_checkpoint_filename: str
    comet_namespace: str
    normalize_training_config: Callable[[Optional[dict]], dict[str, dict[str, Any]]]
    float_or: Callable[[Any, float], float]
    int_or: Callable[[Any, int], int]
    string_or: Callable[[Any, str], str]
    bool_or: Callable[[Any, bool], bool]
    resolve_ttr_runtime_from_model: Callable[[Any], tuple[Any, dict[str, Any]]]
    extract_layer_idx: Callable[[str], Optional[int]]
    resolve_checkpoint_path_or_default: Callable[..., str]


@dataclass(frozen=True)
class ControllerTrainerNodeInputs:
    model_original: Any
    model_ttr: Any
    positive: Any
    negative: Any
    latent: dict[str, Any]
    vae: Any
    training_config: dict[str, Any]
    steps: int
    seed: int
    cfg: float
    sampler_name: str
    scheduler: str
    checkpoint_path: str
    training_iterations: int
    denoise: float
    sigma_aware_training: bool


@dataclass(frozen=True)
class ControllerTrainingConfig:
    normalized_cfg: dict[str, dict[str, Any]]
    loss_cfg: dict[str, Any]
    schedule_cfg: dict[str, Any]
    logging_cfg: dict[str, Any]
    gumbel_start: float
    gumbel_end: float
    lambda_entropy: float
    log_every: int
    checkpoint_path: str
    sigma_aware_training: bool
    total_iterations: int


@dataclass(frozen=True)
class TrainingEnvironment:
    runtime: Any
    flux_cfg: dict[str, Any]
    num_layers: int
    ready_layer_indices: list[int]
    ttr_eligible_mask_cpu: torch.Tensor
    forced_full_layer_count: int


@dataclass(frozen=True)
class TrainerBundle:
    controller: flux2_ttr_controller.TTRController
    trainer: flux2_ttr_controller.ControllerTrainer
    training_wrapper: flux2_ttr_controller.TrainingControllerWrapper


@dataclass(frozen=True)
class RunPlan:
    latent_items: list[dict[str, Any]]
    sigma_value: float
    checkpoint_every: int
    total_steps: int
    comet_experiment: Any


@dataclass
class LoopState:
    global_step: int = 0
    last_original_images: Optional[torch.Tensor] = None
    last_patched_images: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class StudentSample:
    student_latent: dict[str, Any]
    actual_full_attn_ratio_eligible: float
    actual_full_attn_ratio_overall: float
    expected_full_attn_ratio_eligible: float
    expected_full_attn_ratio_overall: float
    effective_mask: Optional[torch.Tensor] = None
    ttr_eligible_mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class SigmaRewardTerms:
    target_ttr_ratio: float
    target_full_attn_ratio: float
    efficiency_penalty: float
    efficiency_penalty_weighted: float
    entropy_bonus: float
    reward_value: float
    baselined_reward: float


@dataclass(frozen=True)
class PolicyOutcome:
    reinforce_metrics: dict[str, float]
    target_ttr_ratio: float
    target_full_attn_ratio: float
    expected_full_attn_ratio_eligible: float
    expected_full_attn_ratio_overall: float


def interpolate_gumbel_temperature(
    *,
    start: float,
    end: float,
    iteration: int,
    total_iterations: int,
) -> float:
    progress = float(iteration) / float(max(1, int(total_iterations) - 1))
    return float(start + (end - start) * progress)


def summarize_step_ttr(
    step_ttr: list[tuple[float, float]],
    *,
    sigma_fallback: float,
) -> dict[str, float]:
    sigma_max = max((float(sigma) for sigma, _ in step_ttr), default=1.0)
    high_ttrs = [fraction for sigma, fraction in step_ttr if float(sigma) > 0.5 * sigma_max]
    low_ttrs = [fraction for sigma, fraction in step_ttr if float(sigma) < 0.2 * sigma_max]
    ttr_ratio_high = float(sum(high_ttrs) / max(1, len(high_ttrs)))
    ttr_ratio_low = float(sum(low_ttrs) / max(1, len(low_ttrs)))
    trajectory_sigma = (
        float(sum(float(sigma) for sigma, _ in step_ttr) / max(1, len(step_ttr)))
        if step_ttr
        else float(sigma_fallback)
    )
    return {
        "ttr_ratio_high_sigma": ttr_ratio_high,
        "ttr_ratio_low_sigma": ttr_ratio_low,
        "ttr_ratio_spread": float(ttr_ratio_high - ttr_ratio_low),
        "steps_per_trajectory": float(len(step_ttr)),
        "sigma": trajectory_sigma,
    }


def build_layer_index_to_key(
    runtime: Any,
    *,
    extract_layer_idx: Callable[[str], Optional[int]],
) -> dict[int, str]:
    layer_index_to_key: dict[int, str] = {}
    for layer_key in runtime.layer_specs:
        if not layer_key.startswith("single:"):
            continue
        idx = extract_layer_idx(layer_key)
        if idx is not None and idx >= 0:
            layer_index_to_key[idx] = layer_key
    if layer_index_to_key:
        return layer_index_to_key
    for layer_key in runtime.layer_ready:
        idx = extract_layer_idx(layer_key)
        if idx is not None and idx >= 0:
            layer_index_to_key[idx] = layer_key
    return layer_index_to_key


def build_ready_layer_indices(runtime: Any, layer_index_to_key: dict[int, str]) -> list[int]:
    ready_indices: list[int] = []
    for idx in sorted(layer_index_to_key.keys()):
        layer_key = layer_index_to_key[idx]
        if runtime._refresh_layer_ready(layer_key):
            ready_indices.append(idx)
    return ready_indices


def require_sampling_stack() -> None:
    if comfy_sample is None or comfy_samplers is None:
        raise RuntimeError("Flux2TTRControllerTrainer requires ComfyUI sampling modules (comfy.sample/comfy.samplers).")


def split_latent_batch(latent: dict[str, Any]) -> list[dict[str, Any]]:
    samples = latent.get("samples")
    if not torch.is_tensor(samples) or samples.ndim < 4:
        raise ValueError("Flux2TTRControllerTrainer: latent must contain tensor key 'samples' with batch dimension.")
    batch_size = int(samples.shape[0])
    items: list[dict[str, Any]] = []
    for idx in range(batch_size):
        item = dict(latent)
        item["samples"] = samples[idx : idx + 1]
        noise_mask = latent.get("noise_mask")
        if torch.is_tensor(noise_mask) and noise_mask.shape[0] == batch_size:
            item["noise_mask"] = noise_mask[idx : idx + 1]
        batch_index = latent.get("batch_index")
        if isinstance(batch_index, (list, tuple)) and len(batch_index) == batch_size:
            item["batch_index"] = [batch_index[idx]]
        items.append(item)
    return items


def sample_model(
    *,
    model: Any,
    latent: dict[str, Any],
    positive: Any,
    negative: Any,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
) -> dict[str, Any]:
    latent_image = latent["samples"]
    latent_image = comfy_sample.fix_empty_latent_channels(model, latent_image, latent.get("downscale_ratio_spacial", None))
    batch_inds = latent.get("batch_index", None)
    noise = comfy_sample.prepare_noise(latent_image, int(seed), batch_inds)
    samples = comfy_sample.sample(
        model,
        noise,
        int(steps),
        float(cfg),
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=float(denoise),
        noise_mask=latent.get("noise_mask", None),
        disable_pbar=True,
        seed=int(seed),
    )
    out = dict(latent)
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return out


def decode_vae(vae: Any, latent_dict: dict[str, Any]) -> torch.Tensor:
    latent_t = latent_dict["samples"]
    if getattr(latent_t, "is_nested", False):
        latent_t = latent_t.unbind()[0]
    images = vae.decode(latent_t)
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def to_lpips_bchw(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError("Flux2TTRControllerTrainer: expected decoded images with shape [B,H,W,C] or [B,C,H,W].")
    if images.shape[-1] == 3:
        x = images.permute(0, 3, 1, 2).contiguous()
    elif images.shape[1] == 3:
        x = images
    else:
        raise ValueError("Flux2TTRControllerTrainer: decoded images must have 3 channels.")
    x = x.float()
    if float(x.min().item()) >= 0.0 and float(x.max().item()) <= 1.0:
        x = x * 2.0 - 1.0
    return x.clamp(-1.0, 1.0)


def maybe_unload_model(model: Any, *, logger: logging.Logger) -> None:
    if comfy_model_management is None:
        return
    try:
        unload_fn = getattr(comfy_model_management, "unload_model_clones", None)
        if callable(unload_fn):
            unload_fn(model)
            return
        free_memory = getattr(comfy_model_management, "free_memory", None)
        if not callable(free_memory):
            return
        device = getattr(model, "load_device", None)
        if device is None and hasattr(comfy_model_management, "get_torch_device"):
            device = comfy_model_management.get_torch_device()
        if device is not None:
            free_memory(1e30, device)
    except Exception as exc:
        logger.debug("Flux2TTRControllerTrainer: failed to unload model from VRAM (%s).", exc)


def empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def representative_sigma(model_ttr: Any, *, scheduler: str, steps: int) -> float:
    if comfy_samplers is None:
        return 0.0
    try:
        model_sampling = model_ttr.get_model_object("model_sampling")
        sigmas = comfy_samplers.calculate_sigmas(model_sampling, scheduler, int(steps))
        if torch.is_tensor(sigmas) and sigmas.numel() > 0:
            return float(sigmas[0].item())
    except Exception:
        pass
    return 0.0


def build_comet_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    payload = flux2_comet_logging.build_prefixed_metrics("flux2ttr_controller", metrics)
    eligible_key = "flux2ttr_controller/full_attn_eligible"
    if eligible_key in payload:
        return payload
    fallback = flux2_comet_logging.sanitize_metric(
        metrics.get("full_attn_ratio", metrics.get("actual_full_attn_ratio"))
    )
    if fallback is not None:
        payload[eligible_key] = fallback
    return payload


def comet_payload_keys_text(payload: dict[str, float]) -> str:
    if not payload:
        return "<empty>"
    return ", ".join(sorted(str(key) for key in payload.keys()))


def safe_comet_log_metrics(
    experiment: Any,
    payload: dict[str, float],
    *,
    step: int,
    logger: logging.Logger,
) -> bool:
    return flux2_comet_logging.safe_log_metrics(
        experiment=experiment,
        payload=payload,
        step=int(step),
        logger=logger,
        component_name="Flux2TTRControllerTrainer",
    )


def inference_tensor_count(module: torch.nn.Module) -> int:
    count = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        checker = getattr(tensor, "is_inference", None)
        if not callable(checker):
            continue
        try:
            if bool(checker()):
                count += 1
        except Exception:
            pass
    return int(count)


def maybe_start_comet(
    *,
    logger: logging.Logger,
    logging_cfg: dict[str, Any],
    training_config: dict[str, dict[str, Any]],
    run_config: dict[str, Any],
    deps: ControllerTrainerNodeDependencies,
) -> Any:
    enabled = deps.bool_or(logging_cfg.get("comet_enabled"), False)
    if not enabled:
        logger.info("Flux2TTRControllerTrainer: Comet logging disabled (training_config.logging_config.comet_enabled=false).")
        return None
    project_name = deps.string_or(logging_cfg.get("comet_project_name"), "ttr-distillation")
    workspace = deps.string_or(logging_cfg.get("comet_workspace"), "comet-workspace")
    experiment_key = _normalize_experiment_key(logging_cfg, logger=logger, deps=deps)
    display_name = _resolve_display_name(logging_cfg)
    params = _build_comet_params(project_name, workspace, experiment_key, display_name, training_config, run_config)
    experiment, disabled = flux2_comet_logging.ensure_experiment(
        namespace=deps.comet_namespace,
        logger=logger,
        component_name="Flux2TTRControllerTrainer",
        enabled=True,
        disabled=False,
        existing_experiment=None,
        api_key=deps.string_or(logging_cfg.get("comet_api_key"), ""),
        project_name=project_name,
        workspace=workspace,
        experiment_key=experiment_key,
        display_name=display_name,
        persist_experiment=True,
        params=params,
    )
    return None if disabled else experiment


def _normalize_experiment_key(
    logging_cfg: dict[str, Any],
    *,
    logger: logging.Logger,
    deps: ControllerTrainerNodeDependencies,
) -> str:
    raw_key = deps.string_or(logging_cfg.get("comet_experiment"), "").strip()
    experiment_key = flux2_comet_logging.normalize_experiment_key(raw_key, allow_empty=True)
    if raw_key != experiment_key:
        logger.info(
            "Flux2TTRControllerTrainer: normalized comet_experiment from %r to %r.",
            raw_key,
            experiment_key,
        )
    if not experiment_key:
        experiment_key = flux2_comet_logging.generate_experiment_key(__file__)
    logging_cfg["comet_experiment"] = experiment_key
    return experiment_key


def _resolve_display_name(logging_cfg: dict[str, Any]) -> str:
    display_name = str(logging_cfg.get("comet_display_name", "")).strip()
    if not display_name:
        display_name = flux2_comet_logging.generate_experiment_display_name(__file__)
    logging_cfg["comet_display_name"] = display_name
    return display_name


def _build_comet_params(
    project_name: str,
    workspace: str,
    experiment_key: str,
    display_name: str,
    training_config: dict[str, dict[str, Any]],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "flux2ttr_phase": "controller_train",
        "project_name": project_name,
        "workspace": workspace,
        "comet_experiment": experiment_key,
        "comet_display_name": display_name,
    }
    flux2_comet_logging.flatten_params("training_config", training_config, params)
    flux2_comet_logging.flatten_params("run_config", run_config, params)
    return params


class ControllerTrainerNodeEngine:
    def __init__(self, *, inputs: ControllerTrainerNodeInputs, deps: ControllerTrainerNodeDependencies):
        self.inputs = inputs
        self.deps = deps
        self.logger = deps.logger
        self.environment: Optional[TrainingEnvironment] = None
        self.config: Optional[ControllerTrainingConfig] = None
        self.bundle: Optional[TrainerBundle] = None
        self.run_plan: Optional[RunPlan] = None
        self.loop_state = LoopState()

    def run(self) -> tuple[torch.Tensor, torch.Tensor, flux2_ttr_controller.TTRController]:
        self._prepare_environment()
        self._prepare_controller_training()
        self._run_training_loop()
        self._save_final_checkpoint()
        if self.loop_state.last_original_images is None or self.loop_state.last_patched_images is None:
            raise RuntimeError("Flux2TTRControllerTrainer: no samples were produced.")
        return self.loop_state.last_original_images, self.loop_state.last_patched_images, self.bundle.controller

    def _prepare_environment(self) -> None:
        require_sampling_stack()
        runtime, flux_cfg = self.deps.resolve_ttr_runtime_from_model(self.inputs.model_ttr)
        runtime.training_mode = False
        runtime.training_enabled = False
        runtime.steps_remaining = 0
        flux_cfg["training_mode"] = False
        flux_cfg["training"] = False
        layer_index_to_key = build_layer_index_to_key(runtime, extract_layer_idx=self.deps.extract_layer_idx)
        if not layer_index_to_key:
            raise ValueError("Flux2TTRControllerTrainer: unable to infer eligible TTR layer count from model_ttr.")
        num_layers = max(layer_index_to_key.keys()) + 1
        ready_indices = build_ready_layer_indices(runtime, layer_index_to_key)
        if not ready_indices:
            raise ValueError(
                "Flux2TTRControllerTrainer requires readiness-qualified TTR layers. "
                "Run Flux2TTRTrainer longer or lower readiness_threshold."
            )
        ttr_eligible_mask_cpu = torch.zeros(num_layers, dtype=torch.bool)
        ttr_eligible_mask_cpu[ready_indices] = True
        forced_full_layer_count = int((~ttr_eligible_mask_cpu).sum().item())
        self.logger.info(
            "Flux2TTRControllerTrainer layer readiness: eligible_ttr=%d/%d forced_full=%d",
            int(len(ready_indices)),
            int(num_layers),
            forced_full_layer_count,
        )
        self.environment = TrainingEnvironment(
            runtime=runtime,
            flux_cfg=flux_cfg,
            num_layers=num_layers,
            ready_layer_indices=ready_indices,
            ttr_eligible_mask_cpu=ttr_eligible_mask_cpu,
            forced_full_layer_count=forced_full_layer_count,
        )

    def _prepare_controller_training(self) -> None:
        normalized_cfg = self.deps.normalize_training_config(self.inputs.training_config)
        config = self._build_training_config(normalized_cfg)
        environment = self.environment
        controller = self._load_controller(config.checkpoint_path, environment.num_layers)
        bundle = self._build_trainer_bundle(controller=controller, config=config)
        self._restore_bundle_training_state(bundle, checkpoint_path=config.checkpoint_path)
        run_plan = self._build_run_plan(config=config, bundle=bundle, environment=environment)
        self.environment = environment
        self.config = config
        self.bundle = bundle
        self.run_plan = run_plan

    def _build_training_config(self, normalized_cfg: dict[str, dict[str, Any]]) -> ControllerTrainingConfig:
        loss_cfg = normalized_cfg["loss_config"]
        schedule_cfg = normalized_cfg["schedule_config"]
        logging_cfg = normalized_cfg["logging_config"]
        gumbel_start = self.deps.float_or(
            schedule_cfg.get("gumbel_temperature_start"),
            self.deps.training_defaults["schedule_config"]["gumbel_temperature_start"],
        )
        gumbel_end = self.deps.float_or(
            schedule_cfg.get("gumbel_temperature_end"),
            self.deps.training_defaults["schedule_config"]["gumbel_temperature_end"],
        )
        lambda_entropy = self.deps.float_or(
            schedule_cfg.get("lambda_entropy"),
            self.deps.training_defaults["schedule_config"]["lambda_entropy"],
        )
        log_every = max(
            1,
            self.deps.int_or(
                logging_cfg.get("log_every"),
                self.deps.training_defaults["logging_config"]["log_every"],
            ),
        )
        checkpoint_path = self.deps.resolve_checkpoint_path_or_default(
            self.inputs.checkpoint_path,
            default_filename=self.deps.default_checkpoint_filename,
            component_name="Flux2TTRControllerTrainer",
            field_name="checkpoint_path",
        )
        return ControllerTrainingConfig(
            normalized_cfg=normalized_cfg,
            loss_cfg=loss_cfg,
            schedule_cfg=schedule_cfg,
            logging_cfg=logging_cfg,
            gumbel_start=float(gumbel_start),
            gumbel_end=float(gumbel_end),
            lambda_entropy=float(lambda_entropy),
            log_every=int(log_every),
            checkpoint_path=checkpoint_path,
            sigma_aware_training=bool(self.inputs.sigma_aware_training),
            total_iterations=max(1, int(self.inputs.training_iterations)),
        )

    def _load_controller(self, checkpoint_path: str, num_layers: int) -> flux2_ttr_controller.TTRController:
        if checkpoint_path and os.path.isfile(checkpoint_path):
            controller = flux2_ttr_controller.load_controller_checkpoint(checkpoint_path, map_location="cpu")
        else:
            controller = flux2_ttr_controller.TTRController(num_layers=num_layers)
        if int(controller.num_layers) != int(num_layers):
            raise ValueError(
                f"Flux2TTRControllerTrainer: controller num_layers={controller.num_layers} does not match model layers={num_layers}."
            )
        return controller

    def _build_trainer_bundle(
        self,
        *,
        controller: flux2_ttr_controller.TTRController,
        config: ControllerTrainingConfig,
    ) -> TrainerBundle:
        environment = self.environment
        device = self._resolve_model_device()
        controller = flux2_ttr_controller.ControllerTrainer._rebuild_controller_trainable_copy(controller)
        trainer = flux2_ttr_controller.ControllerTrainer(
            controller=controller,
            training_config=config.normalized_cfg,
            lambda_entropy=float(config.lambda_entropy),
            device=device,
        )
        controller = trainer.controller
        controller_device, _ = controller._input_device_dtype()
        training_wrapper = flux2_ttr_controller.TrainingControllerWrapper(
            controller,
            temperature=float(config.gumbel_start),
            ready_mask=environment.ttr_eligible_mask_cpu.to(device=controller_device, dtype=torch.float32),
        )
        environment.flux_cfg["controller"] = training_wrapper if config.sigma_aware_training else controller
        environment.flux_cfg["controller_threshold"] = 0.5
        self.logger.info(
            "Flux2TTRControllerTrainer routing mode: sigma_aware_training=%s",
            config.sigma_aware_training,
        )
        return TrainerBundle(controller=controller, trainer=trainer, training_wrapper=training_wrapper)

    def _resolve_model_device(self) -> torch.device:
        device = getattr(self.inputs.model_ttr, "load_device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _restore_bundle_training_state(self, bundle: TrainerBundle, *, checkpoint_path: str) -> None:
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            return
        try:
            payload = flux2_ttr_controller.load_controller_training_state(checkpoint_path, map_location="cpu")
            bundle.trainer.restore_training_state(payload)
            has_optimizer_state = "optimizer_state_dict" in payload and payload["optimizer_state_dict"] is not None
            has_baseline_state = (
                ("reward_baseline" in payload and payload["reward_baseline"] is not None)
                or ("reward_count" in payload and payload["reward_count"] is not None)
            )
            self._log_training_state_restore(bundle.trainer, has_optimizer_state=has_optimizer_state, has_baseline_state=has_baseline_state)
        except Exception as exc:
            self.logger.info(
                "Flux2TTR controller: no training state to restore (%s). Starting fresh.",
                exc,
            )

    def _log_training_state_restore(
        self,
        trainer: flux2_ttr_controller.ControllerTrainer,
        *,
        has_optimizer_state: bool,
        has_baseline_state: bool,
    ) -> None:
        if has_optimizer_state or has_baseline_state:
            self.logger.info(
                "Flux2TTR controller: restored training state (reward_baseline=%.4f, reward_count=%d, optimizer restored=%s)",
                trainer.reward_baseline,
                trainer.reward_count,
                has_optimizer_state,
            )
            return
        self.logger.info(
            "Flux2TTR controller: no persisted training state in checkpoint. Starting fresh optimizer/baseline."
        )

    def _build_run_plan(
        self,
        *,
        config: ControllerTrainingConfig,
        bundle: TrainerBundle,
        environment: TrainingEnvironment,
    ) -> RunPlan:
        latent_items = split_latent_batch(self.inputs.latent)
        sigma_value = representative_sigma(self.inputs.model_ttr, scheduler=self.inputs.scheduler, steps=int(self.inputs.steps))
        total_steps = int(config.total_iterations) * max(1, len(latent_items))
        checkpoint_every = max(1, int(flux2_ttr_controller.DEFAULT_CONTROLLER_CHECKPOINT_EVERY))
        comet_experiment = maybe_start_comet(
            logger=self.logger,
            logging_cfg=config.logging_cfg,
            training_config=config.normalized_cfg,
            run_config=self._build_comet_run_config(config, bundle, environment, total_steps, checkpoint_every, latent_items),
            deps=self.deps,
        )
        return RunPlan(
            latent_items=latent_items,
            sigma_value=float(sigma_value),
            checkpoint_every=int(checkpoint_every),
            total_steps=int(total_steps),
            comet_experiment=comet_experiment,
        )

    def _build_comet_run_config(
        self,
        config: ControllerTrainingConfig,
        bundle: TrainerBundle,
        environment: TrainingEnvironment,
        total_steps: int,
        checkpoint_every: int,
        latent_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "steps": int(self.inputs.steps),
            "seed": int(self.inputs.seed),
            "cfg": float(self.inputs.cfg),
            "sampler_name": str(self.inputs.sampler_name),
            "scheduler": str(self.inputs.scheduler),
            "training_iterations": int(config.total_iterations),
            "latent_batch_size": int(len(latent_items)),
            "total_steps": int(total_steps),
            "denoise": float(self.inputs.denoise),
            "controller_num_layers": int(bundle.controller.num_layers),
            "eligible_ttr_layers": int(len(environment.ready_layer_indices)),
            "forced_full_layers": int(environment.forced_full_layer_count),
            "sigma_aware_training": bool(config.sigma_aware_training),
            "checkpoint_every": int(checkpoint_every),
            "checkpoint_path": config.checkpoint_path,
            "device": str(self._resolve_model_device()),
            "comet_experiment": self.deps.string_or(config.logging_cfg.get("comet_experiment"), ""),
        }

    def _run_training_loop(self) -> None:
        config = self.config
        run_plan = self.run_plan
        for iteration in range(config.total_iterations):
            gumbel_temperature = interpolate_gumbel_temperature(
                start=config.gumbel_start,
                end=config.gumbel_end,
                iteration=iteration,
                total_iterations=config.total_iterations,
            )
            iter_original_images, iter_patched_images = self._run_iteration(iteration=iteration, gumbel_temperature=gumbel_temperature)
            self._update_last_images(iter_original_images, iter_patched_images)
        self._log_comet_training_complete()

    def _run_iteration(self, *, iteration: int, gumbel_temperature: float) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        iter_original_images: list[torch.Tensor] = []
        iter_patched_images: list[torch.Tensor] = []
        for latent_idx, latent_item in enumerate(self.run_plan.latent_items):
            teacher_image, student_image, metrics = self._run_single_sample(
                iteration=iteration,
                latent_idx=latent_idx,
                latent_item=latent_item,
                gumbel_temperature=gumbel_temperature,
            )
            iter_original_images.append(teacher_image.detach().cpu())
            iter_patched_images.append(student_image.detach().cpu())
            self._advance_step(metrics=metrics, iteration=iteration)
            empty_cuda_cache()
        return iter_original_images, iter_patched_images

    def _run_single_sample(
        self,
        *,
        iteration: int,
        latent_idx: int,
        latent_item: dict[str, Any],
        gumbel_temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        sample_seed = int(self.inputs.seed) + int(self.loop_state.global_step)
        latent_sample = latent_item["samples"]
        width = int(latent_sample.shape[-1] * 8)
        height = int(latent_sample.shape[-2] * 8)
        teacher_latent = self._sample_teacher(latent_item=latent_item, sample_seed=sample_seed)
        teacher_image = decode_vae(self.inputs.vae, teacher_latent)
        maybe_unload_model(self.inputs.model_original, logger=self.logger)
        empty_cuda_cache()
        student_sample = self._sample_student(
            latent_item=latent_item,
            sample_seed=sample_seed,
            width=width,
            height=height,
            gumbel_temperature=gumbel_temperature,
        )
        student_image = decode_vae(self.inputs.vae, student_sample.student_latent)
        quality_loss, quality_metrics, reward = self._compute_quality_reward(
            teacher_latent=teacher_latent,
            student_latent=student_sample.student_latent,
            teacher_image=teacher_image,
            student_image=student_image,
            actual_full_attn_ratio_eligible=student_sample.actual_full_attn_ratio_eligible,
        )
        policy_outcome = self._apply_policy_update(
            student_sample=student_sample,
            reward=reward,
            width=width,
            height=height,
            cfg_scale=float(self.inputs.cfg),
        )
        metrics = self._build_step_metrics(
            quality_loss=quality_loss,
            quality_metrics=quality_metrics,
            policy_outcome=policy_outcome,
            student_sample=student_sample,
            iteration=iteration,
            latent_idx=latent_idx,
            gumbel_temperature=gumbel_temperature,
            width=width,
            height=height,
        )
        return teacher_image, student_image, metrics

    def _sample_teacher(self, *, latent_item: dict[str, Any], sample_seed: int) -> dict[str, Any]:
        return sample_model(
            model=self.inputs.model_original,
            latent=latent_item,
            positive=self.inputs.positive,
            negative=self.inputs.negative,
            seed=sample_seed,
            steps=int(self.inputs.steps),
            cfg=float(self.inputs.cfg),
            sampler_name=self.inputs.sampler_name,
            scheduler=self.inputs.scheduler,
            denoise=float(self.inputs.denoise),
        )

    def _sample_student(
        self,
        *,
        latent_item: dict[str, Any],
        sample_seed: int,
        width: int,
        height: int,
        gumbel_temperature: float,
    ) -> StudentSample:
        if self.config.sigma_aware_training:
            return self._sample_student_sigma_aware(
                latent_item=latent_item,
                sample_seed=sample_seed,
                gumbel_temperature=gumbel_temperature,
            )
        return self._sample_student_mask_override(
            latent_item=latent_item,
            sample_seed=sample_seed,
            width=width,
            height=height,
            gumbel_temperature=gumbel_temperature,
        )

    def _sample_student_sigma_aware(
        self,
        *,
        latent_item: dict[str, Any],
        sample_seed: int,
        gumbel_temperature: float,
    ) -> StudentSample:
        training_wrapper = self.bundle.training_wrapper
        self.environment.flux_cfg.pop("controller_mask_override", None)
        self.environment.flux_cfg["controller"] = training_wrapper
        training_wrapper.reset()
        training_wrapper.temperature = float(gumbel_temperature)
        student_latent = sample_model(
            model=self.inputs.model_ttr,
            latent=latent_item,
            positive=self.inputs.positive,
            negative=self.inputs.negative,
            seed=sample_seed,
            steps=int(self.inputs.steps),
            cfg=float(self.inputs.cfg),
            sampler_name=self.inputs.sampler_name,
            scheduler=self.inputs.scheduler,
            denoise=float(self.inputs.denoise),
        )
        if not training_wrapper.step_records:
            raise RuntimeError("Flux2TTRControllerTrainer: no per-step controller records collected during sigma-aware sampling.")
        return StudentSample(
            student_latent=student_latent,
            actual_full_attn_ratio_eligible=float(training_wrapper.mean_full_attn_eligible()),
            actual_full_attn_ratio_overall=float(training_wrapper.mean_full_attn_overall()),
            expected_full_attn_ratio_eligible=float(training_wrapper.mean_expected_full_attn_eligible()),
            expected_full_attn_ratio_overall=float(training_wrapper.mean_expected_full_attn_overall()),
        )

    def _sample_student_mask_override(
        self,
        *,
        latent_item: dict[str, Any],
        sample_seed: int,
        width: int,
        height: int,
        gumbel_temperature: float,
    ) -> StudentSample:
        controller = self.bundle.controller
        effective_mask, ttr_eligible_mask, actual_eligible, actual_overall = self._build_mask_override_tensors(
            controller=controller,
            width=width,
            height=height,
            gumbel_temperature=gumbel_temperature,
        )
        student_latent = self._sample_student_with_mask_override(
            latent_item=latent_item,
            sample_seed=sample_seed,
            effective_mask=effective_mask,
        )
        return StudentSample(
            student_latent=student_latent,
            actual_full_attn_ratio_eligible=actual_eligible,
            actual_full_attn_ratio_overall=actual_overall,
            expected_full_attn_ratio_eligible=actual_eligible,
            expected_full_attn_ratio_overall=actual_overall,
            effective_mask=effective_mask,
            ttr_eligible_mask=ttr_eligible_mask,
        )

    def _build_mask_override_tensors(
        self,
        *,
        controller: flux2_ttr_controller.TTRController,
        width: int,
        height: int,
        gumbel_temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        with torch.no_grad():
            logits = controller(
                sigma=self.run_plan.sigma_value,
                cfg_scale=float(self.inputs.cfg),
                width=width,
                height=height,
            )
            sampled_mask = controller.sample_training_mask(logits, temperature=gumbel_temperature, hard=True)
        sampled_mask = (sampled_mask.detach() > 0.5).to(dtype=torch.float32)
        effective_mask = sampled_mask.clone()
        ttr_eligible_mask = self.environment.ttr_eligible_mask_cpu.to(device=effective_mask.device)
        effective_mask[~ttr_eligible_mask] = 1.0
        eligible_count = int(ttr_eligible_mask.sum().item())
        if eligible_count <= 0:
            raise RuntimeError("Flux2TTRControllerTrainer: no eligible layers available for controller penalty computation.")
        actual_overall = float(effective_mask.mean().item())
        actual_eligible = float(effective_mask[ttr_eligible_mask].mean().item())
        return effective_mask, ttr_eligible_mask, actual_eligible, actual_overall

    def _sample_student_with_mask_override(
        self,
        *,
        latent_item: dict[str, Any],
        sample_seed: int,
        effective_mask: torch.Tensor,
    ) -> dict[str, Any]:
        self.environment.flux_cfg["controller_mask_override"] = effective_mask.detach().cpu()
        try:
            return sample_model(
                model=self.inputs.model_ttr,
                latent=latent_item,
                positive=self.inputs.positive,
                negative=self.inputs.negative,
                seed=sample_seed,
                steps=int(self.inputs.steps),
                cfg=float(self.inputs.cfg),
                sampler_name=self.inputs.sampler_name,
                scheduler=self.inputs.scheduler,
                denoise=float(self.inputs.denoise),
            )
        finally:
            self.environment.flux_cfg.pop("controller_mask_override", None)

    def _compute_quality_reward(
        self,
        *,
        teacher_latent: dict[str, Any],
        student_latent: dict[str, Any],
        teacher_image: torch.Tensor,
        student_image: torch.Tensor,
        actual_full_attn_ratio_eligible: float,
    ) -> tuple[torch.Tensor, dict[str, float], float]:
        teacher_rgb = to_lpips_bchw(teacher_image)
        student_rgb = to_lpips_bchw(student_image)
        quality_loss, quality_metrics = self.bundle.trainer.compute_loss(
            teacher_latent=teacher_latent["samples"],
            student_latent=student_latent["samples"],
            actual_full_attn_ratio=actual_full_attn_ratio_eligible,
            teacher_rgb=teacher_rgb,
            student_rgb=student_rgb,
            include_efficiency_penalty=False,
        )
        reward = -float(quality_loss.detach().item())
        return quality_loss, quality_metrics, reward

    def _apply_policy_update(
        self,
        *,
        student_sample: StudentSample,
        reward: float,
        width: int,
        height: int,
        cfg_scale: float,
    ) -> PolicyOutcome:
        if self.config.sigma_aware_training:
            return self._apply_sigma_aware_policy_update(
                student_sample=student_sample,
                reward=reward,
                width=width,
                height=height,
                cfg_scale=cfg_scale,
            )
        return self._apply_mask_override_policy_update(student_sample=student_sample, reward=reward, width=width, height=height)

    def _apply_sigma_aware_policy_update(
        self,
        *,
        student_sample: StudentSample,
        reward: float,
        width: int,
        height: int,
        cfg_scale: float,
    ) -> PolicyOutcome:
        trainer = self.bundle.trainer
        training_wrapper = self.bundle.training_wrapper
        entropy_value = float(training_wrapper.mean_entropy())
        reward_terms = self._compute_sigma_reward_terms(reward=reward, entropy_value=entropy_value, actual_full_attn_ratio=student_sample.actual_full_attn_ratio_eligible)
        trainer._update_reward_baseline(reward_terms.reward_value)
        sigma_max = max(1e-8, max(float(rec["sigma"]) for rec in training_wrapper.step_records))
        policy_objective_t, policy_loss_t = self._recompute_sigma_policy_loss(
            baselined_reward=reward_terms.baselined_reward,
            sigma_max=sigma_max,
            width=width,
            height=height,
            cfg_scale=cfg_scale,
        )
        policy_loss_t = self._repair_sigma_policy_loss_if_detached(
            policy_objective_t=policy_objective_t,
            policy_loss_t=policy_loss_t,
            baselined_reward=reward_terms.baselined_reward,
            sigma_max=sigma_max,
            width=width,
            height=height,
            cfg_scale=cfg_scale,
        )
        self._optimizer_step_for_sigma_policy(policy_loss_t)
        step_summary = summarize_step_ttr(training_wrapper.per_step_ttr_ratio(), sigma_fallback=self.run_plan.sigma_value)
        reinforce_metrics = self._build_sigma_reinforce_metrics(
            policy_loss_t=policy_loss_t,
            reward=reward,
            entropy_value=entropy_value,
            reward_terms=reward_terms,
            step_summary=step_summary,
            student_sample=student_sample,
        )
        return PolicyOutcome(
            reinforce_metrics=reinforce_metrics,
            target_ttr_ratio=float(reward_terms.target_ttr_ratio),
            target_full_attn_ratio=float(reward_terms.target_full_attn_ratio),
            expected_full_attn_ratio_eligible=float(student_sample.expected_full_attn_ratio_eligible),
            expected_full_attn_ratio_overall=float(student_sample.expected_full_attn_ratio_overall),
        )

    def _compute_sigma_reward_terms(
        self,
        *,
        reward: float,
        entropy_value: float,
        actual_full_attn_ratio: float,
    ) -> SigmaRewardTerms:
        trainer = self.bundle.trainer
        target_ttr_ratio = float(trainer.target_ttr_ratio)
        target_full_attn_ratio = float(trainer._target_full_attn_ratio_from_ttr_ratio(target_ttr_ratio))
        efficiency_penalty = abs(float(actual_full_attn_ratio) - target_full_attn_ratio)
        efficiency_penalty_weighted = float(trainer.lambda_eff * efficiency_penalty)
        entropy_bonus = float(trainer.lambda_entropy * float(entropy_value))
        reward_value = float(reward - efficiency_penalty_weighted + entropy_bonus)
        baselined_reward = float(reward_value - trainer.reward_baseline)
        return SigmaRewardTerms(
            target_ttr_ratio=target_ttr_ratio,
            target_full_attn_ratio=target_full_attn_ratio,
            efficiency_penalty=efficiency_penalty,
            efficiency_penalty_weighted=efficiency_penalty_weighted,
            entropy_bonus=entropy_bonus,
            reward_value=reward_value,
            baselined_reward=baselined_reward,
        )

    def _recompute_sigma_policy_loss(
        self,
        *,
        baselined_reward: float,
        sigma_max: float,
        width: int,
        height: int,
        cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode(False):
            with torch.enable_grad():
                objective = self.bundle.training_wrapper.sigma_weighted_log_prob_recompute(
                    cfg_scale=float(cfg_scale),
                    width=int(width),
                    height=int(height),
                    sigma_max=float(sigma_max),
                    eligible_mask=self.environment.ttr_eligible_mask_cpu,
                )
                policy_loss = -float(baselined_reward) * objective
        return objective, policy_loss

    def _repair_sigma_policy_loss_if_detached(
        self,
        *,
        policy_objective_t: torch.Tensor,
        policy_loss_t: torch.Tensor,
        baselined_reward: float,
        sigma_max: float,
        width: int,
        height: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        if bool(policy_loss_t.requires_grad):
            return policy_loss_t
        repaired = self._enable_controller_gradients()
        if repaired > 0:
            policy_objective_t, policy_loss_t = self._recompute_sigma_policy_loss(
                baselined_reward=baselined_reward,
                sigma_max=sigma_max,
                width=width,
                height=height,
                cfg_scale=cfg_scale,
            )
        if bool(policy_loss_t.requires_grad):
            self.logger.info(
                "Flux2TTRControllerTrainer: restored detached sigma-aware policy loss by re-enabling requires_grad on %d controller params.",
                int(repaired),
            )
            return policy_loss_t
        self._log_sigma_detached_warning(
            repaired=repaired,
            policy_objective_t=policy_objective_t,
            policy_loss_t=policy_loss_t,
            width=width,
            height=height,
            cfg_scale=cfg_scale,
        )
        return policy_loss_t

    def _enable_controller_gradients(self) -> int:
        repaired = 0
        with torch.inference_mode(False):
            for param in self.bundle.trainer.controller.parameters():
                if bool(param.requires_grad):
                    continue
                param.requires_grad_(True)
                repaired += 1
        return int(repaired)

    def _log_sigma_detached_warning(
        self,
        *,
        repaired: int,
        policy_objective_t: torch.Tensor,
        policy_loss_t: torch.Tensor,
        width: int,
        height: int,
        cfg_scale: float,
    ) -> None:
        trainer = self.bundle.trainer
        param_list = list(trainer.controller.parameters())
        trainable_params = int(sum(1 for p in param_list if bool(p.requires_grad)))
        probe_requires_grad, probe_error = self._probe_controller_grad(width=width, height=height, cfg_scale=cfg_scale)
        self.logger.warning(
            "Flux2TTRControllerTrainer: sigma-aware policy loss still detached after recompute; skipping optimizer step for this sample. "
            "diag: trainable_params=%d/%d inference_tensors=%d probe_requires_grad=%s probe_error=%s repaired_params=%d "
            "policy_objective_requires_grad=%s policy_loss_requires_grad=%s recompute_debug=%s",
            trainable_params,
            int(len(param_list)),
            inference_tensor_count(trainer.controller),
            probe_requires_grad,
            probe_error or "<none>",
            int(repaired),
            bool(policy_objective_t.requires_grad),
            bool(policy_loss_t.requires_grad),
            str(getattr(self.bundle.training_wrapper, "last_recompute_debug", {})),
        )

    def _probe_controller_grad(self, *, width: int, height: int, cfg_scale: float) -> tuple[Optional[bool], str]:
        try:
            probe_sigma = float(self.bundle.training_wrapper.step_records[0]["sigma"])
            with torch.inference_mode(False):
                with torch.enable_grad():
                    probe_logits = self.bundle.trainer.controller(
                        sigma=probe_sigma,
                        cfg_scale=float(cfg_scale),
                        width=int(width),
                        height=int(height),
                    )
            if probe_logits.ndim > 1:
                probe_logits = probe_logits.reshape(-1)
            return bool(probe_logits.requires_grad), ""
        except Exception as exc:
            return None, str(exc)

    def _optimizer_step_for_sigma_policy(self, policy_loss_t: torch.Tensor) -> None:
        trainer = self.bundle.trainer
        with torch.inference_mode(False):
            with torch.enable_grad():
                trainer.optimizer.zero_grad(set_to_none=True)
                if bool(policy_loss_t.requires_grad):
                    policy_loss_t.backward()
                    if trainer.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainer.controller.parameters(), trainer.grad_clip_norm)
                    trainer.optimizer.step()

    def _build_sigma_reinforce_metrics(
        self,
        *,
        policy_loss_t: torch.Tensor,
        reward: float,
        entropy_value: float,
        reward_terms: SigmaRewardTerms,
        step_summary: dict[str, float],
        student_sample: StudentSample,
    ) -> dict[str, float]:
        trainer = self.bundle.trainer
        metrics = {
            "policy_loss": float(policy_loss_t.detach().item()),
            "total_loss": float(policy_loss_t.detach().item()),
            "efficiency_penalty": float(reward_terms.efficiency_penalty),
            "efficiency_penalty_weighted": float(reward_terms.efficiency_penalty_weighted),
            "reward": float(reward_terms.reward_value),
            "reward_quality": float(reward),
            "reward_baseline": float(trainer.reward_baseline),
            "baselined_reward": float(reward_terms.baselined_reward),
            "entropy": float(entropy_value),
            "entropy_bonus": float(reward_terms.entropy_bonus),
            "lambda_eff": float(trainer.lambda_eff),
            "lambda_entropy": float(trainer.lambda_entropy),
            "target_ttr_ratio": float(reward_terms.target_ttr_ratio),
            "target_full_attn_ratio": float(reward_terms.target_full_attn_ratio),
            "actual_full_attn_ratio": float(student_sample.actual_full_attn_ratio_eligible),
            "actual_full_attn_ratio_overall": float(student_sample.actual_full_attn_ratio_overall),
            "expected_full_attn_ratio": float(student_sample.expected_full_attn_ratio_eligible),
            "expected_full_attn_ratio_overall": float(student_sample.expected_full_attn_ratio_overall),
        }
        metrics.update(step_summary)
        return metrics

    def _apply_mask_override_policy_update(
        self,
        *,
        student_sample: StudentSample,
        reward: float,
        width: int,
        height: int,
    ) -> PolicyOutcome:
        if student_sample.effective_mask is None or student_sample.ttr_eligible_mask is None:
            raise RuntimeError("Flux2TTRControllerTrainer: missing mask override tensors for non-sigma-aware update.")
        reinforce_metrics = self.bundle.trainer.reinforce_step(
            sigma=self.run_plan.sigma_value,
            cfg_scale=float(self.inputs.cfg),
            width=int(width),
            height=int(height),
            sampled_mask=student_sample.effective_mask,
            reward=float(reward),
            actual_full_attn_ratio=float(student_sample.actual_full_attn_ratio_eligible),
            eligible_layer_mask=student_sample.ttr_eligible_mask,
            actual_full_attn_ratio_overall=float(student_sample.actual_full_attn_ratio_overall),
        )
        target_ttr_ratio = float(reinforce_metrics.get("target_ttr_ratio", self.bundle.trainer.target_ttr_ratio))
        target_full_attn_ratio = float(
            reinforce_metrics.get(
                "target_full_attn_ratio",
                self.bundle.trainer._target_full_attn_ratio_from_ttr_ratio(target_ttr_ratio),
            )
        )
        return PolicyOutcome(
            reinforce_metrics=reinforce_metrics,
            target_ttr_ratio=target_ttr_ratio,
            target_full_attn_ratio=target_full_attn_ratio,
            expected_full_attn_ratio_eligible=float(
                reinforce_metrics.get("expected_full_attn_ratio", student_sample.actual_full_attn_ratio_eligible)
            ),
            expected_full_attn_ratio_overall=float(
                reinforce_metrics.get("expected_full_attn_ratio_overall", student_sample.actual_full_attn_ratio_overall)
            ),
        )

    def _build_step_metrics(
        self,
        *,
        quality_loss: torch.Tensor,
        quality_metrics: dict[str, float],
        policy_outcome: PolicyOutcome,
        student_sample: StudentSample,
        iteration: int,
        latent_idx: int,
        gumbel_temperature: float,
        width: int,
        height: int,
    ) -> dict[str, float]:
        reinforce_metrics = policy_outcome.reinforce_metrics
        full_eligible, full_overall = self._resolve_full_attention_ratios(reinforce_metrics, student_sample)
        metrics = self._build_quality_ratio_metrics(
            quality_loss=quality_loss,
            quality_metrics=quality_metrics,
            reinforce_metrics=reinforce_metrics,
            policy_outcome=policy_outcome,
            full_attn_ratio_eligible=full_eligible,
            full_attn_ratio_overall=full_overall,
        )
        metrics.update(
            self._build_reward_context_metrics(
                reinforce_metrics=reinforce_metrics,
                policy_outcome=policy_outcome,
                iteration=iteration,
                latent_idx=latent_idx,
                gumbel_temperature=gumbel_temperature,
                width=width,
                height=height,
                full_attn_ratio_eligible=full_eligible,
                full_attn_ratio_overall=full_overall,
            )
        )
        self._maybe_attach_sigma_aware_metrics(metrics, reinforce_metrics)
        return metrics

    def _resolve_full_attention_ratios(
        self,
        reinforce_metrics: dict[str, float],
        student_sample: StudentSample,
    ) -> tuple[float, float]:
        full_eligible = float(reinforce_metrics.get("actual_full_attn_ratio", student_sample.actual_full_attn_ratio_eligible))
        full_overall = float(reinforce_metrics.get("actual_full_attn_ratio_overall", student_sample.actual_full_attn_ratio_overall))
        return full_eligible, full_overall

    def _build_quality_ratio_metrics(
        self,
        *,
        quality_loss: torch.Tensor,
        quality_metrics: dict[str, float],
        reinforce_metrics: dict[str, float],
        policy_outcome: PolicyOutcome,
        full_attn_ratio_eligible: float,
        full_attn_ratio_overall: float,
    ) -> dict[str, float]:
        expected_eligible = float(policy_outcome.expected_full_attn_ratio_eligible)
        expected_overall = float(policy_outcome.expected_full_attn_ratio_overall)
        return {
            "loss": float(quality_loss.detach().item()),
            "rmse": float(quality_metrics["rmse"]),
            "cosine_distance": float(quality_metrics["cosine_distance"]),
            "lpips": float(quality_metrics.get("lpips", 0.0)),
            "dreamsim": float(quality_metrics.get("dreamsim", 0.0)),
            "hps_penalty": float(quality_metrics.get("hps_penalty", 0.0)),
            "hps_teacher": float(quality_metrics.get("hps_teacher", 0.0)),
            "hps_student": float(quality_metrics.get("hps_student", 0.0)),
            "biqa_quality_penalty": float(quality_metrics.get("biqa_quality_penalty", 0.0)),
            "biqa_quality_teacher": float(quality_metrics.get("biqa_quality_teacher", 0.0)),
            "biqa_quality_student": float(quality_metrics.get("biqa_quality_student", 0.0)),
            "biqa_aesthetic_penalty": float(quality_metrics.get("biqa_aesthetic_penalty", 0.0)),
            "biqa_aesthetic_teacher": float(quality_metrics.get("biqa_aesthetic_teacher", 0.0)),
            "biqa_aesthetic_student": float(quality_metrics.get("biqa_aesthetic_student", 0.0)),
            "gemini_similarity": float(quality_metrics.get("gemini_similarity", 0.0)),
            "gemini_quality": float(quality_metrics.get("gemini_quality", 0.0)),
            "gemini_similarity_penalty": float(quality_metrics.get("gemini_similarity_penalty", 0.0)),
            "gemini_quality_penalty": float(quality_metrics.get("gemini_quality_penalty", 0.0)),
            "efficiency_penalty": float(reinforce_metrics["efficiency_penalty"]),
            "mask_mean": full_attn_ratio_eligible,
            "probs_mean": expected_eligible,
            "full_attn_ratio": full_attn_ratio_eligible,
            "ttr_ratio": float(max(0.0, 1.0 - full_attn_ratio_eligible)),
            "expected_full_attn_ratio": expected_eligible,
            "expected_ttr_ratio": float(max(0.0, 1.0 - expected_eligible)),
            "full_attn_eligible": full_attn_ratio_eligible,
            "ttr_eligible": float(max(0.0, 1.0 - full_attn_ratio_eligible)),
            "full_attn_overall": full_attn_ratio_overall,
            "ttr_overall": float(max(0.0, 1.0 - full_attn_ratio_overall)),
            "expected_full_attn_eligible": expected_eligible,
            "expected_ttr_eligible": float(max(0.0, 1.0 - expected_eligible)),
            "expected_full_attn_overall": expected_overall,
            "expected_ttr_overall": float(max(0.0, 1.0 - expected_overall)),
        }

    def _build_reward_context_metrics(
        self,
        *,
        reinforce_metrics: dict[str, float],
        policy_outcome: PolicyOutcome,
        iteration: int,
        latent_idx: int,
        gumbel_temperature: float,
        width: int,
        height: int,
        full_attn_ratio_eligible: float,
        full_attn_ratio_overall: float,
    ) -> dict[str, float]:
        return {
            "reward": float(reinforce_metrics.get("reward", 0.0)),
            "reward_quality": float(reinforce_metrics.get("reward_quality", 0.0)),
            "baselined_reward": float(reinforce_metrics.get("baselined_reward", 0.0)),
            "reward_baseline": float(reinforce_metrics["reward_baseline"]),
            "policy_loss": float(reinforce_metrics["policy_loss"]),
            "reinforce_total_loss": float(reinforce_metrics["total_loss"]),
            "efficiency_penalty_weighted": float(reinforce_metrics.get("efficiency_penalty_weighted", 0.0)),
            "lambda_eff": float(reinforce_metrics.get("lambda_eff", 1.0)),
            "lambda_entropy": float(reinforce_metrics.get("lambda_entropy", self.config.lambda_entropy)),
            "entropy": float(reinforce_metrics.get("entropy", 0.0)),
            "entropy_bonus": float(reinforce_metrics.get("entropy_bonus", 0.0)),
            "actual_full_attn_ratio": full_attn_ratio_eligible,
            "actual_ttr_ratio": float(max(0.0, 1.0 - full_attn_ratio_eligible)),
            "actual_full_attn_ratio_overall": full_attn_ratio_overall,
            "actual_ttr_ratio_overall": float(max(0.0, 1.0 - full_attn_ratio_overall)),
            "target_ttr_ratio": float(policy_outcome.target_ttr_ratio),
            "target_full_attn_ratio": float(policy_outcome.target_full_attn_ratio),
            "gumbel_temperature": float(gumbel_temperature),
            "sigma": float(reinforce_metrics.get("sigma", self.run_plan.sigma_value)),
            "iteration": float(iteration + 1),
            "latent_index": float(latent_idx),
            "width": float(width),
            "height": float(height),
            "step": float(self.loop_state.global_step + 1),
            "eligible_ttr_layers": float(len(self.environment.ready_layer_indices)),
            "forced_full_layers": float(self.environment.forced_full_layer_count),
            "sigma_aware_training": float(1.0 if self.config.sigma_aware_training else 0.0),
        }

    def _maybe_attach_sigma_aware_metrics(self, metrics: dict[str, float], reinforce_metrics: dict[str, float]) -> None:
        if not self.config.sigma_aware_training:
            return
        metrics["ttr_ratio_high_sigma"] = float(reinforce_metrics.get("ttr_ratio_high_sigma", 0.0))
        metrics["ttr_ratio_low_sigma"] = float(reinforce_metrics.get("ttr_ratio_low_sigma", 0.0))
        metrics["ttr_ratio_spread"] = float(reinforce_metrics.get("ttr_ratio_spread", 0.0))
        metrics["steps_per_trajectory"] = float(reinforce_metrics.get("steps_per_trajectory", 0.0))

    def _advance_step(self, *, metrics: dict[str, float], iteration: int) -> None:
        self.loop_state.global_step += 1
        self._maybe_save_periodic_checkpoint()
        self._maybe_log_comet_metrics(metrics)
        self._maybe_log_progress(metrics, iteration=iteration)

    def _maybe_save_periodic_checkpoint(self) -> None:
        if not self.config.checkpoint_path or self.loop_state.global_step >= self.run_plan.total_steps:
            return
        should_save = flux2_ttr_controller.should_save_controller_checkpoint_step(
            self.loop_state.global_step,
            checkpoint_every=self.run_plan.checkpoint_every,
        )
        if not should_save:
            return
        flux2_ttr_controller.save_controller_checkpoint(self.bundle.controller, self.config.checkpoint_path, trainer=self.bundle.trainer)
        self.logger.info(
            "Flux2TTRControllerTrainer: periodic controller checkpoint saved at step %d/%d: %s",
            self.loop_state.global_step,
            self.run_plan.total_steps,
            self.config.checkpoint_path,
        )

    def _maybe_log_comet_metrics(self, metrics: dict[str, float]) -> None:
        if self.run_plan.comet_experiment is None:
            return
        should_log = flux2_comet_logging.should_log_step(
            step=self.loop_state.global_step,
            every=self.config.log_every,
            total_steps=self.run_plan.total_steps,
            include_first_step=True,
        )
        if should_log:
            comet_payload = build_comet_metrics(metrics)
            self.logger.debug(
                "Flux2TTRControllerTrainer: Comet payload keys step=%d: %s",
                int(self.loop_state.global_step),
                comet_payload_keys_text(comet_payload),
            )
            safe_comet_log_metrics(
                self.run_plan.comet_experiment,
                comet_payload,
                step=self.loop_state.global_step,
                logger=self.logger,
            )

    def _maybe_log_progress(self, metrics: dict[str, float], *, iteration: int) -> None:
        if self.loop_state.global_step % self.config.log_every != 0 and self.loop_state.global_step != self.run_plan.total_steps:
            return
        if self.config.sigma_aware_training:
            self._log_sigma_aware_progress(metrics, iteration=iteration)
            return
        self._log_standard_progress(metrics, iteration=iteration)

    def _log_sigma_aware_progress(self, metrics: dict[str, float], *, iteration: int) -> None:
        self.logger.info(
            (
                "Flux2TTRControllerTrainer step=%d/%d iter=%d loss=%.6g rmse=%.6g cosine=%.6g "
                "lpips=%.6g full_attn_eligible=%.4g full_attn_overall=%.4g "
                "target_full=%.4g eff_penalty=%.6g entropy=%.6g entropy_bonus=%.6g "
                "spread=%.4g high_ttr=%.4g low_ttr=%.4g traj_steps=%d reward_baseline=%.6g"
            ),
            self.loop_state.global_step,
            self.run_plan.total_steps,
            iteration + 1,
            metrics["loss"],
            metrics["rmse"],
            metrics["cosine_distance"],
            metrics["lpips"],
            metrics["full_attn_eligible"],
            metrics["full_attn_overall"],
            metrics["target_full_attn_ratio"],
            metrics["efficiency_penalty"],
            metrics["entropy"],
            metrics["entropy_bonus"],
            metrics.get("ttr_ratio_spread", 0.0),
            metrics.get("ttr_ratio_high_sigma", 0.0),
            metrics.get("ttr_ratio_low_sigma", 0.0),
            int(metrics.get("steps_per_trajectory", 0.0)),
            metrics["reward_baseline"],
        )

    def _log_standard_progress(self, metrics: dict[str, float], *, iteration: int) -> None:
        self.logger.info(
            (
                "Flux2TTRControllerTrainer step=%d/%d iter=%d loss=%.6g rmse=%.6g cosine=%.6g "
                "lpips=%.6g full_attn_eligible=%.4g full_attn_overall=%.4g "
                "target_full=%.4g eff_penalty=%.6g entropy=%.6g entropy_bonus=%.6g reward_baseline=%.6g"
            ),
            self.loop_state.global_step,
            self.run_plan.total_steps,
            iteration + 1,
            metrics["loss"],
            metrics["rmse"],
            metrics["cosine_distance"],
            metrics["lpips"],
            metrics["full_attn_eligible"],
            metrics["full_attn_overall"],
            metrics["target_full_attn_ratio"],
            metrics["efficiency_penalty"],
            metrics["entropy"],
            metrics["entropy_bonus"],
            metrics["reward_baseline"],
        )

    def _update_last_images(self, iter_original_images: list[torch.Tensor], iter_patched_images: list[torch.Tensor]) -> None:
        if iter_original_images:
            self.loop_state.last_original_images = torch.cat(iter_original_images, dim=0)
        if iter_patched_images:
            self.loop_state.last_patched_images = torch.cat(iter_patched_images, dim=0)

    def _log_comet_training_complete(self) -> None:
        if self.run_plan.comet_experiment is None:
            return
        safe_comet_log_metrics(
            self.run_plan.comet_experiment,
            build_comet_metrics(
                {
                    "training_completed": 1.0,
                    "total_steps": float(self.run_plan.total_steps),
                    "total_iterations": float(self.config.total_iterations),
                    "final_reward_baseline": float(self.bundle.trainer.reward_baseline),
                }
            ),
            step=self.loop_state.global_step if self.loop_state.global_step > 0 else 0,
            logger=self.logger,
        )

    def _save_final_checkpoint(self) -> None:
        if self.config.checkpoint_path:
            flux2_ttr_controller.save_controller_checkpoint(
                self.bundle.controller,
                self.config.checkpoint_path,
                trainer=self.bundle.trainer,
            )


def run_controller_trainer_node(
    *,
    inputs: ControllerTrainerNodeInputs,
    deps: ControllerTrainerNodeDependencies,
) -> tuple[torch.Tensor, torch.Tensor, flux2_ttr_controller.TTRController]:
    engine = ControllerTrainerNodeEngine(inputs=inputs, deps=deps)
    return engine.run()
