import sys
import types

import pytest
import torch

import flux2_ttr_controller


def test_should_save_controller_checkpoint_step_defaults_to_every_ten():
    assert flux2_ttr_controller.should_save_controller_checkpoint_step(0) is False
    assert flux2_ttr_controller.should_save_controller_checkpoint_step(1) is False
    assert flux2_ttr_controller.should_save_controller_checkpoint_step(9) is False
    assert flux2_ttr_controller.should_save_controller_checkpoint_step(10) is True
    assert flux2_ttr_controller.should_save_controller_checkpoint_step(20) is True


def test_should_save_controller_checkpoint_step_respects_custom_interval():
    assert flux2_ttr_controller.should_save_controller_checkpoint_step(6, checkpoint_every=3) is True
    assert flux2_ttr_controller.should_save_controller_checkpoint_step(7, checkpoint_every=3) is False
    assert flux2_ttr_controller.should_save_controller_checkpoint_step(9, checkpoint_every=0) is False


def test_ttr_controller_forward_shape():
    controller = flux2_ttr_controller.TTRController(num_layers=6, embed_dim=16, hidden_dim=32)
    logits = controller(sigma=0.5, cfg_scale=4.0, width=128, height=128)
    assert logits.shape == (6,)
    assert torch.isfinite(logits).all()


def test_training_wrapper_records_steps_and_keeps_log_prob_grad_under_no_grad():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=3, embed_dim=8, hidden_dim=16)
    wrapper = flux2_ttr_controller.TrainingControllerWrapper(controller, temperature=1.0)

    with torch.no_grad():
        wrapper(sigma=0.9, cfg_scale=4.0, width=64, height=64)
        wrapper(sigma=0.5, cfg_scale=4.0, width=64, height=64)
        wrapper(sigma=0.1, cfg_scale=4.0, width=64, height=64)

    assert len(wrapper.step_records) == 3
    assert bool(wrapper.step_records[0]["log_probs"].requires_grad)
    total_log_prob = wrapper.total_log_prob()
    assert bool(total_log_prob.requires_grad)


def test_training_wrapper_uses_trainer_rebuilt_controller_from_inference_mode():
    torch.manual_seed(0)
    with torch.inference_mode():
        controller = flux2_ttr_controller.TTRController(num_layers=3, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=0.0)
    wrapper = flux2_ttr_controller.TrainingControllerWrapper(trainer.controller, temperature=1.0)

    with torch.no_grad():
        wrapper(sigma=0.4, cfg_scale=4.0, width=64, height=64)

    assert len(wrapper.step_records) == 1
    assert bool(wrapper.step_records[0]["log_probs"].requires_grad)


def test_training_wrapper_synthetic_logits_round_trip_to_sampled_mask(monkeypatch):
    def _fixed_sample(logits: torch.Tensor, temperature: float = 1.0, hard: bool = True) -> torch.Tensor:
        del temperature, hard
        return torch.tensor([1.0, 0.0, 1.0], device=logits.device, dtype=logits.dtype)

    monkeypatch.setattr(
        flux2_ttr_controller.TTRController,
        "sample_training_mask",
        staticmethod(_fixed_sample),
    )
    controller = flux2_ttr_controller.TTRController(num_layers=3, embed_dim=8, hidden_dim=16)
    wrapper = flux2_ttr_controller.TrainingControllerWrapper(controller)

    synthetic_logits = wrapper(sigma=0.7, cfg_scale=3.0, width=64, height=64)
    runtime_mask = (torch.sigmoid(synthetic_logits) > 0.5).to(dtype=torch.float32)

    assert torch.equal(runtime_mask, wrapper.step_records[0]["mask"])


def test_training_wrapper_ready_mask_clamps_non_ready_layers_to_full_attention(monkeypatch):
    def _all_ttr_sample(logits: torch.Tensor, temperature: float = 1.0, hard: bool = True) -> torch.Tensor:
        del temperature, hard
        return torch.zeros_like(logits)

    monkeypatch.setattr(
        flux2_ttr_controller.TTRController,
        "sample_training_mask",
        staticmethod(_all_ttr_sample),
    )
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=8, hidden_dim=16)
    ready_mask = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    wrapper = flux2_ttr_controller.TrainingControllerWrapper(controller, ready_mask=ready_mask)

    wrapper(sigma=0.5, cfg_scale=2.0, width=64, height=64)
    mask = wrapper.step_records[0]["mask"]

    assert torch.equal(mask, torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32))
    assert wrapper.mean_full_attn_eligible() == pytest.approx(0.0)
    assert wrapper.mean_full_attn_overall() == pytest.approx(0.5)


def test_training_wrapper_per_step_ttr_ratio_varies_by_sigma(monkeypatch):
    class _SigmaController(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, sigma, cfg_scale, width, height):
            del cfg_scale, width, height
            sigma_f = float(sigma.item()) if torch.is_tensor(sigma) else float(sigma)
            if sigma_f >= 0.5:
                base = torch.tensor([-6.0, -6.0, -6.0, -6.0], device=self.anchor.device)
            else:
                base = torch.tensor([6.0, 6.0, 6.0, 6.0], device=self.anchor.device)
            return base + self.anchor * 0.0

    def _threshold_sample(logits: torch.Tensor, temperature: float = 1.0, hard: bool = True) -> torch.Tensor:
        del temperature, hard
        return (logits > 0).to(dtype=logits.dtype)

    monkeypatch.setattr(
        flux2_ttr_controller.TTRController,
        "sample_training_mask",
        staticmethod(_threshold_sample),
    )

    wrapper = flux2_ttr_controller.TrainingControllerWrapper(_SigmaController())
    wrapper(sigma=0.9, cfg_scale=1.0, width=64, height=64)
    wrapper(sigma=0.1, cfg_scale=1.0, width=64, height=64)
    per_step = wrapper.per_step_ttr_ratio()

    assert len(per_step) == 2
    assert per_step[0][1] == pytest.approx(1.0)
    assert per_step[1][1] == pytest.approx(0.0)


def test_training_wrapper_recompute_sigma_weighted_log_prob_keeps_grad_path():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=8, hidden_dim=16)
    wrapper = flux2_ttr_controller.TrainingControllerWrapper(
        controller,
        ready_mask=torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
    )
    with torch.no_grad():
        wrapper(sigma=0.9, cfg_scale=4.0, width=64, height=64)
        wrapper(sigma=0.1, cfg_scale=4.0, width=64, height=64)
    for rec in wrapper.step_records:
        rec["log_probs"] = rec["log_probs"].detach()

    eligible = torch.tensor([True, True, False, False], dtype=torch.bool)
    recomputed = wrapper.sigma_weighted_log_prob_recompute(
        cfg_scale=4.0,
        width=64,
        height=64,
        eligible_mask=eligible,
    )
    assert bool(recomputed.requires_grad)
    recomputed.backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
        for p in controller.parameters()
    )


def test_training_wrapper_recompute_handles_inference_eligible_mask():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    wrapper = flux2_ttr_controller.TrainingControllerWrapper(controller)
    with torch.no_grad():
        wrapper(sigma=0.4, cfg_scale=4.0, width=64, height=64)
    with torch.inference_mode():
        eligible = torch.tensor([True, False], dtype=torch.bool)

    loss = wrapper.sigma_weighted_log_prob_recompute(
        cfg_scale=4.0,
        width=64,
        height=64,
        eligible_mask=eligible,
    )
    assert bool(loss.requires_grad)


def test_training_wrapper_recompute_all_false_eligible_mask_keeps_grad_path():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=3, embed_dim=8, hidden_dim=16)
    wrapper = flux2_ttr_controller.TrainingControllerWrapper(controller)
    with torch.no_grad():
        wrapper(sigma=0.6, cfg_scale=4.0, width=64, height=64)

    eligible = torch.tensor([False, False, False], dtype=torch.bool)
    loss = wrapper.sigma_weighted_log_prob_recompute(
        cfg_scale=4.0,
        width=64,
        height=64,
        eligible_mask=eligible,
    )
    assert bool(loss.requires_grad)
    assert wrapper.last_recompute_debug.get("result_requires_grad") is True


def test_recomputed_objective_combined_inside_inference_mode_false_keeps_grad():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    wrapper = flux2_ttr_controller.TrainingControllerWrapper(controller)
    with torch.no_grad():
        wrapper(sigma=0.6, cfg_scale=4.0, width=64, height=64)

    with torch.inference_mode():
        objective = wrapper.sigma_weighted_log_prob_recompute(cfg_scale=4.0, width=64, height=64)
        detached_loss = -1.0 * objective
    assert bool(objective.requires_grad)
    assert detached_loss.requires_grad is False

    with torch.inference_mode(False):
        with torch.enable_grad():
            objective_attached = wrapper.sigma_weighted_log_prob_recompute(cfg_scale=4.0, width=64, height=64)
            attached_loss = -1.0 * objective_attached
    assert bool(objective_attached.requires_grad)
    assert bool(attached_loss.requires_grad)


def test_ttr_controller_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=16, hidden_dim=32)
    path = tmp_path / "controller.pt"
    flux2_ttr_controller.save_controller_checkpoint(controller, str(path))

    loaded = flux2_ttr_controller.load_controller_checkpoint(str(path))
    out_a = controller(sigma=0.25, cfg_scale=3.0, width=64, height=64)
    out_b = loaded(sigma=0.25, cfg_scale=3.0, width=64, height=64)
    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_controller_checkpoint_persists_and_restores_trainer_state(tmp_path):
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=16, hidden_dim=32)
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        learning_rate=1e-2,
        target_ttr_ratio=0.6,
        lambda_entropy=0.35,
    )
    trainer.reinforce_step(
        sigma=0.8,
        cfg_scale=3.0,
        width=64,
        height=64,
        sampled_mask=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        reward=1.5,
        actual_full_attn_ratio=0.5,
    )

    path = tmp_path / "controller_with_trainer_state.pt"
    flux2_ttr_controller.save_controller_checkpoint(controller, str(path), trainer=trainer)
    payload = flux2_ttr_controller.load_controller_training_state(str(path))

    assert payload["reward_baseline"] is not None
    assert payload["reward_count"] is not None
    assert payload["reward_baseline_quality_floor"] == pytest.approx(-0.3)
    assert payload["lambda_entropy"] == pytest.approx(0.35)
    assert payload["optimizer_state_dict"] is not None

    loaded_controller = flux2_ttr_controller.load_controller_checkpoint(str(path))
    restored_trainer = flux2_ttr_controller.ControllerTrainer(
        loaded_controller,
        learning_rate=1e-2,
        target_ttr_ratio=0.6,
        lambda_entropy=0.1,
    )
    assert len(restored_trainer.optimizer.state) == 0
    restored_trainer.restore_training_state(payload)

    assert restored_trainer.reward_baseline == pytest.approx(trainer.reward_baseline)
    assert restored_trainer.reward_count == trainer.reward_count
    assert restored_trainer.reward_baseline_quality_floor == pytest.approx(trainer.reward_baseline_quality_floor)
    assert restored_trainer.lambda_entropy == pytest.approx(0.35)
    assert len(restored_trainer.optimizer.state) > 0


def test_controller_trainer_restore_training_state_is_backward_compatible():
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=0.0, lambda_entropy=0.42)
    trainer.restore_training_state(
        {
            "format": "flux2_ttr_controller_v1",
            "num_layers": 2,
            "embed_dim": 8,
            "hidden_dim": 16,
            "state_dict": {k: v.detach().cpu() for k, v in controller.state_dict().items()},
        }
    )
    assert trainer.reward_baseline == pytest.approx(0.0)
    assert trainer.reward_count == 0
    assert trainer.reward_baseline_quality_floor == pytest.approx(-0.3)
    assert trainer.lambda_entropy == pytest.approx(0.42)
    assert len(trainer.optimizer.state) == 0


def test_controller_trainer_restore_training_state_clones_inference_tensors(tmp_path):
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=3, embed_dim=16, hidden_dim=32)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, learning_rate=1e-2, target_ttr_ratio=0.5)
    trainer.reinforce_step(
        sigma=0.6,
        cfg_scale=3.5,
        width=64,
        height=64,
        sampled_mask=torch.tensor([1.0, 0.0, 1.0]),
        reward=0.8,
        actual_full_attn_ratio=2.0 / 3.0,
    )

    path = tmp_path / "controller_inference_restore.pt"
    flux2_ttr_controller.save_controller_checkpoint(controller, str(path), trainer=trainer)

    restored_controller = flux2_ttr_controller.load_controller_checkpoint(str(path))
    restored_trainer = flux2_ttr_controller.ControllerTrainer(
        restored_controller,
        learning_rate=1e-2,
        target_ttr_ratio=0.5,
    )

    with torch.inference_mode():
        payload = flux2_ttr_controller.load_controller_training_state(str(path))
        restored_trainer.restore_training_state(payload)

    for state in restored_trainer.optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                assert not flux2_ttr_controller.ControllerTrainer._is_inference_tensor(value)

    with torch.inference_mode():
        metrics = restored_trainer.reinforce_step(
            sigma=0.6,
            cfg_scale=3.5,
            width=64,
            height=64,
            sampled_mask=torch.tensor([1.0, 0.0, 1.0]),
            reward=0.7,
            actual_full_attn_ratio=2.0 / 3.0,
        )
    assert "total_loss" in metrics


def test_controller_trainer_compute_loss_without_lpips():
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=16, hidden_dim=32)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=0.0)

    teacher = torch.randn(1, 4, 8, 8)
    student = teacher + 0.05 * torch.randn_like(teacher)
    loss, metrics = trainer.compute_loss(
        teacher_latent=teacher,
        student_latent=student,
        actual_full_attn_ratio=0.4,
    )
    assert loss.item() > 0
    assert metrics["actual_full_attn_ratio"] == pytest.approx(0.4)
    assert metrics["actual_ttr_ratio"] == pytest.approx(0.6)
    assert metrics["target_ttr_ratio"] == pytest.approx(0.7)
    assert metrics["target_full_attn_ratio"] == pytest.approx(0.3)
    assert "rmse" in metrics
    assert "cosine_distance" in metrics
    assert metrics["dreamsim"] == pytest.approx(0.0)
    assert metrics["hps_penalty"] == pytest.approx(0.0)
    assert metrics["biqa_quality_penalty"] == pytest.approx(0.0)
    assert metrics["biqa_aesthetic_penalty"] == pytest.approx(0.0)


def test_controller_trainer_efficiency_penalty_uses_ttr_target_semantics():
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=0.0, target_ttr_ratio=0.7)
    teacher = torch.zeros(1, 4, 4, 4)
    student = torch.ones_like(teacher) * 0.25
    _, metrics = trainer.compute_loss(
        teacher_latent=teacher,
        student_latent=student,
        actual_full_attn_ratio=0.9,
        include_efficiency_penalty=False,
    )

    assert metrics["target_ttr_ratio"] == pytest.approx(0.7)
    assert metrics["target_full_attn_ratio"] == pytest.approx(0.3)
    assert metrics["actual_ttr_ratio"] == pytest.approx(0.1)
    assert metrics["efficiency_penalty"] == pytest.approx(0.6)


def test_controller_trainer_lpips_device_sync_for_compute_loss():
    class StrictDeviceLPIPS(torch.nn.Module):
        def __init__(self, initial_device: torch.device):
            super().__init__()
            self.current_device = torch.device(initial_device)

        def to(self, *args, **kwargs):
            device = kwargs.get("device")
            if device is None and args:
                device = args[0]
            if device is not None:
                self.current_device = torch.device(device)
            return self

        def forward(self, in0: torch.Tensor, in1: torch.Tensor) -> torch.Tensor:
            if in0.device != self.current_device or in1.device != self.current_device:
                raise RuntimeError(
                    f"StrictDeviceLPIPS mismatch: model on {self.current_device}, inputs on {in0.device}/{in1.device}"
                )
            return (in0 - in1).abs().mean().reshape(1, 1, 1, 1)

    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=0.0)
    trainer.lpips_weight = 1.0
    trainer.lpips_model = StrictDeviceLPIPS(torch.device("cuda:0"))

    teacher = torch.zeros(1, 4, 4, 4)
    student = torch.zeros_like(teacher)
    teacher_rgb = torch.zeros(1, 3, 8, 8)
    student_rgb = torch.ones(1, 3, 8, 8) * 0.25
    loss, metrics = trainer.compute_loss(
        teacher_latent=teacher,
        student_latent=student,
        actual_full_attn_ratio=0.0,
        teacher_rgb=teacher_rgb,
        student_rgb=student_rgb,
    )

    assert loss.item() >= 0.0
    assert metrics["lpips"] > 0.0
    assert trainer.lpips_model.current_device == teacher_rgb.device


def test_controller_trainer_lpips_requires_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "lpips", None)
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    with pytest.raises(RuntimeError):
        flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=1.0)


def test_controller_trainer_training_config_overrides_defaults(monkeypatch):
    monkeypatch.setattr(flux2_ttr_controller.ControllerTrainer, "_init_dreamsim_model", lambda self: None)
    monkeypatch.setattr(flux2_ttr_controller.ControllerTrainer, "_init_hps_model", lambda self: None)
    monkeypatch.setattr(flux2_ttr_controller.ControllerTrainer, "_init_biqa_models", lambda self: None)
    monkeypatch.setattr(flux2_ttr_controller.ControllerTrainer, "_init_gemini_client", lambda self: None)
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=16, hidden_dim=32)
    training_config = {
        "loss_config": {
            "rmse_weight": 2.5,
            "cosine_weight": 0.75,
            "lpips_weight": 0.0,
            "dreamsim_weight": 0.2,
            "hps_weight": 0.3,
            "biqa_quality_weight": 0.4,
            "biqa_aesthetic_weight": 0.5,
            "gemini_similarity_weight": 0.6,
            "gemini_quality_weight": 0.7,
            "gemini_model": "gemini-3-flash-preview",
            "gemini_api_key_env": "CUSTOM_GEMINI_API_KEY",
            "reward_baseline_quality_floor": -0.25,
        },
        "optimizer_config": {"learning_rate": 3e-4, "grad_clip_norm": 0.25},
        "schedule_config": {"target_ttr_ratio": 0.4, "lambda_eff": 2.25, "lambda_entropy": 0.35},
    }
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        training_config=training_config,
        learning_rate=1e-2,
        rmse_weight=7.0,
        cosine_weight=7.0,
        lpips_weight=0.0,
        dreamsim_weight=0.0,
        hps_weight=0.0,
        biqa_quality_weight=0.0,
        biqa_aesthetic_weight=0.0,
        gemini_similarity_weight=0.0,
        gemini_quality_weight=0.0,
        target_ttr_ratio=0.9,
        lambda_eff=7.0,
        lambda_entropy=0.9,
        grad_clip_norm=3.0,
    )
    assert trainer.rmse_weight == pytest.approx(2.5)
    assert trainer.cosine_weight == pytest.approx(0.75)
    assert trainer.target_ttr_ratio == pytest.approx(0.4)
    assert trainer.lambda_eff == pytest.approx(2.25)
    assert trainer.lambda_entropy == pytest.approx(0.35)
    assert trainer.grad_clip_norm == pytest.approx(0.25)
    assert trainer.dreamsim_weight == pytest.approx(0.2)
    assert trainer.hps_weight == pytest.approx(0.3)
    assert trainer.biqa_quality_weight == pytest.approx(0.4)
    assert trainer.biqa_aesthetic_weight == pytest.approx(0.5)
    assert trainer.gemini_similarity_weight == pytest.approx(0.6)
    assert trainer.gemini_quality_weight == pytest.approx(0.7)
    assert trainer.gemini_model == "gemini-3-flash-preview"
    assert trainer.gemini_api_key_env == "CUSTOM_GEMINI_API_KEY"
    assert trainer.reward_baseline_quality_floor == pytest.approx(-0.25)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


def test_controller_trainer_compute_loss_with_additional_quality_terms():
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=0.0)
    trainer.dreamsim_weight = 0.5
    trainer.hps_weight = 0.25
    trainer.biqa_quality_weight = 0.75
    trainer.biqa_aesthetic_weight = 0.1
    trainer.gemini_similarity_weight = 0.4
    trainer.gemini_quality_weight = 0.2
    trainer._score_dreamsim = lambda **kwargs: torch.tensor(2.0, device=kwargs["loss"].device)
    hps_values = iter((0.8, 0.3))
    biqa_quality_values = iter((0.9, 0.2))
    biqa_aesthetic_values = iter((0.7, 0.4))
    trainer._score_hps_batch = lambda **kwargs: torch.tensor(next(hps_values), device=kwargs["loss"].device)
    trainer._score_biqa_quality = lambda **kwargs: torch.tensor(next(biqa_quality_values), device=kwargs["loss"].device)
    trainer._score_biqa_aesthetic = lambda **kwargs: torch.tensor(next(biqa_aesthetic_values), device=kwargs["loss"].device)
    trainer._score_gemini_batch = lambda **kwargs: (
        torch.tensor(8.0, device=kwargs["loss"].device),
        torch.tensor(2.0, device=kwargs["loss"].device),
    )

    teacher = torch.zeros(1, 4, 4, 4)
    student = torch.zeros_like(teacher)
    teacher_rgb = torch.zeros(1, 3, 8, 8)
    student_rgb = torch.ones(1, 3, 8, 8) * 0.25
    loss, metrics = trainer.compute_loss(
        teacher_latent=teacher,
        student_latent=student,
        actual_full_attn_ratio=0.0,
        teacher_rgb=teacher_rgb,
        student_rgb=student_rgb,
        include_efficiency_penalty=False,
    )

    expected_quality_terms = (
        0.5 * 2.0
        + 0.25 * (0.8 - 0.3)
        + 0.75 * (0.9 - 0.2)
        + 0.1 * (0.7 - 0.4)
        + 0.4 * ((10.0 - 8.0) / 9.0)
        + 0.2 * ((10.0 - 2.0) / 9.0)
    )
    base_terms = trainer.rmse_weight * metrics["rmse"] + trainer.cosine_weight * metrics["cosine_distance"]
    assert metrics["dreamsim"] == pytest.approx(2.0)
    assert metrics["hps_penalty"] == pytest.approx(0.5)
    assert metrics["biqa_quality_penalty"] == pytest.approx(0.7)
    assert metrics["biqa_aesthetic_penalty"] == pytest.approx(0.3)
    assert metrics["gemini_similarity"] == pytest.approx(8.0)
    assert metrics["gemini_quality"] == pytest.approx(2.0)
    assert metrics["gemini_similarity_penalty"] == pytest.approx((10.0 - 8.0) / 9.0)
    assert metrics["gemini_quality_penalty"] == pytest.approx((10.0 - 2.0) / 9.0)
    assert loss.item() == pytest.approx(base_terms + expected_quality_terms, rel=1e-5)


def test_controller_trainer_parse_gemini_json_scores():
    similarity, quality = flux2_ttr_controller.ControllerTrainer._parse_gemini_json_scores(
        "{\"similarity\":8,\"quality\":2}"
    )
    assert similarity == pytest.approx(8.0)
    assert quality == pytest.approx(2.0)


def test_controller_trainer_parse_gemini_json_scores_rejects_invalid_payload():
    with pytest.raises(ValueError):
        flux2_ttr_controller.ControllerTrainer._parse_gemini_json_scores("{\"similarity\":8}")


def test_controller_trainer_gemini_requires_api_key_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    with pytest.raises(RuntimeError, match="environment variable"):
        flux2_ttr_controller.ControllerTrainer(controller, gemini_similarity_weight=1.0)


def test_controller_trainer_dreamsim_recovers_from_shadowed_utils(monkeypatch, tmp_path):
    dreamsim_pkg = tmp_path / "dreamsim"
    dreamsim_pkg.mkdir()
    (dreamsim_pkg / "__init__.py").write_text("from .entry import dreamsim\n", encoding="utf-8")
    (dreamsim_pkg / "entry.py").write_text(
        "import torch\n"
        "\n"
        "from utils import trunc_normal_\n"
        "\n"
        "class FakeDreamSimModel:\n"
        "    def __init__(self):\n"
        "        self.eval_called = False\n"
        "\n"
        "    def eval(self):\n"
        "        self.eval_called = True\n"
        "        return self\n"
        "\n"
        "    def __call__(self, student, teacher):\n"
        "        return 0.0\n"
        "\n"
        "def dreamsim(pretrained=True, device='cpu'):\n"
        "    del pretrained, device\n"
        "    trunc_normal_(torch.empty(1))\n"
        "    return FakeDreamSimModel(), (lambda img: img)\n",
        encoding="utf-8",
    )
    (dreamsim_pkg / "utils.py").write_text(
        "def trunc_normal_(_):\n"
        "    return None\n",
        encoding="utf-8",
    )

    shadow_utils = types.ModuleType("utils")
    monkeypatch.setitem(sys.modules, "utils", shadow_utils)
    monkeypatch.syspath_prepend(str(tmp_path))
    for module_name in ("dreamsim", "dreamsim.entry", "dreamsim.utils"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, dreamsim_weight=1.0)

    assert getattr(trainer.dreamsim_model, "eval_called", False) is True
    assert callable(trainer.dreamsim_preprocess)
    assert sys.modules["utils"] is shadow_utils


def test_controller_trainer_dreamsim_recovers_when_dreamsim_is_single_module(monkeypatch, tmp_path):
    (tmp_path / "dreamsim.py").write_text(
        "from utils import trunc_normal_\n"
        "\n"
        "class FakeDreamSimModel:\n"
        "    def __init__(self):\n"
        "        self.eval_called = False\n"
        "\n"
        "    def eval(self):\n"
        "        self.eval_called = True\n"
        "        return self\n"
        "\n"
        "    def __call__(self, student, teacher):\n"
        "        return 0.0\n"
        "\n"
        "def dreamsim(pretrained=True, device='cpu'):\n"
        "    del pretrained, device\n"
        "    trunc_normal_(None)\n"
        "    return FakeDreamSimModel(), (lambda img: img)\n",
        encoding="utf-8",
    )
    (tmp_path / "utils.py").write_text(
        "def trunc_normal_(_):\n"
        "    return None\n",
        encoding="utf-8",
    )

    shadow_utils = types.ModuleType("utils")
    monkeypatch.setitem(sys.modules, "utils", shadow_utils)
    monkeypatch.syspath_prepend(str(tmp_path))
    for module_name in ("dreamsim",):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, dreamsim_weight=1.0)

    assert getattr(trainer.dreamsim_model, "eval_called", False) is True
    assert callable(trainer.dreamsim_preprocess)
    assert sys.modules["utils"] is shadow_utils


def test_controller_trainer_dreamsim_recovers_by_patching_shadowed_utils(monkeypatch, tmp_path):
    dreamsim_pkg = tmp_path / "dreamsim"
    dreamsim_pkg.mkdir()
    (dreamsim_pkg / "__init__.py").write_text("from .entry import dreamsim\n", encoding="utf-8")
    (dreamsim_pkg / "entry.py").write_text(
        "import torch\n"
        "\n"
        "from utils import trunc_normal_\n"
        "\n"
        "class FakeDreamSimModel:\n"
        "    def __init__(self):\n"
        "        self.eval_called = False\n"
        "\n"
        "    def eval(self):\n"
        "        self.eval_called = True\n"
        "        return self\n"
        "\n"
        "    def __call__(self, student, teacher):\n"
        "        return 0.0\n"
        "\n"
        "def dreamsim(pretrained=True, device='cpu'):\n"
        "    del pretrained, device\n"
        "    trunc_normal_(torch.empty(1))\n"
        "    return FakeDreamSimModel(), (lambda img: img)\n",
        encoding="utf-8",
    )

    shadow_utils = types.ModuleType("utils")
    monkeypatch.setitem(sys.modules, "utils", shadow_utils)
    monkeypatch.syspath_prepend(str(tmp_path))
    for module_name in ("dreamsim", "dreamsim.entry"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, dreamsim_weight=1.0)

    assert getattr(trainer.dreamsim_model, "eval_called", False) is True
    assert callable(trainer.dreamsim_preprocess)
    assert sys.modules["utils"] is shadow_utils
    assert not hasattr(shadow_utils, "trunc_normal_")


def test_controller_trainer_dreamsim_recovers_when_utils_collision_happens_on_factory_call(monkeypatch, tmp_path):
    dreamsim_pkg = tmp_path / "dreamsim"
    dreamsim_pkg.mkdir()
    (dreamsim_pkg / "__init__.py").write_text(
        "def dreamsim(pretrained=True, device='cpu'):\n"
        "    from .runtime import build\n"
        "    return build(pretrained=pretrained, device=device)\n",
        encoding="utf-8",
    )
    (dreamsim_pkg / "runtime.py").write_text(
        "from utils import trunc_normal_\n"
        "\n"
        "class FakeDreamSimModel:\n"
        "    def __init__(self):\n"
        "        self.eval_called = False\n"
        "\n"
        "    def eval(self):\n"
        "        self.eval_called = True\n"
        "        return self\n"
        "\n"
        "    def __call__(self, student, teacher):\n"
        "        return 0.0\n"
        "\n"
        "def build(pretrained=True, device='cpu'):\n"
        "    del pretrained, device\n"
        "    trunc_normal_(None)\n"
        "    return FakeDreamSimModel(), (lambda img: img)\n",
        encoding="utf-8",
    )
    (dreamsim_pkg / "utils.py").write_text(
        "def trunc_normal_(_):\n"
        "    return None\n",
        encoding="utf-8",
    )

    shadow_utils = types.ModuleType("utils")
    monkeypatch.setitem(sys.modules, "utils", shadow_utils)
    monkeypatch.syspath_prepend(str(tmp_path))
    for module_name in ("dreamsim", "dreamsim.runtime", "dreamsim.utils"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, dreamsim_weight=1.0)

    assert getattr(trainer.dreamsim_model, "eval_called", False) is True
    assert callable(trainer.dreamsim_preprocess)
    assert sys.modules["utils"] is shadow_utils


def test_controller_trainer_dreamsim_requires_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "dreamsim", None)
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    with pytest.raises(RuntimeError):
        flux2_ttr_controller.ControllerTrainer(controller, dreamsim_weight=1.0)


def test_controller_trainer_hps_requires_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "hpsv2", None)
    monkeypatch.setitem(sys.modules, "ImageReward", None)
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    with pytest.raises(RuntimeError):
        flux2_ttr_controller.ControllerTrainer(controller, hps_weight=1.0)


def test_controller_trainer_hps_missing_bpe_asset_reports_fix(monkeypatch):
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, hps_weight=0.0)

    class _FakeHpsModule:
        @staticmethod
        def score(*args, **kwargs):
            del args, kwargs
            raise FileNotFoundError(
                "[Errno 2] No such file or directory: "
                "'/venv/main/lib/python3.12/site-packages/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz'"
            )

    trainer.hps_backend = "hpsv2"
    trainer.hps_module = _FakeHpsModule()

    rgb = torch.zeros(1, 3, 8, 8)
    with pytest.raises(RuntimeError, match="missing tokenizer asset"):
        trainer._score_hps_batch(
            rgb_01=rgb,
            prompt="a test prompt",
            loss=torch.tensor(0.0),
        )


def test_controller_trainer_biqa_requires_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyiqa", None)
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    with pytest.raises(RuntimeError):
        flux2_ttr_controller.ControllerTrainer(controller, biqa_quality_weight=1.0)


def test_controller_trainer_compute_loss_can_skip_efficiency_penalty():
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=0.0, target_ttr_ratio=0.2)
    teacher = torch.zeros(1, 4, 4, 4)
    student = torch.ones_like(teacher) * 0.25

    loss_with_penalty, metrics_with_penalty = trainer.compute_loss(
        teacher_latent=teacher,
        student_latent=student,
        actual_full_attn_ratio=0.9,
        include_efficiency_penalty=True,
    )
    loss_without_penalty, metrics_without_penalty = trainer.compute_loss(
        teacher_latent=teacher,
        student_latent=student,
        actual_full_attn_ratio=0.9,
        include_efficiency_penalty=False,
    )

    assert metrics_with_penalty["efficiency_penalty"] == pytest.approx(metrics_without_penalty["efficiency_penalty"])
    assert loss_with_penalty.item() > loss_without_penalty.item()


def test_controller_trainer_reinforce_penalty_is_symmetric_about_target():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=10, embed_dim=16, hidden_dim=32)
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        learning_rate=1e-2,
        target_ttr_ratio=0.5,  # target_full_attn_ratio=0.5
        lambda_eff=1.0,
        lambda_entropy=0.0,
    )
    mask_low = torch.tensor([1.0, 1.0, 1.0] + [0.0] * 7)
    mask_high = torch.tensor([1.0] * 7 + [0.0] * 3)
    metrics_low = trainer.reinforce_step(
        sigma=0.8,
        cfg_scale=3.0,
        width=64,
        height=64,
        sampled_mask=mask_low,
        reward=0.0,
        actual_full_attn_ratio=0.3,
    )
    metrics_high = trainer.reinforce_step(
        sigma=0.8,
        cfg_scale=3.0,
        width=64,
        height=64,
        sampled_mask=mask_high,
        reward=0.0,
        actual_full_attn_ratio=0.7,
    )
    assert metrics_low["target_full_attn_ratio"] == pytest.approx(0.5)
    assert metrics_high["target_full_attn_ratio"] == pytest.approx(0.5)
    assert metrics_low["efficiency_penalty"] == pytest.approx(0.2, abs=1e-6)
    assert metrics_high["efficiency_penalty"] == pytest.approx(0.2, abs=1e-6)


def test_controller_trainer_reward_baseline_quality_floor_clamps_update():
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        lpips_weight=0.0,
        reward_baseline_quality_floor=-0.3,
    )
    for _ in range(200):
        trainer._update_reward_baseline(-0.5)
    assert trainer.reward_baseline == pytest.approx(-0.3, abs=1e-3)


def test_controller_trainer_reinforce_step_updates_parameters():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=16, hidden_dim=32)
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        learning_rate=1e-2,
        target_ttr_ratio=0.6,
        lambda_entropy=0.0,
        grad_clip_norm=1.0,
    )
    before = [p.detach().clone() for p in controller.parameters()]

    metrics_a = trainer.reinforce_step(
        sigma=0.8,
        cfg_scale=3.0,
        width=64,
        height=64,
        sampled_mask=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        reward=1.5,
        actual_full_attn_ratio=0.5,
    )
    metrics_b = trainer.reinforce_step(
        sigma=0.8,
        cfg_scale=3.0,
        width=64,
        height=64,
        sampled_mask=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        reward=-0.5,
        actual_full_attn_ratio=0.25,
    )

    after = list(controller.parameters())
    assert any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))
    assert "policy_loss" in metrics_a
    assert "total_loss" in metrics_a
    assert metrics_a["efficiency_penalty"] == pytest.approx(0.1)
    assert metrics_a["reward_quality"] == pytest.approx(1.5)
    assert metrics_a["reward"] == pytest.approx(1.4)
    assert metrics_a["total_loss"] == pytest.approx(metrics_a["policy_loss"])
    assert metrics_a["reward_baseline"] == pytest.approx(1.4)
    assert metrics_b["reward_baseline"] < metrics_a["reward_baseline"]


def test_controller_trainer_reinforce_penalty_uses_full_attention_budget():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=3, embed_dim=16, hidden_dim=32)
    with torch.no_grad():
        for param in controller.parameters():
            param.zero_()
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        learning_rate=0.0,
        target_ttr_ratio=0.7,
        lambda_entropy=0.0,
        grad_clip_norm=1.0,
    )
    metrics = trainer.reinforce_step(
        sigma=0.8,
        cfg_scale=3.0,
        width=64,
        height=64,
        sampled_mask=torch.tensor([1.0, 1.0, 1.0]),
        reward=0.0,
        actual_full_attn_ratio=1.0,
    )

    assert metrics["target_ttr_ratio"] == pytest.approx(0.7)
    assert metrics["target_full_attn_ratio"] == pytest.approx(0.3)
    assert metrics["probs_mean"] == pytest.approx(0.5)
    assert metrics["expected_full_attn_ratio"] == pytest.approx(0.5)
    assert metrics["expected_ttr_ratio"] == pytest.approx(0.5)
    assert metrics["efficiency_penalty"] == pytest.approx(0.7, abs=1e-6)
    assert metrics["reward_quality"] == pytest.approx(0.0)
    assert metrics["reward"] == pytest.approx(-0.7, abs=1e-6)
    assert metrics["total_loss"] == pytest.approx(metrics["policy_loss"])


def test_controller_trainer_reinforce_penalty_weight_scales_reward():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=3, embed_dim=16, hidden_dim=32)
    with torch.no_grad():
        for param in controller.parameters():
            param.zero_()
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        learning_rate=0.0,
        target_ttr_ratio=0.7,
        lambda_eff=2.0,
        lambda_entropy=0.0,
        grad_clip_norm=1.0,
    )
    metrics = trainer.reinforce_step(
        sigma=0.8,
        cfg_scale=3.0,
        width=64,
        height=64,
        sampled_mask=torch.tensor([1.0, 1.0, 1.0]),
        reward=0.0,
        actual_full_attn_ratio=1.0,
    )

    assert metrics["efficiency_penalty"] == pytest.approx(0.7, abs=1e-6)
    assert metrics["efficiency_penalty_weighted"] == pytest.approx(1.4, abs=1e-6)
    assert metrics["reward"] == pytest.approx(-1.4, abs=1e-6)
    assert metrics["lambda_eff"] == pytest.approx(2.0)


def test_controller_trainer_reinforce_penalty_uses_eligible_layers_only():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=16, hidden_dim=32)
    with torch.no_grad():
        for param in controller.parameters():
            param.zero_()
        # First two layers (eligible) keep p=0.5, last two layers (forced full) keep p~1.0.
        controller.mlp[-1].bias.copy_(torch.tensor([0.0, 0.0, 10.0, 10.0]))
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        learning_rate=0.0,
        target_ttr_ratio=0.5,  # target_full_attn_ratio=0.5 over eligible layers.
        lambda_entropy=0.0,
        grad_clip_norm=1.0,
    )
    metrics = trainer.reinforce_step(
        sigma=0.8,
        cfg_scale=3.0,
        width=64,
        height=64,
        sampled_mask=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        reward=0.0,
        actual_full_attn_ratio=1.0,
        eligible_layer_mask=torch.tensor([True, True, False, False]),
        actual_full_attn_ratio_overall=1.0,
    )

    # Penalty is based on sampled eligible actions (all ones here), not logits/probs.
    assert metrics["target_full_attn_ratio"] == pytest.approx(0.5)
    assert metrics["probs_mean"] == pytest.approx(0.5, abs=1e-4)
    assert metrics["probs_mean_overall"] > 0.7
    assert metrics["efficiency_penalty"] == pytest.approx(0.5, abs=1e-6)
    assert metrics["reward"] == pytest.approx(-0.5, abs=1e-6)
    assert metrics["expected_full_attn_ratio"] == pytest.approx(0.5, abs=1e-4)
    assert metrics["expected_full_attn_ratio_overall"] == pytest.approx(0.75, abs=1e-4)
    assert metrics["actual_full_attn_ratio"] == pytest.approx(1.0)
    assert metrics["actual_full_attn_ratio_overall"] == pytest.approx(1.0)
    assert metrics["eligible_layer_count"] == pytest.approx(2.0)
    assert metrics["forced_full_layer_count"] == pytest.approx(2.0)


def test_controller_trainer_reinforce_entropy_bonus_uses_eligible_layers():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=16, hidden_dim=32)
    with torch.no_grad():
        for param in controller.parameters():
            param.zero_()
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        learning_rate=0.0,
        target_ttr_ratio=1.0,  # target_full_attn_ratio=0.0, so zero mask has no efficiency penalty.
        lambda_eff=1.0,
        lambda_entropy=0.2,
        grad_clip_norm=1.0,
    )
    metrics = trainer.reinforce_step(
        sigma=0.8,
        cfg_scale=3.0,
        width=64,
        height=64,
        sampled_mask=torch.tensor([0.0, 0.0, 1.0, 1.0]),
        reward=0.0,
        actual_full_attn_ratio=0.0,
        eligible_layer_mask=torch.tensor([True, True, False, False]),
        actual_full_attn_ratio_overall=0.5,
    )

    expected_entropy = float(torch.log(torch.tensor(2.0)).item())
    expected_bonus = 0.2 * expected_entropy
    assert metrics["entropy"] == pytest.approx(expected_entropy, abs=1e-6)
    assert metrics["entropy_bonus"] == pytest.approx(expected_bonus, abs=1e-6)
    assert metrics["reward_quality"] == pytest.approx(0.0)
    assert metrics["reward"] == pytest.approx(expected_bonus, abs=1e-6)
    assert metrics["lambda_entropy"] == pytest.approx(0.2)


def test_controller_trainer_reinforce_step_works_under_inference_mode():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=3, embed_dim=16, hidden_dim=32)
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        learning_rate=1e-2,
        target_ttr_ratio=0.5,
        lambda_entropy=0.0,
    )
    before = [p.detach().clone() for p in controller.parameters()]

    with torch.inference_mode():
        metrics = trainer.reinforce_step(
            sigma=0.6,
            cfg_scale=3.5,
            width=64,
            height=64,
            sampled_mask=torch.tensor([1.0, 0.0, 1.0]),
            reward=0.8,
            actual_full_attn_ratio=2.0 / 3.0,
        )

    after = list(controller.parameters())
    assert any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))
    assert "total_loss" in metrics
    assert metrics["reward_quality"] == pytest.approx(0.8)
    assert metrics["reward"] == pytest.approx(0.8 - (2.0 / 3.0 - 0.5), abs=1e-6)


def test_controller_trainer_reinforce_step_with_eligible_mask_works_under_inference_mode():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=3, embed_dim=16, hidden_dim=32)
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        learning_rate=1e-2,
        target_ttr_ratio=0.5,
        lambda_entropy=0.0,
    )
    before = [p.detach().clone() for p in controller.parameters()]

    with torch.inference_mode():
        eligible = torch.tensor([True, True, False], dtype=torch.bool)
        metrics = trainer.reinforce_step(
            sigma=0.6,
            cfg_scale=3.5,
            width=64,
            height=64,
            sampled_mask=torch.tensor([1.0, 0.0, 1.0]),
            reward=0.8,
            actual_full_attn_ratio=0.5,
            eligible_layer_mask=eligible,
            actual_full_attn_ratio_overall=2.0 / 3.0,
        )

    after = list(controller.parameters())
    assert any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))
    assert "total_loss" in metrics
    assert metrics["reward_quality"] == pytest.approx(0.8)
    assert metrics["reward"] == pytest.approx(0.8)
    assert metrics["eligible_layer_count"] == pytest.approx(2.0)


def test_controller_trainer_rebuilds_controller_created_in_inference_mode():
    with torch.inference_mode():
        controller = flux2_ttr_controller.TTRController(num_layers=3, embed_dim=16, hidden_dim=32)

    trainer = flux2_ttr_controller.ControllerTrainer(controller, learning_rate=1e-2, target_ttr_ratio=0.5)
    assert not any(
        flux2_ttr_controller.ControllerTrainer._is_inference_tensor(p)
        for p in trainer.controller.parameters()
    )

    before = [p.detach().clone() for p in trainer.controller.parameters()]
    metrics = trainer.reinforce_step(
        sigma=0.7,
        cfg_scale=2.5,
        width=64,
        height=64,
        sampled_mask=torch.tensor([1.0, 0.0, 1.0]),
        reward=0.4,
        actual_full_attn_ratio=2.0 / 3.0,
    )
    after = list(trainer.controller.parameters())
    assert any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))
    assert "total_loss" in metrics


def test_controller_trainer_train_step_uses_mask_for_gradient_flow():
    torch.manual_seed(0)
    controller = flux2_ttr_controller.TTRController(num_layers=4, embed_dim=16, hidden_dim=32)
    trainer = flux2_ttr_controller.ControllerTrainer(
        controller,
        learning_rate=1e-2,
        rmse_weight=1.0,
        cosine_weight=0.0,
        lpips_weight=0.0,
        target_ttr_ratio=0.0,  # Disable efficiency penalty so RMSE path must drive gradients.
    )
    teacher = torch.zeros(1, 4, 4, 4)
    before = [p.detach().clone() for p in controller.parameters()]

    def _student_forward(mask: torch.Tensor, logits: torch.Tensor):
        del logits
        scale = mask.mean()
        student = scale * torch.ones_like(teacher)
        return {
            "student_latent": student,
            "actual_full_attn_ratio": scale,
        }

    metrics = trainer.train_step(
        sigma=0.5,
        cfg_scale=4.0,
        width=64,
        height=64,
        teacher_latent=teacher,
        student_forward_fn=_student_forward,
    )
    assert metrics["loss"] > 0.0
    assert metrics["mask_mean"] >= 0.0
    assert metrics["layers_full_ratio"] >= 0.0

    after = list(controller.parameters())
    assert any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))


def test_controller_trainer_train_step_requires_student_latent():
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    trainer = flux2_ttr_controller.ControllerTrainer(controller, lpips_weight=0.0)
    teacher = torch.zeros(1, 4, 2, 2)

    def _bad_forward(mask: torch.Tensor, logits: torch.Tensor):
        del mask, logits
        return {"actual_full_attn_ratio": 0.5}

    with pytest.raises(ValueError):
        trainer.train_step(
            sigma=0.3,
            cfg_scale=2.0,
            width=32,
            height=32,
            teacher_latent=teacher,
            student_forward_fn=_bad_forward,
        )
