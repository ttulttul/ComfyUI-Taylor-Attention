import logging
import math
import os
import sys
import types
from collections import deque
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import flux2_comet_logging
import flux2_ttr
import flux2_ttr_controller


def _baseline_flat(q, k, v, mask=None):
    return flux2_ttr._flatten_heads(flux2_ttr._softmax_attention(q.float(), k.float(), v.float(), mask=mask).to(dtype=v.dtype))


def _optimizer_steps(optimizer: torch.optim.Optimizer) -> list[float]:
    steps: list[float] = []
    for state in optimizer.state.values():
        step = state.get("step")
        if torch.is_tensor(step):
            steps.append(float(step.item()))
            continue
        if step is not None:
            steps.append(float(step))
    steps.sort()
    return steps


def _has_inference_params(module: torch.nn.Module) -> bool:
    return any(bool(p.is_inference()) for p in module.parameters())


class _DummyBlock:
    def __init__(self, num_heads: int, hidden_size: int):
        self.num_heads = num_heads
        self.hidden_size = hidden_size


class _DummyDiffusionModel:
    def __init__(self):
        self.single_blocks = [_DummyBlock(num_heads=2, hidden_size=8), _DummyBlock(num_heads=4, hidden_size=16)]


class _DummyBaseModel:
    def __init__(self):
        self.diffusion_model = _DummyDiffusionModel()


class _DummyPatcher:
    def __init__(self):
        self.model = _DummyBaseModel()


def test_validate_feature_dim():
    assert flux2_ttr.validate_feature_dim(256) == 256
    assert flux2_ttr.validate_feature_dim(512) == 512
    with pytest.raises(ValueError):
        flux2_ttr.validate_feature_dim(64)
    with pytest.raises(ValueError):
        flux2_ttr.validate_feature_dim(384)


def test_approximate_attention_models_dir_resolves_from_custom_nodes_anchor(monkeypatch, tmp_path):
    monkeypatch.setattr(flux2_ttr, "folder_paths", None)
    anchor = tmp_path / "ComfyUI" / "custom_nodes" / "ComfyUI-Approximate-Attention" / "flux2_ttr.py"
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text("# anchor\n", encoding="utf-8")

    out = Path(flux2_ttr.approximate_attention_models_dir(anchor_file=str(anchor), ensure_exists=True))
    expected = tmp_path / "ComfyUI" / "models" / "approximate_attention"
    assert out == expected
    assert expected.is_dir()


def test_approximate_attention_models_dir_falls_back_to_anchor_parent(monkeypatch, tmp_path):
    monkeypatch.setattr(flux2_ttr, "folder_paths", None)
    anchor = tmp_path / "plugin_repo" / "flux2_ttr.py"
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text("# anchor\n", encoding="utf-8")

    out = Path(flux2_ttr.approximate_attention_models_dir(anchor_file=str(anchor), ensure_exists=True))
    expected = tmp_path / "plugin_repo" / "models" / "approximate_attention"
    assert out == expected
    assert expected.is_dir()


def test_resolve_default_checkpoint_path_uses_models_approximate_attention(monkeypatch, tmp_path):
    monkeypatch.setattr(flux2_ttr, "folder_paths", None)
    anchor = tmp_path / "ComfyUI" / "custom_nodes" / "ComfyUI-Approximate-Attention" / "flux2_ttr.py"
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text("# anchor\n", encoding="utf-8")

    out = Path(
        flux2_ttr.resolve_default_checkpoint_path(
            "",
            default_filename="flux2_ttr.pt",
            anchor_file=str(anchor),
            ensure_dir=True,
        )
    )
    expected = tmp_path / "ComfyUI" / "models" / "approximate_attention" / "flux2_ttr.pt"
    assert out == expected
    assert expected.parent.is_dir()


def test_resolve_default_checkpoint_path_preserves_explicit_path(tmp_path):
    explicit = str(tmp_path / "manual" / "checkpoint.pt")
    out = flux2_ttr.resolve_default_checkpoint_path(
        explicit,
        default_filename="flux2_ttr.pt",
        anchor_file=__file__,
        ensure_dir=True,
    )
    assert out == explicit


def test_load_checkpoint_feature_dim_reads_saved_value(tmp_path):
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=512,
        learning_rate=1e-3,
        training=False,
        steps=0,
    )
    ckpt = tmp_path / "flux2_ttr_feature_dim_512.pt"
    runtime.save_checkpoint(str(ckpt))
    assert flux2_ttr.load_checkpoint_feature_dim(str(ckpt)) == 512


def test_load_checkpoint_feature_dim_rejects_unsupported_format(tmp_path):
    ckpt = tmp_path / "flux2_ttr_bad_format.pt"
    torch.save({"format": "bad_format", "feature_dim": 512}, ckpt)
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        flux2_ttr.load_checkpoint_feature_dim(str(ckpt))


def test_load_checkpoint_feature_dim_rejects_invalid_feature_dim(tmp_path):
    ckpt = tmp_path / "flux2_ttr_bad_feature_dim.pt"
    torch.save({"format": "flux2_ttr_v2", "feature_dim": 384}, ckpt)
    with pytest.raises(ValueError, match="invalid checkpoint feature_dim"):
        flux2_ttr.load_checkpoint_feature_dim(str(ckpt))


def test_runtime_checkpoint_cache_key_includes_file_identity_and_flags(tmp_path):
    flux2_ttr.clear_runtime_checkpoint_cache()
    ckpt = tmp_path / "flux2_ttr_runtime_cache.pt"
    ckpt.write_bytes(b"checkpoint")
    stat_result = ckpt.stat()
    path_with_dot = str(ckpt.parent / "." / ckpt.name)

    key = flux2_ttr.runtime_checkpoint_cache_key(
        path_with_dot,
        feature_dim=512,
        training=True,
    )

    assert key == (
        os.path.realpath(path_with_dot),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_size),
        512,
        True,
    )
    flux2_ttr.clear_runtime_checkpoint_cache()


def test_runtime_checkpoint_cache_round_trip_is_partitioned_by_training_flag(tmp_path):
    flux2_ttr.clear_runtime_checkpoint_cache()
    ckpt = tmp_path / "flux2_ttr_runtime_cache_partition.pt"
    ckpt.write_bytes(b"checkpoint-v1")

    runtime_infer = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime_train = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=0)

    flux2_ttr.cache_runtime_for_checkpoint(
        str(ckpt),
        feature_dim=256,
        training=False,
        runtime=runtime_infer,
    )
    assert (
        flux2_ttr.get_cached_runtime_for_checkpoint(
            str(ckpt),
            feature_dim=256,
            training=False,
        )
        is runtime_infer
    )
    assert flux2_ttr.get_cached_runtime_for_checkpoint(str(ckpt), feature_dim=256, training=True) is None

    flux2_ttr.cache_runtime_for_checkpoint(
        str(ckpt),
        feature_dim=256,
        training=True,
        runtime=runtime_train,
    )
    assert (
        flux2_ttr.get_cached_runtime_for_checkpoint(
            str(ckpt),
            feature_dim=256,
            training=True,
        )
        is runtime_train
    )
    assert (
        flux2_ttr.get_cached_runtime_for_checkpoint(
            str(ckpt),
            feature_dim=256,
            training=False,
        )
        is runtime_infer
    )


def test_runtime_checkpoint_cache_invalidates_after_checkpoint_metadata_change(tmp_path):
    flux2_ttr.clear_runtime_checkpoint_cache()
    ckpt = tmp_path / "flux2_ttr_runtime_cache_mutation.pt"
    ckpt.write_bytes(b"checkpoint-v1")

    runtime_v1 = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    flux2_ttr.cache_runtime_for_checkpoint(
        str(ckpt),
        feature_dim=256,
        training=False,
        runtime=runtime_v1,
    )
    assert (
        flux2_ttr.get_cached_runtime_for_checkpoint(
            str(ckpt),
            feature_dim=256,
            training=False,
        )
        is runtime_v1
    )

    ckpt.write_bytes(b"checkpoint-v2-with-new-size")
    assert flux2_ttr.get_cached_runtime_for_checkpoint(str(ckpt), feature_dim=256, training=False) is None

    runtime_v2 = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    flux2_ttr.cache_runtime_for_checkpoint(
        str(ckpt),
        feature_dim=256,
        training=False,
        runtime=runtime_v2,
    )
    assert (
        flux2_ttr.get_cached_runtime_for_checkpoint(
            str(ckpt),
            feature_dim=256,
            training=False,
        )
        is runtime_v2
    )
    assert len(flux2_ttr._RUNTIME_CHECKPOINT_CACHE) == 1
    flux2_ttr.clear_runtime_checkpoint_cache()


def test_infer_flux_single_layer_specs():
    specs = flux2_ttr.infer_flux_single_layer_specs(_DummyPatcher())
    assert len(specs) == 2
    assert specs[0].layer_key == "single:0"
    assert specs[0].head_dim == 4
    assert specs[1].layer_key == "single:1"
    assert specs[1].head_dim == 4


def test_kernel_regressor_attention_rectangular_and_masked():
    torch.manual_seed(0)
    layer = flux2_ttr.KernelRegressorAttention(head_dim=8, feature_dim=256)

    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 3, 9, 8)
    v = torch.randn(2, 3, 9, 8)
    key_mask = torch.ones(2, 9, dtype=torch.bool)
    key_mask[:, -2:] = False

    out = layer(q, k, v, key_mask=key_mask, q_chunk=2, k_chunk=4)
    assert out.shape == (2, 3, 5, 8)
    assert torch.isfinite(out).all()
    assert math.isfinite(layer.last_den_min)


def test_kernel_regressor_attention_phi_network_is_three_layer_mlp():
    shared = flux2_ttr.KernelRegressorAttention(head_dim=8, feature_dim=256, split_qk=False)
    split = flux2_ttr.KernelRegressorAttention(head_dim=8, feature_dim=256, split_qk=True)

    linear_q_shared = [m for m in shared.phi_net_q if isinstance(m, torch.nn.Linear)]
    linear_k_shared = [m for m in shared.phi_net_k if isinstance(m, torch.nn.Linear)]
    linear_q_split = [m for m in split.phi_net_q if isinstance(m, torch.nn.Linear)]
    linear_k_split = [m for m in split.phi_net_k if isinstance(m, torch.nn.Linear)]
    hidden_expected = max(8, 2 * 256)

    assert len(linear_q_shared) == 3
    assert len(linear_k_shared) == 3
    assert len(linear_q_split) == 3
    assert len(linear_k_split) == 3
    assert linear_q_shared[0].in_features == 8
    assert linear_q_shared[0].out_features == hidden_expected
    assert linear_q_shared[1].in_features == hidden_expected
    assert linear_q_shared[1].out_features == hidden_expected
    assert linear_q_shared[2].in_features == hidden_expected
    assert linear_q_shared[2].out_features == 256
    assert linear_q_split[0].out_features == hidden_expected
    assert linear_q_split[1].in_features == hidden_expected
    assert linear_q_split[1].out_features == hidden_expected
    assert linear_q_split[2].in_features == hidden_expected
    assert linear_q_split[2].out_features == 256
    assert shared.phi_net_q is shared.phi_net_k
    assert split.phi_net_q is not split.phi_net_k


def test_flux2_hkr_layer_shape_and_landmarks():
    torch.manual_seed(0)
    layer = flux2_ttr.Flux2HKRAttnLayer(
        head_dim=8,
        feature_dim=256,
        landmark_fraction=0.5,
        landmark_min=1,
        landmark_max=6,
        text_tokens_guess=3,
    )

    q = torch.randn(1, 2, 7, 8)
    k = torch.randn(1, 2, 12, 8)
    v = torch.randn(1, 2, 12, 8)
    key_mask = torch.ones(1, 12, dtype=torch.bool)

    out = layer(q, k, v, key_mask=key_mask)
    assert out.shape == (1, 2, 7, 8)
    assert torch.isfinite(out).all()
    assert layer.last_landmark_count == 7


def test_flux2_hkr_select_landmarks_keeps_all_conditioning_tokens():
    layer = flux2_ttr.Flux2HKRAttnLayer(
        head_dim=8,
        feature_dim=256,
        landmark_fraction=0.5,
        landmark_min=1,
        landmark_max=6,
        text_tokens_guess=3,
    )
    idx = layer._select_landmarks(
        num_keys=12,
        device=torch.device("cpu"),
        key_mask=None,
        text_token_count=8,
        conditioning_token_count=None,
    )
    assert idx.numel() == 10
    assert torch.equal(idx[:8], torch.arange(8, dtype=torch.long))


def test_flux2_hkr_conditioning_token_count_overrides_text_token_count():
    torch.manual_seed(0)
    layer = flux2_ttr.Flux2HKRAttnLayer(
        head_dim=8,
        feature_dim=256,
        landmark_fraction=0.5,
        landmark_min=1,
        landmark_max=6,
        text_tokens_guess=3,
    )

    q = torch.randn(1, 2, 7, 8)
    k = torch.randn(1, 2, 12, 8)
    v = torch.randn(1, 2, 12, 8)
    out = layer(q, k, v, text_token_count=2, conditioning_token_count=6)

    assert out.shape == (1, 2, 7, 8)
    assert torch.isfinite(out).all()
    assert layer.last_landmark_count == 9


def test_flux2_hkr_effective_landmark_count_scales_with_image_tokens():
    layer = flux2_ttr.Flux2HKRAttnLayer(
        head_dim=8,
        feature_dim=256,
        landmark_fraction=0.08,
        landmark_min=64,
        landmark_max=512,
    )
    assert layer._effective_landmark_count(0) == 64
    assert layer._effective_landmark_count(1024) == 82
    assert layer._effective_landmark_count(2304) == 184
    assert layer._effective_landmark_count(4096) == 328
    assert layer._effective_landmark_count(10000) == 512


def test_flux2_hkr_effective_landmark_count_unlimited_when_max_is_zero():
    layer = flux2_ttr.Flux2HKRAttnLayer(
        head_dim=8,
        feature_dim=256,
        landmark_fraction=0.08,
        landmark_min=64,
        landmark_max=0,
    )
    assert layer.landmark_max == 0
    assert layer._effective_landmark_count(0) == 64
    assert layer._effective_landmark_count(10000) == 800


def test_flux2_hkr_sigma_cfg_conditioning_identity_by_default():
    torch.manual_seed(0)
    layer = flux2_ttr.Flux2HKRAttnLayer(head_dim=8, feature_dim=256)
    q = torch.randn(1, 2, 7, 8)
    k = torch.randn(1, 2, 12, 8)
    v = torch.randn(1, 2, 12, 8)

    out_base = layer(q, k, v)
    out_cond = layer(q, k, v, sigma=0.7, cfg_scale=3.0)
    assert torch.allclose(out_base, out_cond, atol=1e-6, rtol=1e-6)


def test_flux2_hkr_alpha_init_uses_logit_space_with_adaptive_enabled():
    layer = flux2_ttr.Flux2HKRAttnLayer(head_dim=8, feature_dim=256, alpha_init=0.1)
    assert layer.adaptive_alpha is True
    assert float(torch.sigmoid(layer.alpha).item()) == pytest.approx(0.1, rel=1e-4, abs=1e-4)
    assert float(layer.alpha_max) == pytest.approx(0.2)
    assert float(layer.gamma) == pytest.approx(2.0)


def test_flux2_hkr_adaptive_alpha_gating_modulates_blend_per_token(monkeypatch):
    layer = flux2_ttr.Flux2HKRAttnLayer(head_dim=2, feature_dim=256, alpha_init=0.1)
    q = torch.zeros(1, 1, 2, 2)
    k = torch.zeros(1, 1, 2, 2)
    v = torch.zeros(1, 1, 2, 2)

    out_kernel = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)
    out_land = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0]]]], dtype=torch.float32)

    monkeypatch.setattr(layer.kernel, "forward", lambda *args, **kwargs: out_kernel.clone())
    monkeypatch.setattr(layer, "_landmark_attention", lambda *args, **kwargs: out_land.clone())

    out = layer(q, k, v)

    base_logit = layer.alpha.detach()
    diff = out_kernel.float() - out_land.float()
    disagreement = 1.0 - F.cosine_similarity(out_kernel.float(), out_land.float(), dim=-1)
    d_mean = disagreement.mean(dim=-1, keepdim=True).clamp(min=1e-6)
    d_norm = (disagreement / d_mean).clamp(max=3.0) / 3.0
    alpha_per_token = layer.alpha_max * torch.sigmoid(base_logit + layer.gamma * (d_norm - 0.5))
    expected = out_kernel + alpha_per_token.unsqueeze(-1) * (out_land - out_kernel)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)
    assert float(alpha_per_token.max().item()) <= float(layer.alpha_max) + 1e-6
    assert float(alpha_per_token.min().item()) > 0.0
    assert float(alpha_per_token[0, 0, 1].item()) > float(alpha_per_token[0, 0, 0].item())


def test_flux2_hkr_non_adaptive_alpha_uses_scalar_blend(monkeypatch):
    layer = flux2_ttr.Flux2HKRAttnLayer(head_dim=2, feature_dim=256, alpha_init=0.1)
    layer.adaptive_alpha = False
    q = torch.zeros(1, 1, 2, 2)
    k = torch.zeros(1, 1, 2, 2)
    v = torch.zeros(1, 1, 2, 2)

    out_kernel = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32)
    out_land = torch.tensor([[[[1.0, 0.0], [2.0, 0.0]]]], dtype=torch.float32)

    monkeypatch.setattr(layer.kernel, "forward", lambda *args, **kwargs: out_kernel.clone())
    monkeypatch.setattr(layer, "_landmark_attention", lambda *args, **kwargs: out_land.clone())

    out = layer(q, k, v)

    alpha = layer.alpha_max * torch.sigmoid(layer.alpha.detach())
    expected = out_kernel + alpha.view(1, 1, 1, 1) * (out_land - out_kernel)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_runtime_extracts_sigma_and_cfg_scale():
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=1)
    opts = {"sigmas": torch.tensor([0.35]), "flux2_ttr": {"cfg_scale": 6.5}}
    assert runtime._extract_sigma(opts) == pytest.approx(0.35)
    assert runtime._extract_cfg_scale(opts) == pytest.approx(6.5)


def test_runtime_infers_conditioning_token_count():
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    assert runtime._infer_conditioning_token_count({"conditioning_token_count": 12}, 20) == 12
    assert runtime._infer_conditioning_token_count({"cond_token_count": 9}, 20) == 9
    assert runtime._infer_conditioning_token_count({"prefix_token_count": 11}, 20) == 11
    assert runtime._infer_conditioning_token_count({"flux2_ttr": {"conditioning_token_count": 15}}, 20) == 15
    assert runtime._infer_conditioning_token_count({"text_token_count": 7}, 20) is None


def test_runtime_training_uses_query_subsampling_only():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        training_preview_ttr=False,
        training_query_token_cap=4,
        replay_buffer_size=16,
        train_steps_per_call=1,
    )
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])

    q = torch.randn(1, 2, 10, 4)
    k = torch.randn(1, 2, 10, 4)
    v = torch.randn(1, 2, 10, 4)
    opts = {"block_type": "single", "block_index": 0}

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg, mask=mask)

    baseline = fallback(q, k, v, None)
    out = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)

    assert torch.allclose(out, baseline)
    assert runtime.training_updates_done == 1
    assert runtime.steps_remaining == 0

    buf = runtime.replay_buffers["single:0"]
    assert len(buf) >= 1
    sample = buf[-1]
    assert sample.q_sub.shape[2] == 4
    assert sample.k_full.shape[2] == 10
    assert sample.v_full.shape[2] == 10
    assert sample.k_full.device.type == "cpu"
    assert sample.k_full.dtype == torch.float16


def test_runtime_replay_stores_sigma_and_cfg():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        training_preview_ttr=False,
    )
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])

    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)
    opts = {
        "block_type": "single",
        "block_index": 0,
        "sigmas": torch.tensor([0.42]),
        "conditioning_token_count": 5,
        "flux2_ttr": {"cfg_scale": 4.0},
    }

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg, mask=mask)

    runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)
    sample = runtime.replay_buffers["single:0"][-1]
    assert sample.sigma == pytest.approx(0.42)
    assert sample.cfg_scale == pytest.approx(4.0)
    assert sample.conditioning_token_count == 5


def test_train_from_replay_periodic_flushes_without_sigma_boundary():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=2,
        train_steps_per_call=1,
    )
    layer_key = "single:0"
    runtime.replay_buffers[layer_key] = deque(
        [
            flux2_ttr.ReplaySample(
                q_sub=torch.randn(1, 2, 4, 4),
                k_full=torch.randn(1, 2, 4, 4),
                v_full=torch.randn(1, 2, 4, 4),
                teacher_sub=torch.randn(1, 2, 4, 4),
                key_mask=torch.ones(1, 4, dtype=torch.bool),
                text_token_count=4,
                conditioning_token_count=4,
                sigma=0.5,
                cfg_scale=1.0,
            )
        ]
    )
    runtime._run_ema_accum_loss["single:99"] = [0.2] * 19

    runtime._train_from_replay(layer_key=layer_key, head_dim=4, device=torch.device("cpu"))

    assert runtime.steps_remaining == 1
    assert runtime.training_enabled is True
    assert runtime._run_ema_accum_loss == {}
    assert "single:99" in runtime.layer_ema_loss
    assert layer_key in runtime.layer_ema_loss


def test_train_from_replay_loss_combines_huber_and_cosine(monkeypatch):
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        train_steps_per_call=1,
    )
    layer_key = "single:0"
    runtime.replay_buffers[layer_key] = deque(
        [
            flux2_ttr.ReplaySample(
                q_sub=torch.randn(1, 2, 4, 4),
                k_full=torch.randn(1, 2, 4, 4),
                v_full=torch.randn(1, 2, 4, 4),
                teacher_sub=torch.randn(1, 2, 4, 4),
                key_mask=torch.ones(1, 4, dtype=torch.bool),
                text_token_count=4,
                conditioning_token_count=4,
                sigma=0.5,
                cfg_scale=1.0,
            )
        ]
    )
    layer = runtime._ensure_layer(layer_key=layer_key, head_dim=4, device=torch.device("cpu"))
    layer.adaptive_alpha = False

    seen_dims = []
    seen_eps = []

    def fake_smooth_l1_loss(student, teacher, beta):
        del teacher, beta
        # Keep gradient connectivity while returning a controlled scalar.
        return student.sum() * 0 + 0.25

    def fake_cosine_similarity(student, teacher, dim=1, eps=1e-8):
        del teacher
        seen_dims.append(int(dim))
        seen_eps.append(float(eps))
        out_shape = list(student.shape)
        out_shape.pop(dim)
        return student.new_full(out_shape, 0.8)

    monkeypatch.setattr(flux2_ttr.F, "smooth_l1_loss", fake_smooth_l1_loss)
    monkeypatch.setattr(flux2_ttr.F, "cosine_similarity", fake_cosine_similarity)
    monkeypatch.setattr(runtime, "_compute_distill_metrics", lambda **kwargs: {"loss": float(kwargs["loss_value"])})

    runtime._train_from_replay(layer_key=layer_key, head_dim=4, device=torch.device("cpu"))

    assert seen_dims == [-1]
    assert seen_eps == [1e-8]
    log_var_h_init = 0.0
    log_var_c_init = -1.0
    expected = (
        0.25 / (2.0 * math.exp(log_var_h_init))
        + log_var_h_init / 2.0
        + 0.2 / (2.0 * math.exp(log_var_c_init))
        + log_var_c_init / 2.0
    )
    assert runtime.last_loss == pytest.approx(expected)
    assert runtime.layer_last_loss[layer_key] == pytest.approx(expected)
    assert float(runtime.log_var_huber.item()) != pytest.approx(0.0)
    assert float(runtime.log_var_cosine.item()) != pytest.approx(0.0)


def test_compute_distill_metrics_uses_per_token_cosine_similarity():
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=1)
    layer_key = "single:0"
    teacher = torch.tensor([[[[10.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)
    student = torch.tensor([[[[10.0, 0.0], [-1.0, 0.0]]]], dtype=torch.float32)

    metrics = runtime._compute_distill_metrics(
        student=student,
        teacher=teacher,
        loss_value=0.5,
        layer_key=layer_key,
    )

    # Per-token cosine should average (1 + -1) / 2 = 0; flattening would be ~0.98.
    assert metrics["cosine_similarity"] == pytest.approx(0.0)


def test_loss_log_var_initialization_biases_cosine_weight_higher():
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=1)
    assert float(runtime.log_var_huber.item()) == pytest.approx(0.0)
    assert float(runtime.log_var_cosine.item()) == pytest.approx(-1.0)


def test_loss_weight_optimizer_recovers_from_inference_tensors():
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
    )

    with torch.inference_mode():
        runtime.log_var_huber = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        runtime.log_var_cosine = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    opt = runtime._ensure_loss_weight_optimizer(torch.device("cpu"))
    assert not runtime.log_var_huber.is_inference()
    assert not runtime.log_var_cosine.is_inference()

    opt.zero_grad(set_to_none=True)
    loss = runtime.log_var_huber + runtime.log_var_cosine
    loss.backward()
    opt.step()


def test_ensure_layer_constructs_trainable_layer_inside_inference_mode():
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
    )
    layer_key = "single:0"
    with torch.inference_mode():
        layer = runtime._ensure_layer(layer_key=layer_key, head_dim=4, device=torch.device("cpu"))

    assert _has_inference_params(layer) is False


def test_train_from_replay_rebuilds_stale_inference_layer():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        train_steps_per_call=1,
    )
    layer_key = "single:0"

    with torch.inference_mode():
        stale_layer = flux2_ttr.Flux2HKRAttnLayer(head_dim=4, feature_dim=256).to(device="cpu", dtype=torch.float32)
    assert _has_inference_params(stale_layer) is True

    runtime.layers[layer_key] = stale_layer
    runtime.optimizers[layer_key] = runtime._ensure_optimizer(stale_layer)
    runtime.replay_buffers[layer_key] = deque(
        [
            flux2_ttr.ReplaySample(
                q_sub=torch.randn(1, 2, 4, 4),
                k_full=torch.randn(1, 2, 4, 4),
                v_full=torch.randn(1, 2, 4, 4),
                teacher_sub=torch.randn(1, 2, 4, 4),
                key_mask=torch.ones(1, 4, dtype=torch.bool),
                text_token_count=4,
                conditioning_token_count=4,
                sigma=0.5,
                cfg_scale=1.0,
            )
        ]
    )

    runtime._train_from_replay(layer_key=layer_key, head_dim=4, device=torch.device("cpu"))

    assert runtime.training_updates_done == 1
    assert _has_inference_params(runtime.layers[layer_key]) is False


def test_ensure_layer_rebuild_logs_are_debug_only(caplog):
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
    )
    layer_key = "single:0"

    with torch.inference_mode():
        stale_layer = flux2_ttr.Flux2HKRAttnLayer(head_dim=4, feature_dim=256).to(device="cpu", dtype=torch.float32)
    runtime.layers[layer_key] = stale_layer
    runtime.optimizers[layer_key] = runtime._ensure_optimizer(stale_layer)

    caplog.set_level(logging.INFO, logger="flux2_ttr")
    runtime._ensure_layer(layer_key=layer_key, head_dim=4, device=torch.device("cpu"))

    assert "Flux2TTR: detected inference tensors in layer" not in caplog.text
    assert "Flux2TTR: created HKR layer" not in caplog.text


def test_runtime_training_preview_uses_student_when_layer_ready():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        training_preview_ttr=True,
        readiness_min_updates=1,
        readiness_threshold=10.0,
    )
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])

    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)
    opts = {"block_type": "single", "block_index": 0}

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg, mask=mask)

    baseline = fallback(q, k, v, None)
    out = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)

    assert runtime.layer_ready["single:0"] is True
    assert out.shape == baseline.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, baseline)


def test_runtime_inference_propagates_conditioning_token_count_to_layer():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=False,
        steps=0,
        readiness_min_updates=0,
        readiness_threshold=1.0,
        landmark_fraction=0.5,
        landmark_min=1,
        landmark_max=6,
    )
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])
    runtime.layer_update_count["single:0"] = 1
    runtime.layer_ema_loss["single:0"] = 0.0
    runtime.layer_ema_cosine_dist["single:0"] = 0.0

    q = torch.randn(1, 2, 7, 4)
    k = torch.randn(1, 2, 12, 4)
    v = torch.randn(1, 2, 12, 4)
    opts = {
        "block_type": "single",
        "block_index": 0,
        "text_token_count": 2,
        "conditioning_token_count": 6,
    }

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg, mask=mask)

    out = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)
    assert out.shape == (1, 7, 8)
    assert torch.isfinite(out).all()
    assert runtime.layers["single:0"].last_landmark_count == 9


def test_runtime_training_uses_randomized_swap_set_per_step(monkeypatch):
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=4,
        training_preview_ttr=False,
        min_swap_layers=1,
        max_swap_layers=1,
    )
    runtime.register_layer_specs(
        [
            flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4),
            flux2_ttr.FluxLayerSpec(layer_key="single:1", num_heads=2, head_dim=4),
            flux2_ttr.FluxLayerSpec(layer_key="single:2", num_heads=2, head_dim=4),
        ]
    )

    choices = iter([["single:1"], ["single:0"]])
    monkeypatch.setattr(flux2_ttr.random, "randint", lambda lo, hi: 1)
    monkeypatch.setattr(flux2_ttr.random, "sample", lambda pop, k: list(next(choices)))

    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg, mask=mask)

    opts_step0_l0 = {"block_type": "single", "block_index": 0, "step": 0}
    opts_step0_l1 = {"block_type": "single", "block_index": 1, "step": 0}
    opts_step1_l0 = {"block_type": "single", "block_index": 0, "step": 1}

    runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts_step0_l0, fallback_attention=fallback)
    assert runtime.training_updates_done == 0

    runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts_step0_l1, fallback_attention=fallback)
    assert runtime.training_updates_done == 1

    runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts_step1_l0, fallback_attention=fallback)
    assert runtime.training_updates_done == 2


def test_runtime_inference_falls_back_when_not_ready():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])

    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)
    opts = {"block_type": "single", "block_index": 0}

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg, mask=mask)

    baseline = fallback(q, k, v, None)
    out = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)
    assert torch.allclose(out, baseline)


def test_runtime_inference_controller_mask_routes_teacher_vs_student(monkeypatch):
    class _Controller(torch.nn.Module):
        def forward(self, sigma, cfg_scale, width, height):
            del sigma, cfg_scale, width, height
            return torch.tensor([10.0, -10.0], dtype=torch.float32)

    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime.register_layer_specs(
        [
            flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4),
            flux2_ttr.FluxLayerSpec(layer_key="single:1", num_heads=2, head_dim=4),
        ]
    )
    runtime.layer_update_count["single:0"] = 99
    runtime.layer_ema_loss["single:0"] = 0.0
    runtime.layer_ema_cosine_dist["single:0"] = 0.0
    runtime.layer_ready["single:0"] = True
    runtime.layer_update_count["single:1"] = 99
    runtime.layer_ema_loss["single:1"] = 0.0
    runtime.layer_ema_cosine_dist["single:1"] = 0.0
    runtime.layer_ready["single:1"] = True

    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg, mask=mask)

    def fake_student(**kwargs):
        q_eff = kwargs["q_eff"]
        v_arg = kwargs["v"]
        return torch.full((q_eff.shape[0], q_eff.shape[2], q_eff.shape[1] * q_eff.shape[3]), 42.0, dtype=v_arg.dtype)

    monkeypatch.setattr(runtime, "_student_from_runtime", fake_student)
    controller = _Controller()

    opts0 = {"block_type": "single", "block_index": 0, "flux2_ttr": {"controller": controller}}
    opts1 = {"block_type": "single", "block_index": 1, "flux2_ttr": {"controller": controller}}

    teacher = fallback(q, k, v, None)
    out0 = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts0, fallback_attention=fallback)
    out1 = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts1, fallback_attention=fallback)
    assert torch.allclose(out0, teacher)
    assert torch.all(out1 == 42.0)


def test_runtime_inference_controller_logs_routing_once_per_step(monkeypatch, caplog):
    class _Controller(torch.nn.Module):
        def forward(self, sigma, cfg_scale, width, height):
            del cfg_scale, width, height
            s = float(sigma)
            if s >= 0.7:
                # student on single:1,single:2
                return torch.tensor([2.0, -2.0, -2.0], dtype=torch.float32)
            # student on single:0,single:2
            return torch.tensor([-2.0, 2.0, -2.0], dtype=torch.float32)

    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime.register_layer_specs(
        [
            flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4),
            flux2_ttr.FluxLayerSpec(layer_key="single:1", num_heads=2, head_dim=4),
            flux2_ttr.FluxLayerSpec(layer_key="single:2", num_heads=2, head_dim=4),
        ]
    )
    for key in ("single:0", "single:1", "single:2"):
        runtime.layer_update_count[key] = 99
        runtime.layer_ema_loss[key] = 0.0
        runtime.layer_ema_cosine_dist[key] = 0.0
        runtime.layer_ready[key] = True

    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg, mask=mask)

    def fake_student(**kwargs):
        q_eff = kwargs["q_eff"]
        v_arg = kwargs["v"]
        return torch.full((q_eff.shape[0], q_eff.shape[2], q_eff.shape[1] * q_eff.shape[3]), 19.0, dtype=v_arg.dtype)

    monkeypatch.setattr(runtime, "_student_from_runtime", fake_student)
    controller = _Controller()

    caplog.set_level(logging.INFO, logger="flux2_ttr")

    opts_step0_l0 = {
        "block_type": "single",
        "block_index": 0,
        "step": 0,
        "sigmas": torch.tensor([0.9]),
        "flux2_ttr": {"controller": controller, "controller_threshold": 0.5},
    }
    opts_step0_l1 = {
        "block_type": "single",
        "block_index": 1,
        "step": 0,
        "sigmas": torch.tensor([0.9]),
        "flux2_ttr": {"controller": controller, "controller_threshold": 0.5},
    }
    opts_step1_l2 = {
        "block_type": "single",
        "block_index": 2,
        "step": 1,
        "sigmas": torch.tensor([0.5]),
        "flux2_ttr": {"controller": controller, "controller_threshold": 0.5},
    }

    runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts_step0_l0, fallback_attention=fallback)
    runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts_step0_l1, fallback_attention=fallback)
    runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts_step1_l2, fallback_attention=fallback)

    routing_logs = [rec.getMessage() for rec in caplog.records if "Flux2TTR controller step routing:" in rec.getMessage()]
    assert len(routing_logs) == 2
    assert "step_id=('step', 0)" in routing_logs[0]
    assert "extracted_sigma=0.9" in routing_logs[0]
    assert "controller_policy=threshold" in routing_logs[0]
    assert "controller_threshold=0.5" in routing_logs[0]
    assert "[single:1,single:2]" in routing_logs[0]
    assert "step_id=('step', 1)" in routing_logs[1]
    assert "extracted_sigma=0.5" in routing_logs[1]
    assert "[single:0,single:2]" in routing_logs[1]


def test_runtime_inference_controller_mask_override_routes_without_controller_call(monkeypatch):
    class _Controller(torch.nn.Module):
        def forward(self, sigma, cfg_scale, width, height):
            del sigma, cfg_scale, width, height
            raise AssertionError("controller should not be called when override mask is provided")

    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime.register_layer_specs(
        [
            flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4),
            flux2_ttr.FluxLayerSpec(layer_key="single:1", num_heads=2, head_dim=4),
        ]
    )
    runtime.layer_update_count["single:0"] = 99
    runtime.layer_ema_loss["single:0"] = 0.0
    runtime.layer_ema_cosine_dist["single:0"] = 0.0
    runtime.layer_ready["single:0"] = True
    runtime.layer_update_count["single:1"] = 99
    runtime.layer_ema_loss["single:1"] = 0.0
    runtime.layer_ema_cosine_dist["single:1"] = 0.0
    runtime.layer_ready["single:1"] = True

    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg, mask=mask)

    def fake_student(**kwargs):
        q_eff = kwargs["q_eff"]
        v_arg = kwargs["v"]
        return torch.full((q_eff.shape[0], q_eff.shape[2], q_eff.shape[1] * q_eff.shape[3]), 11.0, dtype=v_arg.dtype)

    monkeypatch.setattr(runtime, "_student_from_runtime", fake_student)
    teacher = fallback(q, k, v, None)
    controller = _Controller()
    mask_override = torch.tensor([1.0, 0.0], dtype=torch.float32)

    opts0 = {
        "block_type": "single",
        "block_index": 0,
        "flux2_ttr": {"controller": controller, "controller_mask_override": mask_override},
    }
    opts1 = {
        "block_type": "single",
        "block_index": 1,
        "flux2_ttr": {"controller": controller, "controller_mask_override": mask_override},
    }

    out0 = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts0, fallback_attention=fallback)
    out1 = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts1, fallback_attention=fallback)
    assert torch.allclose(out0, teacher)
    assert torch.all(out1 == 11.0)


def test_runtime_inference_controller_threshold_is_configurable(monkeypatch):
    class _Controller(torch.nn.Module):
        def forward(self, sigma, cfg_scale, width, height):
            del sigma, cfg_scale, width, height
            # sigmoid(logit(0.6)) -> 0.6 mask probability
            return torch.tensor([torch.logit(torch.tensor(0.6)).item()], dtype=torch.float32)

    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])
    runtime.layer_update_count["single:0"] = 99
    runtime.layer_ema_loss["single:0"] = 0.0
    runtime.layer_ema_cosine_dist["single:0"] = 0.0
    runtime.layer_ready["single:0"] = True

    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg, mask=mask)

    def fake_student(**kwargs):
        q_eff = kwargs["q_eff"]
        v_arg = kwargs["v"]
        return torch.full((q_eff.shape[0], q_eff.shape[2], q_eff.shape[1] * q_eff.shape[3]), 24.0, dtype=v_arg.dtype)

    monkeypatch.setattr(runtime, "_student_from_runtime", fake_student)
    controller = _Controller()
    teacher = fallback(q, k, v, None)

    opts_threshold_high = {
        "block_type": "single",
        "block_index": 0,
        "flux2_ttr": {"controller": controller, "controller_threshold": 0.7},
    }
    opts_threshold_low = {
        "block_type": "single",
        "block_index": 0,
        "flux2_ttr": {"controller": controller, "controller_threshold": 0.5},
    }

    out_student = runtime.run_attention(
        q, k, v, pe=None, mask=None, transformer_options=opts_threshold_high, fallback_attention=fallback
    )
    out_teacher = runtime.run_attention(
        q, k, v, pe=None, mask=None, transformer_options=opts_threshold_low, fallback_attention=fallback
    )
    assert torch.all(out_student == 24.0)
    assert torch.allclose(out_teacher, teacher)


def test_runtime_inference_controller_stochastic_policy_samples_once_per_step(monkeypatch, caplog):
    class _Controller(torch.nn.Module):
        def forward(self, sigma, cfg_scale, width, height):
            del sigma, cfg_scale, width, height
            # Base probability is 0.5 for both layers; stochastic policy drives per-step variation.
            return torch.zeros(2, dtype=torch.float32)

    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime.register_layer_specs(
        [
            flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4),
            flux2_ttr.FluxLayerSpec(layer_key="single:1", num_heads=2, head_dim=4),
        ]
    )
    runtime.layer_update_count["single:0"] = 99
    runtime.layer_ema_loss["single:0"] = 0.0
    runtime.layer_ema_cosine_dist["single:0"] = 0.0
    runtime.layer_ready["single:0"] = True
    runtime.layer_update_count["single:1"] = 99
    runtime.layer_ema_loss["single:1"] = 0.0
    runtime.layer_ema_cosine_dist["single:1"] = 0.0
    runtime.layer_ready["single:1"] = True

    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg, mask=mask)

    def fake_student(**kwargs):
        q_eff = kwargs["q_eff"]
        v_arg = kwargs["v"]
        return torch.full((q_eff.shape[0], q_eff.shape[2], q_eff.shape[1] * q_eff.shape[3]), 31.0, dtype=v_arg.dtype)

    monkeypatch.setattr(runtime, "_student_from_runtime", fake_student)
    controller = _Controller()

    draws = iter(
        (
            torch.tensor([0.2, 0.8], dtype=torch.float32),  # step 0 -> full [1,0], student [0,1]
            torch.tensor([0.8, 0.2], dtype=torch.float32),  # step 1 -> full [0,1], student [1,0]
        )
    )
    draw_calls = {"count": 0}

    def fake_rand_like(ref, *args, **kwargs):
        del args, kwargs
        draw_calls["count"] += 1
        return next(draws).to(device=ref.device, dtype=ref.dtype)

    monkeypatch.setattr(torch, "rand_like", fake_rand_like)
    caplog.set_level(logging.INFO, logger="flux2_ttr")

    opts_step0_l0 = {
        "block_type": "single",
        "block_index": 0,
        "step": 0,
        "sigmas": torch.tensor([0.9]),
        "flux2_ttr": {
            "controller": controller,
            "controller_policy": "stochastic",
            "controller_threshold": 0.5,
            "controller_temperature": 1.0,
        },
    }
    opts_step0_l1 = {
        "block_type": "single",
        "block_index": 1,
        "step": 0,
        "sigmas": torch.tensor([0.9]),
        "flux2_ttr": {
            "controller": controller,
            "controller_policy": "stochastic",
            "controller_threshold": 0.5,
            "controller_temperature": 1.0,
        },
    }
    opts_step1_l0 = {
        "block_type": "single",
        "block_index": 0,
        "step": 1,
        "sigmas": torch.tensor([0.5]),
        "flux2_ttr": {
            "controller": controller,
            "controller_policy": "stochastic",
            "controller_threshold": 0.5,
            "controller_temperature": 1.0,
        },
    }

    teacher = fallback(q, k, v, None)
    out0 = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts_step0_l0, fallback_attention=fallback)
    out1 = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts_step0_l1, fallback_attention=fallback)
    out2 = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts_step1_l0, fallback_attention=fallback)

    assert torch.allclose(out0, teacher)
    assert torch.all(out1 == 31.0)
    assert torch.all(out2 == 31.0)
    assert int(draw_calls["count"]) == 2

    routing_logs = [rec.getMessage() for rec in caplog.records if "Flux2TTR controller step routing:" in rec.getMessage()]
    assert len(routing_logs) == 2
    assert "controller_policy=stochastic" in routing_logs[0]
    assert "decision_threshold=0.5" in routing_logs[0]
    assert "[single:1]" in routing_logs[0]
    assert "[single:0]" in routing_logs[1]


def test_runtime_unsupported_mask_falls_back_to_teacher():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])
    runtime.layer_update_count["single:0"] = 999
    runtime.layer_ema_loss["single:0"] = 0.0
    runtime.layer_ema_cosine_dist["single:0"] = 0.0
    runtime.layer_ready["single:0"] = True

    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)
    # Full per-query mask is unsupported and should force fallback.
    mask = torch.ones(1, 1, 6, 6, dtype=torch.bool)
    opts = {"block_type": "single", "block_index": 0}

    calls = {"count": 0}

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, transformer_options
        calls["count"] += 1
        return _baseline_flat(q_arg, k_arg, v_arg, mask=mask)

    baseline = fallback(q, k, v, None, mask=mask)
    calls["count"] = 0
    out = runtime.run_attention(q, k, v, pe=None, mask=mask, transformer_options=opts, fallback_attention=fallback)
    assert calls["count"] == 1
    assert torch.allclose(out, baseline)


def test_high_loss_inference_falls_back_to_teacher():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime.layer_update_count["single:0"] = 999
    runtime.layer_ema_loss["single:0"] = 0.0
    runtime.layer_ema_cosine_dist["single:0"] = 0.0
    runtime.layer_ready["single:0"] = True
    runtime.last_loss = 10.0
    runtime.max_safe_inference_loss = 0.5

    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)
    opts = {"block_type": "single", "block_index": 0}

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, mask, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg)

    baseline = fallback(q, k, v, None)
    out = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)
    assert torch.allclose(out, baseline)


def test_checkpoint_round_trip_preserves_state(tmp_path):
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=2,
        training_preview_ttr=False,
        landmark_fraction=0.12,
        landmark_min=48,
        landmark_max=640,
        alpha_max=0.33,
        gamma=3.25,
        readiness_min_updates=1,
        readiness_threshold=10.0,
        cfg_scale=3.5,
        min_swap_layers=2,
        max_swap_layers=5,
        comet_experiment="exp_persist_123",
        comet_persist_experiment=True,
    )
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])

    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 2, 8, 4)
    v = torch.randn(1, 2, 8, 4)
    opts = {"block_type": "single", "block_index": 0}

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg, mask=mask)

    runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)
    runtime.log_var_huber.data.fill_(0.23)
    runtime.log_var_cosine.data.fill_(-0.17)

    ckpt = tmp_path / "flux2_ttr_v2.pt"
    runtime.save_checkpoint(str(ckpt))
    assert ckpt.exists()

    runtime_loaded = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime_loaded.load_checkpoint(str(ckpt))

    assert runtime_loaded.pending_state
    assert "single:0" in runtime_loaded.pending_state
    assert "single:0" in runtime_loaded.layer_update_count
    assert runtime_loaded.layer_update_count["single:0"] == runtime.layer_update_count["single:0"]
    assert "single:0" in runtime_loaded.layer_ema_loss
    assert "single:0" in runtime_loaded.layer_ema_cosine_dist
    assert runtime_loaded.landmark_fraction == pytest.approx(0.12)
    assert runtime_loaded.landmark_min == 48
    assert runtime_loaded.landmark_max == 640
    assert runtime_loaded.alpha_max == pytest.approx(0.33)
    assert runtime_loaded.gamma == pytest.approx(3.25)
    assert runtime_loaded.cfg_scale == pytest.approx(3.5)
    assert runtime_loaded.min_swap_layers == 2
    assert runtime_loaded.max_swap_layers == 5
    assert runtime_loaded.comet_experiment == "exp_persist_123"
    assert runtime_loaded.comet_display_name == runtime.comet_display_name
    assert runtime_loaded.comet_persist_experiment is True
    assert float(runtime_loaded.log_var_huber.item()) == pytest.approx(0.23)
    assert float(runtime_loaded.log_var_cosine.item()) == pytest.approx(-0.17)


def test_checkpoint_round_trip_restores_optimizer_states(tmp_path):
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        train_steps_per_call=1,
    )
    layer_key = "single:0"
    runtime.replay_buffers[layer_key] = deque(
        [
            flux2_ttr.ReplaySample(
                q_sub=torch.randn(1, 2, 4, 4),
                k_full=torch.randn(1, 2, 4, 4),
                v_full=torch.randn(1, 2, 4, 4),
                teacher_sub=torch.randn(1, 2, 4, 4),
                key_mask=torch.ones(1, 4, dtype=torch.bool),
                text_token_count=4,
                conditioning_token_count=4,
                sigma=0.5,
                cfg_scale=1.0,
            )
        ]
    )

    runtime._train_from_replay(layer_key=layer_key, head_dim=4, device=torch.device("cpu"))
    assert runtime.loss_weight_optimizer is not None

    layer_steps_before = _optimizer_steps(runtime.optimizers[layer_key])
    loss_steps_before = _optimizer_steps(runtime.loss_weight_optimizer)
    assert layer_steps_before
    assert loss_steps_before

    ckpt = tmp_path / "flux2_ttr_optimizer_state.pt"
    runtime.save_checkpoint(str(ckpt))

    payload = torch.load(ckpt, map_location="cpu")
    assert layer_key in payload["optimizer_states"]
    assert isinstance(payload["loss_weight_optimizer_state"], dict)

    loaded = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    loaded.load_checkpoint(str(ckpt))
    assert layer_key in loaded.pending_optimizer_states
    assert isinstance(loaded.pending_loss_weight_optimizer_state, dict)

    loaded._ensure_layer(layer_key=layer_key, head_dim=4, device=torch.device("cpu"))
    loss_weight_opt = loaded._ensure_loss_weight_optimizer(torch.device("cpu"))

    layer_steps_after = _optimizer_steps(loaded.optimizers[layer_key])
    loss_steps_after = _optimizer_steps(loss_weight_opt)
    assert layer_steps_after == pytest.approx(layer_steps_before)
    assert loss_steps_after == pytest.approx(loss_steps_before)
    assert layer_key not in loaded.pending_optimizer_states
    assert loaded.pending_loss_weight_optimizer_state is None


def test_checkpoint_state_includes_per_layer_readiness_criteria():
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        readiness_threshold=0.12,
        readiness_min_updates=3,
    )
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])
    runtime.layer_update_count["single:0"] = 5
    runtime.layer_ema_loss["single:0"] = 0.1
    runtime.layer_ema_cosine_dist["single:0"] = 0.1
    runtime.layer_readiness_threshold["single:0"] = 0.2
    runtime.layer_readiness_min_updates["single:0"] = 4
    runtime._refresh_layer_ready("single:0")

    payload = runtime.checkpoint_state()
    assert float(payload["alpha_max"]) == pytest.approx(runtime.alpha_max)
    assert float(payload["gamma"]) == pytest.approx(runtime.gamma)
    assert "layer_readiness" in payload
    assert "single:0" in payload["layer_readiness"]
    meta = payload["layer_readiness"]["single:0"]
    assert float(meta["readiness_threshold"]) == pytest.approx(0.2)
    assert int(meta["readiness_min_updates"]) == 4
    assert bool(meta["ready"]) is True


def test_load_checkpoint_restores_per_layer_readiness_thresholds(tmp_path):
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        readiness_threshold=1.0,
        readiness_min_updates=0,
    )
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])
    runtime.layer_update_count["single:0"] = 5
    runtime.layer_ema_loss["single:0"] = 0.1
    runtime.layer_ema_cosine_dist["single:0"] = 0.1
    runtime.layer_readiness_threshold["single:0"] = 0.05
    runtime.layer_readiness_min_updates["single:0"] = 3
    runtime._refresh_layer_ready("single:0")
    assert runtime.layer_ready["single:0"] is False

    ckpt = tmp_path / "flux2_ttr_layer_readiness.pt"
    runtime.save_checkpoint(str(ckpt))

    loaded = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=False,
        steps=0,
        readiness_threshold=10.0,
        readiness_min_updates=0,
    )
    loaded.load_checkpoint(str(ckpt))
    assert loaded.layer_readiness_threshold["single:0"] == pytest.approx(0.05)
    assert loaded.layer_readiness_min_updates["single:0"] == 3
    assert loaded.layer_ready["single:0"] is False


def test_layer_readiness_hysteresis_prevents_boundary_oscillation():
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        readiness_threshold=0.1,
        readiness_min_updates=0,
    )
    layer_key = "single:0"
    runtime.layer_update_count[layer_key] = 10
    runtime.layer_readiness_threshold[layer_key] = 0.1

    runtime.layer_ema_cosine_dist[layer_key] = 0.095
    assert runtime._refresh_layer_ready(layer_key) is True

    # Above entry threshold but below hysteresis-adjusted exit threshold: stays ready.
    runtime.layer_ema_cosine_dist[layer_key] = 0.11
    assert runtime._refresh_layer_ready(layer_key) is True

    # Above exit threshold (0.1 * 1.2): loses readiness.
    runtime.layer_ema_cosine_dist[layer_key] = 0.121
    assert runtime._refresh_layer_ready(layer_key) is False

    # Re-entry still requires crossing the original readiness threshold.
    runtime.layer_ema_cosine_dist[layer_key] = 0.11
    assert runtime._refresh_layer_ready(layer_key) is False
    runtime.layer_ema_cosine_dist[layer_key] = 0.099
    assert runtime._refresh_layer_ready(layer_key) is True


def test_flush_run_emas_updates_from_run_means_and_refreshes_readiness(caplog):
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        readiness_threshold=0.5,
        readiness_min_updates=0,
    )
    layer_key = "single:0"
    runtime.layer_update_count[layer_key] = 2
    runtime._run_ema_accum_loss[layer_key] = [0.2, 0.4]
    runtime._run_ema_accum_cosine_dist[layer_key] = [0.1, 0.3]
    caplog.set_level(logging.INFO, logger="flux2_ttr")

    runtime.flush_run_emas()

    assert runtime.layer_ema_loss[layer_key] == pytest.approx(0.3)
    assert runtime.layer_ema_cosine_dist[layer_key] == pytest.approx(0.2)
    assert runtime.layer_ready[layer_key] is True
    assert runtime._run_ema_accum_loss == {}
    assert runtime._run_ema_accum_cosine_dist == {}
    assert "Flux2TTR: flushed run EMAs layer=single:0" in caplog.text


def test_detect_run_boundary_flushes_only_on_large_sigma_jump():
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        readiness_min_updates=0,
        readiness_threshold=1.0,
    )
    layer_key = "single:0"
    runtime.layer_update_count[layer_key] = 1
    runtime._run_ema_accum_loss[layer_key] = [0.2, 0.4]

    runtime._detect_run_boundary(1.0)
    assert runtime._run_ema_accum_loss[layer_key] == [0.2, 0.4]

    runtime._detect_run_boundary(0.7)
    assert runtime._run_ema_accum_loss[layer_key] == [0.2, 0.4]

    runtime._detect_run_boundary(1.2)
    assert runtime._run_ema_accum_loss == {}
    assert runtime.layer_ema_loss[layer_key] == pytest.approx(0.3)


def test_checkpoint_state_flushes_pending_run_ema_accumulators():
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        readiness_min_updates=0,
        readiness_threshold=1.0,
    )
    layer_key = "single:0"
    runtime.layer_update_count[layer_key] = 1
    runtime._run_ema_accum_loss[layer_key] = [0.2, 0.4]
    runtime._run_ema_accum_cosine_dist[layer_key] = [0.1, 0.3]

    payload = runtime.checkpoint_state()

    assert runtime._run_ema_accum_loss == {}
    assert runtime._run_ema_accum_cosine_dist == {}
    assert runtime.layer_ema_loss[layer_key] == pytest.approx(0.3)
    assert runtime.layer_ema_cosine_dist[layer_key] == pytest.approx(0.2)
    assert payload["layer_ema_loss"][layer_key] == pytest.approx(0.3)
    assert payload["layer_ema_cosine_dist"][layer_key] == pytest.approx(0.2)


def test_load_checkpoint_old_landmark_count_falls_back_to_dynamic_defaults(tmp_path):
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        landmark_fraction=0.2,
        landmark_min=8,
        landmark_max=32,
    )
    payload = runtime.checkpoint_state()
    payload.pop("landmark_fraction", None)
    payload.pop("landmark_min", None)
    payload.pop("landmark_max", None)
    payload["landmark_count"] = 96

    ckpt = tmp_path / "flux2_ttr_old_landmark_count.pt"
    torch.save(payload, ckpt)

    loaded = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=False,
        steps=0,
        landmark_fraction=0.2,
        landmark_min=8,
        landmark_max=32,
    )
    loaded.load_checkpoint(str(ckpt))
    assert loaded.landmark_fraction == pytest.approx(0.08)
    assert loaded.landmark_min == 64
    assert loaded.landmark_max == 0


def test_load_checkpoint_missing_loss_log_vars_uses_default_bias(tmp_path):
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
    )
    payload = runtime.checkpoint_state()
    payload.pop("loss_log_var_huber", None)
    payload.pop("loss_log_var_cosine", None)

    ckpt = tmp_path / "flux2_ttr_missing_loss_log_vars.pt"
    torch.save(payload, ckpt)

    loaded = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=False,
        steps=0,
    )
    loaded.load_checkpoint(str(ckpt))
    assert float(loaded.log_var_huber.item()) == pytest.approx(0.0)
    assert float(loaded.log_var_cosine.item()) == pytest.approx(-1.0)


def test_load_checkpoint_migrates_raw_alpha_to_logit_space(tmp_path):
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
    )
    payload = runtime.checkpoint_state()
    payload.pop("alpha_format", None)
    payload["layers"] = {
        "single:0": {"alpha": torch.tensor(0.1, dtype=torch.float32)},
        "single:1": {"alpha": torch.tensor(-2.1972246, dtype=torch.float32)},
    }

    ckpt = tmp_path / "flux2_ttr_alpha_migration.pt"
    torch.save(payload, ckpt)

    loaded = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=False,
        steps=0,
    )
    loaded.load_checkpoint(str(ckpt))

    expected_logit = math.log(0.1 / (1.0 - 0.1))
    assert float(loaded.pending_state["single:0"]["alpha"].item()) == pytest.approx(expected_logit)
    assert float(loaded.pending_state["single:1"]["alpha"].item()) == pytest.approx(-2.1972246)


def test_load_checkpoint_skips_alpha_migration_for_logit_format(tmp_path):
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
    )
    payload = runtime.checkpoint_state()
    payload["layers"] = {
        "single:0": {"alpha": torch.tensor(0.3, dtype=torch.float32)},
    }

    ckpt = tmp_path / "flux2_ttr_alpha_no_migration.pt"
    torch.save(payload, ckpt)

    loaded = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=False,
        steps=0,
    )
    loaded.load_checkpoint(str(ckpt))
    assert float(loaded.pending_state["single:0"]["alpha"].item()) == pytest.approx(0.3)


def test_recover_runtime_from_config_inference_requires_checkpoint():
    cfg = {
        "training_mode": False,
        "feature_dim": 256,
        "query_chunk_size": 256,
        "key_chunk_size": 1024,
        "checkpoint_path": "",
    }
    runtime = flux2_ttr._recover_runtime_from_config(cfg)
    assert runtime is None


def test_recover_runtime_from_config_training_without_checkpoint():
    cfg = {
        "training_mode": True,
        "training": True,
        "training_preview_ttr": False,
        "training_steps_total": 64,
        "training_steps_remaining": 32,
        "learning_rate": 1e-4,
        "feature_dim": 256,
        "query_chunk_size": 256,
        "key_chunk_size": 1024,
        "landmark_fraction": 0.2,
        "landmark_min": 32,
        "landmark_max": 320,
        "alpha_max": 0.27,
        "gamma": 3.7,
        "cfg_scale": 2.25,
        "min_swap_layers": 3,
        "max_swap_layers": 7,
        "checkpoint_path": "",
    }
    runtime = flux2_ttr._recover_runtime_from_config(cfg)
    assert runtime is not None
    assert runtime.training_mode is True
    assert runtime.training_enabled is True
    assert runtime.training_preview_ttr is False
    assert runtime.steps_remaining == 32
    assert runtime.landmark_fraction == pytest.approx(0.2)
    assert runtime.landmark_min == 32
    assert runtime.landmark_max == 320
    assert runtime.alpha_max == pytest.approx(0.27)
    assert runtime.gamma == pytest.approx(3.7)
    assert runtime.cfg_scale == pytest.approx(2.25)
    assert runtime.min_swap_layers == 3
    assert runtime.max_swap_layers == 7


def test_runtime_ensure_layer_propagates_alpha_max_and_gamma():
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        alpha_max=0.41,
        gamma=4.5,
    )
    layer = runtime._ensure_layer(layer_key="single:0", head_dim=4, device=torch.device("cpu"))
    assert layer.alpha_max == pytest.approx(0.41)
    assert layer.gamma == pytest.approx(4.5)


def test_recover_runtime_from_config_inference_overrides_checkpoint_mode(tmp_path):
    runtime_train = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-4,
        training=True,
        steps=8,
        training_preview_ttr=True,
    )
    runtime_train.layer_update_count["single:0"] = 48
    runtime_train.layer_ema_loss["single:0"] = 0.03
    runtime_train.layer_ema_cosine_dist["single:0"] = 0.03
    runtime_train.layer_ready["single:0"] = True

    ckpt = tmp_path / "flux2_ttr_recover_mode.pt"
    runtime_train.save_checkpoint(str(ckpt))

    cfg = {
        "training_mode": False,
        "training": False,
        "training_preview_ttr": False,
        "training_steps_total": 64,
        "training_steps_remaining": 64,
        "learning_rate": 1e-4,
        "feature_dim": 256,
        "query_chunk_size": 256,
        "key_chunk_size": 1024,
        "checkpoint_path": str(ckpt),
    }
    runtime = flux2_ttr._recover_runtime_from_config(cfg)
    assert runtime is not None
    assert runtime.training_mode is False
    assert runtime.training_enabled is False
    assert runtime.training_preview_ttr is False


def test_recover_runtime_from_config_comet_settings_override_checkpoint(tmp_path):
    runtime_train = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-4,
        training=True,
        steps=4,
        comet_enabled=True,
        comet_project_name="ckpt_proj",
        comet_workspace="ckpt_ws",
        comet_experiment="ckpt_exp",
        comet_persist_experiment=True,
        comet_log_every=77,
    )
    ckpt = tmp_path / "flux2_ttr_recover_comet.pt"
    runtime_train.save_checkpoint(str(ckpt))

    cfg = {
        "training_mode": True,
        "training": True,
        "training_steps_total": 8,
        "training_steps_remaining": 8,
        "learning_rate": 1e-4,
        "feature_dim": 256,
        "query_chunk_size": 256,
        "key_chunk_size": 1024,
        "checkpoint_path": str(ckpt),
        "comet_enabled": False,
        "comet_project_name": "ui_proj",
        "comet_workspace": "ui_ws",
        "comet_experiment": "ui_exp",
        "comet_persist_experiment": True,
        "comet_log_every": 9,
    }

    runtime = flux2_ttr._recover_runtime_from_config(cfg)
    assert runtime is not None
    assert runtime.comet_enabled is False
    assert runtime.comet_project_name == "ui_proj"
    assert runtime.comet_workspace == "ui_ws"
    assert runtime.comet_experiment == "ui_exp"
    assert runtime.comet_persist_experiment is True
    assert runtime.comet_log_every == 9


def test_recover_runtime_from_config_loads_controller_checkpoint(tmp_path):
    controller = flux2_ttr_controller.TTRController(num_layers=2, embed_dim=8, hidden_dim=16)
    controller_ckpt = tmp_path / "controller.pt"
    flux2_ttr_controller.save_controller_checkpoint(controller, str(controller_ckpt))

    cfg = {
        "training_mode": True,
        "training": True,
        "training_steps_total": 2,
        "training_steps_remaining": 2,
        "learning_rate": 1e-4,
        "feature_dim": 256,
        "query_chunk_size": 256,
        "key_chunk_size": 1024,
        "checkpoint_path": "",
        "controller_checkpoint_path": str(controller_ckpt),
    }
    runtime = flux2_ttr._recover_runtime_from_config(cfg)
    assert runtime is not None
    assert isinstance(cfg.get("controller"), flux2_ttr_controller.TTRController)


def test_calibration_switches_to_real_sample_capture_mode():
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=2)
    patcher = _DummyPatcher()

    loss = runtime.calibrate_from_inputs(
        model=patcher,
        latents={"samples": torch.randn(1, 4, 8, 8)},
        conditioning=[[torch.randn(1, 8, 32), {"pooled_output": torch.randn(1, 16)}]],
        steps=7,
        max_tokens=16,
    )

    assert isinstance(loss, float)
    assert runtime.training_mode is True
    assert runtime.training_enabled is True
    assert runtime.training_preview_ttr is False
    assert runtime.steps_remaining == 7


def test_training_progress_logs_every_10_updates(caplog):
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=10,
        training_preview_ttr=False,
        train_steps_per_call=1,
    )
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=1, head_dim=4)])

    q = torch.randn(1, 1, 8, 4)
    k = torch.randn(1, 1, 8, 4)
    v = torch.randn(1, 1, 8, 4)
    opts = {"block_type": "single", "block_index": 0}

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg, mask=mask)

    caplog.set_level(logging.INFO, logger="flux2_ttr")
    for _ in range(10):
        runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)

    assert runtime.training_updates_done == 10
    assert "Flux2TTR distill snapshot: updates=10/10" in caplog.text
    assert "q25-q75 loss=" in caplog.text
    assert "ready_layers=" in caplog.text
    assert "log_vars=(h=" in caplog.text


def test_record_training_metrics_logs_to_comet(monkeypatch):
    start_calls = []
    metric_calls = []
    params_calls = []

    class _FakeExperiment:
        def log_parameters(self, params):
            params_calls.append(dict(params))

        def log_metrics(self, metrics, step=None):
            metric_calls.append((dict(metrics), int(step) if step is not None else None))

        def end(self):
            return None

    def _fake_start(api_key=None, project_name=None, workspace=None, experiment_key=None):
        start_calls.append((api_key, project_name, workspace, experiment_key))
        return _FakeExperiment()

    fake_comet = types.ModuleType("comet_ml")
    fake_comet.start = _fake_start
    monkeypatch.setitem(sys.modules, "comet_ml", fake_comet)
    flux2_ttr._TTR_COMET_EXPERIMENTS.clear()
    flux2_ttr._TTR_COMET_LOGGED_PARAM_KEYS.clear()

    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=10,
        comet_enabled=True,
        comet_api_key="test-key",
        comet_project_name="proj",
        comet_workspace="ws",
        comet_experiment="metrics_test_run",
        comet_persist_experiment=False,
        comet_log_every=1,
    )
    runtime.training_updates_done = 3
    runtime.steps_remaining = 7
    runtime.log_var_huber.data.fill_(0.23)
    runtime.log_var_cosine.data.fill_(-0.17)
    runtime.layers["single:11"] = flux2_ttr.Flux2HKRAttnLayer(head_dim=4, feature_dim=runtime.feature_dim)
    runtime.layers["single:11"].alpha.data.fill_(0.0)
    runtime.layer_ready["single:11"] = True
    runtime._record_training_metrics("single:11", {"loss": 0.5, "mse": 1.0, "nmse": 0.9, "cosine_similarity": 0.8, "ema_loss": 0.7})
    runtime._record_training_metrics("single:10", {"loss": 1.0, "mse": 2.0})

    expected_key = flux2_comet_logging.normalize_experiment_key("metrics_test_run")
    assert start_calls == [("test-key", "proj", "ws", expected_key)]
    assert len(params_calls) == 1
    assert metric_calls
    payload, step = metric_calls[-1]
    assert step == 3
    assert payload["flux2ttr/single:11/loss"] == 0.5
    assert payload["flux2ttr/single:11/ema_cosine_dist"] == pytest.approx(0.2)
    assert payload["flux2ttr/single:11/alpha_sigmoid"] == pytest.approx(0.5)
    assert payload["flux2ttr/single:10/loss"] == 1.0
    assert payload["flux2ttr/single:10/mse"] == 2.0
    assert payload["flux2ttr/single:10/avg_loss"] == 1.0
    assert payload["flux2ttr/single:10/avg_mse"] == 2.0
    assert payload["flux2ttr/global/steps_remaining"] == 7.0
    assert payload["flux2ttr/global/updates_done"] == 3.0
    assert payload["flux2ttr/global/log_var_huber"] == pytest.approx(0.23)
    assert payload["flux2ttr/global/log_var_cosine"] == pytest.approx(-0.17)
    assert payload["flux2ttr/global/layers_tracked"] == 2.0
    assert payload["flux2ttr/global/layers_ready"] == 1.0
    assert payload["flux2ttr/global/layers_ready_ratio"] == 0.5
    assert payload["flux2ttr/global/loss_min"] == 0.5
    assert payload["flux2ttr/global/loss_max"] == 1.0
    assert payload["flux2ttr/global/loss_p50"] == 0.75
    assert payload["flux2ttr/global/mse_min"] == 1.0
    assert payload["flux2ttr/global/mse_max"] == 2.0
    assert payload["flux2ttr/global/pareto_frontier"] == pytest.approx(0.8)


def test_record_training_metrics_accumulates_cosine_distance_per_run():
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=1)
    runtime.layer_ema_cosine_dist["single:0"] = 0.4

    runtime._record_training_metrics("single:0", {"loss": 0.5, "cosine_similarity": 0.8})

    assert runtime.layer_ema_cosine_dist["single:0"] == pytest.approx(0.4)
    assert runtime._run_ema_accum_cosine_dist["single:0"][0] == pytest.approx(0.2)
    assert runtime._layer_metric_latest["single:0"]["ema_cosine_dist"] == pytest.approx(0.4)


def test_record_training_metrics_logs_pareto_frontier_edge_cases(monkeypatch):
    metric_calls = []

    class _FakeExperiment:
        def log_parameters(self, params):
            return None

        def log_metrics(self, metrics, step=None):
            metric_calls.append((dict(metrics), int(step) if step is not None else None))

        def end(self):
            return None

    def _fake_start(api_key=None, project_name=None, workspace=None, experiment_key=None):
        del api_key, project_name, workspace, experiment_key
        return _FakeExperiment()

    fake_comet = types.ModuleType("comet_ml")
    fake_comet.start = _fake_start
    monkeypatch.setitem(sys.modules, "comet_ml", fake_comet)
    flux2_ttr._TTR_COMET_EXPERIMENTS.clear()
    flux2_ttr._TTR_COMET_LOGGED_PARAM_KEYS.clear()

    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=10,
        comet_enabled=True,
        comet_api_key="test-key",
        comet_project_name="proj",
        comet_workspace="ws",
        comet_experiment="pareto_test_run",
        comet_persist_experiment=False,
        comet_log_every=1,
    )
    runtime.training_updates_done = 1
    runtime.steps_remaining = 9
    runtime._record_training_metrics("single:0", {"loss": 1.0, "cosine_similarity": 0.9})
    payload, _ = metric_calls[-1]
    assert payload["flux2ttr/global/pareto_frontier"] == pytest.approx(0.0)

    runtime.layer_ready["single:0"] = True
    runtime.training_updates_done = 2
    runtime.steps_remaining = 8
    runtime._record_training_metrics("single:0", {"loss": 0.9, "cosine_similarity": 0.9})
    payload, _ = metric_calls[-1]
    assert payload["flux2ttr/global/pareto_frontier"] == pytest.approx(0.9)

    runtime.layer_ready["single:1"] = True
    runtime.training_updates_done = 3
    runtime.steps_remaining = 7
    runtime._record_training_metrics("single:1", {"loss": 0.8, "cosine_similarity": 0.7})
    payload, _ = metric_calls[-1]
    assert payload["flux2ttr/global/pareto_frontier"] == pytest.approx(1.4)


def test_record_training_metrics_throttles_comet_logging(monkeypatch):
    metric_calls = []

    class _FakeExperiment:
        def log_parameters(self, params):
            return None

        def log_metrics(self, metrics, step=None):
            metric_calls.append((dict(metrics), int(step) if step is not None else None))

        def end(self):
            return None

    def _fake_start(api_key=None, project_name=None, workspace=None, experiment_key=None):
        del api_key, project_name, workspace, experiment_key
        return _FakeExperiment()

    fake_comet = types.ModuleType("comet_ml")
    fake_comet.start = _fake_start
    monkeypatch.setitem(sys.modules, "comet_ml", fake_comet)
    flux2_ttr._TTR_COMET_EXPERIMENTS.clear()
    flux2_ttr._TTR_COMET_LOGGED_PARAM_KEYS.clear()

    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=100,
        comet_enabled=True,
        comet_api_key="test-key",
        comet_project_name="proj",
        comet_workspace="ws",
        comet_experiment="throttle_test_run",
        comet_persist_experiment=False,
        comet_log_every=50,
    )

    runtime.training_updates_done = 49
    runtime.steps_remaining = 51
    runtime._record_training_metrics("single:0", {"loss": 0.5})
    assert metric_calls == []

    runtime.training_updates_done = 50
    runtime.steps_remaining = 50
    runtime._record_training_metrics("single:0", {"loss": 0.4})
    assert len(metric_calls) == 1
    _, step = metric_calls[-1]
    assert step == 50

    runtime.training_updates_done = 51
    runtime.steps_remaining = 0
    runtime._record_training_metrics("single:0", {"loss": 0.3})
    assert len(metric_calls) == 2
    _, step = metric_calls[-1]
    assert step == 51


def test_comet_experiment_persists_across_runtime_release(monkeypatch):
    start_calls = []
    end_calls = []

    class _FakeExperiment:
        def log_parameters(self, params):
            return None

        def log_metrics(self, metrics, step=None):
            del metrics, step
            return None

        def end(self):
            end_calls.append(True)

    def _fake_start(api_key=None, project_name=None, workspace=None, experiment_key=None):
        start_calls.append((api_key, project_name, workspace, experiment_key))
        return _FakeExperiment()

    fake_comet = types.ModuleType("comet_ml")
    fake_comet.start = _fake_start
    monkeypatch.setitem(sys.modules, "comet_ml", fake_comet)
    flux2_ttr._TTR_COMET_EXPERIMENTS.clear()
    flux2_ttr._TTR_COMET_LOGGED_PARAM_KEYS.clear()

    runtime_a = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        comet_enabled=True,
        comet_api_key="test-key",
        comet_project_name="proj",
        comet_workspace="ws",
        comet_experiment="exp_keep_open",
        comet_persist_experiment=True,
    )
    exp_a = runtime_a._ensure_comet_experiment()
    assert exp_a is not None
    runtime_a.release_resources()
    assert end_calls == []

    runtime_b = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        comet_enabled=True,
        comet_api_key="test-key",
        comet_project_name="proj",
        comet_workspace="ws",
        comet_experiment="exp_keep_open",
        comet_persist_experiment=True,
    )
    exp_b = runtime_b._ensure_comet_experiment()
    assert exp_b is exp_a
    assert len(start_calls) == 1
    runtime_b.comet_persist_experiment = False
    runtime_b.release_resources()
    assert end_calls == [True]


def test_comet_logs_parameters_before_starting_experiment(monkeypatch, caplog):
    class _FakeExperiment:
        def log_parameters(self, params):
            del params
            return None

        def log_metrics(self, metrics, step=None):
            del metrics, step
            return None

        def end(self):
            return None

    def _fake_start(api_key=None, project_name=None, workspace=None, experiment_key=None):
        del api_key, project_name, workspace, experiment_key
        return _FakeExperiment()

    fake_comet = types.ModuleType("comet_ml")
    fake_comet.start = _fake_start
    monkeypatch.setitem(sys.modules, "comet_ml", fake_comet)
    flux2_ttr._TTR_COMET_EXPERIMENTS.clear()
    flux2_ttr._TTR_COMET_LOGGED_PARAM_KEYS.clear()

    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        comet_enabled=True,
        comet_api_key="test-key",
        comet_project_name="proj",
        comet_workspace="ws",
        comet_experiment="preflight_log_test",
        comet_persist_experiment=False,
    )

    caplog.set_level(logging.INFO, logger="flux2_ttr")
    runtime._ensure_comet_experiment()

    messages = [record.getMessage() for record in caplog.records]
    preparing_idx = next(i for i, msg in enumerate(messages) if "preparing Comet logging" in msg)
    enabled_idx = next(i for i, msg in enumerate(messages) if "Comet logging enabled" in msg)
    assert preparing_idx < enabled_idx


def test_comet_sets_display_name_from_runtime(monkeypatch):
    set_name_calls = []

    class _FakeExperiment:
        def set_name(self, name):
            set_name_calls.append(str(name))

        def log_parameters(self, params):
            del params
            return None

        def log_metrics(self, metrics, step=None):
            del metrics, step
            return None

        def end(self):
            return None

    def _fake_start(api_key=None, project_name=None, workspace=None, experiment_key=None):
        del api_key, project_name, workspace, experiment_key
        return _FakeExperiment()

    fake_comet = types.ModuleType("comet_ml")
    fake_comet.start = _fake_start
    monkeypatch.setitem(sys.modules, "comet_ml", fake_comet)
    monkeypatch.setattr(flux2_ttr, "_generate_experiment_display_name", lambda: "2026-02-11-073025-b3ed63")
    flux2_ttr._TTR_COMET_EXPERIMENTS.clear()
    flux2_ttr._TTR_COMET_LOGGED_PARAM_KEYS.clear()

    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        comet_enabled=True,
        comet_api_key="test-key",
        comet_project_name="proj",
        comet_workspace="ws",
        comet_experiment="display_name_test",
        comet_persist_experiment=False,
    )
    runtime._ensure_comet_experiment()

    assert runtime.comet_display_name == "2026-02-11-073025-b3ed63"
    assert set_name_calls == ["2026-02-11-073025-b3ed63"]


def test_runtime_autogenerates_comet_experiment_key_when_missing(monkeypatch):
    monkeypatch.setattr(flux2_ttr, "_generate_experiment_key", lambda: "abc1234-20260211-143022")
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        comet_experiment="",
        comet_persist_experiment=False,
    )
    assert runtime.comet_experiment == "abc1234-20260211-143022"
    assert runtime.comet_persist_experiment is True


def test_runtime_honors_explicit_comet_experiment_key():
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=1,
        comet_experiment="manual_exp_key",
        comet_persist_experiment=False,
    )
    assert runtime.comet_experiment == "manual_exp_key"
    assert runtime.comet_persist_experiment is False


def test_git_short_hash_uses_flux2_ttr_module_directory(monkeypatch):
    call_args = {}

    def _fake_check_output(cmd, stderr=None, text=None, cwd=None):
        call_args["cmd"] = cmd
        call_args["stderr"] = stderr
        call_args["text"] = text
        call_args["cwd"] = cwd
        return "b3ed63\n"

    monkeypatch.setattr(flux2_ttr, "__file__", "/tmp/custom_nodes/ComfyUI-Taylor-Attention/flux2_ttr.py")
    monkeypatch.setattr("subprocess.check_output", _fake_check_output)

    out = flux2_ttr._git_short_hash(6)
    assert out == "b3ed63"
    assert call_args["cmd"] == ["git", "rev-parse", "--short=6", "HEAD"]
    assert call_args["cwd"] == "/tmp/custom_nodes/ComfyUI-Taylor-Attention"
    assert call_args["text"] is True


def test_release_resources_clears_run_ema_accumulator_state():
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=1)
    runtime._run_ema_accum_loss["single:0"] = [0.1]
    runtime._run_ema_accum_cosine_dist["single:0"] = [0.2]
    runtime._run_last_sigma = 0.4

    runtime.release_resources()

    assert runtime._run_ema_accum_loss == {}
    assert runtime._run_ema_accum_cosine_dist == {}
    assert runtime._run_last_sigma is None


def test_memory_reserve_estimate_scales_with_training():
    infer_bytes = flux2_ttr._estimate_flux2_ttr_memory_bytes(
        batch=1,
        heads=24,
        n_query=256,
        n_key=256,
        head_dim=128,
        feature_dim=256,
        q_chunk_size=256,
        k_chunk_size=1024,
        dtype_size=4,
        training=False,
    )
    train_bytes = flux2_ttr._estimate_flux2_ttr_memory_bytes(
        batch=1,
        heads=24,
        n_query=256,
        n_key=256,
        head_dim=128,
        feature_dim=256,
        q_chunk_size=256,
        k_chunk_size=1024,
        dtype_size=4,
        training=True,
    )
    assert infer_bytes > 0
    assert train_bytes > infer_bytes


def test_memory_reserve_estimate_scales_with_landmark_max():
    small = flux2_ttr._estimate_flux2_ttr_memory_bytes(
        batch=1,
        heads=24,
        n_query=256,
        n_key=4096,
        head_dim=128,
        feature_dim=256,
        q_chunk_size=256,
        k_chunk_size=1024,
        dtype_size=4,
        training=False,
        landmark_max=64,
    )
    large = flux2_ttr._estimate_flux2_ttr_memory_bytes(
        batch=1,
        heads=24,
        n_query=256,
        n_key=4096,
        head_dim=128,
        feature_dim=256,
        q_chunk_size=256,
        k_chunk_size=1024,
        dtype_size=4,
        training=False,
        landmark_max=512,
    )
    assert large > small


def test_memory_reserve_estimate_treats_landmark_max_zero_as_unlimited():
    finite = flux2_ttr._estimate_flux2_ttr_memory_bytes(
        batch=1,
        heads=24,
        n_query=256,
        n_key=4096,
        head_dim=128,
        feature_dim=256,
        q_chunk_size=256,
        k_chunk_size=1024,
        dtype_size=4,
        training=False,
        landmark_max=512,
    )
    unlimited = flux2_ttr._estimate_flux2_ttr_memory_bytes(
        batch=1,
        heads=24,
        n_query=256,
        n_key=4096,
        head_dim=128,
        feature_dim=256,
        q_chunk_size=256,
        k_chunk_size=1024,
        dtype_size=4,
        training=False,
        landmark_max=0,
    )
    assert unlimited > finite


def test_memory_reserve_estimate_includes_conditioning_token_landmarks():
    base = flux2_ttr._estimate_flux2_ttr_memory_bytes(
        batch=1,
        heads=24,
        n_query=256,
        n_key=4096,
        head_dim=128,
        feature_dim=256,
        q_chunk_size=256,
        k_chunk_size=1024,
        dtype_size=4,
        training=False,
        landmark_max=128,
        text_count_estimate=0,
    )
    conditioned = flux2_ttr._estimate_flux2_ttr_memory_bytes(
        batch=1,
        heads=24,
        n_query=256,
        n_key=4096,
        head_dim=128,
        feature_dim=256,
        q_chunk_size=256,
        k_chunk_size=1024,
        dtype_size=4,
        training=False,
        landmark_max=128,
        text_count_estimate=256,
    )
    assert conditioned > base


def test_training_oom_recovery_reduces_pressure_and_clears_layer_buffer():
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=4,
        training_query_token_cap=128,
        replay_buffer_size=8,
    )
    layer_key = "single:0"

    runtime._push_replay_sample(
        layer_key=layer_key,
        q_sub=torch.randn(1, 2, 4, 4),
        k_full=torch.randn(1, 2, 8, 4),
        v_full=torch.randn(1, 2, 8, 4),
        teacher_sub=torch.randn(1, 2, 4, 4),
        key_mask=torch.ones(1, 8, dtype=torch.bool),
        text_token_count=4,
        sigma=0.5,
        cfg_scale=3.0,
    )
    assert len(runtime.replay_buffers[layer_key]) == 1

    changed = runtime._handle_training_oom(layer_key, torch.device("cpu"))
    assert changed is True
    assert runtime.training_query_token_cap <= 64
    assert runtime.query_chunk_size <= 128
    assert runtime.key_chunk_size <= 512
    assert runtime.landmark_max > 0
    assert runtime.landmark_max <= 512
    assert len(runtime.replay_buffers[layer_key]) == 0


def test_replay_budget_evicts_old_samples_across_layers():
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=4,
        replay_buffer_size=16,
        replay_max_bytes=1200,
        replay_offload_cpu=True,
        replay_storage_dtype="float32",
    )

    def push(layer_key: str):
        runtime._push_replay_sample(
            layer_key=layer_key,
            q_sub=torch.randn(1, 1, 4, 4),
            k_full=torch.randn(1, 1, 16, 4),
            v_full=torch.randn(1, 1, 16, 4),
            teacher_sub=torch.randn(1, 1, 4, 4),
            key_mask=torch.ones(1, 16, dtype=torch.bool),
            text_token_count=4,
            sigma=0.5,
            cfg_scale=2.0,
        )

    # Push enough samples to force global budget eviction.
    push("single:0")
    push("single:1")
    push("single:0")
    push("single:1")

    assert runtime.replay_total_bytes <= runtime.replay_max_bytes
    total_samples = sum(len(buf) for buf in runtime.replay_buffers.values())
    assert total_samples >= 1
    assert total_samples < 4


def test_maybe_reserve_memory_dedupes(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Memory reservation test requires CUDA device.")
    calls = []

    class _MM:
        @staticmethod
        def free_memory(mem_bytes, device):
            calls.append((int(mem_bytes), str(device)))

    monkeypatch.setattr(flux2_ttr, "model_management", _MM)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    q = torch.randn(1, 2, 32, 4, device="cuda")
    k = torch.randn(1, 2, 32, 4, device="cuda")
    opts = {}

    flux2_ttr._maybe_reserve_memory(runtime, q, k, opts, training=False, dtype_accum=torch.float32)
    flux2_ttr._maybe_reserve_memory(runtime, q, k, opts, training=False, dtype_accum=torch.float32)
    assert len(calls) == 1
    assert "flux2_ttr_memory_reserved" in opts


def test_maybe_reserve_memory_logs_only_when_signature_changes(monkeypatch, caplog):
    calls = []

    class _MM:
        @staticmethod
        def free_memory(mem_bytes, device):
            calls.append((int(mem_bytes), str(device)))

    monkeypatch.setattr(flux2_ttr, "model_management", _MM)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    q_small = torch.empty((1, 2, 32, 4), device="meta")
    k_small = torch.empty((1, 2, 32, 4), device="meta")
    q_large = torch.empty((1, 2, 64, 4), device="meta")
    k_large = torch.empty((1, 2, 64, 4), device="meta")

    caplog.set_level(logging.INFO, logger="flux2_ttr")

    flux2_ttr._maybe_reserve_memory(
        runtime,
        q_small,
        k_small,
        {},
        training=False,
        dtype_accum=torch.float32,
        layer_key="single:0",
    )
    flux2_ttr._maybe_reserve_memory(
        runtime,
        q_small,
        k_small,
        {},
        training=False,
        dtype_accum=torch.float32,
        layer_key="single:1",
    )
    flux2_ttr._maybe_reserve_memory(
        runtime,
        q_large,
        k_large,
        {},
        training=False,
        dtype_accum=torch.float32,
        layer_key="single:2",
    )

    reserve_logs = [rec.getMessage() for rec in caplog.records if "Flux2TTR reserved ~" in rec.getMessage()]
    assert len(calls) == 3
    assert len(reserve_logs) == 2
    assert "inference" in reserve_logs[0]
