import logging
import math
import sys
import types

import pytest
import torch

import flux2_ttr


def _baseline_flat(q, k, v, mask=None):
    return flux2_ttr._flatten_heads(flux2_ttr._softmax_attention(q.float(), k.float(), v.float(), mask=mask).to(dtype=v.dtype))


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


def test_flux2_hkr_layer_shape_and_landmarks():
    torch.manual_seed(0)
    layer = flux2_ttr.Flux2HKRAttnLayer(
        head_dim=8,
        feature_dim=256,
        landmark_count=6,
        text_tokens_guess=3,
    )

    q = torch.randn(1, 2, 7, 8)
    k = torch.randn(1, 2, 12, 8)
    v = torch.randn(1, 2, 12, 8)
    key_mask = torch.ones(1, 12, dtype=torch.bool)

    out = layer(q, k, v, key_mask=key_mask)
    assert out.shape == (1, 2, 7, 8)
    assert torch.isfinite(out).all()
    assert 1 <= layer.last_landmark_count <= 6


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


def test_runtime_unsupported_mask_falls_back_to_teacher():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])
    runtime.layer_update_count["single:0"] = 999
    runtime.layer_ema_loss["single:0"] = 0.0
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

    runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)

    ckpt = tmp_path / "flux2_ttr_v2.pt"
    runtime.save_checkpoint(str(ckpt))
    assert ckpt.exists()

    runtime_loaded = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime_loaded.load_checkpoint(str(ckpt))

    assert runtime_loaded.pending_state
    assert "single:0" in runtime_loaded.pending_state
    assert "single:0" in runtime_loaded.layer_update_count
    assert "single:0" in runtime_loaded.layer_ema_loss


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
        "checkpoint_path": "",
    }
    runtime = flux2_ttr._recover_runtime_from_config(cfg)
    assert runtime is not None
    assert runtime.training_mode is True
    assert runtime.training_enabled is True
    assert runtime.training_preview_ttr is False
    assert runtime.steps_remaining == 32


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

    def _fake_start(api_key, project_name, workspace):
        start_calls.append((api_key, project_name, workspace))
        return _FakeExperiment()

    fake_comet = types.ModuleType("comet_ml")
    fake_comet.start = _fake_start
    monkeypatch.setitem(sys.modules, "comet_ml", fake_comet)

    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=True,
        steps=10,
        comet_enabled=True,
        comet_api_key="test-key",
        comet_project_name="proj",
        comet_workspace="ws",
    )
    runtime.training_updates_done = 3
    runtime.steps_remaining = 7
    runtime._record_training_metrics("single:10", {"loss": 1.0, "mse": 2.0})

    assert start_calls == [("test-key", "proj", "ws")]
    assert len(params_calls) == 1
    assert metric_calls
    payload, step = metric_calls[-1]
    assert step == 3
    assert payload["flux2ttr/single:10/loss"] == 1.0
    assert payload["flux2ttr/single:10/mse"] == 2.0
    assert payload["flux2ttr/single:10/avg_loss"] == 1.0
    assert payload["flux2ttr/single:10/avg_mse"] == 2.0
    assert payload["flux2ttr/global/steps_remaining"] == 7.0
    assert payload["flux2ttr/global/updates_done"] == 3.0


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
    )
    assert len(runtime.replay_buffers[layer_key]) == 1

    changed = runtime._handle_training_oom(layer_key, torch.device("cpu"))
    assert changed is True
    assert runtime.training_query_token_cap <= 64
    assert runtime.query_chunk_size <= 128
    assert runtime.key_chunk_size <= 512
    assert runtime.landmark_count <= 64
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
