import math
import logging
import sys
import types

import pytest
import torch

import flux2_ttr


def _baseline_flat(q, k, v):
    return flux2_ttr._flatten_heads(flux2_ttr._softmax_attention(q.float(), k.float(), v.float()).to(dtype=v.dtype))


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


def test_ttr_flux_layer_shape_and_finite():
    torch.manual_seed(0)
    layer = flux2_ttr.TTRFluxLayer(head_dim=8, feature_dim=256)
    q = torch.randn(2, 3, 12, 8)
    k = torch.randn(2, 3, 12, 8)
    v = torch.randn(2, 3, 12, 8)
    out = layer(q, k, v)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_ttr_scan_chunked_matches_token_scan():
    torch.manual_seed(0)
    cell = flux2_ttr.TTRCell(feature_dim=16, value_dim=8)
    q = torch.randn(3, 19, 16)
    k = torch.randn(3, 19, 16)
    v = torch.randn(3, 19, 8)
    out_token = cell.scan(q, k, v, chunk_size=1)
    out_chunked = cell.scan(q, k, v, chunk_size=7)
    assert out_token.shape == out_chunked.shape
    assert torch.allclose(out_token, out_chunked, atol=1e-6, rtol=1e-5)


def test_runtime_training_previews_ttr_output():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=1)
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])

    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)
    opts = {"block_type": "single", "block_index": 0}

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, mask, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg)

    out_train = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)
    baseline = fallback(q, k, v, None)
    assert out_train.shape == baseline.shape
    assert torch.isfinite(out_train).all()
    assert not torch.allclose(out_train, baseline)
    assert runtime.steps_remaining == 0
    assert runtime.training_enabled is False
    assert not math.isnan(runtime.last_loss)

    out_preview = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)
    assert out_preview.shape == baseline.shape
    assert torch.isfinite(out_preview).all()


def test_runtime_training_can_use_teacher_passthrough_when_preview_disabled():
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
    opts = {"block_type": "single", "block_index": 0}

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, mask, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg)

    baseline = fallback(q, k, v, None)
    out_train = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)
    assert torch.allclose(out_train, baseline)
    assert runtime.steps_remaining == 0
    assert runtime.training_enabled is False
    assert not math.isnan(runtime.last_loss)


def test_runtime_inference_mode_uses_student_output_shape():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime.training_mode = False
    runtime.training_enabled = False
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])

    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)
    opts = {"block_type": "single", "block_index": 0}

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, mask, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg)

    out = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)
    assert out.shape == (q.shape[0], q.shape[2], q.shape[1] * q.shape[3])
    assert torch.isfinite(out).all()


def test_runtime_layer_range_falls_back_to_teacher():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=False,
        steps=0,
        layer_start=2,
        layer_end=4,
    )
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])
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
    assert not runtime.layers


def test_runtime_resolve_inference_dtype_cpu_is_fp32():
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0, inference_mixed_precision=True)
    x = torch.randn(1, 1, 2, 4, dtype=torch.bfloat16)
    assert runtime._resolve_inference_dtype(x) == torch.float32
    runtime.inference_mixed_precision = False
    assert runtime._resolve_inference_dtype(x) == torch.float32


def test_effective_scan_chunk_training_cap():
    runtime = flux2_ttr.Flux2TTRRuntime(
        feature_dim=256,
        learning_rate=1e-3,
        training=False,
        steps=0,
        scan_chunk_size=512,
    )
    assert runtime._effective_scan_chunk(training=False) == 512
    assert runtime._effective_scan_chunk(training=True) == 64


def test_training_token_cap_indices():
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=4)
    idx = runtime._select_training_token_indices(seq_len=1024, device=torch.device("cpu"))
    assert idx is not None
    assert idx.numel() == runtime.training_token_cap
    assert int(idx.min().item()) >= 0
    assert int(idx.max().item()) <= 1023


def test_memory_reserve_estimate_scales_with_training():
    infer_bytes = flux2_ttr._estimate_flux2_ttr_memory_bytes(
        batch=1,
        heads=24,
        seq_len=256,
        head_dim=128,
        feature_dim=256,
        chunk_size=128,
        dtype_size=4,
        training=False,
    )
    train_bytes = flux2_ttr._estimate_flux2_ttr_memory_bytes(
        batch=1,
        heads=24,
        seq_len=256,
        head_dim=128,
        feature_dim=256,
        chunk_size=128,
        dtype_size=4,
        training=True,
    )
    assert infer_bytes > 0
    assert train_bytes > infer_bytes


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
    opts = {}

    flux2_ttr._maybe_reserve_memory(runtime, q, opts, training=False, dtype_accum=torch.float32)
    flux2_ttr._maybe_reserve_memory(runtime, q, opts, training=False, dtype_accum=torch.float32)
    assert len(calls) == 1
    assert "flux2_ttr_memory_reserved" in opts


def test_training_oom_disables_training_and_returns_teacher():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=1)
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])

    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)
    opts = {"block_type": "single", "block_index": 0}

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, mask, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg)

    baseline = fallback(q, k, v, None)
    layer = runtime._ensure_layer("single:0", 4, q.device)

    def oom_forward(*args, **kwargs):
        raise torch.OutOfMemoryError("synthetic oom")

    layer.forward = oom_forward  # type: ignore[method-assign]
    out = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)
    assert torch.allclose(out, baseline)
    assert runtime.training_enabled is False


def test_high_loss_inference_falls_back_to_teacher():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime.training_mode = False
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


def test_cleanup_unregisters_runtime_and_releases_resources(monkeypatch):
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime_id = flux2_ttr.register_runtime(runtime)

    called = {"release": 0, "restore": 0}

    def fake_release():
        called["release"] += 1

    def fake_restore():
        called["restore"] += 1

    runtime.release_resources = fake_release  # type: ignore[method-assign]
    monkeypatch.setattr(flux2_ttr, "restore_flux_attention", fake_restore)

    class DummyPatcher:
        def __init__(self):
            self.model_options = {
                "transformer_options": {"flux2_ttr": {"enabled": True, "runtime_id": runtime_id, "training_mode": False}}
            }

    flux2_ttr.cleanup_callback(DummyPatcher())
    assert called["release"] == 1
    assert called["restore"] == 1
    assert flux2_ttr.get_runtime(runtime_id) is None


def test_recover_runtime_from_config_inference_requires_checkpoint():
    cfg = {
        "training_mode": False,
        "feature_dim": 256,
        "scan_chunk_size": 128,
        "layer_start": 10,
        "layer_end": 19,
        "inference_mixed_precision": True,
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
        "scan_chunk_size": 128,
        "layer_start": 10,
        "layer_end": 19,
        "inference_mixed_precision": True,
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
        "scan_chunk_size": 128,
        "layer_start": 10,
        "layer_end": 19,
        "inference_mixed_precision": True,
        "checkpoint_path": str(ckpt),
    }
    runtime = flux2_ttr._recover_runtime_from_config(cfg)
    assert runtime is not None
    assert runtime.training_mode is False
    assert runtime.training_enabled is False
    assert runtime.training_preview_ttr is False


def test_training_progress_logs_every_10_updates(caplog):
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=10)
    runtime.training_mode = True
    runtime.training_enabled = True
    runtime.training_steps_total = 10
    runtime.training_updates_done = 0
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=1, head_dim=4)])

    q = torch.randn(1, 1, 8, 4)
    k = torch.randn(1, 1, 8, 4)
    v = torch.randn(1, 1, 8, 4)
    opts = {"block_type": "single", "block_index": 0}

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, mask, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg)

    caplog.set_level(logging.INFO, logger="flux2_ttr")
    for _ in range(10):
        runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)

    assert runtime.training_updates_done == 10
    assert "Flux2TTR distill progress: updates=10/10" in caplog.text


def test_compute_distill_metrics_contains_requested_fields():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=1)
    q = torch.randn(1, 2, 12, 4)
    k = torch.randn(1, 2, 12, 4)
    v = torch.randn(1, 2, 12, 4)
    teacher = flux2_ttr._softmax_attention(q.float(), k.float(), v.float())
    student = teacher + 0.05 * torch.randn_like(teacher)
    loss = torch.nn.functional.mse_loss(student, teacher)

    metrics = runtime._compute_distill_metrics(
        student=student,
        teacher=teacher,
        q=q,
        k=k,
        v=v,
        mask=None,
        loss_value=float(loss.item()),
    )

    expected_keys = {
        "loss",
        "nmse",
        "cosine_similarity",
        "norm_ratio",
        "mean_ratio",
        "std_ratio",
        "p95_abs_error",
        "p99_abs_error",
        "attn_kl_div",
        "attn_topk_overlap",
    }
    assert expected_keys.issubset(metrics.keys())
    for key in expected_keys:
        assert math.isfinite(float(metrics[key]))
    assert 0.0 <= float(metrics["attn_topk_overlap"]) <= 1.0


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
    runtime._record_training_metrics("single:10", {"loss": 1.0, "nmse": 2.0})

    assert start_calls == [("test-key", "proj", "ws")]
    assert len(params_calls) == 1
    assert metric_calls
    payload, step = metric_calls[-1]
    assert step == 3
    assert payload["flux2ttr/single:10/loss"] == 1.0
    assert payload["flux2ttr/single:10/nmse"] == 2.0
    assert payload["flux2ttr/single:10/avg_loss"] == 1.0
    assert payload["flux2ttr/single:10/avg_nmse"] == 2.0
    assert payload["flux2ttr/global/steps_remaining"] == 7.0
    assert payload["flux2ttr/global/updates_done"] == 3.0


def test_runtime_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=4)
    patcher = _DummyPatcher()
    latents = {"samples": torch.randn(1, 4, 8, 8)}
    conditioning = [[torch.randn(1, 16, 32), {"pooled_output": torch.randn(1, 16)}]]

    loss = runtime.calibrate_from_inputs(patcher, latents, conditioning, steps=4, max_tokens=32)
    assert isinstance(loss, float)
    assert runtime.layers

    ckpt = tmp_path / "flux2_ttr.pt"
    runtime.save_checkpoint(str(ckpt))
    assert ckpt.exists()

    runtime_loaded = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=False, steps=0)
    runtime_loaded.load_checkpoint(str(ckpt))
    assert runtime_loaded.pending_state
    assert "single:0" in runtime_loaded.pending_state
    assert not math.isnan(runtime_loaded.last_loss)


def test_calibration_works_inside_inference_mode():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=2)
    patcher = _DummyPatcher()
    latents = {"samples": torch.randn(1, 4, 8, 8)}
    conditioning = [[torch.randn(1, 8, 32), {"pooled_output": torch.randn(1, 16)}]]

    with torch.inference_mode():
        loss = runtime.calibrate_from_inputs(patcher, latents, conditioning, steps=2, max_tokens=16)

    assert isinstance(loss, float)
    assert runtime.layers
    assert not math.isnan(runtime.last_loss)


def test_run_attention_training_works_inside_inference_mode():
    torch.manual_seed(0)
    runtime = flux2_ttr.Flux2TTRRuntime(feature_dim=256, learning_rate=1e-3, training=True, steps=1)
    runtime.register_layer_specs([flux2_ttr.FluxLayerSpec(layer_key="single:0", num_heads=2, head_dim=4)])

    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    v = torch.randn(1, 2, 6, 4)
    opts = {"block_type": "single", "block_index": 0}

    def fallback(q_arg, k_arg, v_arg, pe_arg, mask=None, transformer_options=None):
        del pe_arg, mask, transformer_options
        return _baseline_flat(q_arg, k_arg, v_arg)

    with torch.inference_mode():
        out = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)
    baseline = fallback(q, k, v, None)
    assert out.shape == baseline.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, baseline)
    assert not math.isnan(runtime.last_loss)
