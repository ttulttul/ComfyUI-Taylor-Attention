import math

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


def test_runtime_training_uses_teacher_output_then_switches_to_inference():
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
    out_train = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)
    assert torch.allclose(out_train, baseline)
    assert runtime.steps_remaining == 0
    assert runtime.training_enabled is False
    assert not math.isnan(runtime.last_loss)

    out_infer = runtime.run_attention(q, k, v, pe=None, mask=None, transformer_options=opts, fallback_attention=fallback)
    assert out_infer.shape == baseline.shape
    assert torch.isfinite(out_infer).all()


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
