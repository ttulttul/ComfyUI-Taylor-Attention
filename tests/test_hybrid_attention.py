import torch

import hybrid_attention


def test_hybrid_config_resolves():
    cfg = hybrid_attention._resolve_config({"enabled": True, "global_dim": 8, "global_P": 3})
    assert cfg.enabled is True
    assert cfg.global_dim == 8
    assert cfg.global_P == 3


def test_hybrid_global_weight_ramp():
    cfg = hybrid_attention._resolve_config({"enabled": True, "global_weight": 1.0, "global_sigma_low": 0.2, "global_sigma_high": 0.6})
    assert hybrid_attention._compute_global_weight(cfg, 0.1) == 0.0
    assert hybrid_attention._compute_global_weight(cfg, 0.6) == 1.0
    mid = hybrid_attention._compute_global_weight(cfg, 0.4)
    assert 0.4 < mid < 0.6


def test_taylor_feature_map_shape():
    x = torch.randn(2, 3, 5, 4)
    phi = hybrid_attention._taylor_feature_map(x, P=2, scale=1.0)
    assert phi.shape[:3] == x.shape[:3]
    assert phi.shape[-1] > 0


def test_project_pca_falls_back_on_too_few_samples():
    hybrid_attention._PCA_CACHE.clear()
    q = torch.randn(1, 1, 0, 4)
    k = torch.randn(1, 1, 1, 4)
    proj = hybrid_attention._project_pca(q, k, d_low=3, samples=0)
    assert proj.shape == (4, 3)
    eye_slice = torch.eye(4)[:, :3]
    assert torch.allclose(proj.cpu(), eye_slice)


def test_pre_run_callback_reads_model_options(monkeypatch):
    called = {"patch": 0}

    def fake_patch():
        called["patch"] += 1

    monkeypatch.setattr(hybrid_attention, "patch_flux_attention", fake_patch)

    class DummyPatcher:
        def __init__(self):
            self.model_options = {
                "transformer_options": {"hybrid_taylor_attention": {"enabled": True, "log_steps": False}}
            }

    hybrid_attention.pre_run_callback(DummyPatcher())
    assert called["patch"] == 1


def test_pre_run_callback_ignores_disabled(monkeypatch):
    called = {"patch": 0}

    def fake_patch():
        called["patch"] += 1

    monkeypatch.setattr(hybrid_attention, "patch_flux_attention", fake_patch)

    class DummyPatcher:
        def __init__(self):
            self.model_options = {
                "transformer_options": {"hybrid_taylor_attention": {"enabled": False, "log_steps": False}}
            }

    hybrid_attention.pre_run_callback(DummyPatcher())
    assert called["patch"] == 0


def test_cleanup_callback_reads_model_options(monkeypatch):
    called = {"restore": 0}

    def fake_restore():
        called["restore"] += 1

    monkeypatch.setattr(hybrid_attention, "restore_flux_attention", fake_restore)

    class DummyPatcher:
        def __init__(self):
            self.model_options = {
                "transformer_options": {"hybrid_taylor_attention": {"enabled": True, "log_steps": False}}
            }

    hybrid_attention.cleanup_callback(DummyPatcher())
    assert called["restore"] == 1
