import torch

import hybrid_attention


def test_hybrid_config_resolves():
    cfg = hybrid_attention._resolve_config({"enabled": True, "global_dim": 8, "global_P": 3})
    assert cfg.enabled is True
    assert cfg.global_dim == 8
    assert cfg.global_P == 3


def test_hybrid_config_quality_flag():
    cfg = hybrid_attention._resolve_config({"enabled": True, "log_quality_stats": True})
    assert cfg.log_quality_stats is True


def test_compute_local_window_schedule():
    cfg = hybrid_attention._resolve_config(
        {
            "enabled": True,
            "local_window": 256,
            "local_window_min": 128,
            "local_window_max": 0,
            "local_window_sigma_low": 0.5,
            "local_window_sigma_high": 1.0,
        }
    )
    full_window = 512
    assert hybrid_attention._compute_local_window(cfg, 0.4, full_window) == 128
    assert hybrid_attention._compute_local_window(cfg, 1.0, full_window) == full_window
    mid = hybrid_attention._compute_local_window(cfg, 0.75, full_window)
    assert 128 < mid < full_window


def test_hybrid_global_weight_ramp():
    cfg = hybrid_attention._resolve_config({"enabled": True, "global_weight": 1.0, "global_sigma_low": 0.2, "global_sigma_high": 0.6})
    assert hybrid_attention._compute_global_weight(cfg, 0.1) == 1.0
    assert hybrid_attention._compute_global_weight(cfg, 0.6) == 0.0
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


def test_project_pca_respects_dtype_in_cache():
    hybrid_attention._PCA_CACHE.clear()
    q = torch.randn(1, 1, 0, 4, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 1, 4, dtype=torch.bfloat16)
    proj = hybrid_attention._project_pca(q, k, d_low=2, samples=0)
    assert proj.dtype == torch.bfloat16


def test_global_taylor_attention_dtype_matches_input():
    cfg = hybrid_attention._resolve_config({"enabled": True, "global_dim": 2, "global_P": 2, "force_fp32": False})
    q = torch.randn(1, 1, 3, 4, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 3, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 1, 3, 4, dtype=torch.bfloat16)
    out = hybrid_attention._global_taylor_attention(cfg, q, k, v, None, scale=1.0)
    assert out.dtype == v.dtype


def test_format_config_summary_includes_keys():
    cfg = {"local_window": 0, "global_dim": 4, "global_weight": 0.1, "force_fp32": True}
    summary = hybrid_attention._format_config_summary(cfg)
    assert "local_window=0" in summary
    assert "global_dim=4" in summary
    assert "global_weight=0.1" in summary
    assert "force_fp32=True" in summary


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
