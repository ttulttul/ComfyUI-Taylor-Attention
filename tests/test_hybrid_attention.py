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
