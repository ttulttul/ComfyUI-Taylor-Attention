import math
import os
import logging

import pytest
import torch

import taylor_attention


def _baseline_attention(q, k, v, mask=None):
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
    if mask is not None:
        if mask.dtype != torch.bool:
            mask = mask != 0
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("b h i j, b h j d -> b h i d", attn, v)


def _make_qkv(batch, heads, tokens, dim_head, device):
    q = torch.randn(batch, heads, tokens, dim_head, device=device, dtype=torch.float32)
    k = torch.randn(batch, heads, tokens, dim_head, device=device, dtype=torch.float32)
    v = torch.randn(batch, heads, tokens, dim_head, device=device, dtype=torch.float32)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
    return q, k, v


@pytest.mark.parametrize("tokens", [64, 256, 1024, 4096])
@pytest.mark.parametrize("dim_head", [16, 32])
def test_taylor_attention_matches_softmax(tokens, dim_head):
    if tokens > 512 and os.getenv("RUN_LONG_TESTS") != "1":
        pytest.skip("Set RUN_LONG_TESTS=1 to run long attention tests.")

    device = torch.device("cpu")
    batch = 1
    heads = 4

    q, k, v = _make_qkv(batch, heads, tokens, dim_head, device)
    baseline = _baseline_attention(q, k, v)

    config = {
        "enabled": True,
        "P": 4,
        "min_tokens": 0,
        "max_feature_dim_R": 200000,
        "block_size_q": 256,
        "block_size_k": 256,
        "eps": 1e-6,
        "fallback_on_negative": False,
        "allow_cross_attention": True,
        "max_head_dim": 128,
    }

    taylor = taylor_attention.taylor_attention(
        q,
        k,
        v,
        heads,
        skip_reshape=True,
        config=config,
    )

    error = (taylor - baseline).abs()
    mean_error = error.mean().item()
    max_error = error.max().item()

    assert mean_error < 5e-2, f"mean error too high: {mean_error} max: {max_error}"


def test_taylor_attention_with_mask():
    device = torch.device("cpu")
    batch = 1
    heads = 2
    tokens = 128
    dim_head = 32

    q, k, v = _make_qkv(batch, heads, tokens, dim_head, device)
    mask = torch.ones((batch, tokens), dtype=torch.bool, device=device)
    mask[:, tokens // 2 :] = False

    baseline = _baseline_attention(q, k, v, mask=mask)

    config = {
        "enabled": True,
        "P": 4,
        "min_tokens": 0,
        "max_feature_dim_R": 200000,
        "block_size_q": 256,
        "block_size_k": 256,
        "eps": 1e-6,
        "fallback_on_negative": False,
        "allow_cross_attention": True,
        "max_head_dim": 128,
    }

    taylor = taylor_attention.taylor_attention(
        q,
        k,
        v,
        heads,
        mask=mask,
        skip_reshape=True,
        config=config,
    )

    error = (taylor - baseline).abs().mean().item()
    assert error < 5e-2


def test_taylor_attention_skip_reshape_false():
    device = torch.device("cpu")
    batch = 1
    heads = 2
    tokens = 64
    dim_head = 32

    q, k, v = _make_qkv(batch, heads, tokens, dim_head, device)
    q_in = q.permute(0, 2, 1, 3).reshape(batch, tokens, heads * dim_head)
    k_in = k.permute(0, 2, 1, 3).reshape(batch, tokens, heads * dim_head)
    v_in = v.permute(0, 2, 1, 3).reshape(batch, tokens, heads * dim_head)

    baseline = _baseline_attention(q, k, v).permute(0, 2, 1, 3).reshape(batch, tokens, heads * dim_head)

    config = {
        "enabled": True,
        "P": 4,
        "min_tokens": 0,
        "max_feature_dim_R": 200000,
        "block_size_q": 128,
        "block_size_k": 128,
        "eps": 1e-6,
        "fallback_on_negative": False,
        "allow_cross_attention": True,
        "max_head_dim": 128,
    }

    taylor_out = taylor_attention.taylor_attention(
        q_in,
        k_in,
        v_in,
        heads,
        skip_reshape=False,
        config=config,
    )

    error = (taylor_out - baseline).abs().mean().item()
    assert error < 5e-2


def test_feature_dim_too_large_logs(caplog):
    device = torch.device("cpu")
    batch = 1
    heads = 1
    tokens = 1
    dim_head = 8

    q, k, v = _make_qkv(batch, heads, tokens, dim_head, device)

    config = {
        "enabled": True,
        "P": 4,
        "min_tokens": 0,
        "max_feature_dim_R": 10,
        "block_size_q": 1,
        "block_size_k": 1,
        "eps": 1e-6,
        "fallback_on_negative": False,
        "allow_cross_attention": True,
        "max_head_dim": 128,
        "log_fallbacks": True,
    }

    caplog.set_level(logging.WARNING, logger="taylor_attention")
    with pytest.raises(taylor_attention.TaylorAttentionFallback):
        taylor_attention.taylor_attention(
            q,
            k,
            v,
            heads,
            skip_reshape=True,
            config=config,
        )

    assert "feature_dim_too_large" in caplog.text
    assert "max_feature_dim_R" in caplog.text


def test_force_fp32_config_respected():
    cfg = taylor_attention._resolve_config({"enabled": True, "force_fp32": False})
    assert cfg.force_fp32 is False

def test_memory_reserve_config_respected():
    cfg = taylor_attention._resolve_config({"enabled": True, "memory_reserve": False, "memory_reserve_factor": 1.5})
    assert cfg.memory_reserve is False
    assert cfg.memory_reserve_factor == 1.5

def test_probe_config_respected():
    cfg = taylor_attention._resolve_config({"enabled": True, "early_probe": True, "probe_samples": 12, "denom_fp32": True})
    assert cfg.early_probe is True
    assert cfg.probe_samples == 12
    assert cfg.denom_fp32 is True


def test_denom_fallback_frac_limit_config():
    cfg = taylor_attention._resolve_config({"enabled": True, "denom_fallback_frac_limit": 0.001})
    assert cfg.denom_fallback_frac_limit == 0.001


def test_auto_tune_config():
    cfg = taylor_attention._resolve_config({"enabled": True, "auto_tune": True, "auto_tune_steps": 2, "auto_tune_candidates": 4})
    assert cfg.auto_tune is True
    assert cfg.auto_tune_steps == 2
    assert cfg.auto_tune_candidates == 4


def test_fused_kernel_config():
    cfg = taylor_attention._resolve_config({
        "enabled": True,
        "fused_kernel": True,
        "fused_full_kernel": True,
        "fused_feature_chunk_size": 4096,
        "fused_value_chunk_size": 256,
        "s_store_bf16": True,
    })
    assert cfg.fused_kernel is True
    assert cfg.fused_full_kernel is True
    assert cfg.fused_feature_chunk_size == 4096
    assert cfg.fused_value_chunk_size == 256
    assert cfg.s_store_bf16 is True


def test_sub_head_blocks_config_respected():
    cfg = taylor_attention._resolve_config({"enabled": True, "sub_head_blocks": 3})
    assert cfg.sub_head_blocks == 3


def test_sub_head_blocks_output_shape():
    device = torch.device("cpu")
    batch = 1
    heads = 2
    tokens = 64
    dim_head = 32

    q, k, v = _make_qkv(batch, heads, tokens, dim_head, device)

    config = {
        "enabled": True,
        "P": 3,
        "min_tokens": 0,
        "max_feature_dim_R": 200000,
        "block_size_q": 32,
        "block_size_k": 32,
        "eps": 1e-6,
        "fallback_on_negative": False,
        "allow_cross_attention": True,
        "max_head_dim": 128,
        "sub_head_blocks": 2,
    }

    out = taylor_attention.taylor_attention(
        q,
        k,
        v,
        heads,
        skip_reshape=True,
        skip_output_reshape=True,
        config=config,
    )

    assert out.shape == (batch, heads, tokens, dim_head)


def test_qk_normalize_config_respected():
    cfg = taylor_attention._resolve_config({"enabled": True, "qk_normalize": True, "scale_mul": 0.5})
    assert cfg.qk_normalize is True
    assert cfg.scale_mul == 0.5


def test_qk_norm_clip_power_config_respected():
    cfg = taylor_attention._resolve_config({"enabled": True, "qk_norm_clip": 12.5, "qk_norm_power": 0.25})
    assert cfg.qk_norm_clip == 12.5
    assert cfg.qk_norm_power == 0.25


def test_qk_norm_sigma_max_config_respected():
    cfg = taylor_attention._resolve_config({"enabled": True, "qk_norm_sigma_max": 0.75})
    assert cfg.qk_norm_sigma_max == 0.75


def test_taylor_gate_config_respected():
    cfg = taylor_attention._resolve_config({"enabled": True, "taylor_sigma_max": 1.2, "taylor_layer_start": 2, "taylor_layer_end": 4})
    assert cfg.taylor_sigma_max == 1.2
    assert cfg.taylor_layer_start == 2
    assert cfg.taylor_layer_end == 4


def test_qk_norm_clip_power_runs():
    device = torch.device("cpu")
    batch = 1
    heads = 2
    tokens = 32
    dim_head = 16

    q, k, v = _make_qkv(batch, heads, tokens, dim_head, device)

    config = {
        "enabled": True,
        "P": 3,
        "min_tokens": 0,
        "max_feature_dim_R": 200000,
        "block_size_q": 32,
        "block_size_k": 32,
        "eps": 1e-6,
        "fallback_on_negative": False,
        "allow_cross_attention": True,
        "max_head_dim": 128,
        "qk_norm_clip": 10.0,
        "qk_norm_power": 0.5,
    }

    out = taylor_attention.taylor_attention(
        q,
        k,
        v,
        heads,
        skip_reshape=True,
        skip_output_reshape=True,
        config=config,
    )

    assert out.shape == (batch, heads, tokens, dim_head)


def test_qk_normalize_runs():
    device = torch.device("cpu")
    batch = 1
    heads = 2
    tokens = 32
    dim_head = 16

    q, k, v = _make_qkv(batch, heads, tokens, dim_head, device)

    config = {
        "enabled": True,
        "P": 3,
        "min_tokens": 0,
        "max_feature_dim_R": 200000,
        "block_size_q": 32,
        "block_size_k": 32,
        "eps": 1e-6,
        "fallback_on_negative": False,
        "allow_cross_attention": True,
        "max_head_dim": 128,
        "qk_normalize": True,
        "scale_mul": 0.5,
    }

    out = taylor_attention.taylor_attention(
        q,
        k,
        v,
        heads,
        skip_reshape=True,
        skip_output_reshape=True,
        config=config,
    )

    assert out.shape == (batch, heads, tokens, dim_head)


def test_sub_head_blocks_mismatch_fallback():
    device = torch.device("cpu")
    batch = 1
    heads = 2
    tokens = 16
    dim_head = 30

    q, k, v = _make_qkv(batch, heads, tokens, dim_head, device)

    config = {
        "enabled": True,
        "P": 3,
        "min_tokens": 0,
        "max_feature_dim_R": 200000,
        "block_size_q": 16,
        "block_size_k": 16,
        "eps": 1e-6,
        "fallback_on_negative": False,
        "allow_cross_attention": True,
        "max_head_dim": 128,
        "sub_head_blocks": 4,
    }

    with pytest.raises(taylor_attention.TaylorAttentionFallback) as exc:
        taylor_attention.taylor_attention(
            q,
            k,
            v,
            heads,
            skip_reshape=True,
            config=config,
        )

    assert exc.value.reason == "sub_head_block_mismatch"


def test_denominator_stats_logs(caplog):
    device = torch.device("cpu")
    batch = 1
    heads = 2
    tokens = 32
    dim_head = 16

    q, k, v = _make_qkv(batch, heads, tokens, dim_head, device)

    config = {
        "enabled": True,
        "P": 3,
        "min_tokens": 0,
        "max_feature_dim_R": 200000,
        "block_size_q": 32,
        "block_size_k": 32,
        "eps": 1e-6,
        "fallback_on_negative": False,
        "allow_cross_attention": True,
        "max_head_dim": 128,
        "early_probe": False,
    }

    transformer_options = {
        "block_type": "single",
        "block_index": 0,
        "total_blocks": 1,
        "sigmas": torch.tensor([1.0]),
    }

    caplog.set_level(logging.INFO, logger="taylor_attention")
    taylor_attention.taylor_attention(
        q,
        k,
        v,
        heads,
        skip_reshape=True,
        config=config,
        transformer_options=transformer_options,
    )

    assert "Taylor step stats" in caplog.text


def test_quality_check_logs(caplog):
    device = torch.device("cpu")
    batch = 1
    heads = 2
    tokens = 32
    dim_head = 16

    q, k, v = _make_qkv(batch, heads, tokens, dim_head, device)

    config = {
        "enabled": True,
        "P": 3,
        "min_tokens": 0,
        "max_feature_dim_R": 200000,
        "block_size_q": 32,
        "block_size_k": 32,
        "eps": 1e-6,
        "fallback_on_negative": False,
        "allow_cross_attention": True,
        "max_head_dim": 128,
        "quality_check": True,
        "quality_check_samples": 4,
        "quality_check_log_every": 1,
    }

    transformer_options = {
        "block_type": "single",
        "block_index": 0,
        "total_blocks": 1,
        "sigmas": torch.tensor([1.0]),
    }

    caplog.set_level(logging.INFO, logger="taylor_attention")
    taylor_attention.taylor_attention(
        q,
        k,
        v,
        heads,
        skip_reshape=True,
        config=config,
        transformer_options=transformer_options,
    )

    assert "Taylor step stats" in caplog.text


def test_quality_stats_respect_indices():
    device = torch.device("cpu")
    batch = 1
    heads = 2
    tokens = 8
    dim_head = 16

    torch.manual_seed(0)
    q, k, v = _make_qkv(batch, heads, tokens, dim_head, device)
    out = torch.zeros_like(q)
    scale = dim_head ** -0.5

    cfg = taylor_attention._resolve_config({"enabled": True, "quality_check_samples": 4})
    idx_a = torch.tensor([0, 1, 2, 3], device=device)
    idx_b = torch.tensor([4, 5, 6, 7], device=device)

    torch.manual_seed(0)
    stats_a = taylor_attention._compute_quality_stats(cfg, q, k, v, out, None, scale, idx=idx_a)
    torch.manual_seed(0)
    stats_b = taylor_attention._compute_quality_stats(cfg, q, k, v, out, None, scale, idx=idx_b)

    assert stats_a is not None and stats_b is not None
    assert stats_a["sum_abs"] != stats_b["sum_abs"]
