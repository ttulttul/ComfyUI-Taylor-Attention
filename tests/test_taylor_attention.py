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

    caplog.set_level(logging.INFO, logger="taylor_attention")
    taylor_attention.taylor_attention(
        q,
        k,
        v,
        heads,
        skip_reshape=True,
        config=config,
    )

    assert "Taylor denominator stats" in caplog.text


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

    caplog.set_level(logging.INFO, logger="taylor_attention")
    taylor_attention.taylor_attention(
        q,
        k,
        v,
        heads,
        skip_reshape=True,
        config=config,
    )

    assert "Taylor quality check" in caplog.text
