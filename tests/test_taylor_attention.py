import math
import os

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
