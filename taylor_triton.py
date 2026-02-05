import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - optional dependency
    triton = None
    tl = None
    _TRITON_ERROR = exc
else:
    _TRITON_ERROR = None


def is_available(device: torch.device) -> bool:
    if triton is None:
        return False
    if device.type != "cuda":
        return False
    return True


@triton.jit
def _fused_num_den_kernel(
    psi_q_ptr,
    s_ptr,
    z_ptr,
    out_ptr,
    den_ptr,
    stride_bh_q,
    stride_q_q,
    stride_r_q,
    stride_bh_s,
    stride_r_s,
    stride_d_s,
    stride_bh_z,
    stride_r_z,
    stride_bh_o,
    stride_q_o,
    stride_d_o,
    stride_bh_den,
    stride_q_den,
    Q,
    D,
    R,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    COMPUTE_DEN: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_q = offs_q < Q
    mask_d = offs_d < D

    acc_num = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    if tl.constexpr(COMPUTE_DEN):
        acc_den = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    r = 0
    while r < R:
        offs_r = r + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R

        psi_ptrs = psi_q_ptr + (
            pid_bh * stride_bh_q
            + offs_q[:, None] * stride_q_q
            + offs_r[None, :] * stride_r_q
        )
        psi = tl.load(psi_ptrs, mask=mask_q[:, None] & mask_r[None, :], other=0.0)

        s_ptrs = s_ptr + (
            pid_bh * stride_bh_s
            + offs_r[:, None] * stride_r_s
            + offs_d[None, :] * stride_d_s
        )
        s_block = tl.load(s_ptrs, mask=mask_r[:, None] & mask_d[None, :], other=0.0)

        psi = psi.to(tl.float32)
        s_block = s_block.to(tl.float32)

        acc_num += tl.dot(psi, s_block)

        if tl.constexpr(COMPUTE_DEN):
            z_ptrs = z_ptr + pid_bh * stride_bh_z + offs_r * stride_r_z
            z_block = tl.load(z_ptrs, mask=mask_r, other=0.0)
            acc_den += tl.sum(psi * z_block[None, :], axis=1)

        r += BLOCK_R

    out_ptrs = out_ptr + (
        pid_bh * stride_bh_o
        + offs_q[:, None] * stride_q_o
        + offs_d[None, :] * stride_d_o
    )
    out_prev = tl.load(out_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0)
    out_val = out_prev + acc_num
    tl.store(out_ptrs, out_val, mask=mask_q[:, None] & mask_d[None, :])

    if tl.constexpr(COMPUTE_DEN) and pid_d == 0:
        den_ptrs = den_ptr + pid_bh * stride_bh_den + offs_q * stride_q_den
        den_prev = tl.load(den_ptrs, mask=mask_q, other=0.0)
        den_val = den_prev + acc_den
        tl.store(den_ptrs, den_val, mask=mask_q)


def fused_num_den(
    psi_q: torch.Tensor,
    s: torch.Tensor,
    z: torch.Tensor,
    out: torch.Tensor,
    den: torch.Tensor,
    block_q: int = 16,
    block_d: int = 32,
    block_r: int = 32,
    compute_den: bool = True,
) -> None:
    if not is_available(psi_q.device):
        raise RuntimeError("triton_unavailable")
    if psi_q.device != s.device or psi_q.device != z.device:
        raise ValueError("psi_q/s/z must be on the same device")

    if psi_q.ndim != 4 or s.ndim != 4 or z.ndim != 3 or out.ndim != 4 or den.ndim != 3:
        raise ValueError("unexpected tensor ranks for fused kernel")

    b, h, q_len, r = psi_q.shape
    _, _, r_s, d = s.shape
    if r_s != r or z.shape[2] != r:
        raise ValueError("feature dimension mismatch in fused kernel inputs")

    bh = b * h
    psi_q_2d = psi_q.reshape(bh, q_len, r)
    s_2d = s.reshape(bh, r, d)
    z_2d = z.reshape(bh, r)
    out_2d = out.reshape(bh, q_len, d)
    den_2d = den.reshape(bh, q_len)

    grid = (
        bh,
        triton.cdiv(q_len, block_q),
        triton.cdiv(d, block_d),
    )

    _fused_num_den_kernel[grid](
        psi_q_2d,
        s_2d,
        z_2d,
        out_2d,
        den_2d,
        psi_q_2d.stride(0),
        psi_q_2d.stride(1),
        psi_q_2d.stride(2),
        s_2d.stride(0),
        s_2d.stride(1),
        s_2d.stride(2),
        z_2d.stride(0),
        z_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        out_2d.stride(2),
        den_2d.stride(0),
        den_2d.stride(1),
        q_len,
        d,
        r,
        BLOCK_Q=block_q,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        COMPUTE_DEN=compute_den,
        num_warps=4,
    )


@triton.jit
def _fused_s_z_kernel(
    k_ptr,
    v_ptr,
    idx_ptr,
    deg_ptr,
    w_ptr,
    s_ptr,
    z_ptr,
    stride_bh_k,
    stride_n_k,
    stride_d_k,
    stride_bh_v,
    stride_n_v,
    stride_d_v,
    stride_r_idx,
    stride_p_idx,
    stride_bh_s,
    stride_r_s,
    stride_d_s,
    stride_bh_z,
    stride_r_z,
    N,
    D,
    R,
    P_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_r = offs_r < R
    mask_d = offs_d < D

    deg = tl.load(deg_ptr + offs_r, mask=mask_r, other=0).to(tl.int32)
    weight = tl.load(w_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)

    acc_s = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
    acc_z = tl.zeros((BLOCK_R,), dtype=tl.float32)

    n = 0
    while n < N:
        offs_n = n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        psi = tl.full((BLOCK_N, BLOCK_R), 1.0, dtype=tl.float32)
        for j in range(P_MAX):
            idx = tl.load(
                idx_ptr + offs_r * stride_r_idx + j * stride_p_idx,
                mask=mask_r,
                other=0,
            )
            k_ptrs = k_ptr + (
                pid_bh * stride_bh_k
                + offs_n[:, None] * stride_n_k
                + idx[None, :] * stride_d_k
            )
            k_vals = tl.load(k_ptrs, mask=mask_n[:, None] & mask_r[None, :], other=1.0)
            mask_j = j < deg
            k_vals = tl.where(mask_j[None, :], k_vals, 1.0)
            psi *= k_vals.to(tl.float32)

        psi = psi * weight[None, :]

        v_ptrs = v_ptr + (
            pid_bh * stride_bh_v
            + offs_n[:, None] * stride_n_v
            + offs_d[None, :] * stride_d_v
        )
        v_block = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        acc_s += tl.dot(psi.T, v_block)
        acc_z += tl.sum(psi, axis=0)

        n += BLOCK_N

    s_ptrs = s_ptr + (
        pid_bh * stride_bh_s
        + offs_r[:, None] * stride_r_s
        + offs_d[None, :] * stride_d_s
    )
    tl.store(s_ptrs, acc_s, mask=mask_r[:, None] & mask_d[None, :])

    if pid_d == 0:
        z_ptrs = z_ptr + pid_bh * stride_bh_z + offs_r * stride_r_z
        tl.store(z_ptrs, acc_z, mask=mask_r)


@triton.jit
def _fused_out_kernel(
    q_ptr,
    idx_ptr,
    deg_ptr,
    w_ptr,
    s_ptr,
    z_ptr,
    out_ptr,
    den_ptr,
    stride_bh_q,
    stride_q_q,
    stride_d_q,
    stride_r_idx,
    stride_p_idx,
    stride_bh_s,
    stride_r_s,
    stride_d_s,
    stride_bh_z,
    stride_r_z,
    stride_bh_o,
    stride_q_o,
    stride_d_o,
    stride_bh_den,
    stride_q_den,
    Q,
    D,
    R,
    P_MAX: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_q = offs_q < Q
    mask_d = offs_d < D

    acc_num = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    acc_den = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    r = 0
    while r < R:
        offs_r = r + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R
        deg = tl.load(deg_ptr + offs_r, mask=mask_r, other=0).to(tl.int32)
        weight = tl.load(w_ptr + offs_r, mask=mask_r, other=0.0).to(tl.float32)

        psi = tl.full((BLOCK_Q, BLOCK_R), 1.0, dtype=tl.float32)
        for j in range(P_MAX):
            idx = tl.load(
                idx_ptr + offs_r * stride_r_idx + j * stride_p_idx,
                mask=mask_r,
                other=0,
            )
            q_ptrs = q_ptr + (
                pid_bh * stride_bh_q
                + offs_q[:, None] * stride_q_q
                + idx[None, :] * stride_d_q
            )
            q_vals = tl.load(q_ptrs, mask=mask_q[:, None] & mask_r[None, :], other=1.0)
            mask_j = j < deg
            q_vals = tl.where(mask_j[None, :], q_vals, 1.0)
            psi *= q_vals.to(tl.float32)

        psi = psi * weight[None, :]

        s_ptrs = s_ptr + (
            pid_bh * stride_bh_s
            + offs_r[:, None] * stride_r_s
            + offs_d[None, :] * stride_d_s
        )
        s_block = tl.load(s_ptrs, mask=mask_r[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        acc_num += tl.dot(psi, s_block)

        z_ptrs = z_ptr + pid_bh * stride_bh_z + offs_r * stride_r_z
        z_block = tl.load(z_ptrs, mask=mask_r, other=0.0)
        acc_den += tl.sum(psi * z_block[None, :], axis=1)

        r += BLOCK_R

    out_ptrs = out_ptr + (
        pid_bh * stride_bh_o
        + offs_q[:, None] * stride_q_o
        + offs_d[None, :] * stride_d_o
    )
    tl.store(out_ptrs, acc_num, mask=mask_q[:, None] & mask_d[None, :])

    if pid_d == 0:
        den_ptrs = den_ptr + pid_bh * stride_bh_den + offs_q * stride_q_den
        tl.store(den_ptrs, acc_den, mask=mask_q)


def fused_s_z(
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    degree: torch.Tensor,
    weight: torch.Tensor,
    s: torch.Tensor,
    z: torch.Tensor,
    block_n: int = 32,
    block_r: int = 32,
    block_d: int = 32,
) -> None:
    if not is_available(k.device):
        raise RuntimeError("triton_unavailable")
    if k.device != v.device or k.device != indices.device or k.device != degree.device or k.device != weight.device:
        raise ValueError("inputs must be on the same device")
    if k.ndim != 4 or v.ndim != 4 or indices.ndim != 2 or degree.ndim != 1 or weight.ndim != 1:
        raise ValueError("unexpected tensor ranks for fused_s_z")

    b, h, n_k, dim = k.shape
    _, _, _, d_val = v.shape
    r = indices.shape[0]
    if s.shape[:3] != (b, h, r) or s.shape[3] != d_val:
        raise ValueError("s tensor shape mismatch")
    if z.shape[:3] != (b, h, r):
        raise ValueError("z tensor shape mismatch")

    bh = b * h
    k_2d = k.reshape(bh, n_k, dim)
    v_2d = v.reshape(bh, n_k, d_val)
    s_2d = s.reshape(bh, r, d_val)
    z_2d = z.reshape(bh, r)

    grid = (
        bh,
        triton.cdiv(r, block_r),
        triton.cdiv(d_val, block_d),
    )

    _fused_s_z_kernel[grid](
        k_2d,
        v_2d,
        indices,
        degree,
        weight,
        s_2d,
        z_2d,
        k_2d.stride(0),
        k_2d.stride(1),
        k_2d.stride(2),
        v_2d.stride(0),
        v_2d.stride(1),
        v_2d.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.numel() > 0 else 0,
        s_2d.stride(0),
        s_2d.stride(1),
        s_2d.stride(2),
        z_2d.stride(0),
        z_2d.stride(1),
        n_k,
        d_val,
        r,
        P_MAX=indices.shape[1],
        BLOCK_N=block_n,
        BLOCK_R=block_r,
        BLOCK_D=block_d,
        num_warps=4,
    )


def fused_out(
    q: torch.Tensor,
    indices: torch.Tensor,
    degree: torch.Tensor,
    weight: torch.Tensor,
    s: torch.Tensor,
    z: torch.Tensor,
    out: torch.Tensor,
    den: torch.Tensor,
    block_q: int = 32,
    block_r: int = 32,
    block_d: int = 32,
) -> None:
    if not is_available(q.device):
        raise RuntimeError("triton_unavailable")
    if q.device != indices.device or q.device != degree.device or q.device != weight.device or q.device != s.device or q.device != z.device:
        raise ValueError("inputs must be on the same device")
    if q.ndim != 4 or indices.ndim != 2 or degree.ndim != 1 or weight.ndim != 1:
        raise ValueError("unexpected tensor ranks for fused_out")

    b, h, n_q, dim = q.shape
    r = indices.shape[0]
    d_val = s.shape[-1]
    if s.shape[:3] != (b, h, r):
        raise ValueError("s tensor shape mismatch")
    if z.shape[:3] != (b, h, r):
        raise ValueError("z tensor shape mismatch")
    if out.shape[:3] != (b, h, n_q) or out.shape[3] != d_val:
        raise ValueError("out tensor shape mismatch")
    if den.shape[:3] != (b, h, n_q):
        raise ValueError("den tensor shape mismatch")

    bh = b * h
    q_2d = q.reshape(bh, n_q, dim)
    s_2d = s.reshape(bh, r, d_val)
    z_2d = z.reshape(bh, r)
    out_2d = out.reshape(bh, n_q, d_val)
    den_2d = den.reshape(bh, n_q)

    grid = (
        bh,
        triton.cdiv(n_q, block_q),
        triton.cdiv(d_val, block_d),
    )

    _fused_out_kernel[grid](
        q_2d,
        indices,
        degree,
        weight,
        s_2d,
        z_2d,
        out_2d,
        den_2d,
        q_2d.stride(0),
        q_2d.stride(1),
        q_2d.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.numel() > 0 else 0,
        s_2d.stride(0),
        s_2d.stride(1),
        s_2d.stride(2),
        z_2d.stride(0),
        z_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        out_2d.stride(2),
        den_2d.stride(0),
        den_2d.stride(1),
        n_q,
        d_val,
        r,
        P_MAX=indices.shape[1],
        BLOCK_Q=block_q,
        BLOCK_R=block_r,
        BLOCK_D=block_d,
        num_warps=4,
    )
