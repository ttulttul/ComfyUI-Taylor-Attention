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
    out_prev = tl.load(out_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0)
    out_val = out_prev + acc_num
    tl.store(out_ptrs, out_val, mask=mask_q[:, None] & mask_d[None, :])

    if pid_d == 0:
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
        num_warps=4,
    )

