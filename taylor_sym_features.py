import itertools
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSpec:
    degree: int
    indices: torch.Tensor
    sqrt_w: torch.Tensor


@dataclass(frozen=True)
class FeatureTable:
    indices: torch.Tensor
    degree: torch.Tensor
    sqrt_w: torch.Tensor


_CPU_CACHE: Dict[Tuple[int, int], List[FeatureSpec]] = {}
_DEVICE_CACHE: Dict[Tuple[int, int, torch.device], List[FeatureSpec]] = {}
_TABLE_CACHE: Dict[Tuple[int, int, torch.device], FeatureTable] = {}
_BINOM_CACHE: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}


def feature_dim(d: int, P: int) -> int:
    if d <= 0:
        raise ValueError("d must be positive")
    if P <= 0:
        raise ValueError("P must be positive")
    return sum(math.comb(d + p - 1, p) for p in range(P))


def _feature_dim_degree(d: int, p: int) -> int:
    if p < 0:
        raise ValueError("p must be non-negative")
    if d <= 0:
        raise ValueError("d must be positive")
    return math.comb(d + p - 1, p)


def _build_degree_indices(d: int, p: int) -> torch.Tensor:
    if p == 0:
        return torch.empty((1, 0), dtype=torch.long)
    combos = list(itertools.combinations_with_replacement(range(d), p))
    return torch.tensor(combos, dtype=torch.long)


def _get_binom_table(n_max: int, k_max: int, device: torch.device) -> torch.Tensor:
    key = (n_max, k_max, device)
    cached = _BINOM_CACHE.get(key)
    if cached is not None:
        return cached
    table = torch.zeros((n_max + 1, k_max + 1), dtype=torch.long)
    for n in range(n_max + 1):
        max_k = min(n, k_max)
        for k in range(max_k + 1):
            table[n, k] = math.comb(n, k)
    table = table.to(device=device)
    _BINOM_CACHE[key] = table
    return table


def _unrank_multichoose(
    r: torch.Tensor,
    d: int,
    p: int,
    binom_table: torch.Tensor,
) -> torch.Tensor:
    # r: [count] on device, returns indices [count, p] with replacement
    device = r.device
    if p == 0:
        return torch.empty((r.numel(), 0), dtype=torch.long, device=device)
    n = d + p - 1
    xs = torch.arange(n, device=device, dtype=torch.long)
    batch = r.numel()
    comb = torch.zeros((batch, p), dtype=torch.long, device=device)
    start = torch.zeros((batch,), dtype=torch.long, device=device)
    r_work = r.clone()

    for i in range(p):
        k = p - i - 1
        if k == 0:
            counts = torch.ones((n,), dtype=torch.long, device=device)
        else:
            n_minus = n - xs - 1
            counts = torch.where(n_minus >= k, binom_table[n_minus, k], torch.zeros_like(n_minus))
        max_x = n - k - 1
        counts = torch.where(xs <= max_x, counts, torch.zeros_like(counts))
        counts_row = counts.unsqueeze(0).expand(batch, -1)
        mask = xs.unsqueeze(0) >= start.unsqueeze(1)
        counts_row = torch.where(mask, counts_row, torch.zeros_like(counts_row))
        prefix = torch.cumsum(counts_row, dim=1)
        idx = (prefix > r_work.unsqueeze(1)).float().argmax(dim=1)
        comb[:, i] = idx
        prev = torch.zeros_like(r_work)
        has_prev = idx > 0
        if has_prev.any():
            prev[has_prev] = prefix[torch.arange(batch, device=device), idx - 1][has_prev]
        r_work = r_work - prev
        start = idx + 1

    offsets = torch.arange(p, device=device, dtype=torch.long)
    return comb - offsets


def _compute_sqrt_w(indices: torch.Tensor, p: int) -> torch.Tensor:
    if p == 0:
        return torch.ones((indices.shape[0],), dtype=torch.float32, device=indices.device)
    batch = indices.shape[0]
    boundaries = torch.ones((batch, p), dtype=torch.bool, device=indices.device)
    boundaries[:, 1:] = indices[:, 1:] != indices[:, :-1]
    run_ids = boundaries.cumsum(dim=1) - 1
    counts = torch.zeros((batch, p), dtype=torch.long, device=indices.device)
    counts.scatter_add_(1, run_ids, torch.ones_like(run_ids))
    factorial = torch.tensor([math.factorial(i) for i in range(p + 1)], dtype=torch.float32, device=indices.device)
    denom = factorial[counts].prod(dim=1)
    weight = (float(math.factorial(p)) / denom).to(dtype=torch.float32)
    return torch.sqrt(weight)


def iter_feature_specs_streaming(d: int, P: int, device: torch.device, chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    n_max = d + P - 1
    binom_table = _get_binom_table(n_max, P, device)
    for p in range(P):
        total = _feature_dim_degree(d, p)
        if total == 0:
            continue
        for start in range(0, total, chunk_size):
            count = min(chunk_size, total - start)
            r = torch.arange(start, start + count, device=device, dtype=torch.long)
            indices = _unrank_multichoose(r, d, p, binom_table)
            sqrt_w = _compute_sqrt_w(indices, p)
            yield FeatureSpec(degree=p, indices=indices, sqrt_w=sqrt_w)


def _build_degree_weights(indices: torch.Tensor, p: int) -> torch.Tensor:
    if p == 0:
        return torch.ones((1,), dtype=torch.float32)
    weights = []
    for row in indices.tolist():
        counts = {}
        for idx in row:
            counts[idx] = counts.get(idx, 0) + 1
        w = math.factorial(p)
        for c in counts.values():
            w //= math.factorial(c)
        weights.append(w)
    w_tensor = torch.tensor(weights, dtype=torch.float32)
    return torch.sqrt(w_tensor)


def _build_cpu_specs(d: int, P: int) -> List[FeatureSpec]:
    specs: List[FeatureSpec] = []
    for p in range(P):
        indices = _build_degree_indices(d, p)
        sqrt_w = _build_degree_weights(indices, p)
        specs.append(FeatureSpec(degree=p, indices=indices, sqrt_w=sqrt_w))
    return specs


def get_feature_specs(d: int, P: int, device: torch.device) -> List[FeatureSpec]:
    key = (d, P, device)
    if key in _DEVICE_CACHE:
        return _DEVICE_CACHE[key]

    cpu_key = (d, P)
    if cpu_key not in _CPU_CACHE:
        logger.info("Building Taylor feature specs for d=%s P=%s", d, P)
        _CPU_CACHE[cpu_key] = _build_cpu_specs(d, P)

    specs_cpu = _CPU_CACHE[cpu_key]
    specs_device: List[FeatureSpec] = []
    for spec in specs_cpu:
        specs_device.append(
            FeatureSpec(
                degree=spec.degree,
                indices=spec.indices.to(device=device, non_blocking=True),
                sqrt_w=spec.sqrt_w.to(device=device, non_blocking=True),
            )
        )
    _DEVICE_CACHE[key] = specs_device
    return specs_device


def get_feature_table(d: int, P: int, device: torch.device) -> FeatureTable:
    key = (d, P, device)
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    cpu_key = (d, P)
    if cpu_key not in _CPU_CACHE:
        logger.info("Building Taylor feature specs for d=%s P=%s", d, P)
        _CPU_CACHE[cpu_key] = _build_cpu_specs(d, P)

    specs_cpu = _CPU_CACHE[cpu_key]
    p_max = max(P - 1, 0)
    indices_list = []
    degree_list = []
    sqrt_w_list = []
    for spec in specs_cpu:
        m = spec.indices.shape[0]
        if p_max > 0:
            padded = torch.zeros((m, p_max), dtype=torch.int32)
            if spec.indices.numel() > 0:
                padded[:, : spec.indices.shape[1]] = spec.indices.to(dtype=torch.int32)
        else:
            padded = torch.empty((m, 0), dtype=torch.int32)
        indices_list.append(padded)
        degree_list.append(torch.full((m,), spec.degree, dtype=torch.int16))
        sqrt_w_list.append(spec.sqrt_w)

    indices = torch.cat(indices_list, dim=0).to(device=device, non_blocking=True)
    degree = torch.cat(degree_list, dim=0).to(device=device, non_blocking=True)
    sqrt_w = torch.cat(sqrt_w_list, dim=0).to(device=device, non_blocking=True)
    table = FeatureTable(indices=indices, degree=degree, sqrt_w=sqrt_w)
    _TABLE_CACHE[key] = table
    return table


def eval_phi(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.shape[1] == 0:
        return torch.ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)
    gathered = x[..., indices]
    return gathered.prod(dim=-1)
