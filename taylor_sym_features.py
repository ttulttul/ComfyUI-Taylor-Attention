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


_CPU_CACHE: Dict[Tuple[int, int], List[FeatureSpec]] = {}
_DEVICE_CACHE: Dict[Tuple[int, int, torch.device], List[FeatureSpec]] = {}


def feature_dim(d: int, P: int) -> int:
    if d <= 0:
        raise ValueError("d must be positive")
    if P <= 0:
        raise ValueError("P must be positive")
    return sum(math.comb(d + p - 1, p) for p in range(P))


def _build_degree_indices(d: int, p: int) -> torch.Tensor:
    if p == 0:
        return torch.empty((1, 0), dtype=torch.long)
    combos = list(itertools.combinations_with_replacement(range(d), p))
    return torch.tensor(combos, dtype=torch.long)


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


def eval_phi(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.shape[1] == 0:
        return torch.ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)
    gathered = x[..., indices]
    return gathered.prod(dim=-1)
