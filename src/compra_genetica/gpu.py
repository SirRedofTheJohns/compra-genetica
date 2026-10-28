"""Vectorized batch fitness on CPU (NumPy) and GPU (PyTorch) for the CPU vs GPU experiment.

To be vectorizable the route is the gene order itself (stores visited in the
order products appear, like the reference solutions), not the TSP decoder.
"""
from __future__ import annotations

import numpy as np


def batch_cost_numpy(population: np.ndarray, price: np.ndarray, D: np.ndarray,
                     fuel_price: float) -> np.ndarray:
    k = population.shape[1]
    products = price[population, np.arange(k)].sum(axis=1)
    points = np.pad(population + 1, ((0, 0), (1, 1)))            # home ... home
    distance = D[points[:, :-1], points[:, 1:]].sum(axis=1)
    return products + fuel_price * distance


def batch_cost_torch(population, price, D, fuel_price: float):
    import torch
    k = population.shape[1]
    products = price[population, torch.arange(k, device=population.device)].sum(dim=1)
    points = torch.nn.functional.pad(population + 1, (1, 1))
    distance = D[points[:, :-1], points[:, 1:]].sum(dim=1)
    return products + fuel_price * distance
