"""Route heuristics: in which order should we visit a given set of stores?

This is a small asymmetric TSP that starts and ends at home (point 0).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def route_length(order: list[int], D: np.ndarray) -> float:
    """Length of the closed route 0 -> order[0] -> ... -> order[-1] -> 0."""
    path = [0, *order, 0]
    return float(D[path[:-1], path[1:]].sum())


def held_karp(points: list[int], D: np.ndarray) -> tuple[float, list[int]]:
    """Exact TSP by dynamic programming over subsets: O(2^n * n^2).

    dp[mask, j] = shortest path that leaves home, visits exactly the points in
    ``mask`` and ends at point j.
    """
    n = len(points)
    if n == 0:
        return 0.0, []
    sub = D[np.ix_(points, points)]
    full = 1 << n
    dp = np.full((full, n), np.inf)
    parent = np.full((full, n), -1, dtype=int)
    for j in range(n):
        dp[1 << j, j] = D[0, points[j]]
    bits = 1 << np.arange(n)
    for mask in range(1, full):
        cand = dp[mask][:, None] + sub              # cand[k, j]: arrive at j coming from k
        best_k = cand.argmin(axis=0)
        best = cand[best_k, np.arange(n)]
        free = np.flatnonzero((mask & bits) == 0)   # points not visited yet
        new_masks = mask | bits[free]
        better = best[free] < dp[new_masks, free]
        dp[new_masks[better], free[better]] = best[free[better]]
        parent[new_masks[better], free[better]] = best_k[free[better]]
    totals = dp[full - 1] + D[points, 0]
    last = int(totals.argmin())
    order, mask = [], full - 1
    while last != -1:
        order.append(points[last])
        prev = parent[mask, last]
        mask ^= 1 << last
        last = prev
    return float(totals.min()), order[::-1]


def nearest_neighbor(points: list[int], D: np.ndarray) -> list[int]:
    """Greedy route: always go to the closest point not visited yet."""
    pending, order, current = set(points), [], 0
    while pending:
        current = min(pending, key=lambda p: D[current, p])
        order.append(current)
        pending.remove(current)
    return order


def two_opt(order: list[int], D: np.ndarray) -> list[int]:
    """Local search: reverse segments while the route gets shorter."""
    best, best_len = list(order), route_length(order, D)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                length = route_length(candidate, D)
                if length < best_len - 1e-9:
                    best, best_len, improved = candidate, length, True
    return best


class RouteSolver:
    """Finds the visiting order for a set of stores, with memoization.

    Uses Held-Karp (exact) up to ``exact_limit`` stores and nearest neighbor +
    2-opt above that. Stores are given as store indices; point = store + 1.
    """

    def __init__(self, D: np.ndarray, exact_limit: int = 9):
        self.D = D
        self.exact_limit = exact_limit
        self.cache: dict[frozenset, tuple[float, list[int]]] = {}
        self.hits = 0
        self.misses = 0

    def solve(self, stores: Iterable[int]) -> tuple[float, list[int]]:
        key = frozenset(int(s) for s in stores)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        points = sorted(s + 1 for s in key)
        if len(points) <= self.exact_limit:
            length, order = held_karp(points, self.D)
        else:
            order = two_opt(nearest_neighbor(points, self.D), self.D)
            length = route_length(order, self.D)
        result = (length, [p - 1 for p in order])
        self.cache[key] = result
        return result
