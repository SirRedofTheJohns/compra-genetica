"""Reference algorithms to compare the GA against."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .fitness import ShoppingProblem


def greedy_cheapest(problem: ShoppingProblem) -> np.ndarray:
    """Buy every product where it is cheapest, ignoring distances."""
    return problem.price.argmin(axis=0)


def greedy_nearest(problem: ShoppingProblem) -> np.ndarray:
    """From the current position, go to the closest store that sells something
    still pending and buy there everything it can. Ignores prices."""
    chromosome = np.full(problem.k, -1)
    D, current = problem.instance.D, 0
    while (pending := np.flatnonzero(chromosome < 0)).size:
        candidates = {s for i in pending for s in problem.allowed[i]}
        store = min(candidates, key=lambda s: D[current, s + 1])
        for i in pending:
            if np.isfinite(problem.price[store, i]):
                chromosome[i] = store
        current = store + 1
    return chromosome


def exact_search(problem: ShoppingProblem, max_stores: int = 14) -> tuple[np.ndarray, float]:
    """True optimum by enumerating every subset of candidate stores.

    For a fixed subset the best purchase is the cheapest store of the subset for
    each product, and the route is solved exactly. Since distances already are
    shortest paths, visiting fewer stores is never longer, so the minimum over
    all subsets is the global optimum. Cost grows as 2^S: small instances only.
    """
    candidates = sorted({int(s) for a in problem.allowed for s in a})
    if len(candidates) > max_stores:
        raise ValueError(f"{len(candidates)} candidate stores: too many for exhaustive search")
    best, best_cost = None, np.inf
    for size in range(1, len(candidates) + 1):
        for subset in combinations(candidates, size):
            sub_prices = problem.price[list(subset)]
            if not np.isfinite(sub_prices.min(axis=0)).all():
                continue
            chromosome = np.array(subset)[sub_prices.argmin(axis=0)]
            cost = problem.product_cost(chromosome) + problem.transport_cost(chromosome)
            if cost < best_cost:
                best, best_cost = chromosome, cost
    return best, float(best_cost)
