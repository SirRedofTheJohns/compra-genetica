"""Fitness: total cost = product prices + fuel price x route distance."""
from __future__ import annotations

import numpy as np

from .data import ShoppingInstance
from .routing import RouteSolver, route_length


class ShoppingProblem:
    """One shopping query on one catalog.

    Gene syntax: a chromosome is an integer vector of length k (one gene per
    product in the query); gene i is the index of the store where product i is
    bought. Only stores that sell product i are allowed, so every chromosome is
    a valid purchase.
    """

    def __init__(self, instance: ShoppingInstance, query: list[str], fuel_price: float,
                 exact_limit: int = 9):
        missing = [p for p in query if p not in instance.products]
        if missing:
            raise ValueError(f"Products not found in the catalog: {missing}")
        self.instance = instance
        self.query = list(query)
        self.fuel_price = float(fuel_price)
        self.k = len(query)
        cols = [instance.products.index(p) for p in query]
        self.price = instance.price[:, cols]                       # (n_stores, k)
        self.allowed = [np.flatnonzero(np.isfinite(self.price[:, i])) for i in range(self.k)]
        if any(len(a) == 0 for a in self.allowed):
            raise ValueError("Some product is not sold by any store")
        self.router = RouteSolver(instance.D, exact_limit)
        self.evaluations = 0

    def product_cost(self, chromosome: np.ndarray) -> float:
        return float(self.price[chromosome, np.arange(self.k)].sum())

    def transport_cost(self, chromosome: np.ndarray) -> float:
        length, _ = self.router.solve(chromosome)
        return self.fuel_price * length

    def cost(self, chromosome: np.ndarray) -> float:
        """Objective to minimize."""
        self.evaluations += 1
        return self.product_cost(chromosome) + self.transport_cost(chromosome)

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """Cost of every chromosome in the population (one row per individual)."""
        return np.array([self.cost(ind) for ind in population])

    def decode(self, chromosome: np.ndarray) -> list[tuple[str, list[str]]]:
        """Phenotype: the stores in visiting order with what to buy in each one."""
        _, order = self.router.solve(chromosome)
        stores = self.instance.stores
        return [(stores[s], [self.query[i] for i in range(self.k) if chromosome[i] == s])
                for s in order]

    def format_answer(self, chromosome: np.ndarray) -> str:
        parts = []
        for store, items in self.decode(chromosome):
            parts.append(f"({store}, {items[0]})" if len(items) == 1
                         else f"({store}, [{', '.join(items)}])")
        return "RESPUESTA> " + " -> ".join(parts)


def evaluate_sequence(instance: ShoppingInstance, sequence: list[tuple[str, str]],
                      fuel_price: float) -> float:
    """Cost of an explicit (store, product) sequence visited in the given order.

    Used to check our cost model against the reference solutions.
    """
    s_idx = {s: i for i, s in enumerate(instance.stores)}
    p_idx = {p: i for i, p in enumerate(instance.products)}
    products = sum(instance.price[s_idx[s], p_idx[p]] for s, p in sequence)
    route = [s_idx[s] + 1 for s, _ in sequence]
    return float(products + fuel_price * route_length(route, instance.D))
