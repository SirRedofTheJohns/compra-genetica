"""Genetic operators: initialization, selection, crossover, mutation, replacement."""
from __future__ import annotations

import numpy as np

from .fitness import ShoppingProblem


# ---------------------------------------------------------------- initialization

def init_random(problem: ShoppingProblem, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    """Each gene takes a random store among the ones that sell that product."""
    return np.array([[rng.choice(a) for a in problem.allowed] for _ in range(pop_size)])


def init_seeded(problem: ShoppingProblem, pop_size: int, rng: np.random.Generator,
                seeds: list[np.ndarray]) -> np.ndarray:
    """Random population that also contains some known good chromosomes (e.g. greedy)."""
    population = init_random(problem, pop_size, rng)
    for i, seed in enumerate(seeds[:pop_size]):
        population[i] = seed
    return population


# ---------------------------------------------------------------- selection
# Every selection returns the indices of the chosen parents. Lower cost = better.

def selection_tournament(costs: np.ndarray, n: int, rng: np.random.Generator,
                         k: int = 3) -> np.ndarray:
    """Pick k individuals at random; the cheapest one wins. Repeat n times."""
    contenders = rng.integers(0, len(costs), size=(n, k))
    return contenders[np.arange(n), costs[contenders].argmin(axis=1)]


def selection_roulette(costs: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Fitness-proportional selection with fitness = 1 / cost."""
    fitness = 1.0 / costs
    return rng.choice(len(costs), size=n, p=fitness / fitness.sum())


def selection_rank(costs: np.ndarray, n: int, rng: np.random.Generator,
                   pressure: float = 1.8) -> np.ndarray:
    """Linear ranking: probability depends on the position, not on the cost value.

    The best gets ``pressure / N`` and the worst ``(2 - pressure) / N``.
    """
    N = len(costs)
    ranks = np.empty(N)
    ranks[costs.argsort()[::-1]] = np.arange(N)          # worst -> 0, best -> N-1
    probs = ((2 - pressure) + 2 * (pressure - 1) * ranks / max(N - 1, 1)) / N
    return rng.choice(N, size=n, p=probs / probs.sum())


# ---------------------------------------------------------------- crossover

def crossover_one_point(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator):
    """Cut both parents at the same random point and swap the tails."""
    if len(p1) < 2:
        return p1.copy(), p2.copy()
    cut = rng.integers(1, len(p1))
    return (np.concatenate([p1[:cut], p2[cut:]]),
            np.concatenate([p2[:cut], p1[cut:]]))


def crossover_two_point(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator):
    """Swap the middle segment between two random cut points."""
    if len(p1) < 3:
        return crossover_one_point(p1, p2, rng)
    a, b = sorted(rng.choice(np.arange(1, len(p1)), size=2, replace=False))
    c1, c2 = p1.copy(), p2.copy()
    c1[a:b], c2[a:b] = p2[a:b], p1[a:b]
    return c1, c2


def crossover_uniform(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator):
    """Each gene comes from either parent with probability 0.5."""
    mask = rng.random(len(p1)) < 0.5
    return np.where(mask, p1, p2), np.where(mask, p2, p1)


# ---------------------------------------------------------------- mutation
# Both mutations keep genes valid: a product is only moved to a store that sells it.

def mutation_random_reset(chromosome: np.ndarray, problem: ShoppingProblem,
                          rng: np.random.Generator, pm: float) -> np.ndarray:
    """With probability pm, a gene jumps to any store that sells the product."""
    child = chromosome.copy()
    for i in np.flatnonzero(rng.random(len(child)) < pm):
        child[i] = rng.choice(problem.allowed[i])
    return child


def mutation_consolidate(chromosome: np.ndarray, problem: ShoppingProblem,
                         rng: np.random.Generator, pm: float) -> np.ndarray:
    """With probability pm, move a product to a store the route already visits.

    Heuristic mutation: fewer distinct stores means a shorter trip. If no other
    visited store sells the product, it falls back to a random reset.
    """
    child = chromosome.copy()
    for i in np.flatnonzero(rng.random(len(child)) < pm):
        visited = set(np.delete(child, i).tolist())
        options = [s for s in problem.allowed[i] if s in visited and s != child[i]]
        child[i] = rng.choice(options) if options else rng.choice(problem.allowed[i])
    return child


# ---------------------------------------------------------------- replacement

def replacement_elitism(population: np.ndarray, costs: np.ndarray,
                        offspring: np.ndarray, offspring_costs: np.ndarray,
                        n_elite: int) -> tuple[np.ndarray, np.ndarray]:
    """Generational replacement that keeps the n_elite best parents untouched."""
    elite = costs.argsort()[:n_elite]
    keep = offspring_costs.argsort()[:len(population) - n_elite]
    return (np.vstack([population[elite], offspring[keep]]),
            np.concatenate([costs[elite], offspring_costs[keep]]))


# ---------------------------------------------------------------- termination

def should_stop(generation: int, max_generations: int, stall: int, patience: int) -> bool:
    """Stop at the generation limit or after ``patience`` generations without improvement."""
    return generation >= max_generations or stall >= patience
