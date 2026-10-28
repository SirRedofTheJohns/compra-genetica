"""The genetic algorithm: one method per step, hyperparameters in the constructor."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pandas as pd

from .baselines import greedy_cheapest, greedy_nearest
from .fitness import ShoppingProblem
from .operators import (crossover_one_point, crossover_two_point, crossover_uniform,
                        init_random, init_seeded, mutation_consolidate,
                        mutation_random_reset, replacement_elitism, selection_rank,
                        selection_roulette, selection_tournament, should_stop)


@dataclass
class GAResult:
    best_chromosome: np.ndarray
    best_cost: float
    answer: str
    generations: int
    evaluations: int
    elapsed: float
    history: pd.DataFrame = field(repr=False)


class GeneticShoppingOptimizer:
    """Genetic algorithm that minimizes product cost + fuel cost of a shopping trip.

    Hyperparameters
    ---------------
    pop_size      individuals per generation
    generations   maximum number of generations
    pc            crossover probability
    pm            per-gene mutation probability (default 1 / k)
    selection     'tournament' | 'roulette' | 'rank'
    crossover     'one_point' | 'two_point' | 'uniform'
    mutation      'random_reset' | 'consolidate'
    tournament_k  tournament size
    rank_pressure selective pressure for rank selection (1..2)
    elitism       how many of the best parents survive untouched
    patience      stop after this many generations without improvement
    seeding       put the two greedy solutions in the initial population
    seed          random seed (reproducibility)
    """

    CROSSOVERS = {"one_point": crossover_one_point, "two_point": crossover_two_point,
                  "uniform": crossover_uniform}
    MUTATIONS = {"random_reset": mutation_random_reset, "consolidate": mutation_consolidate}

    def __init__(self, problem: ShoppingProblem, pop_size: int = 80, generations: int = 200,
                 pc: float = 0.9, pm: float | None = None, selection: str = "tournament",
                 crossover: str = "uniform", mutation: str = "random_reset",
                 tournament_k: int = 3, rank_pressure: float = 1.8, elitism: int = 2,
                 patience: int = 50, seeding: bool = True, seed: int | None = None):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.pc = pc
        self.pm = pm if pm is not None else 1.0 / problem.k
        self.selection_name, self.crossover_name, self.mutation_name = selection, crossover, mutation
        self.selections = {"tournament": partial(selection_tournament, k=tournament_k),
                           "roulette": selection_roulette,
                           "rank": partial(selection_rank, pressure=rank_pressure)}
        self.elitism = elitism
        self.patience = patience
        self.seeding = seeding
        self.rng = np.random.default_rng(seed)

    # -- one method per step -------------------------------------------------

    def initialize(self) -> np.ndarray:
        if self.seeding:
            seeds = [greedy_cheapest(self.problem), greedy_nearest(self.problem)]
            return init_seeded(self.problem, self.pop_size, self.rng, seeds)
        return init_random(self.problem, self.pop_size, self.rng)

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        return self.problem.evaluate(population)

    def select(self, costs: np.ndarray, n: int) -> np.ndarray:
        return self.selections[self.selection_name](costs, n, self.rng)

    def crossover(self, p1: np.ndarray, p2: np.ndarray):
        if self.rng.random() < self.pc:
            return self.CROSSOVERS[self.crossover_name](p1, p2, self.rng)
        return p1.copy(), p2.copy()

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        return self.MUTATIONS[self.mutation_name](chromosome, self.problem, self.rng, self.pm)

    def replace(self, population, costs, offspring, offspring_costs):
        return replacement_elitism(population, costs, offspring, offspring_costs, self.elitism)

    def should_stop(self, generation: int, stall: int) -> bool:
        return should_stop(generation, self.generations, stall, self.patience)

    # -- main loop -----------------------------------------------------------

    def run(self, verbose: bool = False) -> GAResult:
        start = time.perf_counter()
        self.problem.evaluations = 0
        population = self.initialize()
        costs = self.evaluate(population)
        best_idx = costs.argmin()
        best, best_cost = population[best_idx].copy(), costs[best_idx]
        history, generation, stall = [], 0, 0

        while True:
            history.append({"generation": generation, "best": best_cost,
                            "mean": costs.mean(), "worst": costs.max(),
                            "diversity": len(np.unique(population, axis=0)) / len(population)})
            if verbose and generation % 10 == 0:
                print(f"gen {generation:4d} | best {best_cost:12.2f} | mean {costs.mean():12.2f}")
            if self.should_stop(generation, stall):
                break

            parents = self.select(costs, self.pop_size + self.pop_size % 2)
            offspring = []
            for a, b in zip(parents[::2], parents[1::2]):
                for child in self.crossover(population[a], population[b]):
                    offspring.append(self.mutate(child))
            offspring = np.array(offspring)
            population, costs = self.replace(population, costs, offspring, self.evaluate(offspring))

            generation += 1
            if costs.min() < best_cost - 1e-9:
                best_idx = costs.argmin()
                best, best_cost, stall = population[best_idx].copy(), costs[best_idx], 0
            else:
                stall += 1

        return GAResult(best, float(best_cost), self.problem.format_answer(best), generation,
                        self.problem.evaluations, time.perf_counter() - start,
                        pd.DataFrame(history))
