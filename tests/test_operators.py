"""Every operator must keep chromosomes valid (each product bought where it is sold)."""
import numpy as np
import pytest

from compra_genetica import GeneticShoppingOptimizer, ShoppingProblem, load_instance
from compra_genetica import operators as op

from conftest import DATA_DIR


@pytest.fixture(scope="module")
def problem():
    inst = load_instance(DATA_DIR / "in" / "small-5s-10p-f0_10-q1_10-punico_catalog.xlsx")
    return ShoppingProblem(inst, inst.products, fuel_price=5)


def is_valid(chromosome, problem):
    return all(g in problem.allowed[i] for i, g in enumerate(chromosome))


def test_initial_population_is_valid(problem):
    pop = op.init_random(problem, 50, np.random.default_rng(0))
    assert pop.shape == (50, problem.k)
    assert all(is_valid(ind, problem) for ind in pop)


@pytest.mark.parametrize("crossover", [op.crossover_one_point, op.crossover_two_point,
                                       op.crossover_uniform])
def test_crossover_keeps_genes_per_position(problem, crossover):
    rng = np.random.default_rng(0)
    p1, p2 = op.init_random(problem, 2, rng)
    c1, c2 = crossover(p1, p2, rng)
    assert all(a in (x, y) and b in (x, y) for a, b, x, y in zip(c1, c2, p1, p2))
    assert is_valid(c1, problem) and is_valid(c2, problem)


@pytest.mark.parametrize("mutation", [op.mutation_random_reset, op.mutation_consolidate])
def test_mutation_keeps_validity(problem, mutation):
    rng = np.random.default_rng(0)
    for ind in op.init_random(problem, 30, rng):
        assert is_valid(mutation(ind, problem, rng, pm=0.5), problem)


@pytest.mark.parametrize("selection", [op.selection_tournament, op.selection_roulette,
                                       op.selection_rank])
def test_selection_prefers_cheaper_individuals(selection):
    costs = np.array([100.0, 200.0, 300.0, 400.0])
    picks = selection(costs, 10_000, np.random.default_rng(0))
    counts = np.bincount(picks, minlength=4)
    assert counts[0] > counts[3]


def test_elitism_keeps_the_best(problem):
    rng = np.random.default_rng(0)
    pop = op.init_random(problem, 10, rng)
    costs = problem.evaluate(pop)
    off = op.init_random(problem, 10, rng)
    _, new_costs = op.replacement_elitism(pop, costs, off, problem.evaluate(off) + 1e9, 2)
    assert np.allclose(np.sort(new_costs)[:2], np.sort(costs)[:2])


def test_optimizer_is_reproducible(problem):
    a = GeneticShoppingOptimizer(problem, seed=42, seeding=False).run()
    b = GeneticShoppingOptimizer(problem, seed=42, seeding=False).run()
    assert a.best_cost == b.best_cost and a.generations == b.generations
