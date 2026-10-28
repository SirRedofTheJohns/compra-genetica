"""The cost model must reproduce every reference solution of the validation dataset."""
import numpy as np
import pytest

from compra_genetica import (ShoppingProblem, evaluate_sequence, exact_search,
                             load_instance, load_problemset, parse_solution)

from conftest import DATA_DIR

ALGORITHMS = ["qbf", "ga", "gc", "gn"]


@pytest.fixture(scope="module")
def problemset():
    return load_problemset(DATA_DIR)


@pytest.fixture(scope="module")
def instances(problemset):
    return {p: load_instance(p) for p in problemset["catalog_path"].unique()}


def test_cost_model_matches_all_reference_solutions(problemset, instances):
    checked = 0
    for _, row in problemset.iterrows():
        inst = instances[row.catalog_path]
        for alg in ALGORITHMS:
            solution, value = parse_solution(row[f"{alg} best solution"]), row[f"{alg} best value"]
            if solution is None or np.isnan(value):
                continue
            assert evaluate_sequence(inst, solution, row.fuel_price) == pytest.approx(value)
            checked += 1
    assert checked == 504


def test_exact_search_is_never_worse_than_references(problemset, instances):
    small = problemset[problemset["problem_cluster"].str.startswith("small")]
    for _, row in small.iterrows():
        problem = ShoppingProblem(instances[row.catalog_path], row.query, row.fuel_price)
        _, cost = exact_search(problem)
        assert cost <= row.best_known + 1e-6


def test_decoded_answer_buys_every_product(problemset, instances):
    row = problemset.iloc[1]
    problem = ShoppingProblem(instances[row.catalog_path], row.query, row.fuel_price)
    chromosome, cost = exact_search(problem)
    bought = [p for _, items in problem.decode(chromosome) for p in items]
    assert sorted(bought) == sorted(row.query)
    assert cost == pytest.approx(11270.96)
    assert problem.format_answer(chromosome).startswith("RESPUESTA> ")
