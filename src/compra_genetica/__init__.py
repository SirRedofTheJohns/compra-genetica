"""Compra Genética: shopping-route optimization with genetic algorithms."""
from .baselines import exact_search, greedy_cheapest, greedy_nearest
from .data import ShoppingInstance, load_instance, load_problemset, parse_solution
from .fitness import ShoppingProblem, evaluate_sequence
from .metrics import error_columns, summary_metrics
from .optimizer import GAResult, GeneticShoppingOptimizer

__all__ = ["ShoppingInstance", "ShoppingProblem", "GeneticShoppingOptimizer", "GAResult",
           "load_instance", "load_problemset", "parse_solution", "evaluate_sequence",
           "greedy_cheapest", "greedy_nearest", "exact_search", "error_columns",
           "summary_metrics"]
