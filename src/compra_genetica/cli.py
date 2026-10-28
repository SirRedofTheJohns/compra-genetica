"""Command line entry point.

Example:
    compra-genetica --catalog data/in/small-3s-4p-f1_10-q1_4-random_catalog.xlsx \
        --products "Purificadores de aire,Aceite de girasol,Pasta de dientes,Alfombras" --fuel 1
"""
import argparse

from .data import load_instance
from .fitness import ShoppingProblem
from .optimizer import GeneticShoppingOptimizer


def main() -> None:
    parser = argparse.ArgumentParser(prog="compra-genetica", description="Optimal shopping route with a genetic algorithm")
    parser.add_argument("--catalog", required=True, help="Excel file: sheet 1 catalog, sheet 2 graph edges")
    parser.add_argument("--products", required=True, help="comma-separated shopping list")
    parser.add_argument("--fuel", type=float, required=True, help="fuel price per distance unit")
    parser.add_argument("--pop-size", type=int, default=80)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--pc", type=float, default=0.9)
    parser.add_argument("--pm", type=float, default=None)
    parser.add_argument("--selection", default="tournament", choices=["tournament", "roulette", "rank"])
    parser.add_argument("--crossover", default="uniform", choices=["one_point", "two_point", "uniform"])
    parser.add_argument("--mutation", default="random_reset", choices=["random_reset", "consolidate"])
    parser.add_argument("--elitism", type=int, default=2)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    query = [p.strip() for p in args.products.split(",") if p.strip()]
    problem = ShoppingProblem(load_instance(args.catalog), query, args.fuel)
    result = GeneticShoppingOptimizer(
        problem, pop_size=args.pop_size, generations=args.generations, pc=args.pc, pm=args.pm,
        selection=args.selection, crossover=args.crossover, mutation=args.mutation,
        elitism=args.elitism, patience=args.patience, seed=args.seed).run(verbose=args.verbose)

    print(result.answer)
    print(f"Total cost: {result.best_cost:.2f}  "
          f"(products {problem.product_cost(result.best_chromosome):.2f} + "
          f"transport {problem.transport_cost(result.best_chromosome):.2f})")
    print(f"{result.generations} generations, {result.evaluations} evaluations, {result.elapsed:.2f} s")


if __name__ == "__main__":
    main()
