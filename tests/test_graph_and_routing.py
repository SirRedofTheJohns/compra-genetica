from itertools import permutations

import networkx as nx
import numpy as np
import pytest

from compra_genetica.data import build_adjacency, floyd_warshall, load_catalog
from compra_genetica.routing import held_karp, nearest_neighbor, route_length, two_opt

from conftest import DATA_DIR

CATALOGS = sorted((DATA_DIR / "in").glob("small-*.xlsx"))


@pytest.mark.parametrize("path", CATALOGS, ids=lambda p: p.stem)
def test_floyd_warshall_matches_networkx(path):
    _, graph = load_catalog(path)
    nodes = sorted(set(graph["source"]) | set(graph["target"]))
    D = floyd_warshall(build_adjacency(graph, nodes))
    G = nx.DiGraph()
    for a, b, d in graph.itertuples(index=False):
        if a != b and (not G.has_edge(a, b) or G[a][b]["weight"] > d):
            G.add_edge(a, b, weight=d)
    expected = dict(nx.all_pairs_dijkstra_path_length(G))
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            assert D[i, j] == expected[a][b]


def test_held_karp_is_optimal_on_random_asymmetric_matrices():
    rng = np.random.default_rng(0)
    for _ in range(20):
        D = rng.integers(1, 50, size=(7, 7)).astype(float)
        np.fill_diagonal(D, 0)
        points = [1, 2, 3, 4, 5, 6]
        brute = min(route_length(list(p), D) for p in permutations(points))
        length, order = held_karp(points, D)
        assert length == pytest.approx(brute)
        assert route_length(order, D) == pytest.approx(length)
        assert sorted(order) == points


def test_two_opt_never_worsens_nearest_neighbor():
    rng = np.random.default_rng(1)
    D = rng.integers(1, 100, size=(15, 15)).astype(float)
    np.fill_diagonal(D, 0)
    nn = nearest_neighbor(list(range(1, 15)), D)
    assert route_length(two_opt(nn, D), D) <= route_length(nn, D)
