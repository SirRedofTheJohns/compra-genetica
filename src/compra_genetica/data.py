"""Data loading: catalogs, store graph, shortest paths and the validation problemset."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HOME = "Casa"


@dataclass
class ShoppingInstance:
    """Everything the optimizer needs to know about one catalog file.

    Point indexing used by ``D``: point 0 is the home, point ``s + 1`` is store ``s``.
    """
    stores: list[str]          # store names; the index is the gene value
    products: list[str]        # every product that appears in the catalog
    price: np.ndarray          # (n_stores, n_products) price, np.inf when not sold
    nodes: list[str]           # every node of the raw graph (may include non-store nodes)
    node_dist: np.ndarray      # (n_nodes, n_nodes) shortest directed distances
    D: np.ndarray              # (n_stores + 1, n_stores + 1) distances between home and stores
    catalog: pd.DataFrame      # raw "catalogo" sheet
    graph: pd.DataFrame        # raw "grafo" sheet


def load_catalog(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two sheets of a catalog file: product catalog and graph edges."""
    catalog = pd.read_excel(path, sheet_name=0)
    graph = pd.read_excel(path, sheet_name=1)
    catalog.columns = ["store", "product", "price"]
    graph.columns = ["source", "target", "distance"]
    return catalog, graph


def build_price_matrix(catalog: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray]:
    """Pivot the catalog into a stores x products matrix (np.inf = not sold)."""
    stores = sorted(catalog["store"].unique())
    products = sorted(catalog["product"].unique())
    table = catalog.pivot_table(index="store", columns="product", values="price", aggfunc="min")
    price = table.reindex(index=stores, columns=products).to_numpy(dtype=float)
    price[np.isnan(price)] = np.inf
    return stores, products, price


def build_adjacency(graph: pd.DataFrame, nodes: list[str], directed: bool = True) -> np.ndarray:
    """Adjacency matrix with np.inf where there is no edge.

    Self-loops are ignored (staying in the same place costs 0) and repeated
    edges keep the shortest one.
    """
    idx = {name: i for i, name in enumerate(nodes)}
    A = np.full((len(nodes), len(nodes)), np.inf)
    np.fill_diagonal(A, 0.0)
    for a, b, d in graph.itertuples(index=False):
        if a == b:
            continue
        i, j = idx[a], idx[b]
        A[i, j] = min(A[i, j], d)
        if not directed:
            A[j, i] = min(A[j, i], d)
    return A


def floyd_warshall(A: np.ndarray) -> np.ndarray:
    """All-pairs shortest paths. Turns any connected graph into a complete one.

    D[i, j] = min(D[i, j], D[i, k] + D[k, j]) for every intermediate node k,
    vectorized over (i, j) with NumPy broadcasting.
    """
    D = A.copy()
    for k in range(len(D)):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D


def load_instance(path: str | Path, directed: bool = True) -> ShoppingInstance:
    """Load a catalog file and precompute prices and shortest-path distances."""
    catalog, graph = load_catalog(path)
    stores, products, price = build_price_matrix(catalog)
    nodes = sorted(set(graph["source"]) | set(graph["target"]) | set(stores) | {HOME})
    node_dist = floyd_warshall(build_adjacency(graph, nodes, directed))
    points = [nodes.index(HOME)] + [nodes.index(s) for s in stores]
    D = node_dist[np.ix_(points, points)]
    return ShoppingInstance(stores, products, price, nodes, node_dist, D, catalog, graph)


def resolve_catalog_path(catalog: str, data_dir: str | Path) -> Path:
    """The problemset stores paths like 'problemset/in/X.xlsx'; the files live in data/in/."""
    return Path(data_dir) / "in" / Path(catalog).name


def load_problemset(data_dir: str | Path) -> pd.DataFrame:
    """Read problemset.xlsx, parse the query lists and fix the catalog paths."""
    df = pd.read_excel(Path(data_dir) / "problemset.xlsx")
    df["query"] = df["query"].apply(ast.literal_eval)
    df["catalog_path"] = df["catalog"].apply(lambda c: resolve_catalog_path(c, data_dir))
    value_cols = [c for c in df.columns if c.endswith("best value")]
    df["best_known"] = df[value_cols].min(axis=1)
    return df


_PAIR = re.compile(r"array\(\[(['\"])(.*?)\1, (['\"])(.*?)\3\]")


def parse_solution(text) -> list[tuple[str, str]] | None:
    """Parse the reference solution strings ("[array(['Jumbo', 'Arroz'], ...) ...]")."""
    if not isinstance(text, str):
        return None
    return [(m.group(2), m.group(4)) for m in _PAIR.finditer(text)]
