"""Build (and execute) the Spanish and English notebooks from a single source.

Markdown cells are written in both languages; code cells take the source of the
functions straight from ``src/compra_genetica`` so notebook and package never
drift apart. Inline translations inside code use ⟦es⟧...⟦en⟧...⟦/⟧ markers.

Usage:
    python tools/build_notebooks.py            # build + execute both
    python tools/build_notebooks.py --no-exec  # only build
    python tools/build_notebooks.py --recompute  # rerun the heavy experiments (code changed)
"""
import argparse
import inspect
import os
import re
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compra_genetica import baselines, data, fitness, gpu, metrics, operators, optimizer, routing  # noqa: E402

CELLS: list[tuple[str, str, str | None]] = []
MARK = re.compile(r"⟦es⟧(.*?)⟦en⟧(.*?)⟦/⟧", re.S)


def md(es: str, en: str) -> None:
    CELLS.append(("md", es.strip(), en.strip()))


def code(text: str) -> None:
    CELLS.append(("code", text.strip(), None))


def src(*objects) -> str:
    return "\n\n\n".join(inspect.getsource(o).rstrip() for o in objects)


def localize(text: str, lang: str) -> str:
    return MARK.sub(lambda m: m.group(1) if lang == "es" else m.group(2), text)


# =============================================================================
# 0. Cover
# =============================================================================
md("""
<div align="center">

**🇪🇸 Español** · [🇬🇧 English](compra_genetica_en.ipynb)

# 🧬🛒 Compra Genética
### Optimización de rutas de compra con Algoritmos Genéticos

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-vectorizado-013243?logo=numpy)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest)
![CUDA](https://img.shields.io/badge/GPU-experimento%20CUDA-76B900?logo=nvidia)

</div>

**Problema:** tengo una lista de compras y varias sucursales de supermercado. Cada sucursal vende
productos distintos a precios distintos y está a cierta distancia de mi casa y de las demás.
**¿En qué tiendas compro cada producto y en qué orden las visito para gastar lo mínimo**
(precio de los productos + combustible)?

Este notebook construye la solución **paso a paso**: cada celda hace *una* cosa y muestra su resultado.

| # | Sección | # | Sección |
|---|---|---|---|
| 1 | Configuración del entorno | 10 | Demostración matemática paso a paso |
| 2 | Exploración de los datos | 11 | Operadores genéticos (uno por celda) |
| 3 | Matriz de precios | 12 | La clase `GeneticShoppingOptimizer` |
| 4 | Grafo → Floyd-Warshall | 13 | Ejecución en un problema |
| 5 | Formulación matemática | 14 | Baselines y óptimo exacto |
| 6 | Sintaxis de los genes | 15 | Validación con los 136 problemas |
| 7 | Heurística de ruta (TSP) | 16 | Benchmark de operadores |
| 8 | Función de fitness | 17 | Hiperparámetros y escalabilidad |
| 9 | Validación del modelo de costo | 18 | Experimento CPU vs GPU · 19 Conclusiones · 20 Manual |
""", """
<div align="center">

[🇪🇸 Español](compra_genetica_es.ipynb) · **🇬🇧 English**

# 🧬🛒 Genetic Shopping
### Shopping-route optimization with Genetic Algorithms

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-vectorized-013243?logo=numpy)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest)
![CUDA](https://img.shields.io/badge/GPU-CUDA%20experiment-76B900?logo=nvidia)

</div>

**Problem:** I have a shopping list and several supermarket branches. Each branch sells different
products at different prices and sits at some distance from my home and from the others.
**Where should I buy each product, and in what order should I visit the stores, to spend the least**
(product prices + fuel)?

This notebook builds the solution **step by step**: every cell does *one* thing and shows its result.

| # | Section | # | Section |
|---|---|---|---|
| 1 | Environment setup | 10 | Step-by-step mathematical walkthrough |
| 2 | Data exploration | 11 | Genetic operators (one per cell) |
| 3 | Price matrix | 12 | The `GeneticShoppingOptimizer` class |
| 4 | Graph → Floyd-Warshall | 13 | Running one problem |
| 5 | Mathematical formulation | 14 | Baselines and exact optimum |
| 6 | Gene syntax | 15 | Validation on the 136 problems |
| 7 | Route heuristic (TSP) | 16 | Operator benchmark |
| 8 | Fitness function | 17 | Hyperparameters and scalability |
| 9 | Cost-model validation | 18 | CPU vs GPU experiment · 19 Conclusions · 20 Manual |
""")

# =============================================================================
# 1. Setup
# =============================================================================
md("""
## 1. Configuración del entorno
Importamos librerías, fijamos la semilla y detectamos el hardware. La semilla está ahí porque un AG es aleatorio por
naturaleza: sin ella cada ejecución daría números distintos y no se podrían comparar técnicas ni repetir un resultado.
El hardware lo imprimimos porque los tiempos que aparecen más abajo solo tienen sentido sabiendo en qué máquina se midieron.
""", """
## 1. Environment setup
Imports, a fixed seed and hardware detection. The seed is there because a GA is random by nature: without it every run
would give different numbers and we couldn't compare techniques or repeat a result. We print the hardware because the
timings further down only make sense if you know which machine produced them.
""")
code("""
LANG = "⟦es⟧es⟦en⟧en⟦/⟧"

import ast
import inspect
import os
import platform
import re
import time
import tracemalloc
import warnings
from dataclasses import dataclass, field
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:,.2f}".format)

# Categorical palette in a fixed order (colorblind-validated)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    "axes.titleweight": "bold", "lines.linewidth": 2,
})

DATA_DIR = Path("..") / "data"
RESULTS_DIR = Path("..") / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
N_JOBS = os.cpu_count()


def save_fig(fig, name):
    fig.savefig(FIG_DIR / f"{name}_{LANG}.png", bbox_inches="tight", dpi=130)


# ⟦es⟧False: los experimentos pesados (validación, benchmarks) se cargan de results/ si ya existen.
# True: se recalcula todo desde cero (~15 min usando todos los núcleos).⟦en⟧False: heavy experiments (validation, benchmarks) are loaded from results/ when they exist.
# True: everything is recomputed from scratch (~15 min using every core).⟦/⟧
RECOMPUTE = os.environ.get("COMPRA_RECOMPUTE", "0") == "1"


def cached(name, compute, parallel=True):
    path = RESULTS_DIR / f"{name}.csv"
    if path.exists() and not RECOMPUTE:
        print(f"📂 ⟦es⟧Cargado de⟦en⟧Loaded from⟦/⟧ results/{name}.csv  (RECOMPUTE = True ⟦es⟧para recalcular⟦en⟧to recompute⟦/⟧)")
        return pd.read_csv(path)
    t0 = time.perf_counter()
    df = compute()
    df.to_csv(path, index=False)
    workers = f"{N_JOBS} ⟦es⟧procesos⟦en⟧processes⟦/⟧" if parallel else "⟦es⟧1 proceso (secuencial)⟦en⟧1 process (sequential)⟦/⟧"
    print(f"⚙️ {name}: {len(df)} ⟦es⟧filas en⟦en⟧rows in⟦/⟧ {time.perf_counter() - t0:.1f} s ⟦es⟧usando⟦en⟧using⟦/⟧ {workers}")
    return df


try:
    import torch
    GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except ImportError:
    torch, GPU = None, None

print(f"Python      {platform.python_version()}")
print(f"NumPy       {np.__version__} | pandas {pd.__version__} | networkx {nx.__version__}")
print(f"CPU         {platform.processor()} | {N_JOBS} ⟦es⟧hilos lógicos⟦en⟧logical threads⟦/⟧")
print(f"GPU         {GPU or '⟦es⟧no disponible⟦en⟧not available⟦/⟧'}")
""")

# =============================================================================
# 2. Data exploration
# =============================================================================
md("""
## 2. Exploración de los datos

### 2.1 El dataset de validación (`problemset.xlsx`)
Cada fila es un problema: un catálogo, un precio de combustible y una lista de compras, junto con
las soluciones de cuatro algoritmos de referencia:

| Prefijo | Algoritmo de referencia |
|---|---|
| `qbf` | cuasi fuerza bruta |
| `ga` | algoritmo genético |
| `gc` | greedy: el más barato |
| `gn` | greedy: el más cercano |

La columna `catalog` apunta a `problemset/in/...`, pero en este repo los archivos están en
`data/in/`. Corregimos la ruta y convertimos `query` de texto a lista de Python.
""", """
## 2. Data exploration

### 2.1 The validation dataset (`problemset.xlsx`)
Each row is a problem: a catalog, a fuel price and a shopping list, together with the solutions
of four reference algorithms:

| Prefix | Reference algorithm |
|---|---|
| `qbf` | quasi brute force |
| `ga` | genetic algorithm |
| `gc` | greedy: cheapest |
| `gn` | greedy: nearest |

The `catalog` column points to `problemset/in/...`, but in this repo the files live in
`data/in/`. We fix the path and turn `query` from text into a Python list.
""")
code(src(data.resolve_catalog_path, data.load_problemset) + """


problemset = load_problemset(DATA_DIR)
print("⟦es⟧Forma⟦en⟧Shape⟦/⟧:", problemset.shape)
problemset[["problem_cluster", "fuel_price", "query", "qbf best value", "ga best value",
            "gc best value", "gn best value", "best_known"]].head()
""")

md("""
**¿Cuántos problemas hay de cada tipo y qué falta?** El nombre del cluster codifica el tamaño:
`small-5s-10p-f0_10-q1_10` = 5 tiendas (*stores*), 10 productos, combustible entre 0 y 10 y
listas de 1 a 10 productos. `best_known` es el mejor valor conocido entre las cuatro referencias.
""", """
**How many problems of each kind, and what is missing?** The cluster name encodes the size:
`small-5s-10p-f0_10-q1_10` = 5 stores, 10 products, fuel between 0 and 10 and lists of 1 to 10
products. `best_known` is the best value among the four references.
""")
code("""
problemset["k"] = problemset["query"].apply(len)
overview = problemset.groupby("problem_cluster").agg(
    ⟦es⟧problemas⟦en⟧problems⟦/⟧=("k", "size"),
    k_min=("k", "min"), k_max=("k", "max"),
    fuel_min=("fuel_price", "min"), fuel_max=("fuel_price", "max"),
    qbf_NaN=("qbf best value", lambda s: s.isna().sum()),
    ga_NaN=("ga best value", lambda s: s.isna().sum()),
)
display(overview)
print("⟦es⟧Valores faltantes por columna⟦en⟧Missing values per column⟦/⟧:")
print(problemset.isna().sum()[lambda s: s > 0].to_string())
""")

md("""
### 2.2 Un catálogo por dentro
Cada Excel tiene dos hojas: **`catalogo`** (tienda, producto, precio) y **`grafo`**
(origen, destino, distancia). Empezamos por el problema más pequeño (3 tiendas, 4 productos) porque ahí todavía se puede
revisar cada número a mano; es el mismo que usamos después para la demostración paso a paso.
""", """
### 2.2 Inside a catalog
Each Excel file has two sheets: **`catalogo`** (store, product, price) and **`grafo`**
(source, target, distance). We start with the smallest problem (3 stores, 4 products) because every number can still be
checked by hand there; it is the same one we use later for the step-by-step walkthrough.
""")
code(src(data.load_catalog) + """


SMALL = DATA_DIR / "in" / "small-3s-4p-f1_10-q1_4-random_catalog.xlsx"
catalog_df, graph_df = load_catalog(SMALL)
display(catalog_df.sort_values(["store", "product"]).reset_index(drop=True))
display(graph_df)
""")

md("""
**Observaciones importantes del grafo:**
- Aparecen aristas `A → B` **y** `B → A` con distancias distintas (`Casa → Supermercado Nacional = 5`,
  pero `Supermercado Nacional → Casa = 15`). El grafo es **dirigido**, como calles de un solo sentido.
- Hay *self-loops* (`Casa → Casa = 1`). Quedarse en el mismo lugar no cuesta nada, así que los ignoramos.
""", """
**Important observations about the graph:**
- There are edges `A → B` **and** `B → A` with different distances (`Casa → Supermercado Nacional = 5`
  but `Supermercado Nacional → Casa = 15`). The graph is **directed**, like one-way streets.
- There are self-loops (`Casa → Casa = 1`). Staying in place costs nothing, so we ignore them.
""")
code("""
def draw_graph(graph, ax, title, highlight=None):
    G = nx.DiGraph()
    for a, b, d in graph.itertuples(index=False):
        if a != b:
            G.add_edge(a, b, weight=d)
    pos = nx.circular_layout(G)
    colors = [PALETTE[1] if n == "Casa" else PALETTE[0] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=900, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white", font_weight="bold",
                            labels={n: n.replace(" ", "\\n") for n in G.nodes})
    edge_colors = [PALETTE[7] if highlight and (u, v) in highlight else "#9a9a96" for u, v in G.edges]
    widths = [3 if highlight and (u, v) in highlight else 1 for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=widths, arrowsize=14,
                           connectionstyle="arc3,rad=0.12", node_size=900)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=7, connectionstyle="arc3,rad=0.12",
                                 edge_labels={(u, v): int(d["weight"]) for u, v, d in G.edges(data=True)})
    ax.set_title(title)
    ax.axis("off")


PARQUE = DATA_DIR / "in" / "small-5s-10p-f0_10-q1_10-parque_catalog.xlsx"
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
draw_graph(graph_df, axes[0], "small-3s-4p ⟦es⟧(completo, dirigido)⟦en⟧(complete, directed)⟦/⟧")
draw_graph(load_catalog(PARQUE)[1], axes[1], "small-5s-10p-parque ⟦es⟧(disperso)⟦en⟧(sparse)⟦/⟧")
plt.tight_layout()
save_fig(fig, "graphs")
plt.show()
""")

md("""
El catálogo **`parque`** tiene un nodo **`Parque`** que no es una tienda, y el grafo no está completo:
para ir de una tienda a otra hay que pasar por nodos intermedios. Por eso más adelante lo convertimos
en un grafo **totalmente conectado** con Floyd-Warshall.

### 2.3 Tamaño de cada catálogo
""", """
The **`parque`** catalog has a **`Parque`** node that is not a store, and the graph is not complete:
to go from one store to another you must pass through intermediate nodes. That is why we later turn
it into a **fully connected** graph with Floyd-Warshall.

### 2.3 Size of every catalog
""")
code("""
rows = []
for path in sorted((DATA_DIR / "in").glob("*.xlsx")):
    cat, gr = load_catalog(path)
    n_stores, n_products = cat["store"].nunique(), cat["product"].nunique()
    nodes = set(gr["source"]) | set(gr["target"])
    rows.append({
        "catalog": path.stem.replace("_catalog", ""),
        "⟦es⟧tiendas⟦en⟧stores⟦/⟧": n_stores, "⟦es⟧productos⟦en⟧products⟦/⟧": n_products,
        "⟦es⟧filas_catálogo⟦en⟧catalog_rows⟦/⟧": len(cat),
        "⟦es⟧densidad_%⟦en⟧density_%⟦/⟧": 100 * len(cat) / (n_stores * n_products),
        "⟦es⟧nodos⟦en⟧nodes⟦/⟧": len(nodes), "⟦es⟧aristas⟦en⟧edges⟦/⟧": len(gr),
        "⟦es⟧precio_min⟦en⟧price_min⟦/⟧": cat["price"].min(),
        "⟦es⟧precio_max⟦en⟧price_max⟦/⟧": cat["price"].max(),
        "⟦es⟧dist_max⟦en⟧dist_max⟦/⟧": gr["distance"].max(),
    })
catalog_sizes = pd.DataFrame(rows).set_index("catalog")
catalog_sizes
""")

md("""
💡 **Dato clave para entender los resultados:** los precios van de cientos a miles, mientras que el
costo de transporte es `combustible (0–9) × distancia (decenas)`. **El precio de los productos domina
el costo total.** Por eso la estrategia "comprar donde sea más barato" ya es muy buena, y el valor del
AG está en decidir cuándo conviene pagar un poco más para ahorrarse un viaje.
""", """
💡 **Key fact to understand the results:** prices range from hundreds to thousands, while transport
costs `fuel (0–9) × distance (tens)`. **Product prices dominate the total cost.** That is why the
"buy where it is cheapest" strategy is already very good, and the value of the GA lies in deciding
when it pays to spend a bit more on a product to save a trip.
""")

# =============================================================================
# 3. Price matrix
# =============================================================================
md("""
## 3. Matriz de precios
Pasamos la hoja `catalogo` (formato largo) a una **matriz tiendas × productos**. Si una tienda no vende
un producto, su precio es `∞`, así ese producto nunca se le asigna.

¿Por qué una matriz y no seguir con la tabla? Porque el AG va a consultar precios miles de veces por segundo, y con una
matriz de NumPy sumar el precio de un cromosoma entero es una sola operación vectorizada. Y usamos `∞` en vez de `NaN`
porque `min` y `argmin` lo tratan de forma natural como "la peor opción posible".
""", """
## 3. Price matrix
We pivot the `catalogo` sheet (long format) into a **stores × products matrix**. When a store does not
sell a product its price is `∞`, so the product is never assigned to it.

Why a matrix instead of the table? Because the GA looks up prices thousands of times per second, and with a NumPy matrix
the price of a whole chromosome is a single vectorized operation. We use `∞` instead of `NaN` because `min` and `argmin`
naturally treat it as "the worst possible option".
""")
code(src(data.build_price_matrix) + """


RANDOM5 = DATA_DIR / "in" / "small-5s-10p-f0_10-q1_10-punico_catalog.xlsx"
stores_, products_, price_ = build_price_matrix(load_catalog(RANDOM5)[0])
print("⟦es⟧Forma⟦en⟧Shape⟦/⟧:", price_.shape, "| ⟦es⟧celdas sin venta (∞)⟦en⟧cells not sold (∞)⟦/⟧:", np.isinf(price_).sum())

fig, ax = plt.subplots(figsize=(13, 3.6))
shown = np.where(np.isinf(price_), np.nan, price_)
im = ax.imshow(shown, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(products_)), products_, rotation=35, ha="right", fontsize=8)
ax.set_yticks(range(len(stores_)), stores_, fontsize=8)
ax.grid(False)
for i in range(shown.shape[0]):
    for j in range(shown.shape[1]):
        txt = "—" if np.isnan(shown[i, j]) else f"{shown[i, j]:,.0f}"
        dark = not np.isnan(shown[i, j]) and shown[i, j] > np.nanmean(shown)
        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if dark else "#0b0b0b")
ax.set_title("⟦es⟧Matriz de precios: catálogo 'punico' (— = no se vende)⟦en⟧Price matrix: 'punico' catalog (— = not sold)⟦/⟧")
fig.colorbar(im, ax=ax, label="⟦es⟧precio⟦en⟧price⟦/⟧")
save_fig(fig, "price_matrix")
plt.show()
""")

# =============================================================================
# 4. Graph -> distances
# =============================================================================
md("""
## 4. Del grafo a la matriz de distancias

### 4.1 Matriz de adyacencia (dirigida)
`A[i, j]` es la distancia de la arista `i → j`, o `∞` si no hay arista. La diagonal vale 0 y los
self-loops se ignoran. Si una arista aparece repetida, nos quedamos con la más corta.

La matriz es dirigida (no copiamos `A[i, j]` en `A[j, i]`) por lo que vimos en el grafo: ida y vuelta pueden medir distinto.
Más adelante, en la sección 9, comprobamos con datos que esta decisión es la correcta.
""", """
## 4. From graph to distance matrix

### 4.1 Adjacency matrix (directed)
`A[i, j]` is the length of edge `i → j`, or `∞` if there is no edge. The diagonal is 0 and self-loops
are ignored. If an edge appears twice we keep the shorter one.

The matrix is directed (we don't copy `A[i, j]` into `A[j, i]`) because of what we saw in the graph: going and coming back
can have different lengths. In section 9 we check with data that this was the right call.
""")
code("HOME = \"Casa\"\n\n\n" + src(data.build_adjacency) + """


parque_graph = load_catalog(PARQUE)[1]
parque_nodes = sorted(set(parque_graph["source"]) | set(parque_graph["target"]))
A = build_adjacency(parque_graph, parque_nodes)
print("⟦es⟧Pares (i, j) sin conexión directa⟦en⟧Pairs (i, j) without a direct edge⟦/⟧:", np.isinf(A).sum())
pd.DataFrame(A, index=parque_nodes, columns=parque_nodes)
""")

md(r"""
### 4.2 Floyd-Warshall: grafo totalmente conectado
Para el AG necesitamos saber cuánto cuesta ir de cualquier tienda a cualquier otra, aunque no haya una calle directa.
Por eso convertimos el grafo en uno **totalmente conectado**, donde cada par de nodos guarda su distancia mínima.
Floyd-Warshall prueba cada nodo $k$ como intermediario:

$$D^{(k)}_{ij} = \min\left(D^{(k-1)}_{ij},\; D^{(k-1)}_{ik} + D^{(k-1)}_{kj}\right)$$

Complejidad $O(n^3)$. Con NumPy el doble ciclo sobre $(i, j)$ se vectoriza con *broadcasting*, y solo
queda el ciclo sobre $k$.
""", r"""
### 4.2 Floyd-Warshall: a fully connected graph
The GA needs to know what it costs to go from any store to any other, even without a direct street. So we turn the
graph into a **fully connected** one where every pair of nodes holds its shortest distance. Floyd-Warshall tries every
node $k$ as an intermediate stop:

$$D^{(k)}_{ij} = \min\left(D^{(k-1)}_{ij},\; D^{(k-1)}_{ik} + D^{(k-1)}_{kj}\right)$$

Complexity $O(n^3)$. With NumPy the double loop over $(i, j)$ is vectorized via broadcasting, so only
the loop over $k$ remains.
""")
code(src(data.floyd_warshall) + """


D_parque = floyd_warshall(A)
print("⟦es⟧Pares sin conexión después de Floyd-Warshall⟦en⟧Unconnected pairs after Floyd-Warshall⟦/⟧:", np.isinf(D_parque).sum())

# ⟦es⟧Verificación contra Dijkstra de networkx⟦en⟧Check against networkx's Dijkstra⟦/⟧
G = nx.DiGraph()
for a, b, d in parque_graph.itertuples(index=False):
    if a != b:
        G.add_edge(a, b, weight=min(d, G[a][b]["weight"]) if G.has_edge(a, b) else d)
nx_dist = dict(nx.all_pairs_dijkstra_path_length(G))
same = all(D_parque[i, j] == nx_dist[a][b] for i, a in enumerate(parque_nodes) for j, b in enumerate(parque_nodes))
print("⟦es⟧¿Coincide con networkx?⟦en⟧Matches networkx?⟦/⟧", "✅" if same else "❌")
pd.DataFrame(D_parque, index=parque_nodes, columns=parque_nodes)
""")

md("""
### 4.3 La instancia completa del problema
Juntamos todo en una estructura `ShoppingInstance`. Para el AG solo importan la casa y las tiendas, así
que reducimos la matriz a esos puntos: **el punto 0 es la casa y el punto `s + 1` es la tienda `s`**.
Los nodos intermedios (como `Parque`) ya quedaron "absorbidos" en las distancias mínimas.

La reducción no es solo estética: con la numeración "punto = tienda + 1", pasar de un gen (una tienda) a una fila de la
matriz es una suma, sin diccionarios de por medio en la parte que más se repite.
""", """
### 4.3 The full problem instance
We bundle everything into a `ShoppingInstance`. The GA only cares about the home and the stores, so we
shrink the matrix to those points: **point 0 is the home and point `s + 1` is store `s`**.
Intermediate nodes (like `Parque`) are already "absorbed" into the shortest distances.

The reduction is not just cosmetic: with the "point = store + 1" numbering, going from a gene (a store) to a matrix row is
a plain addition, with no dictionary lookups in the part that runs most often.
""")
code(src(data.ShoppingInstance, data.load_instance) + """


inst_small = load_instance(SMALL)
print("⟦es⟧Tiendas⟦en⟧Stores⟦/⟧:", inst_small.stores)
print("⟦es⟧Productos⟦en⟧Products⟦/⟧:", inst_small.products)
labels = [HOME] + inst_small.stores
pd.DataFrame(inst_small.D, index=labels, columns=labels)
""")
md("""
**Detalle oculto en los datos:** algunos grafos tienen aristas de distancia **0**. Después de Floyd-Warshall, un
cero puede "propagarse" y dejar muchos pares de tiendas a distancia 0. Medimos qué fracción de pares queda así en
cada catálogo:
""", """
**A hidden detail in the data:** some graphs contain edges of distance **0**. After Floyd-Warshall a zero can
"propagate" and leave many store pairs at distance 0. We measure the fraction of such pairs per catalog:
""")
code("""
zero_rows = []
for path in sorted((DATA_DIR / "in").glob("*.xlsx")):
    inst = load_instance(path)
    n = len(inst.D)
    zero_rows.append({"catalog": path.stem.replace("_catalog", ""),
                      "⟦es⟧aristas con distancia 0⟦en⟧edges with distance 0⟦/⟧": int((inst.graph["distance"] == 0).sum()),
                      "⟦es⟧% pares a distancia 0 (Floyd)⟦en⟧% pairs at distance 0 (Floyd)⟦/⟧": 100 * ((inst.D == 0).sum() - n) / (n * n - n)})
pd.DataFrame(zero_rows).set_index("catalog")
""")
md("""
⚠️ En el catálogo **`big`** (100 tiendas) el **97 %** de los pares queda a distancia 0: ahí moverse entre tiendas es
prácticamente gratis, así que el problema se reduce a "comprar cada producto donde sea más barato". Hay que tenerlo
en cuenta al leer los resultados de ese cluster.
""", """
⚠️ In the **`big`** catalog (100 stores) **97 %** of the pairs end up at distance 0: moving between stores is
practically free there, so the problem reduces to "buy each product where it is cheapest". Keep this in mind when
reading that cluster's results.
""")

# =============================================================================
# 5. Math formulation
# =============================================================================
md(r"""
## 5. Formulación matemática

**Datos**
- $Q = \{1, \dots, k\}$: productos de la lista de compras.
- $S$: tiendas, y $A_i \subseteq S$: tiendas que venden el producto $i$.
- $c_{s,i}$: precio del producto $i$ en la tienda $s$.
- $d(u, v)$: distancia mínima dirigida (Floyd-Warshall), con $0$ = casa.
- $\varphi$: precio del combustible por unidad de distancia.

**Variable de decisión:** $x = (x_1, \dots, x_k)$ con $x_i \in A_i$, la tienda donde se compra el producto $i$.
Las tiendas a visitar son $U(x) = \{x_1, \dots, x_k\}$ (sin repetir).

**Objetivo**

$$\min_{x}\; f(x) = \underbrace{\sum_{i=1}^{k} c_{x_i,i}}_{\text{costo de productos}} \;+\; \varphi \cdot \underbrace{\mathrm{TSP}\big(U(x)\big)}_{\text{distancia de la ruta}}$$

$$\mathrm{TSP}(U) = \min_{\pi \in \mathrm{Perm}(U)} \Big[d(0, \pi_1) + \sum_{j=1}^{|U|-1} d(\pi_j, \pi_{j+1}) + d(\pi_{|U|}, 0)\Big]$$

Es un problema **combinatorio, no derivable** y con muchos óptimos locales, justo el tipo de problema donde
los algoritmos genéticos son útiles. El espacio de búsqueda mide $\prod_i |A_i|$ (asignaciones), y además
cada asignación tiene $|U|!$ órdenes posibles.
""", r"""
## 5. Mathematical formulation

**Data**
- $Q = \{1, \dots, k\}$: products in the shopping list.
- $S$: stores, and $A_i \subseteq S$: stores that sell product $i$.
- $c_{s,i}$: price of product $i$ at store $s$.
- $d(u, v)$: shortest directed distance (Floyd-Warshall), with $0$ = home.
- $\varphi$: fuel price per distance unit.

**Decision variable:** $x = (x_1, \dots, x_k)$ with $x_i \in A_i$, the store where product $i$ is bought.
The stores to visit are $U(x) = \{x_1, \dots, x_k\}$ (no repeats).

**Objective**

$$\min_{x}\; f(x) = \underbrace{\sum_{i=1}^{k} c_{x_i,i}}_{\text{product cost}} \;+\; \varphi \cdot \underbrace{\mathrm{TSP}\big(U(x)\big)}_{\text{route distance}}$$

$$\mathrm{TSP}(U) = \min_{\pi \in \mathrm{Perm}(U)} \Big[d(0, \pi_1) + \sum_{j=1}^{|U|-1} d(\pi_j, \pi_{j+1}) + d(\pi_{|U|}, 0)\Big]$$

It is a **combinatorial, non-differentiable** problem with many local optima, exactly the kind of
problem where genetic algorithms shine. The search space has $\prod_i |A_i|$ assignments, and each
assignment has $|U|!$ possible visiting orders.
""")
code("""
instances = {p: load_instance(p) for p in problemset["catalog_path"].unique()}


def search_space_log10(row):
    inst = instances[row.catalog_path]
    cols = [inst.products.index(p) for p in row.query]
    return np.log10(np.isfinite(inst.price[:, cols]).sum(axis=0)).sum()


problemset["log10_space"] = problemset.apply(search_space_log10, axis=1)
problemset.groupby("problem_cluster")["log10_space"].agg(["min", "median", "max"]).rename(
    columns=lambda c: f"log10(⟦es⟧asignaciones⟦en⟧assignments⟦/⟧) {c}")
""")
md("""
Los problemas `big` llegan a $10^{31}$ asignaciones posibles (más que segundos en la edad del universo,
$\\approx 4 \\times 10^{17}$). Enumerarlas todas es imposible: necesitamos una **metaheurística**.
""", """
`big` problems reach $10^{31}$ possible assignments (more than seconds in the age of the universe,
$\\approx 4 \\times 10^{17}$). Enumerating them is impossible: we need a **metaheuristic**.
""")

# =============================================================================
# 6. Gene syntax
# =============================================================================
md("""
## 6. Sintaxis de los genes

| Concepto AG | En este problema |
|---|---|
| **Gen** | entero = índice de la tienda donde compro *un* producto |
| **Cromosoma** (genotipo) | vector de `k` enteros, uno por producto de la lista |
| **Alelos válidos del gen `i`** | solo las tiendas que venden el producto `i` ($A_i$) |
| **Fenotipo** | la ruta: tiendas en orden de visita + qué comprar en cada una |

**¿Por qué no incluir el orden de la ruta en el cromosoma?** Porque, fijado *qué* tiendas visitar, el mejor
orden se puede calcular de forma exacta (y rápida) con una heurística de TSP. Así el AG busca solo en el
espacio de asignaciones y **todo cromosoma es siempre válido**: no hacen falta penalizaciones ni reparaciones.
""", """
## 6. Gene syntax

| GA concept | In this problem |
|---|---|
| **Gene** | integer = index of the store where I buy *one* product |
| **Chromosome** (genotype) | vector of `k` integers, one per product in the list |
| **Valid alleles of gene `i`** | only the stores that sell product `i` ($A_i$) |
| **Phenotype** | the route: stores in visiting order + what to buy at each one |

**Why not put the route order in the chromosome?** Because once we know *which* stores to visit, the best
order can be computed exactly (and fast) with a TSP heuristic. The GA then searches only the space of
assignments and **every chromosome is always valid**: no penalties or repairs needed.
""")
code("""
query_small = problemset.loc[1, "query"]
cols = [inst_small.products.index(p) for p in query_small]
allowed = [np.flatnonzero(np.isfinite(inst_small.price[:, c])) for c in cols]

example = np.array([2, 0, 0, 2])  # ⟦es⟧un cromosoma de ejemplo⟦en⟧an example chromosome⟦/⟧
pd.DataFrame({
    "⟦es⟧gen⟦en⟧gene⟦/⟧ i": range(len(query_small)),
    "⟦es⟧producto⟦en⟧product⟦/⟧": query_small,
    "⟦es⟧alelos válidos⟦en⟧valid alleles⟦/⟧": [[inst_small.stores[s] for s in a] for a in allowed],
    "⟦es⟧valor del gen⟦en⟧gene value⟦/⟧": example,
    "⟦es⟧tienda⟦en⟧store⟦/⟧": [inst_small.stores[s] for s in example],
    "⟦es⟧precio⟦en⟧price⟦/⟧": [inst_small.price[s, c] for s, c in zip(example, cols)],
})
""")

# =============================================================================
# 7. Route heuristic
# =============================================================================
md("""
## 7. Heurística de ruta: ¿en qué orden visito las tiendas?
Dado el conjunto de tiendas del cromosoma, buscamos el recorrido cerrado más corto
`Casa → … → Casa`. Es un **TSP asimétrico** (el grafo es dirigido).

### 7.1 Longitud de una ruta
""", """
## 7. Route heuristic: in what order do I visit the stores?
Given the set of stores in a chromosome we look for the shortest closed tour `Home → … → Home`.
It is an **asymmetric TSP** (the graph is directed).

### 7.1 Length of a route
""")
code(src(routing.route_length) + """


order = [3, 1]  # ⟦es⟧puntos⟦en⟧points⟦/⟧: 3 = Supermercado Nacional, 1 = Jumbo
print("Casa → Supermercado Nacional → Jumbo → Casa =", route_length(order, inst_small.D))
print("Casa → Jumbo → Supermercado Nacional → Casa =", route_length(order[::-1], inst_small.D))
""")
md(r"""
El orden importa: el mismo par de tiendas cuesta 38 o 26 según el sentido.

### 7.2 Held-Karp: TSP exacto por programación dinámica
`dp[mask, j]` = camino más corto que sale de casa, visita exactamente los puntos de `mask` y termina en `j`:

$$dp[mask \cup \{j\}, j] = \min_{k \in mask} \big(dp[mask, k] + d(k, j)\big)$$

Su costo es $O(2^n n^2)$, perfecto para $n \le 9$ tiendas distintas (lo usual en estas rutas).
""", r"""
Order matters: the same pair of stores costs 38 or 26 depending on the direction.

### 7.2 Held-Karp: exact TSP via dynamic programming
`dp[mask, j]` = shortest path that leaves home, visits exactly the points in `mask` and ends at `j`:

$$dp[mask \cup \{j\}, j] = \min_{k \in mask} \big(dp[mask, k] + d(k, j)\big)$$

It costs $O(2^n n^2)$, perfect for $n \le 9$ distinct stores (the usual case in these routes).
""")
code(src(routing.held_karp) + """


length, best_order = held_karp([1, 2, 3], inst_small.D)
print(f"⟦es⟧Ruta óptima visitando las 3 tiendas⟦en⟧Optimal route visiting all 3 stores⟦/⟧: "
      f"Casa → {' → '.join(labels[p] for p in best_order)} → Casa = {length}")
""")
md("""
### 7.3 Vecino más cercano + 2-opt (para rutas largas)
Con más de 9 tiendas, Held-Karp se vuelve caro. Ahí usamos una heurística constructiva (ir siempre a la
tienda más cercana) seguida de una búsqueda local **2-opt** (invertir segmentos mientras mejore).
""", """
### 7.3 Nearest neighbor + 2-opt (for long routes)
With more than 9 stores Held-Karp gets expensive. There we use a constructive heuristic (always go to
the closest store) followed by a **2-opt** local search (reverse segments while it improves).
""")
code(src(routing.nearest_neighbor, routing.two_opt) + """


inst_m30 = load_instance(DATA_DIR / "in" / "medium-30s-80p-f0_10-q1_20-random_catalog.xlsx")
points = list(range(1, 13))  # ⟦es⟧12 tiendas⟦en⟧12 stores⟦/⟧
t = time.perf_counter(); nn = nearest_neighbor(points, inst_m30.D); t_nn = time.perf_counter() - t
t = time.perf_counter(); opt = two_opt(nn, inst_m30.D); t_2opt = time.perf_counter() - t
t = time.perf_counter(); hk_len, _ = held_karp(points, inst_m30.D); t_hk = time.perf_counter() - t
pd.DataFrame({
    "⟦es⟧método⟦en⟧method⟦/⟧": ["⟦es⟧Vecino más cercano⟦en⟧Nearest neighbor⟦/⟧", "NN + 2-opt", "⟦es⟧Held-Karp (exacto)⟦en⟧Held-Karp (exact)⟦/⟧"],
    "⟦es⟧longitud⟦en⟧length⟦/⟧": [route_length(nn, inst_m30.D), route_length(opt, inst_m30.D), hk_len],
    "ms": [1000 * t_nn, 1000 * (t_nn + t_2opt), 1000 * t_hk],
})
""")
md("""
### 7.4 `RouteSolver`: selector de método + memoización
Durante la evolución, muchos cromosomas comparten el mismo **conjunto** de tiendas. Guardamos cada ruta
resuelta en un diccionario indexado por `frozenset`, así cada conjunto se calcula una sola vez.
""", """
### 7.4 `RouteSolver`: method selection + memoization
During evolution many chromosomes share the same **set** of stores. We cache every solved route in a
dictionary keyed by `frozenset`, so each set is computed only once.
""")
code(src(routing.RouteSolver) + """


solver = RouteSolver(inst_small.D)
for stores in [[0, 2], [2, 0], [2, 0, 0, 2], [0, 1, 2]]:
    length, order = solver.solve(stores)
    print(f"{str(stores):14s} → {[inst_small.stores[s] for s in order]} = {length}")
print(f"cache: hits={solver.hits}, misses={solver.misses}")
""")

# =============================================================================
# 8. Fitness
# =============================================================================
md("""
## 8. Función de fitness
`ShoppingProblem` une una instancia con una consulta concreta (lista + combustible). Calcula:
- `product_cost`: suma de precios, vectorizada con *fancy indexing* de NumPy.
- `transport_cost`: combustible × longitud de la ruta óptima del conjunto de tiendas.
- `cost` = la **función objetivo a minimizar**.
- `decode` / `format_answer`: convierten el genotipo en el fenotipo (la respuesta en formato `RESPUESTA> (tienda, [productos]) -> …`).

Separamos el costo en productos y transporte porque así se puede ver *de dónde* viene cada ahorro, y porque más adelante
nos sirve para entender por qué en este dataset el precio pesa mucho más que el combustible.
""", """
## 8. Fitness function
`ShoppingProblem` ties an instance to a concrete query (list + fuel price). It computes:
- `product_cost`: sum of prices, vectorized with NumPy fancy indexing.
- `transport_cost`: fuel × length of the optimal route for the set of stores.
- `cost` = the **objective function to minimize**.
- `decode` / `format_answer`: turn the genotype into the phenotype (the answer as `RESPUESTA> (store, [products]) -> …`).

We split the cost into products and transport so we can see *where* each saving comes from, and because later it helps
explain why, in this dataset, price matters far more than fuel.
""")
code(src(fitness.ShoppingProblem) + """


row = problemset.loc[1]
problem_small = ShoppingProblem(inst_small, row.query, row.fuel_price)
print("⟦es⟧Cromosoma⟦en⟧Chromosome⟦/⟧:", example)
print(f"⟦es⟧Productos⟦en⟧Products⟦/⟧  = {problem_small.product_cost(example):,.2f}")
print(f"⟦es⟧Transporte⟦en⟧Transport⟦/⟧ = {problem_small.transport_cost(example):,.2f}")
print(f"⟦es⟧Total⟦en⟧Total⟦/⟧      = {problem_small.cost(example):,.2f}")
print(problem_small.format_answer(example))
""")

# =============================================================================
# 9. Cost model validation
# =============================================================================
md("""
## 9. Validación del modelo de costo contra 504 soluciones de referencia
Antes de optimizar hay que estar **seguros de que medimos el costo igual que las referencias**. El dataset
trae 4 soluciones por problema, como secuencias `(tienda, producto)` en orden de visita. Las parseamos y
recalculamos su costo con nuestro modelo, primero suponiendo grafo **dirigido** y luego **no dirigido**.
""", """
## 9. Cost-model validation against 504 reference solutions
Before optimizing we must be **sure we measure cost exactly like the references do**. The dataset holds
4 solutions per problem as `(store, product)` sequences in visiting order. We parse them and recompute
their cost with our model, first assuming a **directed** graph and then an **undirected** one.
""")
code("_PAIR = re.compile(r\"array\\(\\[(['\\\"])(.*?)\\1, (['\\\"])(.*?)\\3\\]\")\n\n\n"
     + src(data.parse_solution, fitness.evaluate_sequence) + """


undirected = {p: load_instance(p, directed=False) for p in instances}
checks = []
for _, r in problemset.iterrows():
    for alg in ["qbf", "ga", "gc", "gn"]:
        sol, value = parse_solution(r[f"{alg} best solution"]), r[f"{alg} best value"]
        if sol is None or np.isnan(value):
            continue
        checks.append({
            "⟦es⟧dirigido⟦en⟧directed⟦/⟧": abs(evaluate_sequence(instances[r.catalog_path], sol, r.fuel_price) - value) < 1e-6,
            "⟦es⟧no_dirigido⟦en⟧undirected⟦/⟧": abs(evaluate_sequence(undirected[r.catalog_path], sol, r.fuel_price) - value) < 1e-6,
        })
checks = pd.DataFrame(checks)
print(f"⟦es⟧Soluciones verificadas⟦en⟧Solutions checked⟦/⟧: {len(checks)}")
for col in checks:
    print(f"  {col:12s}: {checks[col].sum():3d} / {len(checks)}  {'✅' if checks[col].all() else '❌'}")
""")
md("""
✅ **El modelo reproduce al centavo las 504 soluciones de referencia** con grafo dirigido, rutas cerradas
y Floyd-Warshall. Con grafo no dirigido no cuadra, lo que confirma que las calles tienen sentido.
Esta celda es la base de confianza de todo lo que sigue, y también está automatizada en `tests/test_fitness.py`.
""", """
✅ **The model reproduces all 504 reference solutions to the cent** with a directed graph, closed routes
and Floyd-Warshall. The undirected version does not match, which confirms the streets are one-way.
This cell is the foundation of trust for everything that follows, and it is also automated in
`tests/test_fitness.py`.
""")

# =============================================================================
# 10. Step-by-step math walkthrough
# =============================================================================
md("""
## 10. Demostración matemática paso a paso: una generación "a mano"
Tomamos el problema 1 (3 tiendas, 4 productos, combustible = 1) y una población de **4 individuos**.
Hacemos una generación completa mostrando cada número.

La idea es quitarle la "magia" al algoritmo: con 4 individuos se puede seguir cada sorteo y cada cuenta con lápiz y papel,
y queda claro que el AG no es más que repetir estos pasos muchas veces.

### 10.1 Inicialización y evaluación
""", """
## 10. Step-by-step mathematical walkthrough: one generation "by hand"
We take problem 1 (3 stores, 4 products, fuel = 1) and a population of **4 individuals**.
We run one full generation showing every number.

The point is to take the "magic" out of the algorithm: with 4 individuals you can follow every draw and every sum with pen
and paper, and it becomes clear that a GA is nothing more than repeating these steps many times.

### 10.1 Initialization and evaluation
""")
code("""
rng_demo = np.random.default_rng(7)
P = problem_small
pop0 = np.array([[rng_demo.choice(a) for a in P.allowed] for _ in range(4)])


def explain(population, problem):
    rows = []
    for idx, ind in enumerate(population):
        length, order = problem.router.solve(ind)
        rows.append({
            "⟦es⟧individuo⟦en⟧individual⟦/⟧": f"x{idx}", "⟦es⟧genes⟦en⟧genes⟦/⟧": ind.tolist(),
            "⟦es⟧ruta⟦en⟧route⟦/⟧": " → ".join(["Casa", *[problem.instance.stores[s] for s in order], "Casa"]),
            "Σ ⟦es⟧precios⟦en⟧prices⟦/⟧": problem.product_cost(ind), "⟦es⟧distancia⟦en⟧distance⟦/⟧": length,
            "φ·⟦es⟧dist⟦en⟧dist⟦/⟧": problem.fuel_price * length, "f(x)": problem.product_cost(ind) + problem.fuel_price * length,
        })
    return pd.DataFrame(rows)


print("⟦es⟧Productos⟦en⟧Products⟦/⟧:", P.query, "| φ =", P.fuel_price)
table0 = explain(pop0, P)
table0
""")
md(r"""
### 10.2 Selección por ruleta (y por qué falla aquí)
Como minimizamos, usamos $\text{fitness}_i = 1/f(x_i)$ y la probabilidad
$p_i = \text{fitness}_i / \sum_j \text{fitness}_j$. Luego se gira la ruleta con números aleatorios
$r \sim U(0,1)$ sobre la probabilidad acumulada.
""", r"""
### 10.2 Roulette selection (and why it struggles here)
Since we minimize, we use $\text{fitness}_i = 1/f(x_i)$ and probability
$p_i = \text{fitness}_i / \sum_j \text{fitness}_j$. Then we spin the wheel with random numbers
$r \sim U(0,1)$ over the cumulative probability.
""")
code("""
costs0 = table0["f(x)"].to_numpy()
fit = 1 / costs0
roulette = pd.DataFrame({"f(x)": costs0, "1/f(x)": fit, "p_i": fit / fit.sum(),
                         "⟦es⟧acumulada⟦en⟧cumulative⟦/⟧": np.cumsum(fit / fit.sum())},
                        index=table0.iloc[:, 0])
display(roulette.style.format({"f(x)": "{:,.2f}", "1/f(x)": "{:.3e}", "p_i": "{:.4f}", "⟦es⟧acumulada⟦en⟧cumulative⟦/⟧": "{:.4f}"}))

r = rng_demo.random(4)
picked = np.searchsorted(np.cumsum(fit / fit.sum()), r)
for ri, pi in zip(r, picked):
    print(f"r = {ri:.4f} → x{pi}")
""")
md("""
Las probabilidades salen casi iguales (≈ 0.25 cada una), porque los costos difieren poco *en proporción*.
La ruleta casi no ejerce **presión selectiva**. Por eso también implementamos **torneo** y **ranking**,
que dependen del *orden* de los costos y no de su magnitud.

### 10.3 Selección por torneo (k = 2)
Se toman `k` individuos al azar y gana el de menor costo.
""", """
The probabilities come out almost equal (≈ 0.25 each) because the costs differ little *in proportion*.
Roulette exerts almost no **selective pressure**. That is why we also implement **tournament** and
**ranking**, which depend on the *order* of the costs and not on their magnitude.

### 10.3 Tournament selection (k = 2)
Pick `k` individuals at random; the one with the lowest cost wins.
""")
code("""
parents = []
for _ in range(4):
    a, b = rng_demo.integers(0, 4, size=2)
    winner = a if costs0[a] <= costs0[b] else b
    parents.append(winner)
    print(f"⟦es⟧torneo⟦en⟧tournament⟦/⟧ x{a} ({costs0[a]:,.2f}) vs x{b} ({costs0[b]:,.2f}) → ⟦es⟧gana⟦en⟧winner⟦/⟧ x{winner}")
""")
md("""
### 10.4 Cruce de un punto
Se elige un punto de corte `c` y los hijos intercambian las colas: `h1 = p1[:c] + p2[c:]`.
""", """
### 10.4 One-point crossover
Pick a cut point `c` and the children swap tails: `c1 = p1[:c] + p2[c:]`.
""")
code("""
children = []
for a, b in [(parents[0], parents[1]), (parents[2], parents[3])]:
    p1, p2 = pop0[a], pop0[b]
    cut = rng_demo.integers(1, P.k)
    h1, h2 = np.concatenate([p1[:cut], p2[cut:]]), np.concatenate([p2[:cut], p1[cut:]])
    print(f"p1=x{a} {p1.tolist()}  p2=x{b} {p2.tolist()}  corte/cut={cut}  →  h1={h1.tolist()}  h2={h2.tolist()}")
    children += [h1, h2]
""")
md("""
### 10.5 Mutación por reinicio aleatorio ($p_m = 1/k = 0.25$)
Por cada gen se sortea $r \\sim U(0,1)$. Si $r < p_m$, el gen toma otra tienda válida al azar.
""", """
### 10.5 Random-reset mutation ($p_m = 1/k = 0.25$)
For each gene draw $r \\sim U(0,1)$. If $r < p_m$, the gene takes a random valid store.
""")
code("""
pm = 1 / P.k
mutated = []
for child in children:
    new = child.copy()
    draws = rng_demo.random(P.k)
    for i in np.flatnonzero(draws < pm):
        new[i] = rng_demo.choice(P.allowed[i])
    print(f"{child.tolist()}  r={np.round(draws, 2).tolist()}  →  {new.tolist()}")
    mutated.append(new)
mutated = np.array(mutated)
""")
md("""
### 10.6 Reemplazo con elitismo (1 élite)
El mejor padre pasa intacto y los 3 mejores hijos completan la nueva población.
""", """
### 10.6 Replacement with elitism (1 elite)
The best parent survives untouched and the 3 best children complete the new population.
""")
code("""
child_costs = P.evaluate(mutated)
elite = costs0.argmin()
keep = child_costs.argsort()[:3]
pop1 = np.vstack([pop0[elite], mutated[keep]])
table1 = explain(pop1, P)
display(table1)
print(f"⟦es⟧Mejor costo⟦en⟧Best cost⟦/⟧  gen 0 = {costs0.min():,.2f}  →  gen 1 = {table1['f(x)'].min():,.2f}")
print(f"⟦es⟧Costo promedio⟦en⟧Mean cost⟦/⟧ gen 0 = {costs0.mean():,.2f}  →  gen 1 = {table1['f(x)'].mean():,.2f}")
print(f"⟦es⟧Óptimo de referencia (qbf)⟦en⟧Reference optimum (qbf)⟦/⟧ = {row['qbf best value']:,.2f}")
""")
md("""
Repetir este ciclo (selección → cruce → mutación → reemplazo) es todo lo que hace un AG. A continuación
implementamos cada paso como una función reutilizable y parametrizada.
""", """
Repeating this cycle (selection → crossover → mutation → replacement) is all a GA does. Next we implement
each step as a reusable, parameterized function.
""")

# =============================================================================
# 11. Operators
# =============================================================================
md("""
## 11. Operadores genéticos

### 11.1 Inicialización
- **Aleatoria:** cada gen toma una tienda válida al azar, lo que da diversidad máxima.
- **Sembrada (seeding):** además se incluyen soluciones heurísticas conocidas. Usamos dos *greedy*, que también
  sirven como baselines: **el más barato** (ignora distancias) y **el más cercano** (ignora precios).

Agregamos el seeding porque arrancar solo con individuos aleatorios desperdicia las primeras generaciones redescubriendo
algo obvio, como "compra donde es más barato". Con las semillas el AG parte de soluciones razonables y nunca termina peor
que ellas. En la sección 15 lo comparamos con y sin seeding para ver cuánto aporta de verdad.
""", """
## 11. Genetic operators

### 11.1 Initialization
- **Random:** every gene takes a random valid store, which gives maximum diversity.
- **Seeded:** known heuristic solutions are added too. We use two *greedy* ones, which also serve as
  baselines: **cheapest** (ignores distances) and **nearest** (ignores prices).

We added seeding because starting from random individuals only wastes the first generations rediscovering something
obvious, like "buy where it is cheapest". With seeds the GA starts from reasonable solutions and can never end up worse than
them. In section 15 we compare with and without seeding to see how much it really adds.
""")
code(src(baselines.greedy_cheapest, baselines.greedy_nearest, operators.init_random, operators.init_seeded) + """


rng = np.random.default_rng(SEED)
pop = init_seeded(P, 6, rng, [greedy_cheapest(P), greedy_nearest(P)])
display(explain(pop, P))
""")

md("""
### 11.2 Selección
Comparamos la **presión selectiva** de las tres técnicas: con una población de 30 individuos de un
problema mediano, seleccionamos 100 000 veces y contamos cuántas veces sale cada individuo según su ranking.
""", """
### 11.2 Selection
We compare the **selective pressure** of the three techniques: with a population of 30 individuals from
a medium problem, we select 100,000 times and count how often each individual is picked by rank.
""")
code(src(operators.selection_tournament, operators.selection_roulette, operators.selection_rank) + """


r_med = problemset[problemset.problem_cluster.str.startswith("medium-30")].iloc[3]
P_med = ShoppingProblem(instances[r_med.catalog_path], r_med.query, r_med.fuel_price)
demo_costs = np.sort(P_med.evaluate(init_random(P_med, 30, rng)))
fig, ax = plt.subplots(figsize=(11, 3.8))
for name, fn in [("⟦es⟧Torneo⟦en⟧Tournament⟦/⟧ (k=3)", selection_tournament), ("⟦es⟧Ruleta⟦en⟧Roulette⟦/⟧", selection_roulette),
                 ("⟦es⟧Ranking⟦en⟧Rank⟦/⟧ (s=1.8)", selection_rank)]:
    counts = np.bincount(fn(demo_costs, 100_000, rng), minlength=30) / 100_000
    ax.plot(range(1, 31), counts, marker="o", markersize=4, label=name)
ax.axhline(1 / 30, color="#9a9a96", ls="--", lw=1, label="⟦es⟧uniforme⟦en⟧uniform⟦/⟧")
ax.set_xlabel("⟦es⟧ranking del individuo (1 = más barato)⟦en⟧individual rank (1 = cheapest)⟦/⟧")
ax.set_ylabel("⟦es⟧probabilidad de selección⟦en⟧selection probability⟦/⟧")
ax.set_title("⟦es⟧Presión selectiva por técnica⟦en⟧Selective pressure by technique⟦/⟧")
ax.legend()
save_fig(fig, "selection_pressure")
plt.show()
print(f"⟦es⟧Rango de costos⟦en⟧Cost range⟦/⟧: {demo_costs.min():,.0f} – {demo_costs.max():,.0f} "
      f"(⟦es⟧el peor cuesta solo⟦en⟧the worst costs only⟦/⟧ {100 * (demo_costs.max() / demo_costs.min() - 1):.0f}% ⟦es⟧más⟦en⟧more⟦/⟧)")
""")

md("""
### 11.3 Cruce (crossover)
Los tres operadores trabajan **posición por posición**, así que el hijo hereda en el gen `i` una tienda que ya
era válida para el producto `i`: **los hijos son siempre válidos**.
""", """
### 11.3 Crossover
All three operators work **position by position**, so a child inherits at gene `i` a store that was already
valid for product `i`: **children are always valid**.
""")
code(src(operators.crossover_one_point, operators.crossover_two_point, operators.crossover_uniform) + """


def plot_genes(rows, title, ax, colors):
    for r, (label, genes, row_colors) in enumerate(rows):
        for c, (g, col) in enumerate(zip(genes, row_colors)):
            ax.add_patch(plt.Rectangle((c, -r), 0.94, 0.88, color=col))
            ax.text(c + 0.47, -r + 0.44, str(g), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(-0.3, -r + 0.44, label, ha="right", va="center", fontsize=9)
    ax.set_xlim(-2.2, len(rows[0][1])); ax.set_ylim(-len(rows) + 0.8, 1); ax.axis("off"); ax.set_title(title, fontsize=10)


p1, p2 = init_random(P_med, 2, np.random.default_rng(3))
fig, axes = plt.subplots(1, 3, figsize=(16, 2.6))
for ax, (name, fn) in zip(axes, [("⟦es⟧Un punto⟦en⟧One-point⟦/⟧", crossover_one_point), ("⟦es⟧Dos puntos⟦en⟧Two-point⟦/⟧", crossover_two_point),
                                 ("⟦es⟧Uniforme⟦en⟧Uniform⟦/⟧", crossover_uniform)]):
    c1, c2 = fn(p1, p2, np.random.default_rng(11))
    src_color = lambda child: [PALETTE[0] if a == b else PALETTE[1] for a, b in zip(child, p1)]
    plot_genes([("p1", p1, [PALETTE[0]] * P_med.k), ("p2", p2, [PALETTE[1]] * P_med.k),
                ("h1", c1, src_color(c1)), ("h2", c2, src_color(c2))], name, ax, PALETTE)
save_fig(fig, "crossover")
plt.show()
""")
md("""
Azul = gen heredado de `p1`, naranja = gen heredado de `p2` (si ambos padres coinciden en una posición, se ve azul).

### 11.4 Mutación
- **Reinicio aleatorio:** el gen salta a *cualquier* tienda válida. Explora.
- **Consolidación (heurística):** el gen se mueve a una tienda **que la ruta ya visita**. Busca rutas con menos
  paradas, es decir, explota.

La segunda la agregamos porque el reinicio aleatorio casi nunca "descubre" por sí solo que conviene juntar productos en
la misma tienda para ahorrarse un viaje. La idea era darle ese empujón. Spoiler: en el benchmark se ve que empuja
demasiado.

Medimos su efecto en el número de tiendas distintas que visita el cromosoma:
""", """
Blue = gene inherited from `p1`, orange = gene inherited from `p2` (where both parents agree it shows blue).

### 11.4 Mutation
- **Random reset:** the gene jumps to *any* valid store. It explores.
- **Consolidation (heuristic):** the gene moves to a store **the route already visits**. It seeks routes with
  fewer stops, so it exploits.

We added the second one because random reset rarely "discovers" on its own that grouping products in one store saves a
trip. The idea was to give it that push. Spoiler: the benchmark shows it pushes too hard.

We measure their effect on the number of distinct stores in the chromosome:
""")
code(src(operators.mutation_random_reset, operators.mutation_consolidate) + """


base = init_random(P_med, 2000, np.random.default_rng(5))
summary = {"⟦es⟧original⟦en⟧original⟦/⟧": [len(set(c)) for c in base]}
for name, fn in [("random_reset", mutation_random_reset), ("consolidate", mutation_consolidate)]:
    summary[name] = [len(set(fn(c, P_med, rng, pm=0.3))) for c in base]
print(f"⟦es⟧Tiendas distintas por cromosoma (k = {P_med.k} productos, pm = 0.3)⟦en⟧Distinct stores per chromosome (k = {P_med.k} products, pm = 0.3)⟦/⟧")
pd.DataFrame(summary).describe().loc[["mean", "min", "max"]]
""")

md("""
### 11.5 Reemplazo con elitismo y 11.6 condición de término
- **Elitismo:** los `n_elite` mejores padres sobreviven intactos y el resto de la población se llena con los
  mejores hijos. Así el mejor costo **nunca empeora** entre generaciones.
- **Término:** se para al llegar al máximo de generaciones **o** tras `patience` generaciones sin mejora.
""", """
### 11.5 Replacement with elitism and 11.6 termination
- **Elitism:** the `n_elite` best parents survive untouched and the rest of the population is filled with the
  best children. The best cost therefore **never gets worse** between generations.
- **Termination:** stop at the generation limit **or** after `patience` generations without improvement.
""")
code(src(operators.replacement_elitism, operators.should_stop) + """


pop_a = init_random(P_med, 6, rng); cost_a = P_med.evaluate(pop_a)
pop_b = init_random(P_med, 6, rng); cost_b = P_med.evaluate(pop_b)
new_pop, new_costs = replacement_elitism(pop_a, cost_a, pop_b, cost_b, n_elite=2)
print("⟦es⟧Padres⟦en⟧Parents⟦/⟧ :", np.sort(cost_a).round(0))
print("⟦es⟧Hijos⟦en⟧Children⟦/⟧ :", np.sort(cost_b).round(0))
print("⟦es⟧Nueva⟦en⟧New⟦/⟧    :", new_costs.round(0), "← ⟦es⟧2 élites + 4 mejores hijos⟦en⟧2 elites + 4 best children⟦/⟧")
print()
for g, stall in [(10, 3), (200, 0), (57, 50)]:
    print(f"should_stop(gen={g}, max=200, stall={stall}, patience=50) = {should_stop(g, 200, stall, 50)}")
""")

# =============================================================================
# 12. Optimizer class
# =============================================================================
md("""
## 12. La clase `GeneticShoppingOptimizer`
Agrupa los pasos anteriores, **un método por paso**, y recibe todos los hiperparámetros en el constructor.
La técnica de selección, cruce y mutación se elige por nombre, así compararlas es trivial.

¿Por qué una clase y no una función gigante? Porque cada paso queda aislado y se puede cambiar o probar por separado, y
porque los benchmarks de más abajo se reducen a crear el optimizador con otros parámetros.
""", """
## 12. The `GeneticShoppingOptimizer` class
It bundles the previous steps, **one method per step**, and takes every hyperparameter in its constructor.
Selection, crossover and mutation are chosen by name, which makes comparing them trivial.

Why a class instead of one big function? Each step stays isolated and can be swapped or tested on its own, and the
benchmarks below boil down to creating the optimizer with different parameters.
""")
code(src(optimizer.GAResult, optimizer.GeneticShoppingOptimizer) + """


# ⟦es⟧Hiperparámetros y sus valores por defecto⟦en⟧Hyperparameters and their default values⟦/⟧
pd.Series({name: p.default for name, p in inspect.signature(GeneticShoppingOptimizer.__init__).parameters.items()
           if p.default is not inspect.Parameter.empty}, name="default").to_frame()
""")

# =============================================================================
# 13. Run one problem
# =============================================================================
md("""
## 13. Ejecución en un problema
Un problema mediano (30 tiendas) **sin seeding**, para ver al AG evolucionar desde una población aleatoria.
""", """
## 13. Running one problem
A medium problem (30 stores) **without seeding**, to watch the GA evolve from a random population.
""")
code("""
r_run = problemset[problemset.problem_cluster.str.startswith("medium-30")].iloc[7]
P_run = ShoppingProblem(instances[r_run.catalog_path], r_run.query, r_run.fuel_price)
print(f"{r_run.problem_cluster} | k = {P_run.k} ⟦es⟧productos⟦en⟧products⟦/⟧ | φ = {P_run.fuel_price}")
ga = GeneticShoppingOptimizer(P_run, pop_size=80, generations=300, seeding=False, seed=SEED)
res = ga.run(verbose=True)
print()
print(res.answer)
print(f"⟦es⟧Costo⟦en⟧Cost⟦/⟧ = {res.best_cost:,.2f} | best_known = {r_run.best_known:,.2f} | "
      f"⟦es⟧generaciones⟦en⟧generations⟦/⟧ = {res.generations} | ⟦es⟧evaluaciones⟦en⟧evaluations⟦/⟧ = {res.evaluations} | {res.elapsed:.2f} s")
""")
code("""
h = res.history
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(h.generation, h["mean"], label="⟦es⟧promedio de la población⟦en⟧population mean⟦/⟧", color=PALETTE[1])
axes[0].plot(h.generation, h["best"], label="⟦es⟧mejor individuo⟦en⟧best individual⟦/⟧", color=PALETTE[0])
axes[0].axhline(r_run.best_known, ls="--", color="#52514e", lw=1, label="best_known")
axes[0].set_xlabel("⟦es⟧generación⟦en⟧generation⟦/⟧"); axes[0].set_ylabel("⟦es⟧costo⟦en⟧cost⟦/⟧")
axes[0].set_title("⟦es⟧Convergencia⟦en⟧Convergence⟦/⟧"); axes[0].legend()
axes[1].plot(h.generation, 100 * h["diversity"], color=PALETTE[2])
axes[1].set_xlabel("⟦es⟧generación⟦en⟧generation⟦/⟧"); axes[1].set_ylabel("⟦es⟧% individuos únicos⟦en⟧% unique individuals⟦/⟧")
axes[1].set_title("⟦es⟧Diversidad de la población⟦en⟧Population diversity⟦/⟧")
plt.tight_layout()
save_fig(fig, "convergence")
plt.show()
""")
md("""
### 13.1 La ruta sobre el grafo real
En el catálogo `parque` (grafo disperso), la ruta óptima entre tiendas puede pasar por nodos intermedios.
Reconstruimos el camino real para dibujarlo. Elegimos este catálogo justamente por eso: es el único donde se ve que la
"ruta" entre dos tiendas en realidad pasa por otros lugares.
""", """
### 13.1 The route on the real graph
In the `parque` catalog (sparse graph) the best path between stores may go through intermediate nodes.
We rebuild the real path to draw it. We picked this catalog precisely for that: it is the only one where you can see that
the "route" between two stores actually goes through other places.
""")
code("""
r_pq = problemset[problemset.problem_cluster.str.endswith("parque")].sort_values("k").iloc[-1]
P_pq = ShoppingProblem(instances[r_pq.catalog_path], r_pq.query, r_pq.fuel_price)
res_pq = GeneticShoppingOptimizer(P_pq, seed=SEED).run()
print(res_pq.answer)
print(f"⟦es⟧Costo⟦en⟧Cost⟦/⟧ = {res_pq.best_cost:,.2f} (best_known = {r_pq.best_known:,.2f}, φ = {r_pq.fuel_price})")

Gp = nx.DiGraph()
for a, b, d in parque_graph.itertuples(index=False):
    if a != b:
        Gp.add_edge(a, b, weight=d)
stops = ["Casa"] + [s for s, _ in P_pq.decode(res_pq.best_chromosome)] + ["Casa"]
path_edges = set()
for a, b in zip(stops, stops[1:]):
    p = nx.shortest_path(Gp, a, b, weight="weight")
    path_edges |= set(zip(p, p[1:]))
fig, ax = plt.subplots(figsize=(8, 6))
draw_graph(parque_graph, ax, "⟦es⟧Ruta óptima (rojo)⟦en⟧Optimal route (red)⟦/⟧: " + " → ".join(stops), highlight=path_edges)
save_fig(fig, "route_parque")
plt.show()
""")

# =============================================================================
# 14. Baselines
# =============================================================================
md("""
## 14. Baselines y óptimo exacto
Para saber qué tan buena es una solución necesitamos el **óptimo real**. En problemas pequeños se puede calcular
enumerando **subconjuntos de tiendas** ($2^S$ en vez de $S^k$ asignaciones): para cada subconjunto se compra cada
producto donde es más barato *dentro* del subconjunto y la ruta se resuelve con Held-Karp.

Como las distancias ya son caminos mínimos, visitar menos tiendas nunca alarga la ruta, así que el mínimo sobre
todos los subconjuntos **es el óptimo global**.
""", """
## 14. Baselines and exact optimum
To know how good a solution is we need the **true optimum**. On small problems it can be computed by enumerating
**subsets of stores** ($2^S$ instead of $S^k$ assignments): for each subset buy every product where it is cheapest
*within* the subset and solve the route with Held-Karp.

Since distances already are shortest paths, visiting fewer stores never makes the route longer, so the minimum
over all subsets **is the global optimum**.
""")
code(src(baselines.exact_search) + """


small_rows = problemset[problemset.problem_cluster.str.startswith("small")]
exact_rows = []
for i, r in small_rows.iterrows():
    Pp = ShoppingProblem(instances[r.catalog_path], r.query, r.fuel_price)
    t = time.perf_counter(); _, opt_cost = exact_search(Pp); t_exact = time.perf_counter() - t
    exact_rows.append({"id": i, "cluster": "-".join(r.problem_cluster.split("-")[i] for i in (1, -1)), "exact": opt_cost,
                       "qbf": r["qbf best value"], "best_known": r.best_known, "t_exact": t_exact})
exact_df = pd.DataFrame(exact_rows).set_index("id")
print(f"⟦es⟧Problemas pequeños⟦en⟧Small problems⟦/⟧: {len(exact_df)} | ⟦es⟧tiempo medio exacto⟦en⟧mean exact time⟦/⟧: {1000 * exact_df.t_exact.mean():.1f} ms")
print(f"⟦es⟧¿exacto ≤ best_known en todos?⟦en⟧exact ≤ best_known everywhere?⟦/⟧ {(exact_df.exact <= exact_df.best_known + 1e-6).all()}")
print(f"⟦es⟧Problemas donde el exacto MEJORA la referencia⟦en⟧Problems where exact BEATS the reference⟦/⟧: {(exact_df.exact < exact_df.best_known - 1e-6).sum()}")
exact_df.assign(gain=exact_df.best_known - exact_df.exact).groupby("cluster").agg(
    n=("gain", "size"), ⟦es⟧mejorados⟦en⟧improved⟦/⟧=("gain", lambda g: (g > 1e-6).sum()),
    ⟦es⟧ahorro_medio⟦en⟧mean_saving⟦/⟧=("gain", "mean"), ⟦es⟧ahorro_max⟦en⟧max_saving⟦/⟧=("gain", "max"))
""")
md("""
La referencia "cuasi fuerza bruta" **no siempre es óptima**: nuestro método exacto encuentra soluciones más
baratas en varios problemas. Esto es importante para interpretar las métricas, porque un **error negativo**
significa que superamos a la referencia.
""", """
The "quasi brute force" reference **is not always optimal**: our exact method finds cheaper solutions on
several problems. This matters when reading the metrics, because a **negative error** means we beat the reference.
""")

# =============================================================================
# 15. Validation
# =============================================================================
md("""
## 15. Validación con los 136 problemas
Corremos el AG sobre todo el dataset **en paralelo** (`joblib`, un proceso por núcleo del CPU), en dos variantes:
- **AG (seeding):** configuración por defecto, con la población inicial sembrada con los greedy.
- **AG (aleatorio):** sin seeding, el AG "puro".

Además recalculamos nuestros greedy (con ruta óptima) y los comparamos con las 4 referencias del dataset.

Corremos dos variantes del AG para ser honestos con el resultado: si solo mostráramos la versión con seeding, no se
sabría cuánto del mérito es del AG y cuánto de las heurísticas con las que arranca. Y lo paralelizamos porque cada
problema es independiente, que es el caso ideal para repartir trabajo entre núcleos.
""", """
## 15. Validation on the 136 problems
We run the GA over the whole dataset **in parallel** (`joblib`, one process per CPU core) in two variants:
- **GA (seeding):** default configuration, initial population seeded with the greedy solutions.
- **GA (random):** no seeding, the "pure" GA.

We also recompute our greedy baselines (with optimal routing) and compare against the dataset's 4 references.

We run two GA variants to be honest about the result: if we only showed the seeded version, nobody could tell how much of
the merit belongs to the GA and how much to the heuristics it starts from. We parallelize because every problem is
independent, which is the ideal case for spreading work across cores.
""")
code(src(metrics.error_columns, metrics.summary_metrics) + """


def run_ga(instance, query, fuel, seed, **config):
    problem = ShoppingProblem(instance, query, fuel)
    result = GeneticShoppingOptimizer(problem, seed=seed, **config).run()
    return {"cost": result.best_cost, "generations": result.generations, "evaluations": result.evaluations,
            "time": result.elapsed, "answer": result.answer,
            "cache_hit_%": 100 * problem.router.hits / max(problem.router.hits + problem.router.misses, 1)}


def run_all(rows, configs, seeds=(SEED,)):
    jobs = [(i, name, s) for i in rows.index for name in configs for s in seeds]
    out = Parallel(n_jobs=N_JOBS)(
        delayed(run_ga)(instances[rows.loc[i, "catalog_path"]], rows.loc[i, "query"], rows.loc[i, "fuel_price"], s, **configs[name])
        for i, name, s in jobs)
    return pd.DataFrame([{"id": i, "config": name, "seed": s, **o} for (i, name, s), o in zip(jobs, out)])


val = cached("runs_validation", lambda: run_all(problemset, {"seeded": {"seeding": True}, "random": {"seeding": False}}))

results = problemset[["problem_cluster", "k", "fuel_price", "log10_space", "best_known", "qbf best value",
                      "ga best value", "gc best value", "gn best value"]].copy()
results["GA_seeded"] = val[val.config == "seeded"].set_index("id")["cost"]
results["GA_random"] = val[val.config == "random"].set_index("id")["cost"]
results["time_seeded"] = val[val.config == "seeded"].set_index("id")["time"]
results["generations"] = val[val.config == "seeded"].set_index("id")["generations"]
results["cache_hit_%"] = val[val.config == "seeded"].set_index("id")["cache_hit_%"]
results["exact"] = exact_df["exact"]
for i, r in problemset.iterrows():
    Pp = ShoppingProblem(instances[r.catalog_path], r.query, r.fuel_price)
    results.loc[i, "greedy_cheapest"] = Pp.cost(greedy_cheapest(Pp))
    results.loc[i, "greedy_nearest"] = Pp.cost(greedy_nearest(Pp))
results["size"] = results.problem_cluster.str.split("-").str[0]
results.to_csv(RESULTS_DIR / "validation_results.csv")
results[["problem_cluster", "k", "best_known", "exact", "GA_seeded", "GA_random", "greedy_cheapest", "greedy_nearest"]].head(10)
""")
md("""
### 15.1 Métricas globales (respecto a `best_known` de las referencias)
- **E** = predicho − referencia (con signo; negativo = mejor que la referencia)
- **AE** = |E| · **RE** = E / referencia · **MAE**, **MSE**, **RMSE**, **MAPE**, **R²**
- **hit rate** = % de problemas donde igualamos o superamos a la referencia

⚠️ Cuando **superamos** a la referencia, el error absoluto también crece. Por eso un MAE > 0 no siempre es malo:
hay que mirarlo junto a **ME** (con signo) y al **hit rate**.
""", """
### 15.1 Global metrics (against the references' `best_known`)
- **E** = predicted − reference (signed; negative = better than the reference)
- **AE** = |E| · **RE** = E / reference · **MAE**, **MSE**, **RMSE**, **MAPE**, **R²**
- **hit rate** = % of problems where we match or beat the reference

⚠️ When we **beat** the reference the absolute error grows too. So an MAE > 0 is not always bad: read it together
with **ME** (signed) and the **hit rate**.
""")
code("""
solvers = {"⟦es⟧AG (seeding) — nuestro⟦en⟧GA (seeding) — ours⟦/⟧": "GA_seeded", "⟦es⟧AG (aleatorio) — nuestro⟦en⟧GA (random) — ours⟦/⟧": "GA_random",
           "⟦es⟧Greedy barato — nuestro⟦en⟧Greedy cheapest — ours⟦/⟧": "greedy_cheapest", "⟦es⟧Greedy cercano — nuestro⟦en⟧Greedy nearest — ours⟦/⟧": "greedy_nearest",
           "⟦es⟧ref: AG⟦en⟧ref: GA⟦/⟧": "ga best value", "⟦es⟧ref: cuasi fuerza bruta⟦en⟧ref: quasi brute force⟦/⟧": "qbf best value",
           "⟦es⟧ref: greedy barato⟦en⟧ref: greedy cheapest⟦/⟧": "gc best value", "⟦es⟧ref: greedy cercano⟦en⟧ref: greedy nearest⟦/⟧": "gn best value"}
global_metrics = pd.DataFrame({name: summary_metrics(results[col], results.best_known) for name, col in solvers.items()}).T
global_metrics.to_csv(RESULTS_DIR / "global_metrics.csv")
global_metrics.style.format("{:,.3f}").format({"n": "{:.0f}"}).background_gradient(subset=["MAE"], cmap="Blues")
""")
code("""
by_cluster = results.groupby("problem_cluster").apply(lambda g: pd.Series({
    "n": len(g),
    "MAE ⟦es⟧AG seeding⟦en⟧GA seeding⟦/⟧": summary_metrics(g.GA_seeded, g.best_known)["MAE"],
    "MAE ⟦es⟧AG aleatorio⟦en⟧GA random⟦/⟧": summary_metrics(g.GA_random, g.best_known)["MAE"],
    "MAE ref GA": summary_metrics(g["ga best value"], g.best_known)["MAE"],
    "ME ⟦es⟧AG seeding⟦en⟧GA seeding⟦/⟧": summary_metrics(g.GA_seeded, g.best_known)["ME"],
    "hit_% ⟦es⟧AG seeding⟦en⟧GA seeding⟦/⟧": summary_metrics(g.GA_seeded, g.best_known)["hit_rate_%"],
    "⟦es⟧% óptimo exacto⟦en⟧% exact optimum⟦/⟧": 100 * (abs(g.GA_seeded - g.exact) < 1e-6).mean() if g.exact.notna().all() else np.nan,
    "⟦es⟧tiempo medio (s)⟦en⟧mean time (s)⟦/⟧": g.time_seeded.mean(),
}))
by_cluster
""")
code("""
small_mask = results.exact.notna()
optimal = (abs(results.loc[small_mask, "GA_seeded"] - results.loc[small_mask, "exact"]) < 1e-6)
improved = (results.GA_seeded < results.best_known - 1e-6)
print(f"⟦es⟧Problemas pequeños donde el AG encuentra el ÓPTIMO EXACTO⟦en⟧Small problems where the GA finds the EXACT OPTIMUM⟦/⟧: {optimal.sum()}/{len(optimal)} ({100 * optimal.mean():.1f}%)")
print(f"⟦es⟧Problemas donde el AG MEJORA la mejor referencia conocida⟦en⟧Problems where the GA BEATS the best known reference⟦/⟧: {improved.sum()}/{len(results)}")
print(f"⟦es⟧Ahorro total frente a best_known⟦en⟧Total savings vs best_known⟦/⟧: {(results.best_known - results.GA_seeded).sum():,.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
err = results.GA_seeded - results.best_known
for j, size in enumerate(["small", "medium", "big"]):
    m = results["size"] == size
    axes[0].scatter(results.loc[m, "best_known"], results.loc[m, "GA_seeded"], s=22, color=PALETTE[j], label=size,
                    edgecolor="white", linewidth=0.8)
lim = [results.best_known.min(), results.best_known.max()]
axes[0].plot(lim, lim, color="#52514e", lw=1, ls="--")
axes[0].set_xlabel("best_known (⟦es⟧referencia⟦en⟧reference⟦/⟧)"); axes[0].set_ylabel("⟦es⟧AG (nuestro)⟦en⟧GA (ours)⟦/⟧")
axes[0].set_title("⟦es⟧Predicho vs referencia⟦en⟧Predicted vs reference⟦/⟧"); axes[0].legend()
data_box = [results.loc[results["size"] == s, "GA_seeded"] - results.loc[results["size"] == s, "best_known"] for s in ["small", "medium", "big"]]
data_box_r = [results.loc[results["size"] == s, "GA_random"] - results.loc[results["size"] == s, "best_known"] for s in ["small", "medium", "big"]]
pos = np.arange(3)
b1 = axes[1].boxplot(data_box, positions=pos - 0.18, widths=0.3, patch_artist=True, showfliers=True)
b2 = axes[1].boxplot(data_box_r, positions=pos + 0.18, widths=0.3, patch_artist=True, showfliers=True)
for b, c in [(b1, PALETTE[0]), (b2, PALETTE[1])]:
    for patch in b["boxes"]:
        patch.set_facecolor(c); patch.set_alpha(0.8)
axes[1].set_xticks(pos, ["small", "medium", "big"]); axes[1].axhline(0, color="#52514e", lw=1)
axes[1].set_ylabel("E = ⟦es⟧AG − referencia⟦en⟧GA − reference⟦/⟧"); axes[1].set_title("⟦es⟧Error por tamaño (negativo = mejor)⟦en⟧Error by size (negative = better)⟦/⟧")
axes[1].legend([b1["boxes"][0], b2["boxes"][0]], ["seeding", "⟦es⟧aleatorio⟦en⟧random⟦/⟧"])
plt.tight_layout()
save_fig(fig, "validation")
plt.show()
""")

# =============================================================================
# 16. Operator benchmark
# =============================================================================
md("""
## 16. Benchmark de técnicas: 3 selecciones × 3 cruces × 2 mutaciones
Comparamos las 18 combinaciones **sin seeding** (para medir el operador, no la heurística inicial) en los 24
problemas medianos y grandes de 30 y 100 tiendas, que es donde hay diferencias, con 2 semillas cada uno.
Métrica de calidad: **RE %** = error relativo frente a `best_known`.
""", """
## 16. Technique benchmark: 3 selections × 3 crossovers × 2 mutations
We compare all 18 combinations **without seeding** (to measure the operator, not the initial heuristic) on the
24 medium and big problems with 30 and 100 stores, which is where differences show, with 2 seeds each.
Quality metric: **RE %** = relative error against `best_known`.
""")
code("""
bench_rows = problemset[problemset.problem_cluster.str.startswith(("medium-30", "big"))]
configs = {f"{s}|{c}|{m}": {"selection": s, "crossover": c, "mutation": m, "seeding": False}
           for s in ["tournament", "roulette", "rank"] for c in ["one_point", "two_point", "uniform"]
           for m in ["random_reset", "consolidate"]}
bench = cached("runs_operator_benchmark", lambda: run_all(bench_rows, configs, seeds=(1, 2)))
bench[["selection", "crossover", "mutation"]] = bench.config.str.split("|", expand=True)
bench["RE_%"] = 100 * (bench.cost - problemset.loc[bench.id, "best_known"].to_numpy()) / problemset.loc[bench.id, "best_known"].to_numpy()
bench.to_csv(RESULTS_DIR / "operator_benchmark.csv", index=False)
ranking = bench.groupby(["selection", "crossover", "mutation"]).agg(
    RE_mean=("RE_%", "mean"), RE_std=("RE_%", "std"), time_s=("time", "mean"),
    generations=("generations", "mean"), evaluations=("evaluations", "mean")).sort_values("RE_mean")
ranking.style.format("{:.3f}").format({"generations": "{:.0f}", "evaluations": "{:,.0f}"}).background_gradient(subset=["RE_mean"], cmap="Blues")
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
for ax, mut in zip(axes, ["random_reset", "consolidate"]):
    pivot = bench[bench.mutation == mut].pivot_table(index="selection", columns="crossover", values="RE_%", aggfunc="mean")
    im = ax.imshow(pivot.to_numpy(), cmap="Blues", vmin=0, vmax=bench.groupby(["selection", "crossover", "mutation"])["RE_%"].mean().max())
    ax.set_xticks(range(3), pivot.columns); ax.set_yticks(range(3), pivot.index); ax.grid(False)
    for i in range(3):
        for j in range(3):
            v = pivot.to_numpy()[i, j]
            ax.text(j, i, f"{v:.2f}%", ha="center", va="center", color="white" if v > pivot.to_numpy().mean() else "#0b0b0b")
    ax.set_title(f"⟦es⟧Mutación⟦en⟧Mutation⟦/⟧: {mut}")
fig.colorbar(im, ax=axes, label="RE % ⟦es⟧medio (menor = mejor)⟦en⟧mean (lower = better)⟦/⟧")
save_fig(fig, "operator_heatmap")
plt.show()
""")
md("""
**Hallazgos del benchmark:**
- **La mutación decide casi todo.** Con *reinicio aleatorio* el RE medio es ≈ 3 %, y con *consolidación* sube a ≈ 49 %.
  La consolidación junta productos en pocas tiendas demasiado rápido, la diversidad colapsa y el AG converge
  **prematuramente** (para antes: ~75 generaciones contra ~120).
- **Torneo** es la mejor selección: ≈ 0 % de RE con reinicio aleatorio, y es además la más rápida. Ruleta y ranking
  quedan en 3–6 %, coherente con la baja presión selectiva vista en la sección 11.2.
- El tipo de **cruce** pesa poco frente a los otros dos factores.
""", """
**Benchmark findings:**
- **Mutation decides almost everything.** With *random reset* the mean RE is ≈ 3 %, and with *consolidation* it rises
  to ≈ 49 %. Consolidation packs products into few stores too fast, diversity collapses and the GA converges
  **prematurely** (it stops earlier: ~75 generations vs ~120).
- **Tournament** is the best selection: ≈ 0 % RE with random reset, and also the fastest. Roulette and rank stay at
  3–6 %, consistent with the low selective pressure seen in section 11.2.
- The **crossover** type matters little compared to the other two factors.
""")
md("""
### 16.1 Memoria y tiempo por técnica
Medimos el **pico de memoria** con `tracemalloc` corriendo cada combinación sobre un problema mediano (20 tiendas)
de forma secuencial, para no mezclar procesos. `tracemalloc` hace todo más lento, así que aquí los tiempos solo
sirven para comparar entre sí.
""", """
### 16.1 Memory and time per technique
We measure **peak memory** with `tracemalloc` by running each combination on a medium problem (20 stores)
sequentially, so processes don't interfere. `tracemalloc` slows everything down, so times here are only
meaningful relative to each other.
""")
code("""
r_mem = problemset[problemset.problem_cluster.str.startswith("medium-20")].sort_values("k").iloc[-1]


def memory_benchmark():
    rows = []
    for name, cfg in configs.items():
        P_b = ShoppingProblem(instances[r_mem.catalog_path], r_mem.query, r_mem.fuel_price)
        tracemalloc.start()
        res_b = GeneticShoppingOptimizer(P_b, seed=1, **cfg).run()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({"config": name, "peak_MB": peak / 2**20, "time_s": res_b.elapsed, "generations": res_b.generations,
                     "RE_%": 100 * (res_b.best_cost - r_mem.best_known) / r_mem.best_known})
    return pd.DataFrame(rows)


mem = cached("memory_benchmark", memory_benchmark, parallel=False).set_index("config").sort_values("RE_%")
mem
""")

# =============================================================================
# 17. Hyperparameters & scalability
# =============================================================================
md("""
## 17. Sensibilidad de hiperparámetros y escalabilidad
Variamos **un hiperparámetro a la vez** sobre la configuración base (torneo + uniforme + reinicio aleatorio,
sin seeding) en los mismos 24 problemas. Cambiar uno a la vez no encuentra la mejor combinación posible, pero sí deja
ver el efecto de cada parámetro por separado, que es lo que nos interesa para entenderlos.
""", """
## 17. Hyperparameter sensitivity and scalability
We vary **one hyperparameter at a time** around the base configuration (tournament + uniform + random reset,
no seeding) on the same 24 problems. Changing one at a time won't find the best possible combination, but it shows the
effect of each parameter on its own, which is what we care about to understand them.
""")
code("""
base_cfg = {"selection": "tournament", "crossover": "uniform", "mutation": "random_reset", "seeding": False}
grid = {
    "pm × k": [("pm", v) for v in [0.25, 0.5, 1.0, 2.0, 4.0]],
    "pop_size": [("pop_size", v) for v in [20, 40, 80, 160]],
    "tournament_k": [("tournament_k", v) for v in [2, 3, 5, 8]],
    "elitism": [("elitism", v) for v in [0, 1, 2, 5, 10]],
}
sens_configs = {}
for param, values in grid.items():
    for key, v in values:
        cfg = dict(base_cfg)
        if key == "pm":
            cfg["pm_factor"] = v
        else:
            cfg[key] = v
        sens_configs[f"{param}={v}"] = cfg


def run_sens(instance, query, fuel, seed, pm_factor=None, **cfg):
    if pm_factor is not None:
        cfg["pm"] = pm_factor / len(query)
    return run_ga(instance, query, fuel, seed, **cfg)


def sensitivity_runs():
    jobs = [(i, n, s) for i in bench_rows.index for n in sens_configs for s in (1, 2)]
    out = Parallel(n_jobs=N_JOBS)(delayed(run_sens)(instances[problemset.loc[i, "catalog_path"]], problemset.loc[i, "query"],
                                                    problemset.loc[i, "fuel_price"], s, **sens_configs[n]) for i, n, s in jobs)
    return pd.DataFrame([{"id": i, "config": n, **o} for (i, n, s), o in zip(jobs, out)])


sens = cached("runs_sensitivity", sensitivity_runs)
sens[["param", "value"]] = sens.config.str.split("=", expand=True)
sens["value"] = sens.value.astype(float)
sens["RE_%"] = 100 * (sens.cost - problemset.loc[sens.id, "best_known"].to_numpy()) / problemset.loc[sens.id, "best_known"].to_numpy()
sens.to_csv(RESULTS_DIR / "hyperparameter_sensitivity.csv", index=False)

fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex="col")
for j, param in enumerate(grid):
    g = sens[sens.param == param].groupby("value").agg(RE=("RE_%", "mean"), t=("time", "mean"))
    axes[0, j].plot(g.index, g.RE, marker="o", color=PALETTE[0]); axes[0, j].set_title(param)
    axes[1, j].plot(g.index, g.t, marker="o", color=PALETTE[1]); axes[1, j].set_xlabel(param)
    if param == "pop_size":
        axes[0, j].set_xscale("log", base=2); axes[1, j].set_xscale("log", base=2)
axes[0, 0].set_ylabel("RE % ⟦es⟧medio⟦en⟧mean⟦/⟧"); axes[1, 0].set_ylabel("⟦es⟧tiempo (s)⟦en⟧time (s)⟦/⟧")
plt.tight_layout()
save_fig(fig, "sensitivity")
plt.show()
""")
md("""
**Lectura de la sensibilidad:**
- **`pm`**: es el parámetro más delicado. Entre 0.25/k y 1/k el AG funciona bien, pero con 2/k el RE sube a ≈ 10 % y con 4/k
  pasa de 100 %. Demasiada mutación convierte la búsqueda en un recorrido aleatorio.
- **Elitismo**: sin élites (0) el mejor individuo se puede perder y el RE sube a ≈ 4.5 %. Con 1 a 10 élites el RE queda cerca
  de 0 y la convergencia es más rápida.
- **`pop_size`**: más población da mejor calidad, pero el tiempo crece casi linealmente. 80 es un buen equilibrio.
- **`tournament_k`**: k = 5 mejora a k = 3 (más presión selectiva), pero k = 8 ya no mejora más.

Como estos valores se ajustaron sobre los mismos 24 problemas, recomendarlos como defaults sería *sobreajustar*.
Los dejamos documentados como recomendación.
""", """
**Reading the sensitivity:**
- **`pm`** is the most delicate parameter. Between 0.25/k and 1/k the GA works well, but at 2/k RE rises to ≈ 10 % and at
  4/k it exceeds 100 %. Too much mutation turns the search into a random walk.
- **Elitism**: with no elites (0) the best individual can be lost and RE rises to ≈ 4.5 %. With 1 to 10 elites RE stays near
  0 and convergence is faster.
- **`pop_size`**: a bigger population gives better quality, but time grows almost linearly. 80 is a good balance.
- **`tournament_k`**: k = 5 beats k = 3 (more selective pressure), but k = 8 brings no further gain.

Since these values were tuned on the same 24 problems, making them the defaults would be *overfitting*. We keep them
documented as a recommendation.
""")
md("""
### 17.1 Escalabilidad
¿Cómo crecen el tiempo y la calidad con el tamaño del problema? Comparamos también el costo teórico del método
exacto ($2^S$ subconjuntos) con el del AG (aproximadamente `pop_size × generaciones` evaluaciones, independiente de $S$).
""", """
### 17.1 Scalability
How do time and quality grow with problem size? We also compare the theoretical cost of the exact method
($2^S$ subsets) with the GA's (about `pop_size × generations` evaluations, independent of $S$).
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
for j, size in enumerate(["small", "medium", "big"]):
    m = results["size"] == size
    axes[0].scatter(results.loc[m, "log10_space"], results.loc[m, "time_seeded"], s=22, color=PALETTE[j], label=size,
                    edgecolor="white", linewidth=0.8)
axes[0].set_xlabel("log10(⟦es⟧espacio de búsqueda⟦en⟧search space⟦/⟧)"); axes[0].set_ylabel("⟦es⟧tiempo del AG (s)⟦en⟧GA time (s)⟦/⟧")
axes[0].set_title("⟦es⟧Tiempo del AG vs tamaño del espacio⟦en⟧GA time vs search-space size⟦/⟧"); axes[0].legend()

S = np.array([3, 5, 10, 20, 30, 100])
axes[1].semilogy(S, 2.0 ** S, marker="o", label="⟦es⟧exacto: 2^S subconjuntos⟦en⟧exact: 2^S subsets⟦/⟧", color=PALETTE[7])
axes[1].semilogy(S, np.full(len(S), results.generations.mean() * 80), marker="o", label="⟦es⟧AG: ~evaluaciones⟦en⟧GA: ~evaluations⟦/⟧", color=PALETTE[0])
axes[1].set_xlabel("⟦es⟧tiendas candidatas S⟦en⟧candidate stores S⟦/⟧"); axes[1].set_ylabel("⟦es⟧operaciones (log)⟦en⟧operations (log)⟦/⟧")
axes[1].set_title("⟦es⟧Escalabilidad teórica⟦en⟧Theoretical scalability⟦/⟧"); axes[1].legend()
plt.tight_layout()
save_fig(fig, "scalability")
plt.show()
results.groupby("size")[["time_seeded", "generations", "cache_hit_%"]].mean()
""")

# =============================================================================
# 18. CPU vs GPU
# =============================================================================
md("""
## 18. Experimento: ¿conviene usar la GPU?
Para que la evaluación sea vectorizable en la GPU, la ruta sigue el **orden de los genes** (igual que las
soluciones de referencia) en lugar del TSP. Evaluamos poblaciones de $10^2$ a $10^6$ individuos con:
NumPy (CPU), PyTorch (CPU) y PyTorch (CUDA).

Este experimento existe porque la pregunta "¿por qué no usar la GPU?" es razonable, y la respuesta correcta no es una
opinión sino una medición.
""", """
## 18. Experiment: is the GPU worth it?
To make evaluation vectorizable on the GPU, the route follows the **gene order** (like the reference
solutions) instead of the TSP. We evaluate populations of $10^2$ to $10^6$ individuals with NumPy (CPU),
PyTorch (CPU) and PyTorch (CUDA).

This experiment exists because "why not use the GPU?" is a fair question, and the right answer is a measurement, not an
opinion.
""")
code(src(gpu.batch_cost_numpy, gpu.batch_cost_torch) + """


def bench_fn(fn, repeat=5, sync=None):
    fn()
    if sync: sync()
    t = time.perf_counter()
    for _ in range(repeat):
        fn()
    if sync: sync()
    return 1000 * (time.perf_counter() - t) / repeat


r_big = problemset[problemset.problem_cluster.str.startswith("big")].sort_values("k").iloc[-1]
P_g = ShoppingProblem(instances[r_big.catalog_path], r_big.query, r_big.fuel_price)
price_np = np.where(np.isinf(P_g.price), 1e9, P_g.price)
gpu_rows = []
for n in [10**2, 10**3, 10**4, 10**5, 10**6]:
    pop_np = init_random(P_g, 1000, rng)[rng.integers(0, 1000, n)]
    rec = {"⟦es⟧población⟦en⟧population⟦/⟧": n, "NumPy CPU (ms)": bench_fn(lambda: batch_cost_numpy(pop_np, price_np, P_g.instance.D, P_g.fuel_price))}
    if torch is not None:
        for dev in ["cpu"] + (["cuda"] if GPU else []):
            pt, pr, dt = (torch.as_tensor(x, device=dev) for x in (pop_np, price_np, P_g.instance.D))
            sync = torch.cuda.synchronize if dev == "cuda" else None
            rec[f"PyTorch {dev.upper()} (ms)"] = bench_fn(lambda: batch_cost_torch(pt, pr, dt, P_g.fuel_price), sync=sync)
            if dev == "cuda":
                rec["⟦es⟧CUDA + copia (ms)⟦en⟧CUDA + transfer (ms)⟦/⟧"] = bench_fn(
                    lambda: batch_cost_torch(torch.as_tensor(pop_np, device="cuda"), pr, dt, P_g.fuel_price).cpu(), sync=sync)
    gpu_rows.append(rec)
gpu_df = pd.DataFrame(gpu_rows).set_index("⟦es⟧población⟦en⟧population⟦/⟧")
gpu_df.to_csv(RESULTS_DIR / "cpu_vs_gpu.csv")

fig, ax = plt.subplots(figsize=(9, 4.2))
for j, col in enumerate(gpu_df.columns):
    ax.loglog(gpu_df.index, gpu_df[col], marker="o", label=col, color=PALETTE[j])
ax.axvline(80, color="#52514e", ls="--", lw=1)
ax.text(90, ax.get_ylim()[1] * 0.4, "⟦es⟧población\\nusada por el AG⟦en⟧population\\nused by the GA⟦/⟧", fontsize=8, color="#52514e")
ax.set_xlabel("⟦es⟧individuos evaluados⟦en⟧individuals evaluated⟦/⟧"); ax.set_ylabel("ms ⟦es⟧por evaluación⟦en⟧per evaluation⟦/⟧")
ax.set_title(f"⟦es⟧Evaluación vectorizada⟦en⟧Vectorized evaluation⟦/⟧ — {GPU or 'CPU only'}"); ax.legend(fontsize=8)
save_fig(fig, "cpu_vs_gpu")
plt.show()
gpu_df
""")
md("""
**Conclusión del experimento:**
- Con **80 individuos** (lo que usa el AG), NumPy en CPU es el más rápido: lanzar trabajo a la GPU cuesta más que el cálculo.
- A partir de **~10⁴ individuos** la GPU gana. Con **10⁶**, CUDA es ~80× más rápido que NumPy, o ~20–25× contando la copia de datos.
- Aun así, el cuello de botella real del AG es el **decodificador TSP** (programación dinámica, diccionarios), que no se
  vectoriza. Por eso se paraleliza *entre problemas* en el CPU.
- **Cuándo valdría la pena:** un AG "masivo" que evolucione cientos de miles de individuos, o todos los problemas a la
  vez, como tensores en la GPU. Queda como mejora futura.
""", """
**Experiment takeaway:**
- With **80 individuals** (what the GA uses), NumPy on the CPU is the fastest: launching work on the GPU costs more than
  the computation itself.
- From **~10⁴ individuals** the GPU wins. At **10⁶**, CUDA is ~80× faster than NumPy, or ~20–25× including the data copy.
- Still, the GA's real bottleneck is the **TSP decoder** (dynamic programming, dictionaries), which does not vectorize.
  That is why we parallelize *across problems* on the CPU.
- **When it would pay off:** a "massive" GA evolving hundreds of thousands of individuals, or all problems at once, as
  GPU tensors. Left as future work.
""")

# =============================================================================
# 19. Data-flow diagram, risks, conclusions
# =============================================================================
md("""
## 19. Diagrama de flujo de datos
El resumen de todo lo anterior en una imagen: qué entra, qué se transforma y qué sale. Es la vista que conviene tener en
la cabeza antes de leer el código del paquete `src/`.
""", """
## 19. Data-flow diagram
Everything above summarized in one picture: what goes in, what gets transformed and what comes out. It is the view worth
keeping in mind before reading the package code in `src/`.
""")
code("""
fig, ax = plt.subplots(figsize=(15, 4.6))
ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 4.6)
boxes = {
    "xlsx": (0.2, 3.0, "⟦es⟧Excel\\ncatalogo + grafo⟦en⟧Excel\\ncatalog + graph⟦/⟧", PALETTE[1]),
    "price": (3.0, 3.6, "⟦es⟧Matriz de precios\\ntiendas × productos⟦en⟧Price matrix\\nstores × products⟦/⟧", PALETTE[0]),
    "dist": (3.0, 1.9, "⟦es⟧Adyacencia dirigida\\n→ Floyd-Warshall⟦en⟧Directed adjacency\\n→ Floyd-Warshall⟦/⟧", PALETTE[0]),
    "query": (0.2, 0.7, "⟦es⟧Lista de compras\\n+ combustible⟦en⟧Shopping list\\n+ fuel price⟦/⟧", PALETTE[1]),
    "problem": (6.0, 2.6, "ShoppingProblem\\n⟦es⟧(fitness)⟦en⟧(fitness)⟦/⟧", PALETTE[2]),
    "ga": (8.8, 2.6, "⟦es⟧AG: init → selección\\n→ cruce → mutación\\n→ reemplazo⟦en⟧GA: init → selection\\n→ crossover → mutation\\n→ replacement⟦/⟧", PALETTE[6]),
    "tsp": (8.8, 0.5, "RouteSolver\\nHeld-Karp / NN+2-opt\\n+ cache", PALETTE[2]),
    "out": (11.9, 2.6, "RESPUESTA>\\n(⟦es⟧tienda⟦en⟧store⟦/⟧, [⟦es⟧productos⟦en⟧products⟦/⟧]) → …", PALETTE[5]),
}
for x, y, text, color in boxes.values():
    ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), 2.5, 1.0, boxstyle="round,pad=0.08", color=color, alpha=0.9))
    ax.text(x + 1.25, y + 0.5, text, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
arrow = dict(arrowstyle="-|>", color="#52514e", lw=1.4)
for a, b in [("xlsx", "price"), ("xlsx", "dist"), ("price", "problem"), ("dist", "problem"), ("query", "problem"),
             ("problem", "ga"), ("ga", "out")]:
    xa, ya = boxes[a][0] + 2.5, boxes[a][1] + 0.5
    xb, yb = boxes[b][0], boxes[b][1] + 0.5
    ax.annotate("", xy=(xb, yb), xytext=(xa, ya), arrowprops=arrow)
ax.annotate("", xy=(10.05, 1.5), xytext=(10.05, 2.6), arrowprops=dict(arrowstyle="<|-|>", color="#52514e", lw=1.4))
save_fig(fig, "dataflow")
plt.show()
""")
md("""
## 20. Riesgos, limitaciones, mejoras y conclusiones

### Riesgos y limitaciones
- **El precio domina el costo.** Con estos datos el transporte pesa poco, así que la heurística "más barato" ya está
  cerca del óptimo. El AG aporta más cuando el combustible es caro o los precios se parecen (cluster `same_price`).
- **Distancias casi nulas en `big`.** El 97 % de los pares queda a distancia 0, así que ese cluster no pone a prueba
  la parte de rutas.
- **Sin garantía de optimalidad.** En problemas grandes no conocemos el óptimo real; solo podemos compararnos con las referencias.
- **Convergencia prematura.** La mutación de consolidación sin seeding reduce la diversidad demasiado rápido en
  problemas grandes (se ve en el benchmark).
- **Ruleta con costos similares.** Casi no ejerce presión selectiva (sección 11.2).
- **TSP heurístico con más de 9 tiendas.** NN + 2-opt no siempre es óptimo; en la práctica las rutas buenas visitan pocas tiendas.
- **Modelo de datos.** Supone stock infinito, una unidad por producto y un grafo dirigido conectado.

### Oportunidades de mejora
- AG **híbrido/memético**: búsqueda local (mover cada producto a su mejor tienda) sobre el mejor individuo.
- Parámetros **adaptativos** (pm que sube si la diversidad cae) o modelo de **islas** distribuido entre núcleos.
- Optimización **multiobjetivo** (NSGA-II): costo vs tiempo/distancia.
- Cantidades por producto, horarios de tiendas, ofertas tipo 2×1.
- Paralelizar la evaluación dentro de una ejecución (hoy se paraleliza *entre* problemas).

### Rendimiento y escalabilidad
El AG evalúa ~`pop_size × generaciones` cromosomas sin importar el tamaño del espacio de búsqueda, y la caché de
rutas evita recalcular conjuntos repetidos (≈ 99 % de aciertos en small y ≈ 70 % en medium). Un problema pequeño se
resuelve en ~0.1 s y uno mediano en ~3 s, y las 272 corridas de la validación toman ~25 s en paralelo usando todos
los núcleos. El método exacto crece como $2^S$ y deja de ser viable alrededor de 25–30 tiendas.

### Conclusiones
1. El modelo de costo se validó **504/504** contra las referencias, lo que reveló que el grafo es **dirigido**.
2. Separar **qué comprar** (genes) de **en qué orden ir** (heurística TSP) produce cromosomas siempre válidos y
   un espacio de búsqueda mucho menor.
3. El AG alcanza el **óptimo exacto en 100/100 problemas pequeños**, **iguala o supera a la mejor referencia en los
   136 problemas** (hit rate 100 %) y **la mejora en 30**, con un ahorro acumulado de 2,286.
4. **Torneo + reinicio aleatorio** es la combinación más robusta sin seeding. La mutación de consolidación causa
   convergencia prematura, y el seeding con greedy acelera la convergencia y garantiza no ser peor que ellos.
5. La GPU solo compensa desde ~10⁴ individuos. Con la población del AG (80), el CPU es más rápido, y el paralelismo
   útil está *entre* problemas.
""", """
## 20. Risks, limitations, improvements and conclusions

### Risks and limitations
- **Price dominates cost.** With this data transport weighs little, so the "cheapest" heuristic is already close to
  optimal. The GA adds more when fuel is expensive or prices are similar (`same_price` cluster).
- **Near-zero distances in `big`.** 97 % of the pairs are at distance 0, so that cluster does not really test the
  routing part.
- **No optimality guarantee.** On big problems we don't know the true optimum; we can only compare with the references.
- **Premature convergence.** Consolidation mutation without seeding kills diversity too fast on big problems
  (visible in the benchmark).
- **Roulette with similar costs.** It exerts almost no selective pressure (section 11.2).
- **Heuristic TSP above 9 stores.** NN + 2-opt is not always optimal; in practice good routes visit few stores.
- **Data model.** Assumes infinite stock, one unit per product and a connected directed graph.

### Improvement opportunities
- **Hybrid/memetic** GA: local search (move each product to its best store) on the best individual.
- **Adaptive** parameters (raise pm when diversity drops) or a distributed **island** model across cores.
- **Multi-objective** optimization (NSGA-II): cost vs time/distance.
- Product quantities, store opening hours, 2-for-1 offers.
- Parallelize evaluation inside one run (today we parallelize *across* problems).

### Performance and scalability
The GA evaluates ~`pop_size × generations` chromosomes regardless of the search-space size, and the route cache
avoids recomputing repeated sets (≈ 99 % hits on small, ≈ 70 % on medium). A small problem takes ~0.1 s and a medium
one ~3 s, and the 272 validation runs take ~25 s in parallel using every core. The exact method grows as $2^S$ and
stops being viable around 25–30 stores.

### Conclusions
1. The cost model was validated **504/504** against the references, which revealed the graph is **directed**.
2. Separating **what to buy** (genes) from **what order to go in** (TSP heuristic) yields always-valid chromosomes
   and a much smaller search space.
3. The GA reaches the **exact optimum on 100/100 small problems**, **matches or beats the best reference on all 136
   problems** (100 % hit rate) and **improves it on 30**, saving 2,286 in total.
4. **Tournament + random reset** is the most robust combination without seeding. Consolidation mutation causes
   premature convergence, and greedy seeding speeds up convergence and guarantees we are never worse than them.
5. The GPU only pays off from ~10⁴ individuals. With the GA's population (80) the CPU is faster, and the useful
   parallelism is *across* problems.
""")
md("""
## 21. Manual de usuario

**Instalación**
```bash
git clone <url-del-repo> && cd CompraGenetica
python -m venv .venv && .venv\\Scripts\\activate      # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -e ".[notebook,dev]"
```

**Desde la línea de comandos**
```bash
compra-genetica --catalog data/in/small-3s-4p-f1_10-q1_4-random_catalog.xlsx \\
                --products "Purificadores de aire,Aceite de girasol,Pasta de dientes,Alfombras" --fuel 1
```
(o `python -m compra_genetica ...` sin instalar el comando). Opciones: `--pop-size --generations --pc --pm --selection {tournament,roulette,rank}
--crossover {one_point,two_point,uniform} --mutation {random_reset,consolidate} --elitism --patience --seed --verbose`

**Desde Python**
```python
from compra_genetica import load_instance, ShoppingProblem, GeneticShoppingOptimizer
problem = ShoppingProblem(load_instance("mi_catalogo.xlsx"), ["Arroz", "Aceite"], fuel_price=5)
print(GeneticShoppingOptimizer(problem, selection="rank", seed=1).run().answer)
```

**Formato del Excel de entrada:** hoja 1 `store | product | price`, hoja 2 `edge 1 | edge 2 | distance`.
La casa debe llamarse `Casa`.

**Resultados guardados:** por defecto el notebook carga los experimentos pesados desde `results/`, así que corre en
segundos. Con `RECOMPUTE = True` (o `python tools/build_notebooks.py --recompute`) se recalcula todo.

**Tests:** `pytest -q` · **Regenerar los notebooks:** `python tools/build_notebooks.py`
""", """
## 21. User manual

**Installation**
```bash
git clone <repo-url> && cd CompraGenetica
python -m venv .venv && .venv\\Scripts\\activate      # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -e ".[notebook,dev]"
```

**From the command line**
```bash
compra-genetica --catalog data/in/small-3s-4p-f1_10-q1_4-random_catalog.xlsx \\
                --products "Purificadores de aire,Aceite de girasol,Pasta de dientes,Alfombras" --fuel 1
```
(or `python -m compra_genetica ...` without installing the command). Options: `--pop-size --generations --pc --pm --selection {tournament,roulette,rank}
--crossover {one_point,two_point,uniform} --mutation {random_reset,consolidate} --elitism --patience --seed --verbose`

**From Python**
```python
from compra_genetica import load_instance, ShoppingProblem, GeneticShoppingOptimizer
problem = ShoppingProblem(load_instance("my_catalog.xlsx"), ["Arroz", "Aceite"], fuel_price=5)
print(GeneticShoppingOptimizer(problem, selection="rank", seed=1).run().answer)
```

**Input Excel format:** sheet 1 `store | product | price`, sheet 2 `edge 1 | edge 2 | distance`.
The home node must be called `Casa`.

**Saved results:** by default the notebook loads the heavy experiments from `results/`, so it runs in seconds.
With `RECOMPUTE = True` (or `python tools/build_notebooks.py --recompute`) everything is recomputed.

**Tests:** `pytest -q` · **Rebuild the notebooks:** `python tools/build_notebooks.py`
""")


# =============================================================================
def build(lang: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    for kind, es, en in CELLS:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(es if lang == "es" else en))
        else:
            nb.cells.append(nbf.v4.new_code_cell(localize(es, lang)))
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}
    return nb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-exec", action="store_true")
    parser.add_argument("--lang", choices=["es", "en"], nargs="*", default=["es", "en"])
    parser.add_argument("--recompute", action="store_true",
                        help="recompute the heavy experiments in the first notebook instead of loading results/*.csv")
    args = parser.parse_args()
    out_dir = ROOT / "notebooks"
    for n, lang in enumerate(args.lang):
        nb = build(lang)
        # Heavy experiments run at most once: the first notebook computes, the next ones reuse results/*.csv
        os.environ["COMPRA_RECOMPUTE"] = "1" if args.recompute and n == 0 else "0"
        if not args.no_exec:
            print(f"Executing {lang} notebook...", flush=True)
            NotebookClient(nb, timeout=3600, kernel_name="python3",
                           resources={"metadata": {"path": str(out_dir)}}).execute()
        path = out_dir / f"compra_genetica_{lang}.ipynb"
        nbf.write(nb, path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
