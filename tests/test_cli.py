import subprocess
import sys

from conftest import DATA_DIR


def test_cli_prints_the_optimal_route():
    catalog = DATA_DIR / "in" / "small-3s-4p-f1_10-q1_4-random_catalog.xlsx"
    out = subprocess.run(
        [sys.executable, "-m", "compra_genetica", "--catalog", str(catalog), "--fuel", "1", "--seed", "0",
         "--products", "Purificadores de aire,Aceite de girasol,Pasta de dientes,Alfombras"],
        capture_output=True, text=True, encoding="utf-8", check=True,
        env={**__import__("os").environ, "PYTHONPATH": str(DATA_DIR.parent / "src")})
    assert out.stdout.startswith("RESPUESTA> (Jumbo, [Aceite de girasol, Pasta de dientes])")
    assert "Total cost: 11270.96" in out.stdout
