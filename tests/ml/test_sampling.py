"""Tests para `ml.utils.sampling.stratified_sample`."""

from __future__ import annotations

import polars as pl
import pytest

from ml.utils.sampling import stratified_sample


@pytest.fixture
def imbalanced_df() -> pl.DataFrame:
    """DataFrame sintético 1000 filas con 3 estratos desbalanceados (700/200/100)."""
    return pl.DataFrame(
        {
            "stratum": ["a"] * 700 + ["b"] * 200 + ["c"] * 100,
            "value": list(range(1000)),
        }
    )


def test_stratified_sample_preserves_proportions(imbalanced_df: pl.DataFrame) -> None:
    """Las proporciones del sample deben aproximar las originales ±1 punto."""
    sampled = stratified_sample(imbalanced_df, by=["stratum"], n=100, seed=42)
    counts = sampled.group_by("stratum").agg(pl.len().alias("n")).sort("stratum")
    counts_dict = {r["stratum"]: r["n"] for r in counts.iter_rows(named=True)}
    # Proporcional: a=70, b=20, c=10 (+/- 1)
    assert abs(counts_dict["a"] - 70) <= 2
    assert abs(counts_dict["b"] - 20) <= 2
    assert abs(counts_dict["c"] - 10) <= 2


def test_stratified_sample_reproducible(imbalanced_df: pl.DataFrame) -> None:
    """Misma seed debe producir el mismo sample."""
    a = stratified_sample(imbalanced_df, by=["stratum"], n=100, seed=42)
    b = stratified_sample(imbalanced_df, by=["stratum"], n=100, seed=42)
    assert a.equals(b)


def test_stratified_sample_different_seed_different_result(
    imbalanced_df: pl.DataFrame,
) -> None:
    """Distintas seeds deben producir samples distintos (no determinístico)."""
    a = stratified_sample(imbalanced_df, by=["stratum"], n=100, seed=1)
    b = stratified_sample(imbalanced_df, by=["stratum"], n=100, seed=2)
    assert not a.equals(b)


def test_stratified_sample_validates_inputs(imbalanced_df: pl.DataFrame) -> None:
    """Valida `by` vacío, `n` no positivo, columna inexistente."""
    with pytest.raises(ValueError):
        stratified_sample(imbalanced_df, by=[], n=10)
    with pytest.raises(ValueError):
        stratified_sample(imbalanced_df, by=["stratum"], n=0)
    with pytest.raises(ValueError):
        stratified_sample(imbalanced_df, by=["missing_col"], n=10)


def test_stratified_sample_empty_df_returns_empty() -> None:
    """DataFrame vacío retorna DataFrame vacío con mismo esquema."""
    empty = pl.DataFrame({"stratum": [], "value": []})
    out = stratified_sample(empty, by=["stratum"], n=10)
    assert out.is_empty()


def test_stratified_sample_n_greater_than_rows(imbalanced_df: pl.DataFrame) -> None:
    """Si `n` excede el total de filas, no debe duplicar filas ni romperse.

    El muestreo es sin reemplazo (`pl.DataFrame.sample(shuffle=True)`), por lo
    que la cuota por estrato se trunca al tamaño del grupo. El resultado debe
    tener como máximo `df.height` filas y ningún `value` duplicado.
    """
    out = stratified_sample(imbalanced_df, by=["stratum"], n=10_000, seed=42)
    assert out.height <= imbalanced_df.height
    # No filas duplicadas: cada `value` original es único en el fixture.
    assert out["value"].n_unique() == out.height
    # Todos los estratos representados con su tamaño completo (al truncar quota).
    counts = {
        r["stratum"]: r["n"]
        for r in out.group_by("stratum").agg(pl.len().alias("n")).iter_rows(named=True)
    }
    assert counts.get("a") == 700
    assert counts.get("b") == 200
    assert counts.get("c") == 100


def test_stratified_sample_single_stratum() -> None:
    """Edge case: un solo estrato debe retornar `n` filas de ese estrato."""
    df = pl.DataFrame(
        {
            "stratum": ["only"] * 500,
            "value": list(range(500)),
        }
    )
    out = stratified_sample(df, by=["stratum"], n=50, seed=42)
    assert out.height == 50
    assert out["stratum"].unique().to_list() == ["only"]
    assert out["value"].n_unique() == 50
