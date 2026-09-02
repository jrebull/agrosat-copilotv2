"""US-015 QA — ejecución automatizada de Flujos 1, 2, 3, 4, 5.

Script temporal (NO commit) para validar end-to-end la US-015 contra Postgres
real con PostGIS + pgvector.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import polars as pl
import xarray as xr
from sqlalchemy import create_engine, text

REPO = Path(__file__).resolve().parents[2]

import importlib.util  # noqa: E402

_spec_tf = importlib.util.spec_from_file_location(
    "_us015_temporal_features", REPO / "ml" / "features" / "temporal_features.py"
)
_tf_mod = importlib.util.module_from_spec(_spec_tf)  # type: ignore[arg-type]
_spec_tf.loader.exec_module(_tf_mod)  # type: ignore[union-attr]
extract_temporal_features = _tf_mod.extract_temporal_features

_spec_pf = importlib.util.spec_from_file_location(
    "_us015_persist_features", REPO / "ml" / "features" / "persist_features.py"
)
_pf_mod = importlib.util.module_from_spec(_spec_pf)  # type: ignore[arg-type]
_pf_mod.__package__ = ""
_spec_pf.loader.exec_module(_pf_mod)  # type: ignore[union-attr]
load_features_parcels = _pf_mod.load_features_parcels

DSN_SYNC = "postgresql+psycopg://agrosat:agrosat@localhost:55432/agrosat"
FIXTURE = REPO / "data" / "test_fixtures" / "parcel_demo_ts.nc"

results: list[tuple[str, str, str]] = []


def _record(flow: str, step: str, status: str) -> None:
    results.append((flow, step, status))
    print(f"[{status:6s}] {flow}  {step}")


def _open_demo_da() -> xr.DataArray:
    ds = xr.open_dataset(FIXTURE)
    da = ds["parcel_indices"]
    da.attrs.setdefault("parcel_id", 42)
    da.attrs.setdefault("year", 2024)
    if da.coords["band"].dtype.kind == "S":
        da = da.assign_coords(
            band=[b.decode() if isinstance(b, bytes) else b for b in da.coords["band"].values]
        )
    return da


# -----------------------------------------------------------------------------
# Flujo 1 — extracción end-to-end
# -----------------------------------------------------------------------------
def flujo_1() -> None:
    da = _open_demo_da()
    df = extract_temporal_features(da)
    _record(
        "Flujo 1",
        f"1.4 shape == (1, 187): got {df.shape}",
        "PASS" if df.shape == (1, 187) else "FAIL",
    )

    row = df.select("peak_doy", "peak_value", "sog_doy", "senescence_doy").row(0, named=True)
    # Fixture con ruido + grilla cada 12 días: tolerancia [150,210]; sin ruido el unitario asserta [178,182].
    peak_ok = 150 <= row["peak_doy"] <= 210
    peak_val_ok = abs(row["peak_value"] - 0.85) < 0.05
    _record(
        "Flujo 1",
        f"1.5 peak_doy in [150,210] (got {row['peak_doy']}), peak_value~0.85 (got {row['peak_value']:.3f})",
        "PASS" if peak_ok and peak_val_ok else "FAIL",
    )

    fft_cols = sorted([c for c in df.columns if c.startswith("NDVI_fft_amp_")])
    fft_vals = [df[c][0] for c in fft_cols]
    armonic_1_max = fft_vals[1] >= max(fft_vals[2:])
    _record(
        "Flujo 1",
        f"1.6 NDVI FFT amp count==4 ({len(fft_vals)}) and armonico_1>=armonicos_2,3 ({fft_vals[1]:.3f} vs {fft_vals[2]:.3f},{fft_vals[3]:.3f})",
        "PASS" if len(fft_vals) == 4 and armonic_1_max else "WARN",
    )

    slopes = df.select("ndvi_slope_pre_peak", "ndvi_slope_post_peak").row(0, named=True)
    slope_signs_ok = (
        (slopes["ndvi_slope_pre_peak"] or 0) > 0 > (slopes["ndvi_slope_post_peak"] or 0)
    )
    _record(
        "Flujo 1",
        f"1.7 slope_pre>0>slope_post (pre={slopes['ndvi_slope_pre_peak']:.5f}, post={slopes['ndvi_slope_post_peak']:.5f})",
        "PASS" if slope_signs_ok else "FAIL",
    )

    df2 = extract_temporal_features(_open_demo_da())
    determinism_ok = df.equals(df2)
    _record(
        "Flujo 1",
        f"AC-7 determinismo: dos llamadas iguales ({determinism_ok})",
        "PASS" if determinism_ok else "FAIL",
    )


# -----------------------------------------------------------------------------
# Flujo 2 — migraciones reversibles + introspección
# -----------------------------------------------------------------------------
def flujo_2(engine) -> None:
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'parcels' ORDER BY ordinal_position"
            )
        ).all()
    cols = {r[0]: r[1] for r in rows}
    expected = {
        "id": "bigint",
        "session_id": "uuid",
        "aoi_id": "bigint",
        "geom": "USER-DEFINED",
        "crop_class": "text",
        "confidence": "real",
        "area_ha": "real",
        "year": "smallint",
        "created_at": "timestamp with time zone",
        "updated_at": "timestamp with time zone",
    }
    ok = all(cols.get(k) == v for k, v in expected.items()) and len(cols) == 10
    _record(
        "Flujo 2",
        f"2.3 \\d parcels: 10 cols con tipos correctos ({len(cols)} found)",
        "PASS" if ok else "FAIL",
    )

    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name, data_type, udt_name FROM information_schema.columns "
                "WHERE table_name = 'features_parcels' ORDER BY ordinal_position"
            )
        ).all()
    cols_fp = {r[0]: (r[1], r[2]) for r in rows}
    expected_fp = {
        "id",
        "parcel_id",
        "year",
        "alphaearth_embedding",
        "ndvi_stats",
        "phenology",
        "sog_doy",
        "peak_doy",
        "peak_value",
        "senescence_doy",
        "ndvi_auc",
        "ndvi_slope_pre_peak",
        "ndvi_slope_post_peak",
        "maturity_duration_days",
        "created_at",
        "updated_at",
    }
    has_vector = cols_fp.get("alphaearth_embedding", (None, None))[1] == "vector"
    all_cols = expected_fp.issubset(set(cols_fp.keys()))
    _record(
        "Flujo 2",
        f"2.4 \\d features_parcels: 14+ cols (got {len(cols_fp)}) incluye vector y JSONB",
        "PASS" if all_cols and has_vector else "FAIL",
    )

    with engine.begin() as conn:
        u = conn.execute(
            text(
                "SELECT constraint_name FROM information_schema.table_constraints "
                "WHERE table_name='features_parcels' AND constraint_type='UNIQUE'"
            )
        ).all()
    unique_ok = any("parcel_year" in r[0] for r in u)
    _record(
        "Flujo 2",
        f"2.4b UNIQUE constraint presente: {[r[0] for r in u]}",
        "PASS" if unique_ok else "FAIL",
    )

    with engine.begin() as conn:
        idx = conn.execute(
            text(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE tablename IN ('parcels', 'features_parcels')"
            )
        ).all()
    gist_ok = any(("using gist" in i[1].lower()) and ("geom" in i[1].lower()) for i in idx)
    btree_year = any("parcels_year_idx" in i[0] for i in idx)
    btree_fp = any("features_parcels_parcel_id_idx" in i[0] for i in idx)
    _record(
        "Flujo 2",
        f"2.5 índices: GIST geom={gist_ok}, BTREE year={btree_year}, BTREE FK={btree_fp}",
        "PASS" if (gist_ok and btree_year and btree_fp) else "FAIL",
    )

    with engine.begin() as conn:
        sess_id = conn.execute(
            text("INSERT INTO chat_sessions (user_id) VALUES ('qa-test') RETURNING id")
        ).scalar()
        parcel_id = conn.execute(
            text(
                "INSERT INTO parcels (session_id, geom, year) "
                "VALUES (:sid, ST_GeomFromText('POLYGON((0 0,1 0,1 1,0 1,0 0))', 4326), 2024) "
                "RETURNING id"
            ),
            {"sid": str(sess_id)},
        ).scalar()
        conn.execute(
            text("INSERT INTO features_parcels (parcel_id, year) VALUES (:p, 2024)"),
            {"p": parcel_id},
        )

    from sqlalchemy.exc import IntegrityError

    unique_raised = False
    try:
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO features_parcels (parcel_id, year) VALUES (:p, 2024)"),
                {"p": parcel_id},
            )
    except IntegrityError:
        unique_raised = True
    _record(
        "Flujo 2",
        "2.6 UNIQUE viola al duplicar (parcel_id, year)",
        "PASS" if unique_raised else "FAIL",
    )

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM parcels WHERE id = :p"), {"p": parcel_id})
        remaining = conn.execute(
            text("SELECT COUNT(*) FROM features_parcels WHERE parcel_id = :p"), {"p": parcel_id}
        ).scalar()
    _record(
        "Flujo 2",
        f"2.7 FK CASCADE: features borrado al borrar parcel ({remaining} restantes)",
        "PASS" if remaining == 0 else "FAIL",
    )

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM chat_sessions WHERE id = :s"), {"s": str(sess_id)})


# -----------------------------------------------------------------------------
# Flujo 3 — UPSERT 3 modos
# -----------------------------------------------------------------------------
def flujo_3(engine) -> None:
    with engine.begin() as conn:
        sess_id = conn.execute(
            text("INSERT INTO chat_sessions (user_id) VALUES ('qa-test-upsert') RETURNING id")
        ).scalar()
        parcel_id = conn.execute(
            text(
                "INSERT INTO parcels (session_id, geom, year) "
                "VALUES (:sid, ST_GeomFromText('POLYGON((0 0,1 0,1 1,0 1,0 0))', 4326), 2024) "
                "RETURNING id"
            ),
            {"sid": str(sess_id)},
        ).scalar()

    df = pl.DataFrame(
        {
            "parcel_id": [int(parcel_id)],
            "year": [2024],
            "sog_doy": [120],
            "peak_doy": [180],
            "peak_value": [0.85],
            "senescence_doy": [240],
            "ndvi_auc": [50.0],
            "ndvi_slope_pre_peak": [0.005],
            "ndvi_slope_post_peak": [-0.005],
            "maturity_duration_days": [40],
            "NDVI_mean": [0.5],
            "NDVI_fft_amp_0": [0.5],
        }
    )

    n = load_features_parcels(df, engine)
    _record(
        "Flujo 3",
        f"3.4 insert: load_features_parcels devuelve {n} (esperado 1)",
        "PASS" if n == 1 else "FAIL",
    )

    with engine.begin() as conn:
        pv = conn.execute(
            text("SELECT peak_value, updated_at FROM features_parcels WHERE parcel_id=:p"),
            {"p": parcel_id},
        ).one()
    first_updated = pv[1]
    _record(
        "Flujo 3",
        f"3.4b peak_value persistido = {pv[0]}",
        "PASS" if abs(pv[0] - 0.85) < 1e-5 else "FAIL",
    )

    df2 = df.with_columns(pl.lit(0.90).alias("peak_value"))
    n2 = load_features_parcels(df2, engine, on_conflict="update")
    with engine.begin() as conn:
        pv2 = conn.execute(
            text("SELECT peak_value, updated_at FROM features_parcels WHERE parcel_id=:p"),
            {"p": parcel_id},
        ).one()
    updated_changed = pv2[1] > first_updated
    _record(
        "Flujo 3",
        f"3.5 update: peak_value pasa a {pv2[0]:.2f}, updated_at avanzó ({updated_changed})",
        "PASS" if (abs(pv2[0] - 0.90) < 1e-5 and updated_changed) else "FAIL",
    )

    n3 = load_features_parcels(df2, engine, on_conflict="skip")
    _record("Flujo 3", f"3.6 skip: devuelve {n3} (esperado 0)", "PASS" if n3 == 0 else "FAIL")

    from sqlalchemy.exc import IntegrityError

    raised = False
    try:
        load_features_parcels(df2, engine, on_conflict="raise")
    except IntegrityError:
        raised = True
    _record(
        "Flujo 3", f"3.7 raise: IntegrityError propagado ({raised})", "PASS" if raised else "FAIL"
    )

    empty = pl.DataFrame(schema={"parcel_id": pl.Int64, "year": pl.Int64})
    n4 = load_features_parcels(empty, engine)
    _record(
        "Flujo 3", f"3.8 empty frame: devuelve {n4} (esperado 0)", "PASS" if n4 == 0 else "FAIL"
    )

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM chat_sessions WHERE id = :s"), {"s": str(sess_id)})


# -----------------------------------------------------------------------------
# Flujo 4 — regeneración determinista del fixture
# -----------------------------------------------------------------------------
def flujo_4() -> None:
    h_before = hashlib.md5(FIXTURE.read_bytes()).hexdigest()
    backup = FIXTURE.with_suffix(".nc.bak")
    backup.write_bytes(FIXTURE.read_bytes())

    r = subprocess.run(
        ["poetry", "run", "python", "scripts/generate_demo_parcel_ts.py", "--output", str(FIXTURE)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    regen_ok = r.returncode == 0
    _record("Flujo 4", f"4.2 regenera fixture: rc={r.returncode}", "PASS" if regen_ok else "FAIL")

    size = FIXTURE.stat().st_size
    _record("Flujo 4", f"4.3 size {size} bytes < 100000", "PASS" if size < 100_000 else "FAIL")

    h_after = hashlib.md5(FIXTURE.read_bytes()).hexdigest()
    _record(
        "Flujo 4",
        f"4.4 hash bit-exact ({h_before[:8]} vs {h_after[:8]})",
        "PASS" if h_before == h_after else "FAIL",
    )
    backup.unlink()

    da = _open_demo_da()
    df = extract_temporal_features(da)
    _record(
        "Flujo 4", f"4.5 re-extract shape={df.shape}", "PASS" if df.shape == (1, 187) else "FAIL"
    )


# -----------------------------------------------------------------------------
# Flujo 5 — docs agronómica
# -----------------------------------------------------------------------------
def flujo_5() -> None:
    spec = (REPO / "docs" / "spectral_indices.md").read_text(encoding="utf-8")
    has_section = ("Temporal aggregation" in spec) or ("Agregación temporal" in spec)
    _record(
        "Flujo 5", "5.2 sección 'Temporal aggregation' presente", "PASS" if has_section else "FAIL"
    )

    refs_spec = sum(r in spec for r in ["White", "Reed", "Jönsson", "Eklundh", "TIMESAT"])
    _record(
        "Flujo 5",
        f"5.3 refs académicas en spec ({refs_spec}/5 keywords)",
        "PASS" if refs_spec >= 3 else "FAIL",
    )

    src = (REPO / "ml" / "features" / "temporal_features.py").read_text(encoding="utf-8")
    refs_src = sum(r in src for r in ["White", "Reed", "Jönsson", "Eklundh", "TIMESAT"])
    has_doi = "10.1016" in src or "10.1029" in src
    _record(
        "Flujo 5",
        f"5.4 docstring cita refs ({refs_src}/5 keywords) + DOIs ({has_doi})",
        "PASS" if refs_src >= 3 and has_doi else "FAIL",
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("US-015 QA — Ejecución de Flujos 1-5")
    print("=" * 70)

    engine = create_engine(DSN_SYNC)
    flujo_1()
    flujo_2(engine)
    flujo_3(engine)
    flujo_4()
    flujo_5()
    engine.dispose()

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    passed = sum(1 for _, _, s in results if s == "PASS")
    failed = sum(1 for _, _, s in results if s == "FAIL")
    warned = sum(1 for _, _, s in results if s == "WARN")
    print(f"PASS: {passed}   FAIL: {failed}   WARN: {warned}   TOTAL: {len(results)}")
    print()
    for flow, step, status in results:
        marker = "OK" if status == "PASS" else status
        print(f"  [{marker:4s}] {flow} {step}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
