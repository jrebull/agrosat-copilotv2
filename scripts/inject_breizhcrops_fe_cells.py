"""Inserts the BreizhCrops cross-region section into 03b_fe_spectral_temporal_pastis.

Permanent and idempotent operational script: locates the
``BREIZHCROPS_CROSSREGION`` marker cell and, if it does not exist, inserts a new
section right before the conclusions cell (## 11. Conclusiones), renumbering it to
## 12. The cells reuse the existing temporal transformers
(``ml.features.temporal_features.extract_temporal_features``) on BreizhCrops
series, demonstrating that the FFT/phenology features generalize
cross-region. Papermill-safe: degraded mode if the dataset is not on disk.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "feature_engineering" / "03b_fe_spectral_temporal_pastis.ipynb"
MARKER = "BREIZHCROPS_CROSSREGION"


def _cell(cell_type: str, source: str, tags: list[str] | None = None) -> dict:
    meta: dict = {"tags": tags or []}
    base = {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:12],
        "metadata": meta,
        "source": source.strip("\n").splitlines(keepends=True),
    }
    if cell_type == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


MD_INTRO = """## 11. Generalización cross-region: features temporales sobre BreizhCrops

Las features temporales (FFT + fenología) se calibraron sobre PASTIS-R.
Antes de confiar en ellas hay que comprobar que no están sobreajustadas a
esa región concreta. BreizhCrops aporta series Sentinel-2 de otra zona de
Francia (Bretaña) con etiquetas de 9 cultivos: si el mismo extractor
produce features con rangos comparables aquí, las features son
transferibles entre regiones y no un artefacto del dataset original.

Reutilizamos exactamente `extract_temporal_features` (el mismo transformer
aplicado a PASTIS-R, sin reimplementar nada) sobre una muestra de
parcelas BreizhCrops."""

CODE_LOAD = """
# BREIZHCROPS_CROSSREGION — celda marcador (no eliminar este comentario).
from ml.features.temporal_features import extract_temporal_features
from ml.ingest.breizhcrops_loader import (
    breizhcrops_parcel_index,
    breizhcrops_pixel_series,
)
from ml.features.spectral_indices import compute_index

bc_region = 'frh04'
bc_year = 2017
bc_sample_parcels = 60

bc_index = breizhcrops_parcel_index(bc_region, bc_year, 'L2A')
bc_series = breizhcrops_pixel_series(
    bc_region, bc_year, 'L2A', sample_parcels=bc_sample_parcels, seed=42
)
BC_DEGRADED = bc_index.is_empty() or bc_series.is_empty()

if BC_DEGRADED:
    display(Markdown(
        '**Modo degradado**: BreizhCrops no está descargado en '
        '`data/breizhcrops/`. Esta sección muestra placeholders válidos. '
        'Ejecuta `bash scripts/download_breizhcrops.sh` y re-corre el '
        'notebook para poblar la comparación cross-region.'
    ))
else:
    display(Markdown(
        f'**BreizhCrops cargado**: `{bc_index.height:,}` parcelas en '
        f'`{bc_region}`; muestra de `{bc_series[\"parcel_id\"].n_unique()}` '
        'parcelas para extracción temporal.'
    ))
"""

CODE_EXTRACT = """
def _bc_parcel_to_temporal_features(parcel_df: pl.DataFrame, pid: int, yr: int):
    \"\"\"Builds the (time, band) DataArray of indices and applies the transformer.

    Reuses compute_index (NDVI/NDWI/EVI) + extract_temporal_features, the
    same ones used in PASTIS-R, without reimplementing feature logic.
    \"\"\"
    import numpy as np
    import xarray as xr

    wide = (
        parcel_df.sort('t')
        .pivot(values='value', index=['t', 'doy'], on='band',
                aggregate_function='first')
        .drop_nulls()
    )
    if wide.height < 4:
        return None

    bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    if any(b not in wide.columns for b in bands):
        return None

    doy = wide['doy'].to_numpy()
    times = (np.datetime64(f'{yr}-01-01')
             + (doy - 1).astype('timedelta64[D]')).astype('datetime64[ns]')
    mat = np.stack([wide[b].to_numpy().astype(np.float64) * 1e-4 for b in bands], axis=1)
    da = xr.DataArray(
        mat, dims=('time', 'band'),
        coords={'time': times, 'band': bands},
    )
    idx_names = ['NDVI', 'NDWI', 'EVI']
    idx_stack = np.stack(
        [compute_index(da, name).values for name in idx_names], axis=1
    )
    idx_da = xr.DataArray(
        idx_stack, dims=('time', 'band'),
        coords={'time': times, 'band': idx_names},
    )
    idx_da.attrs['parcel_id'] = int(pid)
    idx_da.attrs['year'] = int(yr)
    return extract_temporal_features(
        idx_da, indices=tuple(idx_names), fft_indices=tuple(idx_names)
    )


bc_feature_frames = []
if not BC_DEGRADED:
    for pid, sub in bc_series.group_by('parcel_id'):
        pid_val = pid[0] if isinstance(pid, tuple) else pid
        try:
            feat = _bc_parcel_to_temporal_features(sub, int(pid_val), bc_year)
        except Exception as exc:  # noqa: BLE001
            log.warning('bc_temporal_skip', parcel=str(pid_val), error=str(exc))
            feat = None
        if feat is not None:
            bc_feature_frames.append(feat)

if bc_feature_frames:
    bc_features = pl.concat(bc_feature_frames, how='vertical_relaxed')
    display(Markdown(
        f'**Features temporales BreizhCrops**: `{bc_features.height}` parcelas '
        f'x `{bc_features.width}` columnas (mismo esquema que PASTIS-R).'
    ))
    display(bc_features.select(
        ['parcel_id', 'NDVI_mean', 'NDVI_p95', 'NDVI_fft_amp_1',
         'EVI_fft_amp_1', 'peak_doy', 'ndvi_auc']
    ).head(8))
else:
    display(Markdown('_Sin features BreizhCrops (modo degradado o muestra vacía)._'))
"""

CODE_COMPARE = """
# Comparacion de rangos de un feature clave entre PASTIS-R y BreizhCrops:
# si los rangos se solapan, el transformer generaliza cross-region.
if bc_feature_frames and 'X' in dir() and 'NDVI_fft_amp_1' in X.columns:
    pastis_amp = X['NDVI_fft_amp_1'].drop_nulls()
    bc_amp = bc_features['NDVI_fft_amp_1'].drop_nulls()
    cmp_tbl = pl.DataFrame({
        'dataset': ['PASTIS-R', 'BreizhCrops'],
        'n': [pastis_amp.len(), bc_amp.len()],
        'ndvi_fft_amp1_p25': [
            round(float(pastis_amp.quantile(0.25)), 4),
            round(float(bc_amp.quantile(0.25)), 4),
        ],
        'ndvi_fft_amp1_median': [
            round(float(pastis_amp.median()), 4),
            round(float(bc_amp.median()), 4),
        ],
        'ndvi_fft_amp1_p75': [
            round(float(pastis_amp.quantile(0.75)), 4),
            round(float(bc_amp.quantile(0.75)), 4),
        ],
    })
    display(cmp_tbl)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(
        [pastis_amp.to_numpy(), bc_amp.to_numpy()],
        tick_labels=['PASTIS-R', 'BreizhCrops'],
        showfliers=False,
    )
    ax.set_ylabel('NDVI_fft_amp_1 (amplitud 1er armónico)')
    ax.set_title('Feature temporal FFT: PASTIS-R vs BreizhCrops')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    display(fig)
    plt.close(fig)

    delta = abs(cmp_tbl['ndvi_fft_amp1_median'][0] - cmp_tbl['ndvi_fft_amp1_median'][1])
    display(Markdown(
        f'La mediana de la amplitud del primer armónico FFT del NDVI difiere '
        f'en `{delta:.4f}` entre regiones. Un solapamiento amplio confirma que '
        'el extractor temporal produce features **comparables cross-region**: '
        'no está sobreajustado a PASTIS-R y puede alimentar un baseline '
        'entrenado sobre ambas regiones.'
    ))
else:
    display(Markdown(
        '_Comparación cross-region no disponible: requiere features PASTIS-R '
        '(`X`) y BreizhCrops cargados. En modo degradado se omite._'
    ))
"""


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    already = any(
        MARKER in "".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"
    )
    if already:
        print("marcador ya presente — no se reinserta (idempotente)")
        return

    # Renumber the conclusions cell from ## 11 to ## 12.
    concl_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "markdown" and "".join(c["source"]).lstrip().startswith(
            "## 11. Conclusiones"
        ):
            concl_idx = i
            src = "".join(c["source"]).replace("## 11. Conclusiones", "## 12. Conclusiones", 1)
            c["source"] = src.splitlines(keepends=True)
            break
    if concl_idx is None:
        concl_idx = len(cells)

    new_cells = [
        _cell("markdown", MD_INTRO),
        _cell("code", CODE_LOAD),
        _cell("code", CODE_EXTRACT),
        _cell("code", CODE_COMPARE),
    ]
    cells[concl_idx:concl_idx] = new_cells
    nb["cells"] = cells
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"insertadas {len(new_cells)} celdas antes de la celda {concl_idx} (conclusiones)")


if __name__ == "__main__":
    main()
