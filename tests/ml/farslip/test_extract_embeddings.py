"""Tests de ``ml.farslip.extract_embeddings`` (US-022-c P1 B-4).

Cubre los AC del plan canonico seccion 2.1:

- Smoke con student mockeado (sin descargar CLIP HF).
- Shape determinista ``(N, 514)`` con ``seed=42``.
- CLI argparse parsing (--rois, --output, --device, --seed).
- Device fallback ``cuda -> cpu`` con warning si CUDA no disponible.
- Output parquet schema (parcel_id int64, year int32, 512 float32).
- MLflow URI resolution ``mlflow://Models/...@Production`` (mockeada).

Cobertura objetivo: >= 75 % del modulo.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import torch

from ml.farslip import extract_embeddings as ee

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_parcels_parquet(path: Path, n_rows: int = 16, with_roi: bool = True) -> Path:
    """Crea un parquet pequenio de parcelas para los tests."""
    data: dict[str, list[int] | list[str]] = {
        "parcel_id": list(range(n_rows)),
        "year": [2024] * n_rows,
    }
    if with_roi:
        rois_cycle = ["pianura_padana", "toscana", "puglia"]
        data["roi"] = [rois_cycle[i % 3] for i in range(n_rows)]
    df = pl.DataFrame(data)
    df.write_parquet(path)
    return path


class _MockStudent(torch.nn.Module):
    """Mock minimal del CLIPVisionModel student (no descarga HF)."""

    def __init__(self, embed_dim: int = ee.EMBED_DIM) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, embed_dim)
        self.embed_dim = embed_dim

    def eval(self) -> _MockStudent:
        super().eval()
        return self

    def to(self, device: str | torch.device) -> _MockStudent:  # type: ignore[override]
        super().to(device)
        return self


@pytest.fixture
def mock_student_loader() -> MagicMock:
    """Patchea ``_load_student`` para evitar descarga HF en tests."""
    with patch.object(ee, "_load_student", return_value=_MockStudent()) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Tests 1-2: device fallback + seed determinism
# ---------------------------------------------------------------------------


def test_resolve_device_cuda_unavailable_falls_back_to_cpu_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``device='cuda'`` con CUDA off -> ``cpu`` + warning estructurado."""
    if torch.cuda.is_available():
        pytest.skip("CUDA disponible; este test valida el fallback CPU-only.")
    with patch.object(ee, "_log") as mock_log:
        resolved = ee._resolve_device("cuda")
        assert resolved.type == "cpu"
        mock_log.warning.assert_called_once()
        # El mensaje debe identificar la causa
        args, _kwargs = mock_log.warning.call_args
        assert "cuda_requested_but_unavailable_fallback_cpu" in args[0]


def test_resolve_device_auto_returns_cpu_when_no_cuda() -> None:
    """``device='auto'`` -> ``cpu`` si CUDA no disponible (sin warning)."""
    if torch.cuda.is_available():
        pytest.skip("CUDA disponible; comportamiento auto -> cuda no testeado aqui.")
    resolved = ee._resolve_device("auto")
    assert resolved.type == "cpu"


def test_project_parcels_deterministic_seed_42() -> None:
    """Misma ``seed=42`` -> mismo tensor (reproducibilidad B-4)."""
    mock_model = _MockStudent()
    cpu = torch.device("cpu")
    t1 = ee._project_parcels_to_embeddings(
        mock_model, n_parcels=8, batch_size=4, device=cpu, seed=42
    )
    t2 = ee._project_parcels_to_embeddings(
        mock_model, n_parcels=8, batch_size=4, device=cpu, seed=42
    )
    assert t1.shape == (8, ee.EMBED_DIM)
    assert torch.allclose(t1, t2, atol=1e-7)


# ---------------------------------------------------------------------------
# Tests 3-4: end-to-end con student mockeado + shape parquet
# ---------------------------------------------------------------------------


def test_extract_embeddings_smoke_with_mocked_student(
    tmp_path: Path, mock_student_loader: MagicMock
) -> None:
    """Smoke: corre el pipeline completo con student mockeado y verifica shape."""
    parcels = _make_parcels_parquet(tmp_path / "parcels.parquet", n_rows=12)
    output = tmp_path / "embeddings.parquet"
    result = ee.extract_farslip_embeddings(
        student_checkpoint_path=tmp_path / "ckpt.safetensors",
        parcels_parquet=parcels,
        rois=("italy",),
        output_path=output,
        batch_size=4,
        device="cpu",
        seed=42,
    )
    assert isinstance(result, ee.ExtractEmbeddingsResult)
    assert result.n_parcels == 12
    assert result.n_dims == ee.EMBED_DIM
    assert result.output_path == output.resolve()
    assert result.device_used == "cpu"
    assert output.exists()
    mock_student_loader.assert_called_once()


def test_output_parquet_schema_is_stable(tmp_path: Path, mock_student_loader: MagicMock) -> None:
    """Schema: parcel_id Int64 + year Int32 + 512 float32 cols = 514 cols."""
    parcels = _make_parcels_parquet(tmp_path / "parcels.parquet", n_rows=6)
    output = tmp_path / "embeddings.parquet"
    ee.extract_farslip_embeddings(
        student_checkpoint_path=tmp_path / "ckpt.safetensors",
        parcels_parquet=parcels,
        rois=("italy",),
        output_path=output,
        batch_size=2,
        device="cpu",
        seed=42,
    )
    df = pl.read_parquet(output)
    assert df.width == ee.TOTAL_COLS == 514
    assert df.height == 6
    assert df.schema["parcel_id"] == pl.Int64
    assert df.schema["year"] == pl.Int32
    # Las 512 columnas de embeddings deben ser float32
    embed_cols = [c for c in df.columns if c.startswith(ee.EMBED_COL_PREFIX)]
    assert len(embed_cols) == ee.EMBED_DIM
    for col in embed_cols[:5]:  # muestreo
        assert df.schema[col] == pl.Float32


# ---------------------------------------------------------------------------
# Test 5: CLI argparse
# ---------------------------------------------------------------------------


def test_cli_argparse_parses_required_and_default_flags(tmp_path: Path) -> None:
    """Argparser construye namespace coherente con los defaults documentados."""
    parser = ee._build_arg_parser()
    args = parser.parse_args(
        [
            "--student-checkpoint",
            "mlflow://Models/farslip-clip-italy-v1@Production",
            "--parcels-parquet",
            str(tmp_path / "p.parquet"),
            "--output",
            str(tmp_path / "out.parquet"),
        ]
    )
    assert args.student_checkpoint == ("mlflow://Models/farslip-clip-italy-v1@Production")
    assert args.parcels_parquet == tmp_path / "p.parquet"
    assert args.output == tmp_path / "out.parquet"
    assert args.rois == "italy"  # default
    assert args.batch_size == 256  # default
    assert args.device == "auto"  # default
    assert args.seed == 42  # default


def test_cli_argparse_rejects_missing_required_flags() -> None:
    """Falta de ``--student-checkpoint`` provoca SystemExit (argparse standard)."""
    parser = ee._build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--parcels-parquet", "x", "--output", "y"])


# ---------------------------------------------------------------------------
# Test 6: MLflow URI resolution
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_mlflow_uri_delegates_to_mlflow_artifacts() -> None:
    """``mlflow://Models/<name>@<stage>`` se resuelve via mlflow.artifacts."""
    with patch.object(
        ee, "_resolve_mlflow_uri", return_value=Path("/tmp/fake/student.safetensors")
    ) as mock_resolve:
        local, data_version = ee._resolve_checkpoint(
            "mlflow://Models/farslip-clip-italy-v1@Production"
        )
        assert local == Path("/tmp/fake/student.safetensors")
        assert data_version == "mlflow://Models/farslip-clip-italy-v1@Production"
        mock_resolve.assert_called_once_with("mlflow://Models/farslip-clip-italy-v1@Production")


def test_resolve_checkpoint_local_path_passes_through() -> None:
    """Ruta local devuelve la ruta como Path + str como data_version."""
    local, data_version = ee._resolve_checkpoint("/tmp/local/ckpt.pt")
    assert local == Path("/tmp/local/ckpt.pt")
    assert data_version == "/tmp/local/ckpt.pt"


# ---------------------------------------------------------------------------
# Tests adicionales (cobertura util / edge cases)
# ---------------------------------------------------------------------------


def test_resolve_rois_alias_italy_expands_to_three() -> None:
    """``("italy",)`` se expande a la tupla canonica de 3 ROIs."""
    assert ee._resolve_rois(("italy",)) == (
        "pianura_padana",
        "toscana",
        "puglia",
    )


def test_resolve_rois_passthrough_when_no_alias_matches() -> None:
    """Sin alias conocido la tupla se devuelve tal cual."""
    assert ee._resolve_rois(("custom_roi",)) == ("custom_roi",)


def test_load_parcels_filtered_drops_unknown_rois(tmp_path: Path) -> None:
    """Filtra parcelas por ROI cuando la columna existe."""
    p = _make_parcels_parquet(tmp_path / "p.parquet", n_rows=9, with_roi=True)
    df = ee._load_parcels_filtered(p, rois=("toscana",))
    # 9 filas con cycle de 3 ROIs -> 3 filas con roi='toscana'
    assert df.height == 3
    assert (df["roi"] == "toscana").all()


def test_load_parcels_filtered_no_roi_column_returns_all(tmp_path: Path) -> None:
    """Si no hay columna roi/region, no se filtra."""
    p = _make_parcels_parquet(tmp_path / "p.parquet", n_rows=5, with_roi=False)
    df = ee._load_parcels_filtered(p, rois=("toscana",))
    assert df.height == 5


def test_load_parcels_missing_parcel_id_raises(tmp_path: Path) -> None:
    """Parquet sin columna parcel_id provoca ValueError explicito."""
    bad = tmp_path / "bad.parquet"
    pl.DataFrame({"year": [2024]}).write_parquet(bad)
    with pytest.raises(ValueError, match="parcel_id"):
        ee._load_parcels_filtered(bad, rois=("italy",))


def test_embed_columns_count_and_format() -> None:
    """Genera 512 columnas con prefijo y zero-pad de 3 digitos."""
    cols = ee._embed_columns()
    assert len(cols) == ee.EMBED_DIM
    assert cols[0] == "farslip_emb_000"
    assert cols[-1] == "farslip_emb_511"
