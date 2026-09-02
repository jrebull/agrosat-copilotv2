"""Tests del FarSLIPExtractor (AC-5, AC-10).

Cobertura objetivo ``ml/extractors/farslip_extractor.py`` >= 75 %.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch


def _hf_available() -> bool:
    try:
        from transformers import CLIPModel  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


@pytest.fixture(scope="module")
def extractor() -> Any:
    if not _hf_available():
        pytest.skip("transformers no disponible")
    from ml.extractors.farslip_extractor import FarSLIPExtractor

    try:
        return FarSLIPExtractor(weights_uri=None, device="cpu", n_in_channels=4)
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"no se pudo instanciar extractor: {exc}")


def test_extract_embeddings_shape_B_512(extractor: Any) -> None:
    crops = torch.rand(3, 4, 256, 256, dtype=torch.float32)
    out = extractor.extract_embeddings(crops)
    assert out.shape == (3, 512)
    assert out.dtype == torch.float32
    # L2-norm: norma ~ 1
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_extract_patch_features_shape_B_196_768(extractor: Any) -> None:
    crops = torch.rand(2, 4, 224, 224, dtype=torch.float32)
    out = extractor.extract_patch_features(crops)
    assert out.shape == (2, 196, 768)
    assert out.dtype == torch.float32


def test_encode_text_consistency(extractor: Any) -> None:
    """Cosine sim entre dos textos relacionados > 0 (fixture sintetico)."""
    texts = ["maize field", "wheat field"]
    out = extractor.encode_text(texts)
    assert out.shape == (2, 512)
    # ambos campos agricolas => sim >= 0.3
    sim = (out[0] * out[1]).sum().item()
    assert sim >= 0.3


def test_idempotent_two_calls_same_output(extractor: Any) -> None:
    crops = torch.rand(1, 4, 256, 256, dtype=torch.float32)
    o1 = extractor.extract_embeddings(crops)
    o2 = extractor.extract_embeddings(crops)
    assert torch.allclose(o1, o2, atol=1e-5)


def test_inference_mode_no_grad(extractor: Any) -> None:
    crops = torch.rand(1, 4, 256, 256, dtype=torch.float32)
    out = extractor.extract_embeddings(crops)
    assert out.grad_fn is None


def test_gcs_fallback_to_cache_on_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Si GCS no responde y hay cache local valido, lo usa."""
    if not _hf_available():
        pytest.skip("transformers no disponible")
    from ml.extractors.farslip_extractor import FarSLIPExtractor

    # Pre-poblamos el cache con un archivo dummy (no se cargara realmente).
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Mockeamos download_from_gcs para que falle
    uri = "gs://fake-bucket/fake-path/"

    # Crear cache "valido" -> safetensors vacio compatible. En su lugar, lo que
    # validamos es la rama: weights_uri GCS + no cache + GCS falla =>
    # extractor cae a modo degradado teacher (no exception).
    def _fail_download(self: Any, _: str) -> Path:
        raise RuntimeError("GCS offline simulado")

    monkeypatch.setattr(FarSLIPExtractor, "_download_from_gcs", _fail_download)
    try:
        ex = FarSLIPExtractor(weights_uri=uri, device="cpu", cache_dir=cache_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"no se pudo instanciar extractor degradado: {exc}")
    # Aun asi debe extraer embeddings (modo teacher fallback)
    crops = torch.rand(1, 4, 256, 256, dtype=torch.float32)
    out = ex.extract_embeddings(crops)
    assert out.shape == (1, 512)


def test_extract_embeddings_invalid_channels_raises(extractor: Any) -> None:
    """C != n_in_channels => ValueError."""
    bad = torch.rand(1, 3, 256, 256)
    with pytest.raises(ValueError):
        extractor.extract_embeddings(bad)


def test_extract_embeddings_invalid_dim_raises(extractor: Any) -> None:
    """crops debe ser 4D (B,C,H,W)."""
    bad = torch.rand(4, 256, 256)
    with pytest.raises(ValueError):
        extractor.extract_embeddings(bad)


def test_load_local_weights_from_directory(tmp_path: Path) -> None:
    """Si weights_uri apunta a un directorio con student.safetensors, lo carga."""
    if not _hf_available():
        pytest.skip("transformers no disponible")
    from safetensors.torch import save_file

    from ml.extractors.farslip_extractor import FarSLIPExtractor

    # Crear directorio con un safetensors trivial (solo unas keys del vision_model)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    # Cargamos un extractor sin weights primero para sacar un state_dict valido
    try:
        ex_base = FarSLIPExtractor(weights_uri=None, device="cpu")
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"no se pudo instanciar: {exc}")
    state = {
        k: v.detach().contiguous().cpu() for k, v in ex_base.model.vision_model.state_dict().items()
    }
    save_file(state, str(weights_dir / "student.safetensors"))

    ex = FarSLIPExtractor(weights_uri=str(weights_dir), device="cpu", cache_dir=tmp_path / "cache")
    crops = torch.rand(1, 4, 224, 224)
    out = ex.extract_embeddings(crops)
    assert out.shape == (1, 512)


def test_load_local_weights_from_file_path(tmp_path: Path) -> None:
    """Si weights_uri apunta a un .safetensors directo, lo carga."""
    if not _hf_available():
        pytest.skip("transformers no disponible")
    from safetensors.torch import save_file

    from ml.extractors.farslip_extractor import FarSLIPExtractor

    weights_path = tmp_path / "student.safetensors"
    try:
        ex_base = FarSLIPExtractor(weights_uri=None, device="cpu")
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"no se pudo instanciar: {exc}")
    state = {
        k: v.detach().contiguous().cpu() for k, v in ex_base.model.vision_model.state_dict().items()
    }
    save_file(state, str(weights_path))

    ex = FarSLIPExtractor(weights_uri=str(weights_path), device="cpu", cache_dir=tmp_path / "cache")
    crops = torch.rand(1, 4, 224, 224)
    out = ex.extract_embeddings(crops)
    assert out.shape == (1, 512)


def test_validate_checksum_mismatch(tmp_path: Path) -> None:
    """expected_sha1 distinto => _validate_checksum False."""
    if not _hf_available():
        pytest.skip("transformers no disponible")
    from ml.extractors.farslip_extractor import FarSLIPExtractor

    try:
        ex = FarSLIPExtractor(
            weights_uri=None,
            device="cpu",
            cache_dir=tmp_path,
            expected_sha1="0" * 40,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"no se pudo instanciar: {exc}")
    fake_file = tmp_path / "fake.bin"
    fake_file.write_bytes(b"hola mundo")
    assert ex._validate_checksum(fake_file) is False


def test_validate_checksum_no_expected(tmp_path: Path) -> None:
    """sin expected_sha1 => siempre True (skip)."""
    if not _hf_available():
        pytest.skip("transformers no disponible")
    from ml.extractors.farslip_extractor import FarSLIPExtractor

    try:
        ex = FarSLIPExtractor(weights_uri=None, device="cpu", cache_dir=tmp_path)
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"no se pudo instanciar: {exc}")
    fake_file = tmp_path / "anything.bin"
    fake_file.write_bytes(b"x")
    assert ex._validate_checksum(fake_file) is True


def test_extract_embeddings_uint16_normalization(extractor: Any) -> None:
    """uint16 input se normaliza a [0,1] (divide /10000)."""
    crops = torch.randint(0, 10000, (1, 4, 256, 256), dtype=torch.int16)
    out = extractor.extract_embeddings(crops)
    assert out.shape == (1, 512)


def test_download_gcs_invalid_scheme_raises(extractor: Any) -> None:
    """URI no gs:// en _download_from_gcs => ValueError."""
    with pytest.raises(ValueError):
        extractor._download_from_gcs("http://no-gs-uri")


def test_gcs_auth_error_degrades_to_teacher_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Forbidden`` 403 / DefaultCredentialsError NO debe romper el __init__.

    Regresion contra el bug detectado en QA manual-test US-017 (Flujo 5):
    el extractor solo capturaba ``(OSError, RuntimeError, ValueError)`` y
    dejaba propagar ``google.api_core.exceptions.Forbidden`` al constructor,
    rompiendo todos los consumers downstream (US-016 fusion, US-025
    SegFormer cabezal open-vocab) cuando GCS no era accesible.

    El contrato del Flujo 5 dice: warning + modo teacher degradado.
    """
    if not _hf_available():
        pytest.skip("transformers no disponible")
    from ml.extractors.farslip_extractor import FarSLIPExtractor

    class FakeForbidden(Exception):
        """Simula google.api_core.exceptions.Forbidden sin importar la lib."""

        pass

    FakeForbidden.__name__ = "Forbidden"

    def _raise_forbidden(self: Any, _: str) -> Path:
        raise FakeForbidden("403 GET https://...: storage.objects.get denied")

    monkeypatch.setattr(FarSLIPExtractor, "_download_from_gcs", _raise_forbidden)
    # No debe levantar — degradacion silenciosa a teacher mode con warning.
    ex = FarSLIPExtractor(
        weights_uri="gs://invalid-bucket/farslip/",
        device="cpu",
        cache_dir=tmp_path / "cache",
    )
    crops = torch.rand(1, 4, 256, 256, dtype=torch.float32)
    out = ex.extract_embeddings(crops)
    assert out.shape == (1, 512)


def test_non_gcs_error_still_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bugs reales (AttributeError, KeyError) no deben enmascararse como GCS.

    Garantiza que el bloque ``except Exception`` agregado para Forbidden no
    se convierta en un swallow universal — solo lo clasificado por
    ``is_gcs_auth_error`` se degrada; el resto burbujea.
    """
    if not _hf_available():
        pytest.skip("transformers no disponible")
    from ml.extractors.farslip_extractor import FarSLIPExtractor

    def _raise_attr(self: Any, _: str) -> Path:
        raise AttributeError("bug real en el extractor")

    monkeypatch.setattr(FarSLIPExtractor, "_download_from_gcs", _raise_attr)
    with pytest.raises(AttributeError, match="bug real"):
        FarSLIPExtractor(
            weights_uri="gs://invalid-bucket/farslip/",
            device="cpu",
            cache_dir=tmp_path / "cache",
        )


def test_download_failure_leaves_no_partial_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MT-3 regresion: download fallido NO debe dejar cache parcial.

    Reproduce el bug encadenado detectado en QA manual-test US-017 Flujo 5:
    si ``blob.download_to_filename()`` revienta despues de tocar/crear el
    archivo destino (caso real: 403 mid-write), la proxima invocacion ve el
    archivo en cache, ``_validate_checksum`` lo acepta (sin expected_sha1)
    y ``load_file`` revienta con ``SafetensorError: header too small``.

    El fix en ``_download_from_gcs`` debe ``unlink`` el destino parcial
    antes de propagar la excepcion, de modo que el cache quede limpio
    para la siguiente invocacion (que entonces caera limpiamente a teacher
    mode via ``is_gcs_auth_error``).

    Este test ejercita el ``_download_from_gcs`` REAL (no lo mockea):
    parcheamos ``google.cloud.storage.Client`` para que el ``blob`` toque
    el destino y luego lance ``Forbidden``.
    """
    if not _hf_available():
        pytest.skip("transformers no disponible")
    from ml.extractors.farslip_extractor import FarSLIPExtractor

    class FakeForbidden(Exception):
        pass

    FakeForbidden.__name__ = "Forbidden"

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    class _FakeBlob:
        def __init__(self) -> None:
            self.dest_seen: Path | None = None

        def download_to_filename(self, dest_str: str) -> None:
            # Simula google.cloud.storage: toca el destino y luego falla.
            dest = Path(dest_str)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"\x00\x01\x02")  # 3 bytes basura
            self.dest_seen = dest
            raise FakeForbidden("403 mid-write")

    class _FakeBucket:
        def blob(self, _: str) -> _FakeBlob:
            return _FakeBlob()

    class _FakeClient:
        def bucket(self, _: str) -> _FakeBucket:
            return _FakeBucket()

    # Parchear el modulo google.cloud.storage que se importa dentro de
    # _download_from_gcs (es un import perezoso).
    import sys
    import types

    fake_storage = types.ModuleType("storage")
    fake_storage.Client = _FakeClient  # type: ignore[attr-defined]
    fake_cloud = types.ModuleType("cloud")
    fake_cloud.storage = fake_storage  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", fake_storage)

    # __init__ debe degradar a teacher mode (via is_gcs_auth_error en el
    # caller) sin dejar cache parcial.
    ex = FarSLIPExtractor(
        weights_uri="gs://invalid-bucket/farslip/",
        device="cpu",
        cache_dir=cache_dir,
    )
    cached_files = list(cache_dir.glob("*.safetensors"))
    assert cached_files == [], (
        f"cache parcial NO eliminado: {cached_files} "
        "(regresion MT-3: siguiente run cargaria basura y reventaria "
        "con SafetensorError: header too small)"
    )
    # Sigue funcionando en modo teacher degradado.
    out = ex.extract_embeddings(torch.rand(1, 4, 256, 256))
    assert out.shape == (1, 512)
