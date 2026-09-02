"""CPU/offline tests for the FarSLIP ``L_glo`` caption generator (US-036-a v2, T1).

Covers ``ml/farslip/caption_generator.py`` and ``ml/farslip/caption_cache.py``
(T1 write-set): the anti-leakage prompt builder, the deterministic 2-98
percentile RGB enhancement to an 896 px PNG, the Gemma client payload
(``/api/chat`` + ``think=false`` + inline base64 image), response parsing /
retries, the idempotent captions cache (``resume=True``), and the anti-leakage
audit (regex). The Gemma client and all PASTIS disk reads are mocked: no network,
no GPU, no real PASTIS / parquet required.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ml.farslip.caption_cache as cc
import ml.farslip.caption_generator as cg
from ml.farslip.caption_cache import (
    CAPTIONS_SCHEMA,
    audit_captions,
    generate_captions_parquet,
    load_captions,
)
from ml.farslip.caption_generator import (
    GemmaCaptionClient,
    build_caption_prompt,
    composite_to_png896,
    composite_to_png896_b64,
)

_LEAK_PATTERNS = list(cc._LEAKAGE_PATTERNS.values())


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _make_composite(h: int = 32, w: int = 32, seed: int = 7) -> np.ndarray:
    """Builds a deterministic 4-band composite ``(4, H, W)`` in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.random((4, h, w)).astype(np.float32)


def _png_size(png_bytes: bytes) -> tuple[int, int]:
    """Returns the ``(width, height)`` of a PNG byte string."""
    from PIL import Image

    with Image.open(io.BytesIO(png_bytes)) as image:
        return image.size


# ---------------------------------------------------------------------------
# 1. Prompt builder: no leakage, includes required context.
# ---------------------------------------------------------------------------


def test_prompt_builder_no_leakage() -> None:
    prompt = build_caption_prompt(
        present_class_names=["Meadow", "Corn"],
        spatial_composition="norte: Meadow; centro: Corn; sur: Meadow",
        n_parcels=4,
        total_area_px=900,
        tile_mgrs="T31TFM",
        composite_date="20190715",
        typical_phenology={
            "Meadow": "vegetacion perenne con vigor sostenido en verano",
            "Corn": "siembra en primavera, pico de vigor a mediados de verano",
        },
    )
    # The prompt never injects a numeric NDVI value computed from the patch
    # (the circular-leak field). The prompt legitimately NAMES the other
    # forbidden tokens (NDVI/AlphaEarth/"la clase es") in its prohibition
    # instructions; the zero-leak guarantee is enforced on the CAPTION output
    # by audit_captions, not on the prompt text.
    assert not cc._LEAKAGE_PATTERNS["ndvi_numeric"].search(prompt)
    # The required context is present.
    assert "Meadow" in prompt
    assert "Corn" in prompt
    assert "T31TFM" in prompt
    assert "20190715" in prompt
    assert "norte: Meadow" in prompt
    assert "4" in prompt
    assert "900" in prompt
    # It explicitly forbids the leaky fields in its own instructions.
    assert "NDVI" in prompt
    assert "AlphaEarth" in prompt
    assert "la clase es" in prompt


def test_prompt_builder_handles_empty_phenology() -> None:
    prompt = build_caption_prompt(
        present_class_names=[],
        spatial_composition="sin cultivo declarado",
        n_parcels=0,
        total_area_px=0,
        tile_mgrs="T30UXV",
        composite_date="",
        typical_phenology={},
    )
    assert "no determinadas" in prompt
    assert "sin descripcion fenologica tipica" in prompt
    # No numeric NDVI is injected into the prompt (circular-leak field).
    assert not cc._LEAKAGE_PATTERNS["ndvi_numeric"].search(prompt)


# ---------------------------------------------------------------------------
# 2. Image enhancement: deterministic p2-98 stretch, 896 px RGB.
# ---------------------------------------------------------------------------


def test_composite_to_png896_size_and_channels() -> None:
    composite = _make_composite()
    png = composite_to_png896(composite)
    assert _png_size(png) == (896, 896)

    from PIL import Image

    with Image.open(io.BytesIO(png)) as image:
        assert image.mode == "RGB"


def test_composite_to_png896_custom_side() -> None:
    composite = _make_composite()
    png = composite_to_png896(composite, side=224)
    assert _png_size(png) == (224, 224)


def test_composite_to_png896_deterministic() -> None:
    composite = _make_composite(seed=11)
    assert composite_to_png896(composite) == composite_to_png896(composite)


def test_percentile_stretch_maps_full_range() -> None:
    # A clean ramp: the p2-98 stretch must span [0, 1] (min 0, max 1).
    channel = np.linspace(0.0, 1.0, 10000, dtype=np.float32).reshape(100, 100)
    out = cg._percentile_stretch(channel)
    assert float(out.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(out.max()) == pytest.approx(1.0, abs=1e-6)


def test_percentile_stretch_degenerate_channel() -> None:
    # A constant channel has no contrast to recover -> all zeros, no NaN/inf.
    channel = np.full((16, 16), 0.5, dtype=np.float32)
    out = cg._percentile_stretch(channel)
    assert np.all(out == 0.0)
    assert np.isfinite(out).all()


def test_composite_to_png896_uses_rgb_band_order() -> None:
    # B04 channel (idx 2) bright, others dark -> the rendered image is reddish
    # (R = B04). Verifies the B04,B03,B02 -> R,G,B mapping.
    composite = np.zeros((4, 16, 16), dtype=np.float32)
    rng = np.random.default_rng(0)
    composite[2] = rng.uniform(0.4, 1.0, size=(16, 16))  # B04 -> red
    composite[0] = rng.uniform(0.0, 0.1, size=(16, 16))  # B02 -> blue
    composite[1] = rng.uniform(0.0, 0.1, size=(16, 16))  # B03 -> green
    png = composite_to_png896(composite, side=32)

    from PIL import Image

    with Image.open(io.BytesIO(png)) as image:
        arr = np.asarray(image)
    assert arr[..., 0].mean() > arr[..., 1].mean()
    assert arr[..., 0].mean() > arr[..., 2].mean()


def test_composite_to_png896_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"\(>=3, H, W\)"):
        composite_to_png896(np.zeros((2, 8, 8), dtype=np.float32))


def test_composite_to_png896_b64_roundtrips() -> None:
    import base64

    composite = _make_composite(seed=3)
    b64 = composite_to_png896_b64(composite, side=64)
    decoded = base64.b64decode(b64)
    assert _png_size(decoded) == (64, 64)


# ---------------------------------------------------------------------------
# 3. Gemma client: /api/chat, think=false, inline base64 image, parsing.
# ---------------------------------------------------------------------------


class _FakePost:
    """Captures the URL and parsed payload passed to ``urlopen`` for one call."""

    def __init__(self, response_body: dict[str, object]) -> None:
        self.response_body = response_body
        self.url: str | None = None
        self.payload: dict[str, object] | None = None

    def __call__(self, request: object, timeout: float = 0.0) -> object:
        self.url = request.full_url  # type: ignore[attr-defined]
        self.payload = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        body = json.dumps(self.response_body).encode("utf-8")
        return _FakeResponse(body)


class _FakeResponse:
    """Minimal context-manager stand-in for an ``http.client.HTTPResponse``."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_gemma_client_uses_chat_think_false(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePost({"message": {"role": "assistant", "content": "Una pradera."}})
    monkeypatch.setattr(cg.urllib.request, "urlopen", fake)

    client = GemmaCaptionClient(base_url="http://127.0.0.1:11434")
    caption, gen_seconds = client.caption("describe la escena", "QkFTRTY0UE5H")

    assert caption == "Una pradera."
    assert gen_seconds >= 0.0
    # Hits /api/chat, NOT /api/generate.
    assert fake.url == "http://127.0.0.1:11434/api/chat"
    payload = fake.payload
    assert payload is not None
    assert payload["model"] == "gemma4:31b-it-q8_0"
    assert payload["think"] is False  # OBLIGATORY: without it the model times out.
    assert payload["stream"] is False
    message = payload["messages"][0]
    assert message["role"] == "user"
    assert message["content"] == "describe la escena"
    assert message["images"] == ["QkFTRTY0UE5H"]
    assert payload["options"]["temperature"] == pytest.approx(0.4)
    assert payload["options"]["num_predict"] == 400


def test_gemma_client_parses_and_trims(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePost({"message": {"content": "  texto con espacios  "}})
    monkeypatch.setattr(cg.urllib.request, "urlopen", fake)
    client = GemmaCaptionClient()
    caption, _ = client.caption("p", "b64")
    assert caption == "texto con espacios"


def test_gemma_client_raises_on_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePost({"message": {"content": "   "}})
    monkeypatch.setattr(cg.urllib.request, "urlopen", fake)
    client = GemmaCaptionClient(max_retries=0)
    with pytest.raises(RuntimeError, match="failed after"):
        client.caption("p", "b64")


def test_gemma_client_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _flaky(request: object, timeout: float = 0.0) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise cg.urllib.error.URLError("connection refused")
        body = json.dumps({"message": {"content": "ok"}}).encode("utf-8")
        return _FakeResponse(body)

    monkeypatch.setattr(cg.urllib.request, "urlopen", _flaky)
    monkeypatch.setattr(cg.time, "sleep", lambda _s: None)
    client = GemmaCaptionClient(max_retries=2)
    caption, _ = client.caption("p", "b64")
    assert caption == "ok"
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# 4. Captions cache: idempotent resume, schema, audit.
# ---------------------------------------------------------------------------


class _MockGemmaClient:
    """Deterministic stand-in for ``GemmaCaptionClient`` (counts invocations)."""

    def __init__(self, text: str = "Escena agricola con parcelas.") -> None:
        self.model = "gemma4:31b-it-q8_0"
        self.text = text
        self.calls = 0

    def caption(self, prompt: str, image_png_b64: str) -> tuple[str, float]:
        self.calls += 1
        return self.text, 3.1


def _make_s2(ndvi: float = 0.6, h: int = 16, w: int = 16) -> np.ndarray:
    """Single-timestep ``(1, 10, H, W)`` patch with a target spatial-mean NDVI."""
    s2 = np.zeros((1, 10, h, w), dtype=np.int16)
    red = 1000.0
    nir = red * (1.0 + ndvi) / (1.0 - ndvi)
    s2[0, 0] = 200  # B02
    s2[0, 1] = 400  # B03
    s2[0, 2] = int(red)  # B04
    s2[0, 6] = round(nir)  # B08
    return s2


def _make_semantic(counts: dict[int, int], h: int = 16, w: int = 16) -> np.ndarray:
    """``(H, W)`` semantic mask with a known per-class pixel histogram."""
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for cid, n in counts.items():
        flat[pos : pos + n] = cid
        pos += n
    return flat.reshape(h, w)


def _make_instance(n_parcels: int, h: int = 16, w: int = 16) -> np.ndarray:
    """``(H, W)`` ParcelIDs mask with ``n_parcels`` distinct non-zero instances."""
    flat = np.zeros(h * w, dtype=np.int32)
    per = (h * w) // (n_parcels + 1)
    for i in range(1, n_parcels + 1):
        flat[(i - 1) * per : i * per] = i
    return flat.reshape(h, w)


def _patch_pastis(monkeypatch: pytest.MonkeyPatch, patches: dict[str, dict[str, object]]) -> None:
    """Mocks PASTIS disk access on ``caption_cache`` (loader, index, phenology)."""

    def _fake_loader(
        patch_id: object, root: object = None, load_annotations: bool = True
    ) -> dict[str, object]:
        return patches[str(patch_id)]

    def _fake_index(meta: object = None) -> pl.DataFrame:
        rows = [
            {"patch_id": pid, "TILE": str(p["tile"]), "Fold": int(p["fold"])}
            for pid, p in patches.items()
        ]
        return pl.DataFrame(rows, schema={"patch_id": pl.Utf8, "TILE": pl.Utf8, "Fold": pl.Int64})

    monkeypatch.setattr(cc, "load_pastis_patch", _fake_loader)
    monkeypatch.setattr(cc, "pastis_patch_index", _fake_index)
    monkeypatch.setattr(cc, "load_typical_phenology", lambda path=None: {})


def _two_patch_fixture() -> dict[str, dict[str, object]]:
    """Two synthetic patches in fold 1 (Meadow+Corn and Grapevine)."""
    return {
        "100": {
            "s2": _make_s2(),
            "semantic": _make_semantic({1: 120, 3: 100}),
            "instance": _make_instance(3),
            "dates_s2": [20190715],
            "tile": "T31TFM",
            "fold": 1,
        },
        "200": {
            "s2": _make_s2(),
            "semantic": _make_semantic({8: 200}),
            "instance": _make_instance(2),
            "dates_s2": [20190810],
            "tile": "T30UXV",
            "fold": 1,
        },
    }


def test_generate_captions_parquet_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_pastis(monkeypatch, _two_patch_fixture())
    client = _MockGemmaClient()
    out = tmp_path / "captions.parquet"

    generate_captions_parquet(
        pastis_root=tmp_path / "PASTIS-R",
        out_path=out,
        folds=(1,),
        client=client,  # type: ignore[arg-type]
    )
    df = pl.read_parquet(out)
    assert set(df.columns) == set(CAPTIONS_SCHEMA)
    assert df.height == 2
    assert client.calls == 2
    row = df.filter(pl.col("patch_id") == "100").to_dicts()[0]
    assert row["tile"] == "T31TFM"
    assert row["composite_date"] == "20190715"
    assert sorted(row["present_class_ids"]) == [1, 3]
    assert row["n_regions"] == 3
    assert row["caption_model"] == "gemma4:31b-it-q8_0"
    assert row["prompt_version"] == "v2"


def test_generate_captions_parquet_resume_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_pastis(monkeypatch, _two_patch_fixture())
    out = tmp_path / "captions.parquet"

    first = _MockGemmaClient()
    generate_captions_parquet(
        tmp_path / "PASTIS-R",
        out,
        (1,),
        first,  # type: ignore[arg-type]
    )
    assert first.calls == 2

    # Second run with resume=True must NOT invoke the client (both already done).
    second = _MockGemmaClient()
    generate_captions_parquet(
        tmp_path / "PASTIS-R",
        out,
        (1,),
        second,
        resume=True,  # type: ignore[arg-type]
    )
    assert second.calls == 0
    assert pl.read_parquet(out).height == 2


def test_generate_captions_parquet_resume_only_new(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    patches = _two_patch_fixture()
    out = tmp_path / "captions.parquet"

    # First pass: only patch 100 visible.
    _patch_pastis(monkeypatch, {"100": patches["100"]})
    first = _MockGemmaClient()
    generate_captions_parquet(
        tmp_path / "PASTIS-R",
        out,
        (1,),
        first,  # type: ignore[arg-type]
    )
    assert first.calls == 1

    # Second pass: both patches visible; only 200 should be generated.
    _patch_pastis(monkeypatch, patches)
    second = _MockGemmaClient()
    generate_captions_parquet(
        tmp_path / "PASTIS-R",
        out,
        (1,),
        second,
        resume=True,  # type: ignore[arg-type]
    )
    assert second.calls == 1
    assert pl.read_parquet(out).height == 2


def test_generate_captions_parquet_flushes_incrementally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # With flush_every=1 the parquet exists after the FIRST caption, so a crash
    # mid-run never loses everything (the bug that wiped the previous run).
    _patch_pastis(monkeypatch, _two_patch_fixture())
    out = tmp_path / "captions.parquet"
    flushed_heights: list[int] = []
    real_flush = cc._flush_captions

    def _spy_flush(existing: object, new_rows: object, path: object) -> object:
        merged = real_flush(existing, new_rows, path)  # type: ignore[arg-type]
        if path.exists():  # type: ignore[attr-defined]
            flushed_heights.append(pl.read_parquet(path).height)
        return merged

    monkeypatch.setattr(cc, "_flush_captions", _spy_flush)
    generate_captions_parquet(
        tmp_path / "PASTIS-R",
        out,
        (1,),
        _MockGemmaClient(),
        flush_every=1,  # type: ignore[arg-type]
    )
    # Persisted progressively (1 then 2), not only at the very end.
    assert flushed_heights[0] == 1
    assert pl.read_parquet(out).height == 2


def test_generate_captions_parquet_resumes_after_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Simulate a crash after the first caption was flushed: the second client
    # must only generate the missing patch (resume reads the partial parquet).
    patches = _two_patch_fixture()
    out = tmp_path / "captions.parquet"

    class _CrashAfterFirst(_MockGemmaClient):
        def caption(self, prompt: str, image_png_b64: str) -> tuple[str, float]:
            if self.calls >= 1:
                raise RuntimeError("simulated SSH/tunnel drop")
            return super().caption(prompt, image_png_b64)

    _patch_pastis(monkeypatch, patches)
    with pytest.raises(RuntimeError, match="simulated"):
        generate_captions_parquet(
            tmp_path / "PASTIS-R",
            out,
            (1,),
            _CrashAfterFirst(),
            flush_every=1,  # type: ignore[arg-type]
        )
    # The first caption survived the crash on disk.
    assert pl.read_parquet(out).height == 1

    # Relaunch: only the missing patch is generated.
    resume_client = _MockGemmaClient()
    generate_captions_parquet(
        tmp_path / "PASTIS-R",
        out,
        (1,),
        resume_client,
        resume=True,  # type: ignore[arg-type]
    )
    assert resume_client.calls == 1
    assert pl.read_parquet(out).height == 2


def test_load_captions_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_pastis(monkeypatch, _two_patch_fixture())
    out = tmp_path / "captions.parquet"
    generate_captions_parquet(
        tmp_path / "PASTIS-R",
        out,
        (1,),
        _MockGemmaClient(),  # type: ignore[arg-type]
    )
    captions = load_captions(out)
    assert set(captions) == {"100", "200"}
    assert all(isinstance(v, str) and v for v in captions.values())


def test_load_captions_missing_returns_empty(tmp_path: Path) -> None:
    assert load_captions(tmp_path / "nope.parquet") == {}


# ---------------------------------------------------------------------------
# 5. Audit: clean cache passes, injected leak detected.
# ---------------------------------------------------------------------------


def _write_captions(path: Path, captions: list[str]) -> None:
    """Writes a minimal captions parquet with the canonical schema."""
    rows = [
        {
            "patch_id": str(i),
            "caption_glo": text,
            "caption_model": "gemma4:31b-it-q8_0",
            "prompt_version": "v2",
            "tile": "T31TFM",
            "composite_date": "20190715",
            "present_class_ids": [1],
            "n_regions": 1,
            "clases": "Meadow",
            "gen_seconds": 3.1,
        }
        for i, text in enumerate(captions)
    ]
    pl.DataFrame(rows, schema=CAPTIONS_SCHEMA).write_parquet(path)


def test_audit_captions_clean(tmp_path: Path) -> None:
    out = tmp_path / "captions.parquet"
    _write_captions(
        out,
        [
            "Una escena de praderas verdes con parcelas fragmentadas en verano.",
            "Vinedos en hileras con vigor moderado a finales de la temporada.",
        ],
    )
    counts = audit_captions(out)
    assert sum(counts.values()) == 0


def test_audit_captions_detects_injected_leak(tmp_path: Path) -> None:
    out = tmp_path / "captions.parquet"
    _write_captions(
        out,
        [
            "Escena limpia sin fugas.",
            "El NDVI=0.82 indica vegetacion densa.",  # numeric NDVI leak
            "Segun AlphaEarth la textura es uniforme.",  # AlphaEarth leak
            "En conclusion, la clase es Meadow.",  # label-as-answer leak
            "The class is Corn according to the embedding.",  # english leak
            "Computed from a satellite embedding bank.",  # satellite embedding leak
        ],
    )
    counts = audit_captions(out)
    assert counts["ndvi_numeric"] >= 1
    assert counts["alphaearth"] >= 1
    assert counts["la_clase_es"] >= 1
    assert counts["the_class_is"] >= 1
    assert counts["satellite_embedding"] >= 1
    assert sum(counts.values()) >= 5


def test_audit_captions_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="captions parquet not found"):
        audit_captions(tmp_path / "missing.parquet")


# ---------------------------------------------------------------------------
# 6. Spatial composition + parcel/area helpers (caption inputs, no leakage).
# ---------------------------------------------------------------------------


def test_spatial_composition_north_center_south() -> None:
    # Top band = Corn (3), middle = Meadow (1), bottom = Grapevine (8).
    semantic = np.zeros((9, 9), dtype=np.uint8)
    semantic[0:3] = 3
    semantic[3:6] = 1
    semantic[6:9] = 8
    phrase = cg._spatial_composition(semantic, tuple(range(1, 19)))
    assert "norte: Corn" in phrase
    assert "centro: Meadow" in phrase
    assert "sur: Grapevine" in phrase
    for pattern in _LEAK_PATTERNS:
        assert not pattern.search(phrase)


def test_patch_n_parcels_uses_instance_mask() -> None:
    semantic = _make_semantic({1: 120, 3: 100})
    instance = _make_instance(4)
    assert cc._patch_n_parcels(instance, semantic) == 4


def test_patch_n_parcels_fallback_without_instance() -> None:
    semantic = _make_semantic({1: 120, 3: 100})
    # Without an instance mask, falls back to the number of present crop classes.
    assert cc._patch_n_parcels(None, semantic) == 2


def test_patch_total_area_excludes_bg_void() -> None:
    semantic = _make_semantic({1: 50, 3: 30, 19: 10})  # void excluded
    # 256 total px, 50+30 crop, the rest background (0) + 10 void -> 80 crop.
    assert cc._patch_total_area_px(semantic) == 80


def test_present_classes_excludes_bg_void_and_inactive() -> None:
    semantic = _make_semantic({0: 20, 1: 50, 3: 30, 19: 10})
    present = cc._patch_present_classes(semantic, tuple(range(1, 19)))
    assert present == [1, 3]
