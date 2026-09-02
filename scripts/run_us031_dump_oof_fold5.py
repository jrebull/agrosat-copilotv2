"""US-031 closure: real softmax/OOF dump of the 6 segmentation checkpoints (fold-5).

Runs ``dump_oof(fold=5)`` over the full held-out PASTIS fold-5, persisting the
per-pixel softmax (post-softmax, float16) and the per-parcel reconciliation as
parquet under ``ml/eval/oof/``, plus a manifest. These are the ensemble inputs
for US-040/041/042. Invoked once for the US-031 closure on real data (RTX 4070);
not part of the test suite.

Usage (from repo root):
    poetry run python scripts/run_us031_dump_oof_fold5.py
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from ml.eval.oof.dump_oof import dump_oof

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OOF_DIR = _REPO_ROOT / "ml" / "eval" / "oof"


def main() -> int:
    """Run the full fold-5 OOF dump and report sizes. Returns exit code."""
    logger.info("us031_dump_start", fold=5)
    manifest = dump_oof(
        fold=5,
        device="auto",
        out_dir=str(_OOF_DIR),
        dtype="float16",
        skip_missing=True,
        write_parcel=True,
    )

    total_mb = sum(p.stat().st_size for p in _OOF_DIR.glob("*.parquet")) / (1024 * 1024)
    n_ok = sum(1 for m in manifest.get("models", {}).values() if m.get("status") == "ok")
    logger.info(
        "us031_dump_done",
        models_ok=n_ok,
        total_parquet_mb=round(total_mb, 1),
        manifest=str(_OOF_DIR / "manifest.json"),
    )
    print(json.dumps(manifest, indent=2, default=str))
    print(f"TOTAL parquet size: {total_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
