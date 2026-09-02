#!/usr/bin/env python3
"""
US-022b-C — Train TempCNN + InceptionTime phenology models with spatial CV.

Usage:
    poetry run python scripts/train_phenology_models.py [--n-parcels N] [--device DEVICE]
"""

import json
import sys
from pathlib import Path

import polars as pl

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.train.phenology_models import train_temporal_model


def main():
    """Train temporal models and print results as JSON."""
    import argparse

    parser = argparse.ArgumentParser(description="Train phenology temporal models")
    parser.add_argument("--n-parcels", type=int, default=4000, help="Number of parcels to sample")
    parser.add_argument("--n-epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--k-folds", type=int, default=5, help="K-fold spatial CV splits")
    parser.add_argument("--buffer-km", type=float, default=1.0, help="Buffer in km for spatial CV")
    args = parser.parse_args()

    # Load and sample data
    parquet_path = Path("data/test_fixtures/feature_selection_parcels_subset.parquet")
    if not parquet_path.exists():
        print(
            f"Error: {parquet_path} not found. Run: make feature-selection-subset",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pl.read_parquet(parquet_path).sample(n=args.n_parcels, seed=42)
    results = {}

    # Train models
    for model_kind in ("tempcnn", "inceptiontime"):
        print(f"Training {model_kind}...", file=sys.stderr, flush=True)
        r = train_temporal_model(
            df=df,
            model_kind=model_kind,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            seed=42,
            device=args.device,
            k_folds=args.k_folds,
            buffer_km=args.buffer_km,
        )
        results[model_kind] = {
            "f1_macro": r.f1_macro,
            "miou": r.miou,
            "n_parcels": r.n_parcels,
            "n_classes": r.n_classes,
            "train_time_s": r.train_time_s,
        }
        print(f"OK {model_kind} complete", file=sys.stderr)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
