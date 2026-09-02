"""Phenological temporal models: TempCNN + InceptionTime (US-022b-C).

Polars-in / PyTorch wrapper to train the official architectures of the
BreizhCrops benchmark (Russwurm et al. 2020) on the daily interpolated
NDVI/NDWI/EVI curve of each PASTIS-R parcel. Reuses
:func:`build_spatial_kfold` and :func:`compute_baseline_metrics` so that the
results are comparable with the tabular baseline closed in US-022
(commit ``87b7c57``, F1-macro 0.32).

Canonical decisions (plan ``docs/us-planning/us-022b.md`` §6.1 + ADR-006
D-ARQ-2 updated):

- **D-ARQ-2 (updated 2026-05-22)**: TempCNN and InceptionTime are PORTED
  natively into the repo in :mod:`ml.models.temporal` (based on Pelletier 2019 +
  Fawaz 2020, MIT license). The wrapper adapts I/O (Polars DataFrame ->
  tensor ``(B, T, C)``), builds the models via
  :func:`ml.models.temporal.build_temporal_model`, logs to MLflow and
  resolves the device prioritizing CUDA.
- **Spatial CV mandatory** (not random): reuses
  :func:`ml.train.baseline._build_cv_splits` (with cache).
- **Spatial CV 5-fold**, same partitions as the baseline (thanks to the cache
  keyed by ``n_rows + k + buffer + seed``).
- **MLflow tags**: ``data_version`` (DVC hash) + ``code_version`` (git sha)
  whenever ``mlflow_uri`` is passed and the run is opened; if the mlflow
  library is not available or the URI is ``None``, the wrapper degrades to
  "no tracking" without failing (CPU CI testability).
- **Lightweight architectures**: ADR-006 D3 confirms L4 24 GB as the target;
  Wen et al. 2025 trained heavier variants on RTX 3090. CPU smoke
  for 2 batches works for tests.

Agronomic / architecture references
-----------------------------------
- Pelletier, Webb & Petitjean 2019 — TempCNN. DOI 10.3390/rs11050523.
- Fawaz et al. 2020 — InceptionTime. DOI 10.1007/s10618-020-00710-y.
- Russwurm et al. 2020 — BreizhCrops dataset + benchmark.
  DOI 10.1109/IGARSS39084.2020.9324249.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import structlog

from ml.eval.metrics import compute_baseline_metrics
from ml.train.baseline import _DROP_CLASS_IDS, _build_cv_splits

logger = structlog.get_logger(__name__)

__all__ = [
    "TemporalDataset",
    "TemporalModelKind",
    "TemporalModelResult",
    "build_temporal_tensor",
    "train_temporal_model",
]


#: Supported temporal models (both live in :mod:`ml.models.temporal`).
TemporalModelKind = Literal["tempcnn", "inceptiontime"]

#: Canonical indices used as C channels of the time series (same as
#: :data:`ml.features.temporal_features.DEFAULT_FFT_INDICES`).
DEFAULT_TEMPORAL_INDICES: tuple[str, ...] = ("NDVI", "NDWI", "EVI")

#: Number of FFT harmonics (4 amps + 4 phases per index) present in the
#: US-018 subset. The tensor fed to TempCNN/InceptionTime reconstructs a
#: synthetic daily series from the FFT representation via partial inverse;
#: alternatively it accepts a pre-materialized daily curve.
DEFAULT_FFT_HARMONICS: int = 3

#: Default temporal length of the series (one agronomic year, T=72 ~5d
#: cadence; balance between cost and seasonal resolution).
DEFAULT_SEQUENCE_LENGTH: int = 72


# ---------------------------------------------------------------------------
# Output dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalModelResult:
    """Result of training a temporal model with spatial CV.

    Attributes:
        model_kind: ``"tempcnn"`` or ``"inceptiontime"``.
        f1_macro: Out-of-fold F1-macro of the spatial CV.
        f1_weighted: Support-weighted F1.
        miou: Mean IoU (macro Jaccard) — parcel-level proxy.
        cohen_kappa: Cohen agreement index.
        train_time_s: Total training wall-clock (sum of folds).
        n_parcels: Number of effective parcels after filtering out
            non-agronomic classes (``_DROP_CLASS_IDS`` inherited from the baseline).
        n_classes: Number of effective classes after filtering.
        mlflow_run_id: MLflow run ID if logged, ``None`` otherwise.
    """

    model_kind: TemporalModelKind
    f1_macro: float
    f1_weighted: float
    miou: float
    cohen_kappa: float
    train_time_s: float
    n_parcels: int
    n_classes: int
    mlflow_run_id: str | None
    y_true_oof: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    y_pred_oof: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    checkpoint_path: Path | None = None


# ---------------------------------------------------------------------------
# Polars -> tensor adapter.
# ---------------------------------------------------------------------------


class TemporalDataset:
    """Minimal Polars -> PyTorch tensor adapter for time series.

    Converts a :class:`polars.DataFrame` of phenological features into a
    tensor ``(B, T, C)`` ready for the ``forward`` of TempCNN / InceptionTime.

    Series reconstruction strategy:

    - If the DataFrame contains pre-materialized columns
      ``{idx}_t_{i:02d}`` (i in [0, T)) it uses them directly.
    - Otherwise, it reconstructs a daily pseudo-curve from the FFT
      coefficients ``{idx}_fft_amp_k`` and ``{idx}_fft_phase_k`` (inverse
      decomposition truncated to the number of available harmonics). It is a
      compact and agronomically faithful representation: 1 DC + 3 harmonics
      reconstruct the dominant seasonal signal.

    Args:
        df: Polars DataFrame with temporal features (output of
            :func:`ml.features.temporal_features.extract_temporal_features`
            or the US-018 subset).
        indices: C channels to use (default ``("NDVI", "NDWI", "EVI")``).
        sequence_length: T (series length); default 72.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        indices: tuple[str, ...] = DEFAULT_TEMPORAL_INDICES,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ) -> None:
        self.df = df
        self.indices = indices
        self.sequence_length = sequence_length

    def to_tensor(self) -> np.ndarray:
        """Returns the ``(B, T, C)`` matrix as a float32 ``np.ndarray``.

        It is the caller's responsibility to wrap it in ``torch.from_numpy(...)``.
        """
        n = self.df.height
        out = np.zeros((n, self.sequence_length, len(self.indices)), dtype=np.float32)
        for c_idx, idx in enumerate(self.indices):
            curve = _reconstruct_curve(
                self.df, index_name=idx, sequence_length=self.sequence_length
            )
            out[:, :, c_idx] = curve
        return out


def build_temporal_tensor(
    df: pl.DataFrame,
    *,
    indices: tuple[str, ...] = DEFAULT_TEMPORAL_INDICES,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Functional shortcut over :class:`TemporalDataset`.

    Args:
        df: Polars DataFrame with temporal features.
        indices: C channels (default NDVI/NDWI/EVI).
        sequence_length: T (default 72).

    Returns:
        ``np.ndarray`` of shape ``(n_rows, sequence_length, len(indices))``
        in ``float32``.
    """
    return TemporalDataset(df, indices=indices, sequence_length=sequence_length).to_tensor()


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def train_temporal_model(
    features_path: Path | str | None = None,
    *,
    df: pl.DataFrame | None = None,
    model_kind: TemporalModelKind,
    n_epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
    mlflow_uri: str | None = None,
    indices: tuple[str, ...] = DEFAULT_TEMPORAL_INDICES,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    k_folds: int = 5,
    buffer_km: float = 1.0,
    max_samples: int | None = None,
    checkpoint_dir: Path | str | None = None,
    use_class_weights: bool = True,
    use_weighted_sampler: bool = True,
    use_lr_scheduler: bool = True,
    early_stopping_patience: int = 20,
    val_fraction: float = 0.15,
    dropout: float | None = None,
    warmup_epochs: int = 5,
) -> TemporalModelResult:
    """Trains TempCNN or InceptionTime on the temporal FE with spatial CV.

    Args:
        features_path: Path to the phenological features parquet (US-018 /
            US-015). If ``df`` is passed, it is ignored.
        df: Already-loaded Polars DataFrame (shortcut for tests/notebooks).
        model_kind: ``"tempcnn"`` or ``"inceptiontime"``.
        n_epochs: Number of epochs per fold (default 30).
        batch_size: Batch size (default 256). On CPU dev it drops to 64.
        learning_rate: LR of the Adam optimizer (default 1e-3).
        seed: Deterministic seed (``np.random.default_rng``, ``torch.manual_seed``).
        device: ``"cuda"``, ``"cpu"`` or ``None`` (autodetect). In CI without GPU
            ``"cpu"`` is forced.
        mlflow_uri: If not ``None``, an MLflow run is attempted. If the
            mlflow library is not installed or the URI does not respond, it degrades
            to "no tracking" with a warning.
        indices: C channels (default ``("NDVI", "NDWI", "EVI")``).
        sequence_length: T (default 72).
        k_folds: Number of spatial CV folds (default 5).
        buffer_km: Anti-leakage buffer in km (default 1.0).
        max_samples: Deterministic uniform subsample. ``None`` = full
            dataset.
        checkpoint_dir: If passed, persists the ``state_dict`` of the last
            fold's model to disk with embedded metadata.
        use_class_weights: If ``True`` (default) weights the loss inversely
            to each class frequency to address the ~31x imbalance of the
            US-018 subset. Formula: ``w_k = N_total / (N_classes * N_k)``.
        use_weighted_sampler: If ``True`` (default) uses
            ``WeightedRandomSampler`` so each batch sees all classes
            proportionally. Crucial for F1-macro under strong imbalance.
        use_lr_scheduler: If ``True`` (default) applies linear warmup
            (``warmup_epochs`` epochs) + ``CosineAnnealingLR`` for the rest
            of training. Stabilizes convergence with large datasets.
        early_stopping_patience: Epochs without improvement in val F1-macro
            before stopping the fold. ``0`` = no early stopping. Default 20.
        val_fraction: Fraction of the fold's train reserved for
            intra-fold validation (early stopping + best epoch). Default 0.15.
            Stratified by class so as not to lose minority classes.
        dropout: Override of the model's dropout. ``None`` = paper default
            (0.5 TempCNN, 0.2 InceptionTime). For short series (T=72), lowering
            to 0.2-0.3 helps.
        warmup_epochs: Epochs of linear LR warmup (from 0 to ``learning_rate``)
            before activating cosine decay. Default 5.

    Returns:
        A :class:`TemporalModelResult` with out-of-fold metrics and metadata.

    Raises:
        ImportError: if ``torch`` is not installed.
        ValueError: if neither ``features_path`` nor ``df`` is passed, or if
            ``model_kind`` is not supported.
    """
    if model_kind not in ("tempcnn", "inceptiontime"):
        raise ValueError(f"`model_kind` must be 'tempcnn' or 'inceptiontime'; got {model_kind!r}.")
    if df is None:
        if features_path is None:
            raise ValueError("You must pass `features_path` or `df`.")
        df = pl.read_parquet(Path(features_path))

    clean_df = _prepare_temporal_dataframe(df)
    if max_samples is not None and max_samples > 0 and clean_df.height > max_samples:
        clean_df = clean_df.sample(n=max_samples, seed=seed, with_replacement=False)
        logger.info("temporal_subsampled", max_samples=max_samples, n=clean_df.height)

    label_encoder, y_encoded = _encode_labels(clean_df)
    n_classes = len(label_encoder)

    # Build the tensor (B, T, C) and the spatial splits (shared cache).
    X = build_temporal_tensor(clean_df, indices=indices, sequence_length=sequence_length)
    cv_splits = _build_cv_splits(clean_df, k_folds=k_folds, buffer_km=buffer_km, random_state=seed)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch is not installed. Run `poetry install --with ml,ml-gpu` "
            "or `poetry install --with ml` to train temporal models."
        ) from exc

    resolved_device = _resolve_device(device)
    torch.manual_seed(seed)

    mlflow_run_id: str | None = None
    mlflow_ctx = _try_mlflow_run(mlflow_uri, model_kind=model_kind)

    per_fold_metrics: list[dict[str, float]] = []
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    t0 = time.perf_counter()
    with mlflow_ctx as run_ctx:
        mlflow_run_id = run_ctx.run_id if run_ctx is not None else None
        if run_ctx is not None:
            run_ctx.log_params(
                {
                    "model_kind": model_kind,
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "k_folds": k_folds,
                    "buffer_km": buffer_km,
                    "sequence_length": sequence_length,
                    "indices": ",".join(indices),
                    "device": resolved_device,
                    "n_parcels": clean_df.height,
                    "n_classes": n_classes,
                    "use_class_weights": use_class_weights,
                    "use_weighted_sampler": use_weighted_sampler,
                    "use_lr_scheduler": use_lr_scheduler,
                    "early_stopping_patience": early_stopping_patience,
                    "val_fraction": val_fraction,
                    "warmup_epochs": warmup_epochs,
                    "dropout_override": dropout if dropout is not None else "default",
                }
            )

        # Global class weights (same for all folds for consistency;
        # computed over the full dataset).
        class_weights_t: Any = None
        if use_class_weights:
            counts = np.bincount(y_encoded, minlength=n_classes).astype(np.float64)
            counts = np.where(counts > 0, counts, 1.0)
            weights = float(y_encoded.size) / (float(n_classes) * counts)
            class_weights_t = torch.from_numpy(weights.astype(np.float32)).to(resolved_device)
            logger.info(
                "temporal_class_weights",
                min_weight=float(weights.min()),
                max_weight=float(weights.max()),
                imbalance_ratio=float(counts.max() / counts.min()),
            )

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            if train_idx.size == 0 or test_idx.size == 0:
                logger.warning("temporal_cv_fold_skipped", fold=fold_idx)
                continue

            # Intra-fold split train -> (train_inner, val_inner) stratified
            # by class for early stopping. If a class has only 1 sample it
            # goes to train_inner (it cannot be used for validation).
            rng = np.random.default_rng(seed + fold_idx)
            train_inner_idx, val_inner_idx = _stratified_inner_split(
                y_encoded[train_idx],
                val_fraction=val_fraction,
                rng=rng,
            )
            train_inner_global = train_idx[train_inner_idx]
            val_inner_global = train_idx[val_inner_idx]

            x_train = X[train_inner_global]
            x_val = X[val_inner_global]
            x_test = X[test_idx]
            y_train = y_encoded[train_inner_global]
            y_val = y_encoded[val_inner_global]
            y_test = y_encoded[test_idx]

            model_kwargs: dict[str, Any] = {}
            if dropout is not None:
                model_kwargs["dropout"] = dropout
            model = _build_temporal_model_native(
                model_kind=model_kind,
                input_dim=len(indices),
                num_classes=n_classes,
                sequence_length=sequence_length,
                device=resolved_device,
                **model_kwargs,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t)

            x_train_t = torch.from_numpy(x_train).to(resolved_device)
            y_train_t = torch.from_numpy(y_train).long().to(resolved_device)
            x_val_t = torch.from_numpy(x_val).to(resolved_device)
            x_test_t = torch.from_numpy(x_test).to(resolved_device)

            # WeightedRandomSampler: probability inverse to each class
            # frequency. Each batch sees all classes proportionally.
            n_train = x_train_t.shape[0]
            if use_weighted_sampler:
                fold_counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
                fold_counts = np.where(fold_counts > 0, fold_counts, 1.0)
                sample_weights = 1.0 / fold_counts[y_train]
                sample_weights_t = torch.from_numpy(sample_weights.astype(np.float64)).to(
                    resolved_device
                )
            else:
                sample_weights_t = None

            # LR scheduler: linear warmup + cosine annealing for the rest.
            scheduler = None
            if use_lr_scheduler and n_epochs > warmup_epochs:
                cosine_epochs = max(1, n_epochs - warmup_epochs)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cosine_epochs, eta_min=learning_rate * 0.01
                )

            best_val_f1 = -1.0
            best_state_dict: dict[str, Any] | None = None
            epochs_since_improve = 0

            for epoch in range(n_epochs):
                # Manual warmup: LR rises linearly from ~0 to learning_rate.
                if use_lr_scheduler and epoch < warmup_epochs:
                    warm_lr = learning_rate * (epoch + 1) / max(1, warmup_epochs)
                    for pg in optimizer.param_groups:
                        pg["lr"] = warm_lr

                model.train()
                epoch_loss = 0.0
                n_batches = 0
                # Sample selection: either weighted sampler or uniform permutation.
                if use_weighted_sampler and sample_weights_t is not None:
                    # As many samples as n_train (with replacement, by design
                    # of WeightedRandomSampler).
                    indices_iter = torch.multinomial(sample_weights_t, n_train, replacement=True)
                else:
                    indices_iter = torch.randperm(n_train, device=resolved_device)

                for batch_start in range(0, n_train, batch_size):
                    sel = indices_iter[batch_start : batch_start + batch_size]
                    if sel.numel() < 2:
                        continue
                    optimizer.zero_grad()
                    logits = model(x_train_t[sel])
                    loss = criterion(logits, y_train_t[sel])
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.item())
                    n_batches += 1
                avg_loss = epoch_loss / max(1, n_batches)

                # Cosine annealing after warmup.
                if scheduler is not None and epoch >= warmup_epochs:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]

                # Intra-fold validation: F1-macro for early stopping.
                model.eval()
                with torch.no_grad():
                    val_logits = model(x_val_t)
                    val_pred = val_logits.argmax(dim=-1).cpu().numpy()
                val_metrics = compute_baseline_metrics(
                    y_val, val_pred, labels=list(range(n_classes))
                )
                val_f1 = float(val_metrics["f1_macro"])

                if run_ctx is not None:
                    run_ctx.log_metric(f"fold{fold_idx}_train_loss", avg_loss, step=epoch)
                    run_ctx.log_metric(f"fold{fold_idx}_val_f1_macro", val_f1, step=epoch)
                    run_ctx.log_metric(f"fold{fold_idx}_lr", current_lr, step=epoch)

                # Early stopping: keep the best state_dict by val F1-macro.
                if val_f1 > best_val_f1 + 1e-6:
                    best_val_f1 = val_f1
                    best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1

                if early_stopping_patience > 0 and epochs_since_improve >= early_stopping_patience:
                    logger.info(
                        "temporal_early_stop",
                        fold=fold_idx,
                        epoch=epoch + 1,
                        best_val_f1=round(best_val_f1, 4),
                    )
                    break

            # Load the best fold checkpoint before evaluating on test.
            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)

            model.eval()
            with torch.no_grad():
                y_pred = model(x_test_t).argmax(dim=-1).cpu().numpy()
            fold_metrics = compute_baseline_metrics(y_test, y_pred, labels=list(range(n_classes)))
            per_fold_metrics.append(fold_metrics)
            y_true_chunks.append(y_test)
            y_pred_chunks.append(np.asarray(y_pred))
            logger.info(
                "temporal_fold_done",
                model_kind=model_kind,
                fold=f"{fold_idx + 1}/{len(cv_splits)}",
                f1_macro=round(fold_metrics["f1_macro"], 4),
            )

        y_true_oof = (
            np.concatenate(y_true_chunks) if y_true_chunks else np.array([], dtype=np.int64)
        )
        y_pred_oof = (
            np.concatenate(y_pred_chunks) if y_pred_chunks else np.array([], dtype=np.int64)
        )
        if y_true_oof.size > 0:
            oof_metrics = compute_baseline_metrics(
                y_true_oof, y_pred_oof, labels=list(range(n_classes))
            )
        else:
            oof_metrics = {
                "f1_macro": float("nan"),
                "f1_weighted": float("nan"),
                "miou": float("nan"),
                "accuracy": float("nan"),
                "cohen_kappa": float("nan"),
            }
        train_time_s = time.perf_counter() - t0
        if run_ctx is not None:
            run_ctx.log_metric("oof_f1_macro", oof_metrics["f1_macro"])
            run_ctx.log_metric("oof_f1_weighted", oof_metrics["f1_weighted"])
            run_ctx.log_metric("oof_miou", oof_metrics["miou"])
            run_ctx.log_metric("oof_cohen_kappa", oof_metrics["cohen_kappa"])
            run_ctx.log_metric("train_time_s", train_time_s)
            # Persist the last fold model's state_dict as an artifact.
            # Allows later reload with torch.load(...) for inference.
            if hasattr(run_ctx, "log_state_dict"):
                try:
                    run_ctx.log_state_dict(model, name=f"{model_kind}_last_fold.pt")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("mlflow_state_dict_save_failed", error=str(exc))

    # On-disk persistence of the state_dict (independent of MLflow).
    # Allows reloading the model with torch.load(...) without retraining.
    checkpoint_path: Path | None = None
    if checkpoint_dir is not None:
        try:
            ckpt_dir = Path(checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            code_v = _resolve_code_version()[:7]
            f1_str = f"{float(oof_metrics['f1_macro']):.4f}".replace(".", "p")
            checkpoint_path = ckpt_dir / f"{model_kind}_{code_v}_f1_{f1_str}_seed{seed}.pt"
            torch.save(
                {
                    "model_kind": model_kind,
                    "state_dict": model.state_dict(),
                    "input_dim": len(indices),
                    "num_classes": n_classes,
                    "sequence_length": sequence_length,
                    "indices": list(indices),
                    "label_encoder": label_encoder,
                    "f1_macro": float(oof_metrics["f1_macro"]),
                    "miou": float(oof_metrics["miou"]),
                    "n_parcels": int(clean_df.height),
                    "seed": seed,
                    "code_version": _resolve_code_version(),
                    "data_version": _resolve_data_version(),
                    "mlflow_run_id": mlflow_run_id,
                },
                checkpoint_path,
            )
            logger.info(
                "temporal_checkpoint_saved",
                path=str(checkpoint_path),
                f1_macro=round(float(oof_metrics["f1_macro"]), 4),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "temporal_checkpoint_save_failed",
                checkpoint_dir=str(checkpoint_dir),
                error=str(exc),
            )

    result = TemporalModelResult(
        model_kind=model_kind,
        f1_macro=float(oof_metrics["f1_macro"]),
        f1_weighted=float(oof_metrics["f1_weighted"]),
        miou=float(oof_metrics["miou"]),
        cohen_kappa=float(oof_metrics["cohen_kappa"]),
        train_time_s=float(train_time_s),
        n_parcels=int(clean_df.height),
        n_classes=n_classes,
        mlflow_run_id=mlflow_run_id,
        y_true_oof=y_true_oof.astype(np.int64, copy=False),
        y_pred_oof=y_pred_oof.astype(np.int64, copy=False),
        checkpoint_path=checkpoint_path,
    )
    logger.info(
        "temporal_train_done",
        **{
            "model_kind": model_kind,
            "f1_macro": round(result.f1_macro, 4),
            "n_parcels": result.n_parcels,
            "n_classes": result.n_classes,
            "train_time_s": round(result.train_time_s, 2),
        },
    )
    return result


# ---------------------------------------------------------------------------
# Private helpers.
# ---------------------------------------------------------------------------


def _prepare_temporal_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Filters out non-agronomic classes (baseline patch)."""
    if "class_id" not in df.columns:
        raise ValueError("`df` must contain the `class_id` column.")
    clean = df.filter(
        pl.col("class_id").is_not_null() & ~pl.col("class_id").is_in(list(_DROP_CLASS_IDS))
    )
    if clean.height == 0:
        raise ValueError("After filtering out non-agronomic classes the DataFrame was empty.")
    return clean


def _encode_labels(df: pl.DataFrame) -> tuple[list[int], np.ndarray]:
    """Re-maps `class_id` to contiguous labels ``[0, n_classes)``."""
    raw = df.get_column("class_id").to_numpy().astype(np.int64)
    unique_classes = sorted(int(c) for c in np.unique(raw).tolist())
    mapping = {c: i for i, c in enumerate(unique_classes)}
    y_encoded = np.array([mapping[int(v)] for v in raw], dtype=np.int64)
    return unique_classes, y_encoded


def _stratified_inner_split(
    y: np.ndarray,
    *,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Class-stratified split of a fold's train for early stopping.

    Guarantees that each class with >= 2 samples has at least one in val.
    Classes with a single sample go entirely to train.

    Args:
        y: Train labels (1D, encoded).
        val_fraction: Target fraction for validation (0 < x < 1).
        rng: Per-fold deterministic generator.

    Returns:
        Tuple ``(train_inner_idx, val_inner_idx)``, indices relative to ``y``.
    """
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_val = round(len(cls_idx) * val_fraction)
        # Guarantee >= 1 in val if there are >= 2 samples of the class.
        if len(cls_idx) >= 2 and n_val == 0:
            n_val = 1
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
    )


def _build_temporal_model_native(
    *,
    model_kind: TemporalModelKind,
    input_dim: int,
    num_classes: int,
    sequence_length: int,
    device: str,
    **model_overrides: Any,
) -> Any:
    """Builds the TempCNN or InceptionTime model from ``ml.models.temporal``.

    Own implementation (not breizhcrops) after the porting of the updated
    ADR-006 D-ARQ-2. Lazy import of torch (~3s) and of the architectures;
    tests can monkeypatch this helper to inject mock models in CI without
    touching the real architecture.

    Args:
        model_kind: ``"tempcnn"`` or ``"inceptiontime"``.
        input_dim: Number of C channels.
        num_classes: Effective classes.
        sequence_length: T.
        device: device string.
        **model_overrides: Additional hyperparameters (``dropout``,
            ``hidden_dim``, ``depth``, etc.) passed to the constructor.
    """
    import torch

    from ml.models.temporal import build_temporal_model

    model = build_temporal_model(
        model_kind,
        input_dim=input_dim,
        num_classes=num_classes,
        sequence_length=sequence_length,
        **model_overrides,
    )
    return model.to(torch.device(device))


def _resolve_device(requested: str | None) -> str:
    """Resolves the desired device prioritizing CUDA when ``"auto"``.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``None``. ``None``
            is equivalent to ``"auto"``.

    Returns:
        String ready for ``torch.device(...)``: ``"cuda"`` if CUDA is
        available and ``requested`` allows it, ``"cpu"`` otherwise.
    """
    import torch

    if requested in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda_requested_but_unavailable_fallback_cpu")
        return "cpu"
    return requested


def _reconstruct_curve(
    df: pl.DataFrame,
    *,
    index_name: str,
    sequence_length: int,
) -> np.ndarray:
    """Reconstructs the daily curve of an index as a ``(N, T)`` matrix.

    Source priority:
      1. Pre-materialized columns ``{idx}_t_{i:02d}``.
      2. Inverse FFT reconstruction from ``{idx}_fft_amp_k`` and
         ``{idx}_fft_phase_k``.
      3. Fallback: repeats ``{idx}_mean`` constant over T (degenerate
         model, returns a flat series).

    Columns may have nulls; they are imputed to 0.0 per T column.
    """
    n = df.height
    T = sequence_length

    # 1) pre-materialized curves.
    cols_t = [f"{index_name}_t_{i:02d}" for i in range(T)]
    if all(c in df.columns for c in cols_t):
        matrix = df.select(cols_t).fill_null(0.0).to_numpy().astype(np.float32)
        return matrix

    # 2) FFT inverse reconstruction.
    amp_cols = [f"{index_name}_fft_amp_{k}" for k in range(DEFAULT_FFT_HARMONICS + 1)]
    phase_cols = [f"{index_name}_fft_phase_{k}" for k in range(DEFAULT_FFT_HARMONICS + 1)]
    if all(c in df.columns for c in amp_cols) and all(c in df.columns for c in phase_cols):
        amps = df.select(amp_cols).fill_null(0.0).to_numpy().astype(np.float64)
        phases = df.select(phase_cols).fill_null(0.0).to_numpy().astype(np.float64)
        # Frequencies per harmonic (1 cycle per year for k=1, 2 for k=2, ...).
        t_axis = np.arange(T, dtype=np.float64)
        curve = np.zeros((n, T), dtype=np.float32)
        # k=0 (DC): constant signal = DC amplitude.
        curve += amps[:, 0:1].astype(np.float32)
        for k in range(1, DEFAULT_FFT_HARMONICS + 1):
            freq = 2.0 * np.pi * k * t_axis / T  # shape (T,)
            # amplitude * cos(freq + phase). Single-sided (consistent with FFT).
            phase_k = phases[:, k : k + 1]  # (N, 1)
            amp_k = amps[:, k : k + 1]  # (N, 1)
            curve += (amp_k * np.cos(freq[None, :] + phase_k)).astype(np.float32)
        return curve

    # 3) Flat fallback: repeat the mean.
    mean_col = f"{index_name}_mean"
    if mean_col in df.columns:
        means = df.get_column(mean_col).fill_null(0.0).to_numpy().astype(np.float32)
        return np.broadcast_to(means[:, None], (n, T)).copy()

    # Degenerate case: 0.0 series (tests should avoid it).
    return np.zeros((n, T), dtype=np.float32)


class _NullMlflowRun:
    """Null context manager used when MLflow is not available."""

    run_id: str | None = None

    def __enter__(self) -> _NullMlflowRun | None:
        return None

    def __exit__(self, *args: object) -> None:
        return None

    def log_params(self, params: dict[str, object]) -> None:  # pragma: no cover
        return None

    def log_metric(
        self, key: str, value: float, *, step: int | None = None
    ) -> None:  # pragma: no cover
        return None


class _MlflowRun:
    """Thin context manager over mlflow.start_run; logging + standard tags."""

    def __init__(self, uri: str, model_kind: TemporalModelKind) -> None:
        self.uri = uri
        self.model_kind = model_kind
        self.run_id: str | None = None
        self._mlflow: Any = None
        self._run: Any = None

    def __enter__(self) -> _MlflowRun:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.uri)
        run = mlflow.start_run(run_name=f"phenology_{self.model_kind}")
        self._run = run
        self.run_id = run.info.run_id
        mlflow.set_tags(
            {
                "data_version": _resolve_data_version(),
                "code_version": _resolve_code_version(),
                "module": "ml.train.phenology_models",
                "model_kind": self.model_kind,
            }
        )
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._mlflow is not None:
            self._mlflow.end_run()

    def log_params(self, params: dict[str, object]) -> None:
        if self._mlflow is not None:
            self._mlflow.log_params(params)

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        if self._mlflow is not None and value == value:  # filter out NaN
            self._mlflow.log_metric(key, value, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Logs a file as an artifact in the current run."""
        if self._mlflow is not None:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_state_dict(self, model: Any, name: str = "model_state_dict.pt") -> None:
        """Persists the model's ``state_dict`` as a serialized artifact."""
        if self._mlflow is None:
            return
        import tempfile

        import torch as _torch

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            _torch.save(model.state_dict(), path)
            self._mlflow.log_artifact(str(path), artifact_path="checkpoints")


def _try_mlflow_run(uri: str | None, *, model_kind: TemporalModelKind):  # type: ignore[no-untyped-def]
    """Returns a context manager: real if MLflow available, null otherwise."""
    if uri is None:
        return _NullMlflowRun()
    try:
        import mlflow  # noqa: F401
    except ImportError:  # pragma: no cover
        logger.warning("mlflow_not_available", uri=uri)
        return _NullMlflowRun()
    return _MlflowRun(uri=uri, model_kind=model_kind)


def _resolve_data_version() -> str:
    """Resolves the ``data_version`` tag (short DVC hash if .dvc available)."""
    try:
        import subprocess

        repo_root = Path(__file__).resolve().parents[2]
        dvc_file = (
            repo_root / "data" / "test_fixtures" / "feature_selection_parcels_subset.parquet.dvc"
        )
        if dvc_file.exists():
            content = dvc_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                if "md5:" in line:
                    return line.split("md5:")[-1].strip()[:12]
        # Fallback: git rev of the data/ folder.
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", "data/"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            check=False,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _resolve_code_version() -> str:
    """Resolves the ``code_version`` tag (short git HEAD sha)."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


# Sentinel-free helper to override imports in tests.
_TEMPORAL_BUILDER = _build_temporal_model_native


def _set_model_builder(builder):  # type: ignore[no-untyped-def]  # pragma: no cover - test util
    """Injects an alternative builder (test-only use)."""
    global _TEMPORAL_BUILDER
    _TEMPORAL_BUILDER = builder
