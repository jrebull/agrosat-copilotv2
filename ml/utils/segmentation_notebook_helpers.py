"""DRY helpers for the segmentation notebooks ``notebooks/segmentation/5*.ipynb``.

Centralizes the patterns repeated in ``5a_deeplabv3plus.ipynb`` and
``5b_tsvit.ipynb`` so each cell becomes a composition of calls + markdown +
display, with no inline code. Mirrors the pattern of
:mod:`ml.utils.baseline_notebook_helpers`.

Covers:

- :func:`run_training_or_load` — skip-if-trained shortcut: if the trained
  ``best.pt`` exists, reads its metrics instead of re-training; otherwise it
  launches the :mod:`ml.train.train_segmentation` CLI via subprocess and
  parses its log.
- :func:`training_results_table` — Polars DataFrame of one or several results.
- :func:`build_variant_comparison` — base-vs-variant table with delta (5b).
- :func:`segmentation_eval_table` — table of evaluation metrics per checkpoint.
- :func:`per_class_table` / :func:`per_class_comparison_table` — IoU/F1 per class.
- :func:`plot_confusion_matrix` — row-normalized confusion matrix.
- :func:`read_segmentation_lineage` — robust read of the MLflow lineage.
- :func:`pastis_class_names` — training index map ``[0..17]`` -> name.
- Re-exports from :mod:`ml.eval.segmentation_inference`:
  :func:`load_segmentation_model`, :func:`evaluate_checkpoint`,
  :func:`predict_examples` (the logic lives there; here it is only re-exposed so
  the notebook imports everything from a single module).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, SupportsFloat, cast

import polars as pl
import structlog

# Re-export: the implementation remains in ml.eval.segmentation_inference.
# The dependency is one-directional (utils -> eval), no cycle.
from ml.eval.segmentation_inference import (
    evaluate_checkpoint,
    load_segmentation_model,
    predict_examples,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from matplotlib.figure import Figure

logger = structlog.get_logger(__name__)

__all__ = [
    "TrainingResult",
    "build_variant_comparison",
    "evaluate_checkpoint",
    "load_segmentation_model",
    "pastis_class_names",
    "per_class_comparison_table",
    "per_class_table",
    "plot_confusion_matrix",
    "predict_examples",
    "read_segmentation_lineage",
    "run_training_or_load",
    "segmentation_eval_table",
    "training_results_table",
]

#: Default MLflow run_name per architecture (must match the CLI).
_DEFAULT_RUN_NAMES = {
    "deeplabv3plus": "alt-deeplabv3plus-mobilenet-v1",
    "tsvit": "alt-tsvit-v1",
    "tsvit-pheno": "alt-tsvit-pheno-v1",
}

#: Checkpoint subdirectory per architecture + target.
_CKPT_SUBDIR = {
    ("deeplabv3plus", "semantic18"): "deeplab-18",
    ("deeplabv3plus", "hcat6"): "deeplab-6",
    ("tsvit", "semantic18"): "tsvit-v1",
    ("tsvit", "hcat6"): "tsvit-v1",
    ("tsvit-pheno", "semantic18"): "tsvit-pheno-v1",
    ("tsvit-pheno", "hcat6"): "tsvit-pheno-v1",
}

#: Temporal architectures (receive ``--n-timesteps`` and a time series).
_TEMPORAL_KINDS = frozenset({"tsvit", "tsvit-pheno"})


@dataclass(frozen=True)
class TrainingResult:
    """Result of :func:`run_training_or_load` for a variant.

    Attributes:
        model: Architecture (``deeplabv3plus`` / ``tsvit`` / ``tsvit-pheno``).
        miou: mIoU of the best epoch, or ``None`` if the run failed.
        f1_macro: F1-macro of the best epoch, or ``None``.
        pixel_acc: Pixel accuracy of the best epoch, or ``None``.
        returncode: Return code of the subprocess (``0`` if the checkpoint was
            reused), or ``None`` if it was not even launched.
        error: Error message in degraded mode, or ``None``.
        from_checkpoint: ``True`` if ``best.pt`` was reused without re-training.
        best_epoch: Epoch of the best checkpoint, or ``None``.
        cli_command: Documented CLI command (to display in the notebook).
    """

    model: str
    miou: float | None
    f1_macro: float | None
    pixel_acc: float | None
    returncode: int | None
    error: str | None
    from_checkpoint: bool
    best_epoch: int | None
    cli_command: str


def _resolve_run_name(model_kind: str, run_name: str | None) -> str:
    return run_name or _DEFAULT_RUN_NAMES.get(model_kind, model_kind)


def _documented_cli(
    model_kind: str, target: str, run_name: str, epochs: int, batch_size: int
) -> str:
    """Build the text of the documented CLI command (for display)."""
    cmd = (
        f"python -m ml.train.train_segmentation --model {model_kind} "
        f"--epochs {epochs} --batch-size {batch_size} --target {target} "
        f"--run-name {run_name}"
    )
    if model_kind in _TEMPORAL_KINDS:
        cmd += " --n-timesteps 10"
    return cmd


def run_training_or_load(
    model_kind: Literal["deeplabv3plus", "tsvit", "tsvit-pheno"],
    *,
    n_epochs: int,
    target: Literal["semantic18", "hcat6"] = "semantic18",
    run_name: str | None = None,
    checkpoint_dir: Path | str = Path("checkpoints/segmentation"),
    repo_root: Path | None = None,
    batch_size: int = 4,
    n_timesteps: int = 10,
    device: str = "auto",
    run_full: bool = False,
    documented_epochs: int | None = None,
    documented_batch_size: int | None = None,
    python_executable: str | None = None,
    on_message: Callable[[str], None] | None = None,
) -> TrainingResult:
    """Reuse the trained checkpoint or launch the training CLI.

    Skip-if-trained shortcut: if ``run_full`` is ``False`` and the variant's
    ``best.pt`` exists (``checkpoint_dir/<sub>/best.pt``), loads its
    ``best_metrics`` without re-training. Otherwise it builds and executes the
    CLI command via subprocess (the run is recorded in MLflow) and parses its
    structlog log looking for the last ``cli_done`` line. Robust degraded mode:
    subprocess errors, ``returncode != 0`` or unparseable metrics return a
    :class:`TrainingResult` with ``None`` fields without breaking execution.

    Args:
        model_kind: Architecture to train.
        n_epochs: Epochs to train if re-training.
        target: ``semantic18`` (18 classes) or ``hcat6`` (6 HCAT groups).
        run_name: MLflow run name; if ``None`` uses the default per kind.
        checkpoint_dir: Root of segmentation checkpoints.
        repo_root: Repo root for the subprocess ``cwd`` (default: CWD).
        batch_size: Training batch size.
        n_timesteps: Temporal steps (temporal architectures only).
        device: ``auto`` / ``cuda`` / ``cpu``.
        run_full: If ``True``, ignores the shortcut and forces training.
        documented_epochs: Epochs to show in the documented command (default
            ``n_epochs``); useful to reflect the actual run (30/15).
        documented_batch_size: Batch to show in the documented command.
        python_executable: Interpreter to use (default ``sys.executable``).
        on_message: Callback that receives the markdown to display in the notebook.

    Returns:
        :class:`TrainingResult` with the metrics of the best epoch.
    """
    run_name = _resolve_run_name(model_kind, run_name)
    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    py = python_executable or sys.executable
    doc_epochs = documented_epochs if documented_epochs is not None else n_epochs
    doc_batch = documented_batch_size if documented_batch_size is not None else batch_size
    cli_doc = _documented_cli(model_kind, target, run_name, doc_epochs, doc_batch)

    def _emit(msg: str) -> None:
        if on_message is not None:
            on_message(msg)

    # 1. Skip-if-trained shortcut.
    if not run_full:
        sub = _CKPT_SUBDIR.get((model_kind, target))
        if sub is not None:
            ckpt = Path(checkpoint_dir) / sub / "best.pt"
            ckpt_abs = ckpt if ckpt.is_absolute() else repo / ckpt
            if ckpt_abs.is_file():
                import torch

                ck = torch.load(ckpt_abs, map_location="cpu", weights_only=False)
                bm = ck.get("best_metrics", {})
                _emit(
                    f"Checkpoint entrenado ya presente (`{ckpt_abs.relative_to(repo)}`, "
                    f"mejor epoch {bm.get('best_epoch')}): se omite el re-entrenamiento. "
                    f"Comando CLI documentado:\n\n`{cli_doc}`"
                )
                logger.info(
                    "training_loaded_from_checkpoint",
                    model_kind=model_kind,
                    target=target,
                    best_epoch=bm.get("best_epoch"),
                )
                return TrainingResult(
                    model=model_kind,
                    miou=float(bm["miou"]) if "miou" in bm else None,
                    f1_macro=float(bm["f1_macro"]) if "f1_macro" in bm else None,
                    pixel_acc=float(bm["pixel_acc"]) if "pixel_acc" in bm else None,
                    returncode=0,
                    error=None,
                    from_checkpoint=True,
                    best_epoch=int(bm["best_epoch"]) if "best_epoch" in bm else None,
                    cli_command=cli_doc,
                )

    # 2. Launch the training CLI.
    cmd = [
        py,
        "-m",
        "ml.train.train_segmentation",
        "--model",
        model_kind,
        "--epochs",
        str(n_epochs),
        "--batch-size",
        str(batch_size),
        "--target",
        target,
        "--device",
        device,
        "--run-name",
        run_name,
    ]
    if model_kind in _TEMPORAL_KINDS:
        cmd += ["--n-timesteps", str(n_timesteps)]
    _emit(f"`{' '.join(cmd)}`")

    miss = TrainingResult(
        model=model_kind,
        miou=None,
        f1_macro=None,
        pixel_acc=None,
        returncode=None,
        error=None,
        from_checkpoint=False,
        best_epoch=None,
        cli_command=cli_doc,
    )
    try:
        proc = subprocess.run(  # noqa: S603 - cmd is built from controlled literals, not external input
            cmd, cwd=str(repo), capture_output=True, text=True, check=False
        )
    except OSError as exc:
        _emit(f"> Subprocess no disponible: `{exc}`. Modo degradado.")
        logger.warning("training_subprocess_failed", model_kind=model_kind, error=str(exc))
        return TrainingResult(**{**miss.__dict__, "error": f"subprocess: {exc}"})

    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        tail = "\n".join(log.strip().splitlines()[-12:])
        _emit(f"> Entrenamiento fallido (returncode={proc.returncode}).\n\n```\n{tail}\n```")
        logger.warning("training_returncode_nonzero", model_kind=model_kind, rc=proc.returncode)
        return TrainingResult(
            **{
                **miss.__dict__,
                "returncode": proc.returncode,
                "error": f"returncode={proc.returncode}",
            }
        )

    parsed = _parse_cli_done(log)
    logger.info(
        "training_completed",
        model_kind=model_kind,
        **{k: parsed.get(k) for k in ("miou", "f1_macro", "pixel_acc")},
    )
    return TrainingResult(
        model=model_kind,
        miou=parsed.get("miou"),
        f1_macro=parsed.get("f1_macro"),
        pixel_acc=parsed.get("pixel_acc"),
        returncode=0,
        error=None if parsed.get("miou") is not None else "metricas no parseables",
        from_checkpoint=False,
        best_epoch=None,
        cli_command=cli_doc,
    )


def _parse_cli_done(log: str) -> dict[str, float | None]:
    """Extract miou/f1_macro/pixel_acc from the last ``cli_done`` line of the log.

    The CLI logs with structlog (``key=value`` format). Looks for the last line
    that contains ``cli_done`` and parses the three metric tokens.

    Args:
        log: Combined text of the subprocess stdout + stderr.

    Returns:
        Dict with ``miou``, ``f1_macro``, ``pixel_acc`` (``None`` if not found).
    """
    result: dict[str, float | None] = {"miou": None, "f1_macro": None, "pixel_acc": None}
    for line in reversed(log.splitlines()):
        if "cli_done" not in line:
            continue
        for key in result:
            token = f"{key}="
            if token in line:
                raw = line.split(token, 1)[1].split()[0].rstrip(",")
                try:
                    result[key] = float(raw)
                except ValueError:
                    result[key] = None
        break
    return result


def training_results_table(
    results: TrainingResult | Sequence[TrainingResult],
) -> pl.DataFrame:
    """Convert one or several :class:`TrainingResult` into a Polars DataFrame.

    Args:
        results: A single result or a sequence of results.

    Returns:
        DataFrame with columns ``model, miou, f1_macro, pixel_acc, returncode``.
    """
    rows = [results] if isinstance(results, TrainingResult) else list(results)
    return pl.DataFrame(
        [
            {
                "model": r.model,
                "miou": r.miou,
                "f1_macro": r.f1_macro,
                "pixel_acc": r.pixel_acc,
                "returncode": r.returncode,
            }
            for r in rows
        ],
        schema={
            "model": pl.Utf8,
            "miou": pl.Float64,
            "f1_macro": pl.Float64,
            "pixel_acc": pl.Float64,
            "returncode": pl.Int64,
        },
    )


def build_variant_comparison(
    results: Sequence[TrainingResult],
    *,
    baseline_model: str = "tsvit",
    variant_model: str = "tsvit-pheno",
    metrics: Sequence[str] = ("miou", "f1_macro", "pixel_acc"),
) -> pl.DataFrame | None:
    """Base-vs-variant comparison table with per-metric delta.

    Args:
        results: Results of both variants.
        baseline_model: Name of the base model.
        variant_model: Name of the variant model.
        metrics: Metrics to compare.

    Returns:
        DataFrame with columns ``metrica, <baseline>, <variant>, delta``, or
        ``None`` if either of the two models is missing or any metric is ``None``.
    """
    by_model = {r.model: r for r in results}
    base = by_model.get(baseline_model)
    variant = by_model.get(variant_model)
    if base is None or variant is None:
        return None
    rows = []
    for metric in metrics:
        b = getattr(base, metric)
        v = getattr(variant, metric)
        if b is None or v is None:
            return None
        rows.append(
            {
                "metrica": metric,
                baseline_model: round(float(b), 4),
                variant_model: round(float(v), 4),
                "delta": round(float(v) - float(b), 4),
            }
        )
    return pl.DataFrame(rows)


def segmentation_eval_table(
    results: dict[str, dict[str, object]],
    *,
    label_col: str = "variante",
) -> pl.DataFrame:
    """Table of evaluation metrics for one or several checkpoints.

    Args:
        results: ``{name: metrics_dict}`` (metrics from
            :func:`ml.eval.metrics.dense_metrics_from_cm`).
        label_col: Name of the label column.

    Returns:
        DataFrame with ``label_col, mIoU, F1_macro, pixel_acc, balanced_acc,
        cohen_kappa``.
    """

    def _f(m: dict[str, object], key: str) -> float:
        return float(cast("SupportsFloat", m[key]))

    rows = [
        {
            label_col: name,
            "mIoU": round(_f(m, "miou"), 4),
            "F1_macro": round(_f(m, "f1_macro"), 4),
            "pixel_acc": round(_f(m, "pixel_acc"), 4),
            "balanced_acc": round(_f(m, "balanced_acc"), 4),
            "cohen_kappa": round(_f(m, "cohen_kappa"), 4),
        }
        for name, m in results.items()
    ]
    return pl.DataFrame(rows)


def _class_label(c: int, class_names: dict[int, str] | None) -> str:
    if class_names is not None:
        return class_names.get(c, f"clase_{c}")
    return f"grupo_{c}"


def per_class_table(
    metrics: dict[str, object],
    *,
    class_names: dict[int, str] | None = None,
    num_classes: int = 18,
    out_csv: Path | str | None = None,
) -> pl.DataFrame:
    """Per-class IoU/F1 table of a single checkpoint, sorted by IoU desc.

    Args:
        metrics: metrics_dict with ``per_class_iou`` and ``per_class_f1``.
        class_names: Index -> name map (``None`` for ``grupo_c``).
        num_classes: Number of classes.
        out_csv: If given, persists the table.

    Returns:
        DataFrame with columns ``clase, IoU, F1``.
    """
    rows = [
        {
            "clase": _class_label(c, class_names),
            "IoU": round(float(metrics["per_class_iou"][c]), 4),  # type: ignore[index]
            "F1": round(float(metrics["per_class_f1"][c]), 4),  # type: ignore[index]
        }
        for c in range(num_classes)
    ]
    df = pl.DataFrame(rows).sort("IoU", descending=True)
    if out_csv is not None:
        df.write_csv(out_csv)
    return df


def per_class_comparison_table(
    baseline_metrics: dict[str, object],
    variant_metrics: dict[str, object],
    *,
    class_names: dict[int, str] | None = None,
    num_classes: int = 18,
    out_csv: Path | str | None = None,
) -> pl.DataFrame:
    """Per-class IoU/F1 table of two variants side by side with IoU delta.

    Args:
        baseline_metrics: metrics_dict of the base variant.
        variant_metrics: metrics_dict of the variant to compare.
        class_names: Index -> name map (``None`` for ``grupo_c``).
        num_classes: Number of classes.
        out_csv: If given, persists the table.

    Returns:
        DataFrame sorted by ``delta_IoU`` desc with columns
        ``clase, IoU_base, IoU_pheno, delta_IoU, F1_base, F1_pheno``.
    """
    rows = []
    for c in range(num_classes):
        b_iou = float(baseline_metrics["per_class_iou"][c])  # type: ignore[index]
        v_iou = float(variant_metrics["per_class_iou"][c])  # type: ignore[index]
        rows.append(
            {
                "clase": _class_label(c, class_names),
                "IoU_base": round(b_iou, 4),
                "IoU_pheno": round(v_iou, 4),
                "delta_IoU": round(v_iou - b_iou, 4),
                "F1_base": round(float(baseline_metrics["per_class_f1"][c]), 4),  # type: ignore[index]
                "F1_pheno": round(float(variant_metrics["per_class_f1"][c]), 4),  # type: ignore[index]
            }
        )
    df = pl.DataFrame(rows).sort("delta_IoU", descending=True)
    if out_csv is not None:
        df.write_csv(out_csv)
    return df


def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    class_names: dict[int, str] | None = None,
    title: str = "Matriz de confusion (normalizada)",
    num_classes: int | None = None,
    out_path: Path | str | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Draw the row-normalized confusion matrix.

    Args:
        cm: Unnormalized confusion matrix ``(C, C)``.
        class_names: Index -> name map for the ticks (``None`` = no labels).
        title: Title of the figure.
        num_classes: Number of classes (default: ``cm.shape[0]``).
        out_path: If given, saves the figure.
        cmap: Colormap.

    Returns:
        The matplotlib figure (the cell does ``display(fig)`` + ``plt.close(fig)``).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n = num_classes if num_classes is not None else int(cm.shape[0])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    # Explicit layout engine: `ml.report.paper_figures` switches rcParams to
    # constrained layout at import, and `tight_layout()` on a figure whose
    # colorbar was placed by that engine raises RuntimeError.
    fig, ax = plt.subplots(figsize=(8, 7), layout="tight")
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Verdad")
    if class_names is not None:
        labels = [class_names.get(c, str(c)) for c in range(n)]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
    return fig


def read_segmentation_lineage(
    run_names: str | Sequence[str],
    *,
    experiment_name: str = "agrosat-segmentation",
    tracking_uri: str | None = None,
    max_results: int = 50,
) -> pl.DataFrame | None:
    """Read the MLflow lineage of the segmentation runs (degraded mode).

    Retrieves the runs of the experiment, filters by ``run_names`` client-side
    (avoids the fragility of the server-side ``IN`` operator over tags) and
    returns a DataFrame with best metrics and version tags. Returns ``None``
    (does not raise) on any failure, to preserve the end-to-end execution of
    the notebook in papermill/CI.

    Args:
        run_names: Run name or list of run names to retrieve.
        experiment_name: MLflow experiment.
        tracking_uri: Tracking URI; if ``None`` uses
            :func:`ml.utils.mlflow_utils.resolve_tracking_uri`.
        max_results: Maximum runs to fetch before filtering.

    Returns:
        DataFrame with ``run_name, miou, f1_macro, pixel_acc, code_version,
        data_version`` (columns present), or ``None`` in degraded mode.
    """
    wanted = {run_names} if isinstance(run_names, str) else set(run_names)
    try:
        import mlflow

        from ml.utils.mlflow_utils import resolve_tracking_uri

        mlflow.set_tracking_uri(tracking_uri or resolve_tracking_uri())
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            logger.info("lineage_experiment_absent", experiment=experiment_name)
            return None
        # search_runs with output_format="pandas" (default) returns a DataFrame;
        # the mlflow stub types it as list[Run], hence the cast.
        runs_pd = cast(
            "pd.DataFrame",
            mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=max_results,
            ),
        )
        if runs_pd.empty or "tags.mlflow.runName" not in runs_pd.columns:
            return None
        runs_pd = runs_pd[runs_pd["tags.mlflow.runName"].isin(wanted)]
        if runs_pd.empty:
            return None
        rename = {
            "tags.mlflow.runName": "run_name",
            "metrics.best_val_miou": "miou",
            "metrics.best_val_f1_macro": "f1_macro",
            "metrics.best_val_pixel_acc": "pixel_acc",
            "tags.code_version": "code_version",
            "tags.data_version": "data_version",
        }
        keep = [c for c in rename if c in runs_pd.columns]
        return pl.from_pandas(runs_pd[keep]).rename({c: rename[c] for c in keep})
    except Exception as exc:  # noqa: BLE001 - degraded mode in notebook
        logger.warning("lineage_read_failed", error=str(exc))
        return None


def pastis_class_names(num_classes: int = 18) -> dict[int, str]:
    """Training index map ``[0..17]`` -> PASTIS crop name.

    The dataset remaps the original PASTIS class ``cid`` (1..18) to the training
    index ``cid-1`` (0..17); here that offset is inverted to name each model
    index. Only covers ``semantic18``: for ``hcat6`` the caller must pass
    ``class_names=None`` to the helpers (generates ``grupo_c``).

    Args:
        num_classes: Must be 18 (semantic18).

    Returns:
        Dict ``{0: 'Meadow', 1: 'Soft winter wheat', ...}``.

    Raises:
        ValueError: if ``num_classes != 18`` (guard against misuse in hcat6).
    """
    if num_classes != 18:
        raise ValueError(
            f"pastis_class_names only covers semantic18 (18 classes), received "
            f"{num_classes}. For hcat6 pass class_names=None to the helpers."
        )
    from ml.features.phenology_class_prototypes import load_class_names

    orig = load_class_names()
    return {c: orig.get(c + 1, f"clase_{c}") for c in range(18)}
