"""E1 -- Homogeneous pixel-level Voting ensemble (US-040, EPIC 6).

The first of the four rubric ensembles: a homogeneous soft-vote over the three
dense temporal segmenters that share the per-pixel ``(18, 128, 128)`` output
space. The vote is the arithmetic mean of the POST-softmax probabilities of the
members at pixel granularity; the hard prediction is the ``argmax`` over the
averaged class axis. The ensemble NEVER averages logits -- by design it only ever
sees the US-031 OOF dump, which is already post-softmax, and every member matrix
is re-checked with :meth:`EnsembleModel.validate_probs` before being averaged
(anti-leakage R-LEAK, plan Section 9).

Member selection (R-VOTE, plan Section "Correcciones" point 3). The plan v8 lists
"TSViT base" as a third dense voter, but US-031 only dumped ``tsvit-pheno``,
``utae``, ``unet``, ``deeplabv3plus``, ``segformer`` and ``anysat`` -- TSViT base
has no OOF parquet. The default homogeneous terna is therefore
``("tsvit-pheno", "utae", "unet")`` (three dense segmenters over PASTIS-R). The
members are a substitutable constructor argument: pass ``members=(...)`` to swap
``unet`` for ``deeplabv3plus`` (or for TSViT base once its OOF is dumped).

Ground truth (anti-leakage). The semantic target is NOT stored inside the OOF
parquet (the dump discards it), so :meth:`VotingEnsemble.load_ground_truth`
re-reads the PASTIS-R semantic18 fold-5 labels directly from
:class:`ml.data.pastis_seg_dataset.PASTISSegmentationDataset` and aligns them to
the exact ``patch_ids`` order used by :meth:`VotingEnsemble.predict_proba`, so the
dense metrics in :meth:`VotingEnsemble.evaluate` are computed pixel-for-pixel on
fold-5 only.

Project conventions: ``polars`` (never pandas) for tabular access, ``numpy`` only
at the array boundary, ``structlog`` for logging, type hints and Google-style
docstrings everywhere; visible prose is Spanish, code identifiers English.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import structlog

from ml.ensemble.base import EnsembleModel

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_VOTING_MEMBERS",
    "VotingEnsemble",
]

#: Default homogeneous dense terna (R-VOTE: TSViT base is not dumped in US-031,
#: so the third voter defaults to U-Net; it is a substitutable argument).
DEFAULT_VOTING_MEMBERS: tuple[str, ...] = ("tsvit-pheno", "utae", "unet")

#: Number of agronomic classes in the harness 18-class space.
_NUM_CLASSES: int = 18

#: Class axis of a dense ``(18, H, W)`` softmax map.
_CLASS_AXIS: int = 0

#: Sum-to-1 tolerance for a member map BEFORE renormalization. The US-031 OOF
#: dump stores the softmax as float16 (manifest ``dtype="float16"``), whose
#: round-trip drift can reach ~3e-4 per pixel -- looser than the base's strict
#: 1e-4 default, but still far from logits (which are negative and never sum near
#: 1). Each member map is checked at this tolerance and then RENORMALIZED so the
#: averaged output passes the strict default of :meth:`EnsembleModel.validate_probs`.
_FLOAT16_SUM_TOL: float = 5e-3

#: Floor avoiding divide-by-zero when renormalizing a pixel's class vector.
_RENORM_EPS: float = 1e-12


class VotingEnsemble(EnsembleModel):
    """Homogeneous pixel-level soft-voting ensemble (E1).

    Averages the post-softmax ``(18, 128, 128)`` maps of the dense members at
    pixel granularity and predicts the per-pixel ``argmax`` of the average. It is
    parameter-free: :meth:`fit` is a no-op (there is nothing to learn -- the vote
    is a fixed arithmetic mean). Reports F1-macro / accuracy on fold-5 ONLY,
    enforced by the base :meth:`EnsembleModel.evaluate`.

    Attributes:
        members: Ordered dense base-learner names averaged at the pixel level.
        data_root: PASTIS-R root used to load the semantic18 fold-5 ground truth
            for :meth:`evaluate` (the OOF parquet does not store the target).
    """

    def __init__(
        self,
        members: Sequence[str] = DEFAULT_VOTING_MEMBERS,
        *,
        data_root: Path | str | None = None,
        **kw: object,
    ) -> None:
        """Initialize the homogeneous pixel voting ensemble.

        Args:
            members: Ordered dense member names whose OOF pixel softmax maps are
                averaged. Defaults to ``("tsvit-pheno", "utae", "unet")``
                (R-VOTE: TSViT base is not in the US-031 OOF dump). Provide a
                different terna (e.g. ``("tsvit-pheno", "utae", "deeplabv3plus")``)
                to substitute the third voter.
            data_root: PASTIS-R dataset root forwarded to the ground-truth loader
                in :meth:`evaluate`; ``None`` uses the dataset's default root.
            **kw: Forwarded to :class:`EnsembleModel` (``oof_dir``,
                ``random_state``).

        Raises:
            ValueError: if fewer than two members are given (a single member is
                not an ensemble).
        """
        super().__init__(**kw)  # type: ignore[arg-type]
        members_tuple = tuple(str(m) for m in members)
        if len(members_tuple) < 2:
            raise ValueError(
                f"VotingEnsemble needs at least 2 members, got {members_tuple!r}; "
                "a single member is not an ensemble."
            )
        self.members: tuple[str, ...] = members_tuple
        self.data_root = Path(data_root) if data_root is not None else None
        logger.debug("voting_init", members=self.members)

    # ------------------------------------------------------------------
    # fit: voting is parameter-free.
    # ------------------------------------------------------------------

    def fit(self, *args: object, **kwargs: object) -> VotingEnsemble:
        """No-op fit -- the soft-vote is a fixed arithmetic mean, nothing to learn.

        Present to satisfy the :class:`EnsembleModel` contract; voting has no
        trainable state, so it returns ``self`` immediately.

        Returns:
            ``self`` for chaining.
        """
        return self

    # ------------------------------------------------------------------
    # predict_proba: mean of post-softmax pixel maps over members.
    # ------------------------------------------------------------------

    def predict_proba(self, patch_ids: Sequence[str]) -> np.ndarray:
        """Average the members' post-softmax ``(18, 128, 128)`` maps per patch.

        Loads each member's pixel-space OOF parquet via
        :meth:`EnsembleModel.load_oof_members` (``space="pixel"``), validates that
        every per-patch map is post-softmax (sum-to-1 over the class axis -- this
        is the programmatic guard that the ensemble averages PROBABILITIES, never
        logits), and returns the arithmetic mean over the members aligned to the
        requested ``patch_ids`` order.

        Args:
            patch_ids: Ordered PASTIS-R patch ids to vote on. The output rows
                follow this exact order so the caller can align the ground truth
                (see :meth:`load_ground_truth`).

        Returns:
            A ``float64`` array. Shape ``(18, 128, 128)`` when a single patch id
            is given, else ``(N, 18, 128, 128)`` with ``N == len(patch_ids)``.
            Every per-patch map is post-softmax (sums to 1 over the class axis).

        Raises:
            ValueError: if ``patch_ids`` is empty, if a requested patch id is
                absent from a member's OOF, if member maps have inconsistent
                shapes, or if any loaded map is not post-softmax (i.e. logits).
            FileNotFoundError: if a member's OOF parquet is missing (run
                ``dvc pull ml/eval/oof``; do not regenerate).
        """
        ids = [str(p) for p in patch_ids]
        if not ids:
            raise ValueError("predict_proba needs at least one patch_id.")

        loaded = self.load_oof_members(self.members, space="pixel")
        per_member_maps = {
            member: self._patch_softmax_index(member, df) for member, df in loaded.items()
        }

        per_patch_means: list[np.ndarray] = [
            self._vote_one_patch(pid, per_member_maps) for pid in ids
        ]
        stacked = np.stack(per_patch_means, axis=0)
        logger.info(
            "voting_predict_proba",
            members=self.members,
            n_patches=len(ids),
            shape=tuple(stacked.shape),
        )
        if stacked.shape[0] == 1:
            return cast("np.ndarray", stacked[0])
        return stacked

    def _vote_one_patch(
        self,
        patch_id: str,
        per_member_maps: dict[str, dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Mean of the post-softmax maps of every member for one patch.

        Args:
            patch_id: PASTIS-R patch id to average across members.
            per_member_maps: ``{member: {patch_id: softmax (18, H, W)}}`` index.

        Returns:
            The averaged ``float64`` map ``(18, H, W)`` (post-softmax).

        Raises:
            ValueError: if the patch is missing from a member, if the member maps
                disagree in shape, or if a member map is not post-softmax.
        """
        maps: list[np.ndarray] = []
        ref_shape: tuple[int, ...] | None = None
        for member in self.members:
            member_index = per_member_maps[member]
            sm = member_index.get(patch_id)
            if sm is None:
                raise ValueError(
                    f"patch_id {patch_id!r} is absent from member {member!r} OOF; "
                    "predict_proba can only vote on patches present in every "
                    "member's fold-5 dump."
                )
            arr = np.asarray(sm, dtype=np.float64)
            if ref_shape is None:
                ref_shape = arr.shape
            elif arr.shape != ref_shape:
                raise ValueError(
                    f"member {member!r} softmax shape {arr.shape} != "
                    f"{ref_shape} for patch {patch_id!r}; members must share the "
                    "dense output space to vote at the pixel level."
                )
            # Anti-leakage: reject anything that is not post-softmax (logits).
            # The check uses a float16-storage tolerance (the OOF dump is
            # float16); true logits are negative and never sum near 1, so they
            # are still rejected. The map is then renormalized so float16 drift
            # does not propagate into the averaged output.
            self.validate_probs(
                arr,
                class_axis=_CLASS_AXIS,
                name=f"{member}:{patch_id}",
                tol=_FLOAT16_SUM_TOL,
            )
            maps.append(self._renormalize(arr))
        mean_map = np.mean(np.stack(maps, axis=0), axis=0)
        # The mean of sum-to-1 distributions is sum-to-1; re-validate (strict
        # default tolerance) to guarantee the contract for downstream callers.
        return self.validate_probs(mean_map, class_axis=_CLASS_AXIS, name="vote_mean")

    @staticmethod
    def _renormalize(softmax_18: np.ndarray) -> np.ndarray:
        """Renormalize a ``(18, H, W)`` map so each pixel's class axis sums to 1.

        Absorbs the float16 storage drift of the US-031 OOF dump (the raw maps
        sum to ~1 +/- 3e-4) so the averaged vote satisfies the strict sum-to-1
        contract. A no-op (up to float64 epsilon) for an already-exact softmax.

        Args:
            softmax_18: Post-softmax map ``(18, H, W)`` (float64).

        Returns:
            The renormalized ``float64`` map summing to 1 over the class axis.
        """
        denom: np.ndarray = softmax_18.sum(axis=_CLASS_AXIS, keepdims=True)
        renormalized: np.ndarray = softmax_18 / np.where(denom < _RENORM_EPS, 1.0, denom)
        return renormalized

    @staticmethod
    def _patch_softmax_index(member: str, df: object) -> dict[str, np.ndarray]:
        """Build a ``{patch_id: softmax}`` index from a member's pixel OOF frame.

        Args:
            member: Member name (for error messages).
            df: Pixel-space OOF DataFrame from
                :func:`ml.eval.oof.parquet_io.read_softmax_parquet` (columns
                ``patch_id`` + object ``softmax`` arrays).

        Returns:
            Mapping ``{patch_id (str): softmax (18, H, W) numpy}`` skipping rows
            whose ``softmax`` is ``None`` (missing-checkpoint sentinels).

        Raises:
            ValueError: if the frame lacks the ``patch_id`` or ``softmax`` column.
        """
        import polars as pl

        assert isinstance(df, pl.DataFrame)
        if "patch_id" not in df.columns or "softmax" not in df.columns:
            raise ValueError(
                f"member {member!r} pixel OOF frame must carry 'patch_id' and "
                f"'softmax' columns; got {df.columns}."
            )
        index: dict[str, np.ndarray] = {}
        for pid, sm in zip(df["patch_id"].to_list(), df["softmax"].to_list(), strict=True):
            if sm is None:
                continue
            index[str(pid)] = np.asarray(sm)
        return index

    def predict(self, patch_ids: Sequence[str]) -> np.ndarray:
        """Hard per-pixel labels: ``argmax`` over the averaged class axis.

        Args:
            patch_ids: Ordered PASTIS-R patch ids (see :meth:`predict_proba`).

        Returns:
            An ``int64`` array of class ids: ``(128, 128)`` for a single patch id
            or ``(N, 128, 128)`` otherwise.
        """
        proba = self.predict_proba(patch_ids)
        class_axis = 0 if proba.ndim == 3 else 1
        labels: np.ndarray = proba.argmax(axis=class_axis).astype(np.int64)
        return labels

    # ------------------------------------------------------------------
    # Ground truth: PASTIS-R semantic18 fold-5 (not stored in the OOF).
    # ------------------------------------------------------------------

    def load_ground_truth(self, patch_ids: Sequence[str]) -> np.ndarray:
        """Load the PASTIS-R semantic18 fold-5 labels aligned to ``patch_ids``.

        The OOF parquet discards the semantic target, so the ground truth is read
        directly from :class:`ml.data.pastis_seg_dataset.PASTISSegmentationDataset`
        (``folds=(5,)``, ``target="semantic18"``, ``ignore_index=255``) -- the
        SAME configuration the US-031 dump iterated, so labels and predictions
        share the ``[0..17]`` plus ``255`` ignore space. The patches are indexed
        by id and stacked in the exact ``patch_ids`` order, so each ground-truth
        map lines up pixel-for-pixel with :meth:`predict_proba`.

        Args:
            patch_ids: Ordered PASTIS-R patch ids whose labels to load (same order
                passed to :meth:`predict_proba`).

        Returns:
            An ``int64`` array of labels ``(N, 128, 128)`` (or ``(128, 128)`` for
            a single id), values in ``[0..17]`` plus ``255`` (ignore).

        Raises:
            ValueError: if ``patch_ids`` is empty or a requested id is not in the
                fold-5 split.
            FileNotFoundError: if the PASTIS-R dataset root is unavailable.
        """
        from ml.data.pastis_seg_dataset import PASTISSegmentationDataset

        ids = [str(p) for p in patch_ids]
        if not ids:
            raise ValueError("load_ground_truth needs at least one patch_id.")

        ds_kwargs: dict[str, object] = {
            "folds": (self.HELD_OUT_FOLD,),
            "collapse_time": "median",
            "target": "semantic18",
            "ignore_index": 255,
        }
        if self.data_root is not None:
            ds_kwargs["root"] = self.data_root
        dataset = PASTISSegmentationDataset(**ds_kwargs)  # type: ignore[arg-type]
        pos_of = {pid: i for i, pid in enumerate(dataset.patch_ids)}

        labels: list[np.ndarray] = []
        for pid in ids:
            pos = pos_of.get(pid)
            if pos is None:
                raise ValueError(
                    f"patch_id {pid!r} is not in the fold-{self.HELD_OUT_FOLD} "
                    "split; ground truth can only be loaded for held-out patches."
                )
            _x, y = dataset[pos]
            labels.append(np.asarray(y, dtype=np.int64))
        stacked = np.stack(labels, axis=0)
        logger.info(
            "voting_ground_truth_loaded",
            n_patches=len(ids),
            fold=self.HELD_OUT_FOLD,
            shape=tuple(stacked.shape),
        )
        if stacked.shape[0] == 1:
            return cast("np.ndarray", stacked[0])
        return stacked

    # ------------------------------------------------------------------
    # evaluate: dense F1-macro / accuracy on fold-5.
    # ------------------------------------------------------------------

    def evaluate_patches(
        self, patch_ids: Sequence[str], *, fold: int = EnsembleModel.HELD_OUT_FOLD
    ) -> dict[str, float]:
        """Vote on ``patch_ids`` and score against the real fold-5 ground truth.

        Convenience wrapper that loads the PASTIS-R semantic18 fold-5 labels via
        :meth:`load_ground_truth`, runs the soft-vote via :meth:`predict`, and
        delegates the metric to :meth:`EnsembleModel.evaluate` (which rejects any
        fold other than 5 -- anti-leakage). The dense maps are flattened to
        per-pixel label vectors by ``compute_metrics`` (255 is the ignore index).

        Args:
            patch_ids: Ordered PASTIS-R patch ids to vote on and score.
            fold: Must be the held-out fold 5; any other value raises in the base.

        Returns:
            ``{"f1_macro": float, "accuracy": float}`` over the fold-5 pixels.

        Raises:
            ValueError: if ``fold != 5`` (delegated to the base ``evaluate``).
        """
        y_true = self.load_ground_truth(patch_ids)
        y_pred = self.predict(patch_ids)
        return self.evaluate(
            y_true=np.asarray(y_true).reshape(-1),
            y_pred=np.asarray(y_pred).reshape(-1),
            fold=fold,
        )
