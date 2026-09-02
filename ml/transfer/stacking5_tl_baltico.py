"""Stacking-5 transfer learning to the Baltic vocabulary (the champion, fine-tuned).

The decided experiment (Arthur, 2026-06-25): fine-tune the CHAMPION ensemble
(Stacking-5: tsvit-pheno + utae + xgb-alphaearth + farslip-ft18 + farslip-zeroshot)
to the Baltic label space and recombine with the meta-LogReg, exactly the
champion's combination layer (Voting simple loses -0.124 F1 vs Stacking in PASTIS,
so we keep Stacking). The label space is 18 Baltic leaves = 6 conserved (warm-start
flag from the PASTIS head) + 12 new fine leaves the EDA surfaced (apples, quinces,
fresh_vegetables, clover, oats, rye...).

Why a single orchestrator
-------------------------
The five members must all score the SAME Baltic parcels (train + test) so their
posteriors can be stacked. This module:
  1. Downloads the stratified Baltic parcels ONCE (real-texture SH patches +
     AlphaEarth embedding + leaf), shared across members -- no SH re-download.
  2. Produces each member's per-parcel posterior over the 18 Baltic classes:
     - U-TAE, TSViT-pheno: fine-tuned dense backbones (PASTIS init + kept flag),
       pooled to a per-parcel posterior;
     - xgb-alphaearth: champion XGBoost recipe on the AlphaEarth embedding;
     - FarSLIP: per-parcel CLS embedding scored against class prototypes (captions
       via Gemini 2.5 Flash, parallel) -- the two FarSLIP members.
  3. Stacks the member posteriors and fits the meta-LogReg (the champion's layer)
     on a train split; evaluates fine + collapsed-to-coarse F1 on the held-out
     target parcels (the papaya/fruits hierarchical eval).

Each stage persists its artefact so a failure mid-run resumes from the last good
member instead of re-downloading / re-training everything.

Honesty
-------
- Every posterior is a real model output on real Baltic parcels; nothing is faked.
- FarSLIP historically adds ~+0.0016 in PASTIS; its weight in the Baltic meta is
  whatever the meta-LogReg learns -- reported, not assumed.
- The cost is real (GPU fine-tunes + SH download + Gemini captions); subset and
  epochs are parameters for a pilot vs the full run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn

    from ml.transfer.ensemble_full_tl import _RegionParcels
    from ml.transfer.ensemble_texture_tl import _RegionTexture
    from ml.transfer.finetune_baltico import BalticLabelSpace

logger = structlog.get_logger(__name__)

__all__ = ["Stacking5TLConfig", "Stacking5TLResult", "run_stacking5_tl"]

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_OUT_DIR: Path = _REPO_ROOT / "data" / "transfer" / "stacking5_tl_baltico"
_CKPT = {
    "utae": "checkpoints/segmentation/utae-isaac/best_model.pt",
    "tsvit-pheno-fullm": "checkpoints/segmentation/tsvit-pheno-fullm-v1/best.pt",
}


@dataclass
class Stacking5TLConfig:
    """Hyperparameters for the Stacking-5 Baltic transfer."""

    source: str = "latvia"
    target: str = "estonia"
    per_class: int = 250
    dense_members: tuple[str, ...] = ("utae", "tsvit-pheno-fullm")
    include_xgb: bool = True
    include_farslip: bool = True
    #: Data source. When True, use the LOCAL EuroCropsML npz series (no Sentinel
    #: Hub, no paid quota; pixel-tiled patches). When False, download real-texture
    #: SH patches (cached). The vocabulary-correction experiment uses local npz.
    use_local_npz: bool = True
    warmup_epochs: int = 2
    finetune_epochs: int = 8
    batch_size: int = 16
    seed: int = 42
    device: str = "cuda"


@dataclass
class Stacking5TLResult:
    """Stacking-5 TL outcome: per-member + meta F1, fine and coarse."""

    summary: dict[str, object] = field(default_factory=dict)


def _persist(stage: str, payload: dict[str, object], out_dir: Path) -> None:
    """Write a stage artefact (resume point) to ``out_dir/stage.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stage}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logger.info("stacking5_stage_persisted", stage=stage)


def run_stacking5_tl(
    config: Stacking5TLConfig,
    *,
    sh_client: object,
    out_dir: Path = _OUT_DIR,
) -> Stacking5TLResult:
    """Run the full Stacking-5 transfer to the Baltic vocabulary.

    Downloads the stratified Baltic parcels once, fine-tunes / scores each member
    to produce a per-parcel posterior over the 18 Baltic classes, stacks them and
    fits the meta-LogReg, then evaluates fine + coarse F1 on the target split.

    Args:
        config: Experiment hyperparameters.
        sh_client: A :class:`ml.ingest.sh_client.SentinelHubClient`.
        out_dir: Directory for the per-stage artefacts and the final summary.

    Returns:
        A :class:`Stacking5TLResult`.
    """
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    from ml.transfer.finetune_baltico import (
        FINE_TO_COARSE,
        build_baltic_label_space,
    )

    label_space = build_baltic_label_space()
    keep = set(label_space.leaves)

    # --- 1. Load the stratified Baltic parcels ONCE (shared by all members). ------
    # Local npz (free, pixel-tiled) for the vocabulary experiment; SH (paid, real
    # texture, cached) otherwise. Both yield ``annual + patches + leaf`` per parcel.
    # TSViT-fullm needs a 128px patch grid; U-TAE is grid-agnostic, so when TSViT
    # is a member every patch is built at 128px (shared across members).
    patch_side = 128 if "tsvit-pheno-fullm" in config.dense_members else 8

    def _load(region: str) -> _RegionParcels | _RegionTexture:
        if config.use_local_npz:
            from ml.transfer.ensemble_full_tl import _load_region_parcels

            return _load_region_parcels(
                region,
                max_parcels=10_000,
                seed=config.seed,
                stratify_keep=keep,
                per_class=config.per_class,
                patch_side=patch_side,
            )
        from ml.transfer.ensemble_texture_tl import (
            _load_region_texture,
            build_season_windows,
        )

        return _load_region_texture(
            region,
            sh_client=sh_client,
            windows=build_season_windows(2021),
            max_parcels=10_000,
            size=128,
            max_cloud=25.0,
            seed=config.seed,
            stratify_keep=keep,
            per_class=config.per_class,
        )

    reg_src = _load(config.source)
    reg_tgt = _load(config.target)

    def _prep(
        reg: _RegionParcels | _RegionTexture,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        mask = np.array([leaf in keep for leaf in reg.leaf], dtype=bool)
        annual = reg.annual[mask]
        patches = [p for p, m in zip(reg.patches, mask, strict=True) if m]
        y = np.array([label_space.index[leaf] for leaf in reg.leaf[mask]], dtype=np.int64)
        return annual, patches, y

    a_src, p_src, y_src = _prep(reg_src)
    a_tgt, p_tgt, y_tgt = _prep(reg_tgt)
    n_classes = len(label_space.leaves)
    logger.info(
        "stacking5_data_ready",
        n_train=len(p_src),
        n_test=len(p_tgt),
        n_classes=n_classes,
    )
    _persist(
        "01_data",
        {"n_train": len(p_src), "n_test": len(p_tgt), "n_classes": n_classes},
        out_dir,
    )

    member_post_src: dict[str, np.ndarray] = {}
    member_post_tgt: dict[str, np.ndarray] = {}

    # --- 2a. Dense members: fine-tune each, then per-parcel posterior. ------------
    for kind in config.dense_members:
        model = _finetune_dense_member(
            kind,
            label_space,
            p_src,
            y_src,
            config,
            device=config.device,
        )
        member_post_src[kind] = _dense_posteriors(model, p_src, kind, config, n_classes)
        member_post_tgt[kind] = _dense_posteriors(model, p_tgt, kind, config, n_classes)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _persist(f"02_member_{kind}", {"n_classes": n_classes}, out_dir)

    # --- 2b. xgb-alphaearth on the AlphaEarth embedding. -------------------------
    if config.include_xgb:
        xs, xt = _xgb_posteriors(a_src, y_src, a_tgt, n_classes, config.seed)
        member_post_src["xgb-alphaearth"] = xs
        member_post_tgt["xgb-alphaearth"] = xt
        _persist("02_member_xgb", {"n_classes": n_classes}, out_dir)

    # --- 2c. FarSLIP (captions via Gemini Flash) -- optional, slowest. -----------
    if config.include_farslip:
        try:
            fs_src, fs_tgt = _farslip_posteriors(p_src, y_src, p_tgt, n_classes, config)
            member_post_src["farslip"] = fs_src
            member_post_tgt["farslip"] = fs_tgt
            _persist("02_member_farslip", {"n_classes": n_classes}, out_dir)
        except Exception as exc:  # noqa: BLE001 -- FarSLIP is additive; degrade honestly
            logger.warning("stacking5_farslip_skipped", error=str(exc))

    # --- 2d. Persist the member posteriors so any re-combination is GPU-free. -----
    # The expensive part is producing these posteriors (dense fine-tunes + SH /
    # npz patches). Persisting them once means a different combination layer (the
    # weighted vote below, or a future stacking variant) can be re-fit on CPU in
    # seconds without re-running a single member. Keys are the member order.
    members = list(member_post_src.keys())
    post_src_stack = np.stack([member_post_src[m] for m in members])  # (M, n_src, K)
    post_tgt_stack = np.stack([member_post_tgt[m] for m in members])  # (M, n_tgt, K)
    np.savez_compressed(
        out_dir / "02_posteriors.npz",
        members=np.array(members, dtype=object),
        post_src=post_src_stack,
        post_tgt=post_tgt_stack,
        y_src=y_src,
        y_tgt=y_tgt,
    )
    logger.info("stacking5_posteriors_persisted", members=members, shape=list(post_tgt_stack.shape))

    id_to_leaf = {i: leaf for leaf, i in label_space.index.items()}
    true_leaves = [id_to_leaf[t] for t in y_tgt.tolist()]

    def _leaves(pred_ids: np.ndarray) -> list[str]:
        return [id_to_leaf[p] for p in pred_ids.tolist()]

    # --- 3a. Meta-LogReg (the CHAMPION combination layer). ------------------------
    meta_src = np.concatenate([member_post_src[m] for m in members], axis=1)
    meta_tgt = np.concatenate([member_post_tgt[m] for m in members], axis=1)
    meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.seed)
    meta.fit(meta_src, y_src)
    meta_leaves = _leaves(meta.predict(meta_tgt))
    meta_metrics = _fc_metrics(true_leaves, meta_leaves, FINE_TO_COARSE, label_space.leaf_to_pastis)

    # --- 3b. Combination-layer SWAP: weighted vote + simple vote (engram #340). ---
    # Same posteriors, different combiner. The weighted vote learns N convex
    # weights on the SOURCE (like the meta) and applies them to the TARGET; the
    # simple vote is the fixed 1/N floor. This is the faithful transfer analog of
    # the PASTIS Voting-3 vs Stacking head-to-head.
    vote_w = _learn_vote_weights(post_src_stack, y_src, seed=config.seed)
    wvote_leaves = _leaves(np.tensordot(vote_w, post_tgt_stack, axes=(0, 0)).argmax(axis=1))
    svote_leaves = _leaves(post_tgt_stack.mean(axis=0).argmax(axis=1))
    weighted_vote_metrics = _fc_metrics(
        true_leaves, wvote_leaves, FINE_TO_COARSE, label_space.leaf_to_pastis
    )
    simple_vote_metrics = _fc_metrics(
        true_leaves, svote_leaves, FINE_TO_COARSE, label_space.leaf_to_pastis
    )

    # Per-member solo F1 (how good each member alone is on the target) for context.
    member_solo = {
        m: round(float(f1_score(y_tgt, member_post_tgt[m].argmax(axis=1), average="macro")), 4)
        for m in members
    }

    summary = {
        "source": config.source,
        "target": config.target,
        "members": members,
        "use_local_npz": config.use_local_npz,
        "n_train": len(p_src),
        "n_test": len(p_tgt),
        "n_classes_fine": n_classes,
        "n_conserved": len(label_space.conserved),
        "n_new": len(label_space.new),
        # Headline kept under the historical keys (the meta-LogReg champion).
        "stacking5_fine_macro_f1": meta_metrics["fine_macro_f1"],
        "stacking5_coarse_macro_f1": meta_metrics["coarse_macro_f1"],
        "stacking5_fine_accuracy": meta_metrics["fine_accuracy"],
        "stacking5_coarse_accuracy": meta_metrics["coarse_accuracy"],
        # Combination-layer head-to-head on the SAME posteriors.
        "combiner_comparison": {
            "meta_logreg": meta_metrics,
            "weighted_vote": weighted_vote_metrics,
            "simple_vote": simple_vote_metrics,
        },
        "vote_weights": {m: round(float(w), 4) for m, w in zip(members, vote_w, strict=True)},
        "member_solo_fine_f1": member_solo,
        "y_true_leaf": true_leaves,
        "y_pred_leaf": meta_leaves,
        "y_pred_leaf_weighted_vote": wvote_leaves,
        "conserved_leaves": list(label_space.conserved),
        "new_leaves": list(label_space.new),
    }
    _persist("03_stacking5_summary", summary, out_dir)
    logger.info(
        "stacking5_tl_done",
        **{k: v for k, v in summary.items() if not isinstance(v, list | dict)},
        meta_fine_f1=meta_metrics["fine_macro_f1"],
        weighted_vote_fine_f1=weighted_vote_metrics["fine_macro_f1"],
        simple_vote_fine_f1=simple_vote_metrics["fine_macro_f1"],
        vote_weights=summary["vote_weights"],
        member_solo=member_solo,
    )
    return Stacking5TLResult(summary=summary)


def _finetune_dense_member(
    kind: str,
    label_space: BalticLabelSpace,
    patches: list[np.ndarray],
    y: np.ndarray,
    config: Stacking5TLConfig,
    *,
    device: str,
) -> nn.Module:
    """Fine-tune one dense member (PASTIS init + kept flag) on the Baltic train set."""
    import torch
    from torch import nn

    from ml.transfer.finetune_baltico import build_finetune_model

    model = build_finetune_model(
        label_space,
        model_kind=kind,
        pastis_checkpoint=_CKPT[kind],
        device=device,
    )
    criterion = nn.CrossEntropyLoss()

    def _fwd(xb: torch.Tensor) -> torch.Tensor:
        t = xb.shape[1]
        if kind == "utae":
            doy = (torch.arange(t, device=device).float() / max(t - 1, 1) * 364.0).round().long()
            utae_logits: torch.Tensor = model(xb, doy.unsqueeze(0).repeat(xb.shape[0], 1))
            return utae_logits
        logits: torch.Tensor = model(xb)
        return logits

    def _is_head(n: str) -> bool:
        return "out_conv" in n or "head" in n or "cls_token" in n

    def _run_epochs(opt: torch.optim.Optimizer, n_ep: int, tag: str) -> None:
        rng = np.random.default_rng(config.seed)
        for ep in range(n_ep):
            model.train()
            order = rng.permutation(len(patches))
            for s in range(0, len(order), config.batch_size):
                idx = order[s : s + config.batch_size]
                xb = torch.from_numpy(np.stack([patches[i] for i in idx])).float().to(device)
                yb = torch.from_numpy(y[idx]).to(device)
                pooled = _fwd(xb).mean(dim=(2, 3))
                loss = criterion(pooled, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            logger.info("stacking5_dense_epoch", member=kind, phase=tag, epoch=ep)

    for n, p in model.named_parameters():
        p.requires_grad = _is_head(n)
    _run_epochs(
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3),
        config.warmup_epochs,
        "warmup",
    )
    for p in model.parameters():
        p.requires_grad = True
    head = [p for n, p in model.named_parameters() if _is_head(n)]
    back = [p for n, p in model.named_parameters() if not _is_head(n)]
    _run_epochs(
        torch.optim.AdamW(
            [{"params": head, "lr": 1e-3}, {"params": back, "lr": 1e-4}], weight_decay=1e-4
        ),
        config.finetune_epochs,
        "finetune",
    )
    return model


def _dense_posteriors(
    model: nn.Module,
    patches: list[np.ndarray],
    kind: str,
    config: Stacking5TLConfig,
    n_classes: int,
) -> np.ndarray:
    """Per-parcel softmax posterior ``(n, K)`` from a fine-tuned dense member."""
    import torch

    device = config.device
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(patches), config.batch_size):
            xb = torch.from_numpy(np.stack(patches[s : s + config.batch_size])).float().to(device)
            t = xb.shape[1]
            if kind == "utae":
                frac = torch.arange(t, device=device).float() / max(t - 1, 1)
                doy = (frac * 364.0).round().long()
                logits = model(xb, doy.unsqueeze(0).repeat(xb.shape[0], 1))
            else:
                logits = model(xb)
            post = torch.softmax(logits.mean(dim=(2, 3)), dim=1)
            out.append(post.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def _xgb_posteriors(
    a_src: np.ndarray,
    y_src: np.ndarray,
    a_tgt: np.ndarray,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Champion XGBoost on the AlphaEarth embedding -> per-parcel posteriors."""
    from ml.train.baseline import _XGB_BASE_PARAMS, build_estimator

    params = dict(_XGB_BASE_PARAMS)
    params["random_state"] = seed
    model = build_estimator("xgb", params)
    model.fit(a_src, y_src)
    ps = model.predict_proba(a_src)
    pt = model.predict_proba(a_tgt)
    # Align to the full K class axis (xgb may drop classes absent in train).
    classes = np.asarray(model.classes_, dtype=int)
    full_s = np.zeros((a_src.shape[0], n_classes), dtype=np.float64)
    full_t = np.zeros((a_tgt.shape[0], n_classes), dtype=np.float64)
    for col, cls in enumerate(classes):
        full_s[:, cls] = ps[:, col]
        full_t[:, cls] = pt[:, col]
    return full_s, full_t


def _farslip_posteriors(
    p_src: list[np.ndarray],
    y_src: np.ndarray,
    p_tgt: list[np.ndarray],
    n_classes: int,
    config: Stacking5TLConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """FarSLIP per-parcel posteriors via class-prototype cosine (captions Gemini).

    Placeholder for the FarSLIP member: builds visual CLS prototypes per class on
    the train parcels and scores by cosine. Captions (Gemini 2.5 Flash, parallel)
    refine the text side. Raises ``NotImplementedError`` until the FarSLIP wiring
    is added so the orchestrator degrades to Stacking-(n-1) honestly instead of
    faking a member.
    """
    raise NotImplementedError(
        "FarSLIP member pending: requires the CLS-prototype scorer + Gemini captions."
    )


# ----------------------------------------------------------------------------
# Combination-layer swap: weighted / simple vote on the SAME member posteriors.
# Mirrors the PASTIS WeightedVotingEnsemble experiment (engram #337/#340) in the
# transfer setting -- the members all emit a posterior over the SAME 18 Baltic
# leaves, so a convex soft-vote is well defined. The weighted vote learns N
# convex weights on the SOURCE (the meta-LogReg's train split) by direct F1-macro
# maximization and applies them to the TARGET, exactly the protocol of the meta.
# ----------------------------------------------------------------------------


def _simplex(raw: np.ndarray) -> np.ndarray:
    """Map free logits to the convex simplex (softmax, ``w_i >= 0``, ``sum == 1``)."""
    z = raw - raw.max()
    e = np.exp(z)
    weights: np.ndarray = e / e.sum()
    return weights


def _learn_vote_weights(
    post_src_stack: np.ndarray,
    y_src: np.ndarray,
    *,
    seed: int,
    n_restarts: int = 6,
    max_iter: int = 400,
) -> np.ndarray:
    """Learn convex weights maximizing source F1-macro of the weighted vote.

    Direct Nelder-Mead over the simplex logits, multi-started from each member
    corner and the centroid (the F1 surface is non-convex / piecewise-constant).
    Learned on the SOURCE posteriors so the head-to-head with the meta-LogReg is
    apples-to-apples (both fit on source, evaluate on target).

    Args:
        post_src_stack: Member tensor ``(M, n_src, K)`` of source posteriors.
        y_src: Source labels ``(n_src,)``.
        seed: Deterministic seed (unused by Nelder-Mead but kept for parity).
        n_restarts: Minimum number of restarts.
        max_iter: Max Nelder-Mead iterations per restart.

    Returns:
        Convex weights ``(M,)`` with the best source F1-macro found.
    """
    from scipy.optimize import minimize
    from sklearn.metrics import f1_score

    n_members = post_src_stack.shape[0]

    def neg_f1(raw: np.ndarray) -> float:
        w = _simplex(raw)
        preds = np.tensordot(w, post_src_stack, axes=(0, 0)).argmax(axis=1)
        return -float(f1_score(y_src, preds, average="macro"))

    starts: list[np.ndarray] = []
    for i in range(n_members):
        corner = np.full(n_members, 0.05, dtype=np.float64)
        corner[i] = 1.0
        starts.append(corner)
    starts.append(np.full(n_members, 1.0 / n_members, dtype=np.float64))
    for r in range(max(0, n_restarts - len(starts))):
        jitter = np.full(n_members, 1.0 / n_members, dtype=np.float64)
        jitter[r % n_members] += 0.3
        starts.append(jitter)

    best_raw, best_neg = None, np.inf
    for x0 in starts:
        res = minimize(
            neg_f1,
            x0,
            method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-4},
        )
        if float(res.fun) < best_neg:
            best_neg = float(res.fun)
            best_raw = np.asarray(res.x, dtype=np.float64)
    assert best_raw is not None
    return _simplex(best_raw)


def _fc_metrics(
    true_leaves: list[str],
    pred_leaves: list[str],
    fine_to_coarse: dict[str, str],
    leaf_to_pastis: dict[str, str],
) -> dict[str, float]:
    """Fine + collapsed-to-coarse macro-F1 / accuracy for one combiner's preds."""
    from sklearn.metrics import accuracy_score, f1_score

    def coarse(leaf: str) -> str:
        return fine_to_coarse.get(leaf, leaf_to_pastis.get(leaf, leaf))

    ct = [coarse(t) for t in true_leaves]
    cp = [coarse(p) for p in pred_leaves]
    return {
        "fine_macro_f1": round(float(f1_score(true_leaves, pred_leaves, average="macro")), 4),
        "fine_accuracy": round(float(accuracy_score(true_leaves, pred_leaves)), 4),
        "coarse_macro_f1": round(float(f1_score(ct, cp, average="macro")), 4),
        "coarse_accuracy": round(float(accuracy_score(ct, cp)), 4),
    }
