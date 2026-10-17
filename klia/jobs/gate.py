"""klia/jobs/gate.py — Model promotion gate.

Trains a candidate bundle on new rows (train split), evaluates both the
candidate and the existing incumbent on a chronologically held-out
validation split, and promotes only if the candidate strictly improves
quality without regressing either metric beyond a configurable tolerance.

Scoring (mirrors tune.py):
    quality_score = 0.4 × ROC-AUC + 0.6 × F1

This module has NO store or MLflow dependencies.  It is a pure evaluation
function called by update.py inside its store.lock() context.

Lifecycle (called from update.py):
    1. update.py clones incumbent bytes → eval_incumbent, candidate
    2. update.py calls apply_rows(candidate, train_ok, cfg)    — trains candidate
    3. update.py calls advance_state_only(eval_incumbent, train_ok) — advances
       incumbent FeatureState to the same feature-horizon as candidate,
       WITHOUT training the model
    4. update.py calls run_gate(eval_incumbent, candidate, val_ok, cfg)
    5. update.py commits the winner; watermark advances regardless of outcome
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger("klia.gate")

# Mirrors tune.py weights exactly.
_AUC_W = 0.4
_F1_W  = 0.6


# ─────────────────────────── public helpers ───────────────────────────────────

def quality_score(auc: float, f1: float) -> float:
    """0.4·AUC + 0.6·F1 — same convention as tune.py."""
    return _AUC_W * auc + _F1_W * f1


def advance_state_only(bundle, ok) -> None:
    """Advance bundle.state through *ok* rows WITHOUT training the model.

    This brings eval_incumbent's FeatureState to the same feature-horizon
    as candidate (which saw these rows via apply_rows), so both models
    predict from identical context on the validation set.

    bundle.state.features(rec) is read-only — it reads running statistics
    but never mutates them (that is exclusively done by state.update()).
    """
    for rec in ok.to_dict("records"):
        label = int(rec["is_delayed"])
        bundle.state.features(rec)       # read-only; keeps state consistent
        bundle.state.update(rec, label)  # advance running statistics


# ────────────────────────── result types ─────────────────────────────────────

@dataclass
class GateMetrics:
    auc:       float
    f1:        float
    score:     float
    threshold: float
    val_rows:  int


@dataclass
class GateResult:
    promoted:   bool
    reason:     str
    incumbent:  Optional[GateMetrics] = None
    candidate:  Optional[GateMetrics] = None
    score_delta: Optional[float]      = None
    val_rows:   int                   = 0

    def as_dict(self) -> dict:
        """Flat dict for MLflow / run-log; all keys prefixed `gate_`."""
        d: dict = {
            "gate_promoted": int(self.promoted),
            "gate_reason":   self.reason,
            "gate_val_rows": self.val_rows,
        }
        if self.incumbent:
            d.update({
                "gate_inc_auc":   self.incumbent.auc,
                "gate_inc_f1":    self.incumbent.f1,
                "gate_inc_score": self.incumbent.score,
            })
        if self.candidate:
            d.update({
                "gate_cand_auc":   self.candidate.auc,
                "gate_cand_f1":    self.candidate.f1,
                "gate_cand_score": self.candidate.score,
            })
        if self.score_delta is not None:
            d["gate_score_delta"] = round(self.score_delta, 4)
        return d


# ─────────────────────── internal evaluation ─────────────────────────────────

def _extract_features(bundle, val_ok):
    """(X, y) from val_ok WITHOUT updating bundle.state.

    state.features() is explicitly read-only (no side effects).
    state.update() is never called here.
    """
    X: list = []
    y: list[int] = []
    for rec in val_ok.to_dict("records"):
        feat = bundle.state.features(rec)
        if bundle.selector.is_fitted:
            feat = bundle.selector.filter(feat)
        X.append(feat)
        y.append(int(rec["is_delayed"]))
    return X, y


def _eval(bundle, val_ok) -> Optional[GateMetrics]:
    """Compute AUC + F1 for *bundle* on *val_ok*.  Returns None on invalid data."""
    from sklearn.metrics import f1_score, roc_auc_score  # local: stays out of serve deps

    X, y = _extract_features(bundle, val_ok)
    n    = len(y)

    if n == 0:
        log.warning("gate._eval: empty validation set")
        return None

    y_arr = np.array(y, dtype=int)
    if y_arr.min() == y_arr.max():
        log.warning("gate._eval: single-class validation set (all label=%d)", int(y_arr[0]))
        return None

    base = float(getattr(bundle.state, "base_rate", 0.3))
    raw  = bundle.model.predict_proba(X, fallback=base)
    p    = np.atleast_1d(np.asarray(raw, dtype=float))

    if p.shape[0] != n:
        log.warning("gate._eval: predict_proba length mismatch (%d vs %d)", p.shape[0], n)
        return None

    if not np.all(np.isfinite(p)):
        log.warning("gate._eval: non-finite probabilities")
        return None

    t   = float(bundle.threshold)
    auc = float(roc_auc_score(y_arr, p))
    f1  = float(f1_score(y_arr, (p >= t).astype(int), zero_division=0))

    if not (math.isfinite(auc) and math.isfinite(f1)):
        log.warning("gate._eval: non-finite AUC=%.4f or F1=%.4f", auc, f1)
        return None

    sc = quality_score(auc, f1)
    log.debug("gate._eval: n=%d  auc=%.4f  f1=%.4f  score=%.4f  t=%.2f", n, auc, f1, sc, t)
    return GateMetrics(
        auc=round(auc, 4),
        f1=round(f1,  4),
        score=round(sc, 4),
        threshold=round(t, 4),
        val_rows=n,
    )


# ──────────────────────────── gate ────────────────────────────────────────────

def run_gate(
    eval_incumbent,   # Bundle whose state was advanced through train_rows
    candidate,        # Bundle trained on train_rows via apply_rows()
    val_ok,           # pd.DataFrame — chronological hold-out, not seen by candidate
    cfg: dict,
) -> GateResult:
    """Evaluate candidate vs eval_incumbent on val_ok and return a GateResult.

    Promotion requires ALL of:
      * candidate quality_score > incumbent quality_score  (strictly)
      * AUC does not regress by more than metric_tolerance
      * F1  does not regress by more than metric_tolerance
      * valid metrics for both models on val_ok
    """
    gate_cfg = cfg.get("gate", {})
    min_val  = int(gate_cfg.get("min_val_rows", 200))
    tol      = float(gate_cfg.get("metric_tolerance", 0.0))
    warmup   = int(cfg.get("model", {}).get("warmup_rows", 500))
    n_val    = len(val_ok) if val_ok is not None else 0

    # ── guard: not enough validation rows ────────────────────────────────────
    if n_val < min_val:
        return GateResult(
            promoted=False,
            reason=f"insufficient_val_rows:{n_val}<{min_val}",
            val_rows=n_val,
        )

    # ── guard: incumbent still warming up ────────────────────────────────────
    inc_learned = getattr(eval_incumbent.model, "n_learned", warmup + 1)
    if inc_learned < warmup:
        return GateResult(
            promoted=False,
            reason=f"incumbent_warming_up:n_learned={inc_learned}<{warmup}",
            val_rows=n_val,
        )

    # ── evaluate ──────────────────────────────────────────────────────────────
    inc_m  = _eval(eval_incumbent, val_ok)
    cand_m = _eval(candidate,      val_ok)

    if inc_m is None:
        return GateResult(
            promoted=False,
            reason="invalid_val_data_incumbent",
            candidate=cand_m,
            val_rows=n_val,
        )
    if cand_m is None:
        return GateResult(
            promoted=False,
            reason="invalid_val_data_candidate",
            incumbent=inc_m,
            val_rows=n_val,
        )

    delta = cand_m.score - inc_m.score

    log.info(
        "gate | incumbent score=%.4f (auc=%.4f f1=%.4f) | "
        "candidate score=%.4f (auc=%.4f f1=%.4f) | Δ=%.4f | tol=%.4f | n_val=%d",
        inc_m.score,  inc_m.auc,  inc_m.f1,
        cand_m.score, cand_m.auc, cand_m.f1,
        delta, tol, n_val,
    )

    # ── promotion rule: strictly better score ─────────────────────────────────
    if cand_m.score <= inc_m.score:
        return GateResult(
            promoted=False,
            reason=f"score_not_improved:Δ={delta:.4f}",
            incumbent=inc_m, candidate=cand_m,
            score_delta=delta, val_rows=n_val,
        )

    # ── regression checks: neither metric regresses beyond tol ───────────────
    auc_drop = inc_m.auc - cand_m.auc
    f1_drop  = inc_m.f1  - cand_m.f1

    if auc_drop > tol:
        return GateResult(
            promoted=False,
            reason=f"auc_regression:{auc_drop:.4f}>tol={tol}",
            incumbent=inc_m, candidate=cand_m,
            score_delta=delta, val_rows=n_val,
        )
    if f1_drop > tol:
        return GateResult(
            promoted=False,
            reason=f"f1_regression:{f1_drop:.4f}>tol={tol}",
            incumbent=inc_m, candidate=cand_m,
            score_delta=delta, val_rows=n_val,
        )

    # ── PROMOTE ───────────────────────────────────────────────────────────────
    log.info("gate | PROMOTED (Δ=%.4f)", delta)
    return GateResult(
        promoted=True,
        reason="candidate_passes_gate",
        incumbent=inc_m, candidate=cand_m,
        score_delta=delta, val_rows=n_val,
    )
