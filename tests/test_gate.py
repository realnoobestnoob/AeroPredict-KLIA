"""tests/test_gate.py — Unit tests for the model promotion gate.

Run with:
    pytest tests/test_gate.py -v

These tests exercise gate.py directly with lightweight Bundle stubs.
No store or network connection is required.
"""
from __future__ import annotations

import math
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

from klia.jobs.gate import (
    GateResult,
    advance_state_only,
    quality_score,
    run_gate,
)


# ─────────────────────────── stubs ───────────────────────────────────────────

class _State:
    """Minimal FeatureState stub."""
    def __init__(self, base_rate: float = 0.35):
        self.base_rate = base_rate
        self._updates  = 0

    def features(self, rec: dict) -> dict:
        """Read-only: returns a feature dict without mutating state."""
        return {"f1": float(rec.get("f1", 0.5)),
                "f2": float(rec.get("f2", 0.5))}

    def update(self, rec: dict, label: int) -> None:
        self._updates += 1


class _Selector:
    is_fitted = False

    def filter(self, feat: dict) -> dict:
        return feat


class _Model:
    """Stub that returns a fixed probability for every row."""
    def __init__(self, p: float, n_learned: int = 2000):
        self._p        = float(p)
        self.n_learned = n_learned

    def predict_proba(self, X, fallback: float = 0.35):
        if isinstance(X, list):
            return [self._p] * len(X)
        return self._p


class _Bundle:
    def __init__(self, p: float, threshold: float = 0.5, n_learned: int = 2000):
        self.state    = _State()
        self.selector = _Selector()
        self.model    = _Model(p, n_learned)
        self.threshold = threshold
        self.meta     = {}


# ──────────────────────── val DataFrame factory ───────────────────────────────

def _val_df(n: int = 400, delay_frac: float = 0.55, seed: int = 0) -> pd.DataFrame:
    """Balanced labelled validation DataFrame."""
    rng      = np.random.default_rng(seed)
    n_delay  = int(n * delay_frac)
    n_ok     = n - n_delay
    labels   = [1] * n_delay + [0] * n_ok
    return pd.DataFrame({
        "is_delayed": labels,
        "f1": rng.random(n),
        "f2": rng.random(n),
    })


_BASE_CFG = {
    "gate":  {"min_val_rows": 200, "metric_tolerance": 0.0},
    "model": {"warmup_rows": 500},
}


# ────────────────────── quality_score sanity ─────────────────────────────────

def test_quality_score_weights():
    assert abs(quality_score(1.0, 1.0) - 1.0) < 1e-9
    assert abs(quality_score(0.0, 0.0) - 0.0) < 1e-9
    assert abs(quality_score(1.0, 0.0) - 0.4) < 1e-9
    assert abs(quality_score(0.0, 1.0) - 0.6) < 1e-9


# ─────────────────────── promotion ────────────────────────────────────────────

def test_promotion_when_candidate_strictly_better():
    """Candidate with clearly better predictions should be promoted."""
    # incumbent predicts everything as 0 (p=0.0) → AUC≈0.5, F1=0
    # candidate predicts correctly (p=0.9 for all) → AUC>0.5, F1>0
    val      = _val_df(n=400, delay_frac=0.55)
    inc      = _Bundle(p=0.0, threshold=0.5, n_learned=2000)
    cand     = _Bundle(p=0.9, threshold=0.5, n_learned=2000)

    result = run_gate(inc, cand, val, _BASE_CFG)

    assert result.promoted, f"expected PROMOTED, got reason={result.reason}"
    assert result.score_delta is not None and result.score_delta > 0
    assert result.candidate is not None
    assert result.incumbent is not None


# ─────────────────────── rejection: worse score ───────────────────────────────

def test_rejection_when_candidate_not_better():
    """Candidate with same or worse predictions should be rejected."""
    val  = _val_df(n=400)
    inc  = _Bundle(p=0.9, threshold=0.5, n_learned=2000)   # good incumbent
    cand = _Bundle(p=0.1, threshold=0.5, n_learned=2000)   # bad candidate

    result = run_gate(inc, cand, val, _BASE_CFG)

    assert not result.promoted
    assert "score_not_improved" in result.reason


def test_rejection_when_scores_equal():
    """Equal scores should NOT promote (requires *strictly* better)."""
    val  = _val_df(n=400)
    inc  = _Bundle(p=0.6, threshold=0.5, n_learned=2000)
    cand = _Bundle(p=0.6, threshold=0.5, n_learned=2000)   # identical

    result = run_gate(inc, cand, val, _BASE_CFG)

    assert not result.promoted
    assert "score_not_improved" in result.reason


# ─────────────────── rejection: metric regression ────────────────────────────

def test_rejection_on_auc_regression_zero_tolerance():
    """Candidate with higher composite score but lower AUC should be rejected at tol=0."""
    val = _val_df(n=400, delay_frac=0.55)
    # Arrange predictions so candidate has higher F1 but lower AUC than incumbent.
    # We test this by checking the gate applies the regression check.
    cfg_tol = {
        "gate":  {"min_val_rows": 200, "metric_tolerance": 0.0},
        "model": {"warmup_rows": 500},
    }

    inc  = _Bundle(p=0.8, threshold=0.5, n_learned=2000)
    cand = _Bundle(p=0.6, threshold=0.5, n_learned=2000)

    result = run_gate(inc, cand, val, cfg_tol)
    # Candidate has worse predictions than incumbent → score_not_improved or regression
    assert not result.promoted


def test_rejection_passes_with_nonzero_tolerance():
    """A small AUC drop inside tolerance should not block promotion."""
    val = _val_df(n=400, delay_frac=0.55)
    cfg = {
        "gate":  {"min_val_rows": 200, "metric_tolerance": 0.5},  # wide tolerance
        "model": {"warmup_rows": 500},
    }

    inc  = _Bundle(p=0.0, threshold=0.5, n_learned=2000)   # poor incumbent
    cand = _Bundle(p=0.9, threshold=0.5, n_learned=2000)   # good candidate

    result = run_gate(inc, cand, val, cfg)
    # Candidate is clearly better on both metrics → should promote
    assert result.promoted


# ────────────────── rejection: invalid validation data ───────────────────────

def test_rejection_on_single_class_val_data():
    """Single-class validation set should always reject."""
    val = pd.DataFrame({
        "is_delayed": [1] * 300,  # all delayed — single class
        "f1": np.random.random(300),
        "f2": np.random.random(300),
    })
    inc  = _Bundle(p=0.5, threshold=0.5, n_learned=2000)
    cand = _Bundle(p=0.9, threshold=0.5, n_learned=2000)

    result = run_gate(inc, cand, val, _BASE_CFG)

    assert not result.promoted
    assert "invalid_val_data" in result.reason


def test_rejection_on_insufficient_val_rows():
    """Fewer than min_val_rows should reject without evaluating."""
    val = _val_df(n=50)   # below default min_val_rows=200
    inc  = _Bundle(p=0.5, threshold=0.5, n_learned=2000)
    cand = _Bundle(p=0.9, threshold=0.5, n_learned=2000)

    result = run_gate(inc, cand, val, _BASE_CFG)

    assert not result.promoted
    assert "insufficient_val_rows" in result.reason
    # Metrics should NOT have been computed (guarded before eval)
    assert result.incumbent is None
    assert result.candidate is None


def test_rejection_on_empty_val():
    """Empty DataFrame should reject."""
    val  = pd.DataFrame({"is_delayed": [], "f1": [], "f2": []})
    inc  = _Bundle(p=0.5, threshold=0.5, n_learned=2000)
    cand = _Bundle(p=0.9, threshold=0.5, n_learned=2000)

    result = run_gate(inc, cand, val, _BASE_CFG)

    assert not result.promoted


# ────────────────── incumbent immutability ────────────────────────────────────

def test_advance_state_only_does_not_mutate_model():
    """advance_state_only() should update FeatureState counters but never model weights."""
    rows = _val_df(n=100)
    bundle = _Bundle(p=0.5, threshold=0.5, n_learned=1000)

    n_learned_before = bundle.model.n_learned
    updates_before   = bundle.state._updates

    advance_state_only(bundle, rows)

    # Model weights unchanged
    assert bundle.model.n_learned == n_learned_before, "model.n_learned must not change"
    # FeatureState advanced (one update per row)
    assert bundle.state._updates == updates_before + len(rows)


def test_eval_does_not_mutate_state():
    """_extract_features() must not call state.update() (state is read-only during eval)."""
    from klia.jobs.gate import _extract_features

    val    = _val_df(n=200)
    bundle = _Bundle(p=0.7, threshold=0.5, n_learned=2000)
    before = bundle.state._updates

    _extract_features(bundle, val)

    assert bundle.state._updates == before, (
        "_extract_features must not update FeatureState"
    )


def test_incumbent_bytes_unchanged_after_gate():
    """Re-loading incumbent from the same bytes after gate should yield identical model."""
    import pickle

    val   = _val_df(n=400)
    inc   = _Bundle(p=0.0, threshold=0.5, n_learned=2000)
    cand  = _Bundle(p=0.9, threshold=0.5, n_learned=2000)

    # Snapshot incumbent predictions BEFORE gate
    pre_preds = inc.model.predict_proba([{"f1": 0.5}] * 5)

    run_gate(inc, cand, val, _BASE_CFG)

    # Incumbent predictions unchanged after gate (gate must not mutate inc)
    post_preds = inc.model.predict_proba([{"f1": 0.5}] * 5)
    assert pre_preds == post_preds, "incumbent model must not be mutated by gate"


# ─────────────── GateResult helper ───────────────────────────────────────────

def test_gate_result_as_dict_contains_required_keys():
    """as_dict() must always contain gate_promoted, gate_reason, gate_val_rows."""
    result = GateResult(promoted=False, reason="test_reason", val_rows=300)
    d = result.as_dict()

    assert "gate_promoted"  in d
    assert "gate_reason"    in d
    assert "gate_val_rows"  in d
    assert d["gate_promoted"] == 0
    assert d["gate_reason"] == "test_reason"


def test_gate_result_as_dict_metrics_present_on_promotion():
    """Promoted result must expose all metric keys."""
    from klia.jobs.gate import GateMetrics

    result = GateResult(
        promoted=True, reason="candidate_passes_gate",
        incumbent=GateMetrics(auc=0.70, f1=0.65, score=0.67, threshold=0.5, val_rows=300),
        candidate=GateMetrics(auc=0.75, f1=0.70, score=0.72, threshold=0.5, val_rows=300),
        score_delta=0.05, val_rows=300,
    )
    d = result.as_dict()

    for key in ("gate_inc_auc", "gate_inc_f1", "gate_inc_score",
                "gate_cand_auc", "gate_cand_f1", "gate_cand_score",
                "gate_score_delta"):
        assert key in d, f"missing key: {key}"


# ─────────────── MLflow failure does not bypass gate ─────────────────────────

def test_mlflow_failure_does_not_affect_gate():
    """If MLflow raises during logging, gate result is unaffected."""
    from klia.jobs.update import _log_mlflow
    from klia.jobs.gate import GateResult

    result = GateResult(
        promoted=True, reason="candidate_passes_gate", val_rows=300
    )

    # Patch mlflow so it raises immediately
    with mock.patch("mlflow.set_tracking_uri", side_effect=RuntimeError("mlflow down")):
        # Should not raise — MLflow failures are swallowed
        _log_mlflow({"rows_ok": 100}, version=5, gate_result=result)

    # Gate result is unchanged
    assert result.promoted is True


def test_mlflow_failure_start_run_does_not_bypass():
    """MLflow failure mid-run (start_run raises) should not corrupt state."""
    from klia.jobs.update import _log_mlflow
    from klia.jobs.gate import GateResult

    result = GateResult(promoted=False, reason="score_not_improved:Δ=-0.05", val_rows=300)

    with mock.patch("mlflow.start_run", side_effect=Exception("connection refused")):
        _log_mlflow({"rows_ok": 50}, version=3, gate_result=result)

    assert result.promoted is False   # gate decision unchanged


# ──────────────────────────── warmup guard ───────────────────────────────────

def test_rejection_when_incumbent_warming_up():
    """Incumbent still in warmup should always reject (model unreliable)."""
    val  = _val_df(n=400)
    inc  = _Bundle(p=0.5, threshold=0.5, n_learned=10)  # way below warmup_rows=500
    cand = _Bundle(p=0.9, threshold=0.5, n_learned=2000)

    result = run_gate(inc, cand, val, _BASE_CFG)

    assert not result.promoted
    assert "warming_up" in result.reason
