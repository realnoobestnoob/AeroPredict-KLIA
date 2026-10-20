"""Optuna hyperparameter tuning for the KLIA flight delay predictor.

Tunes XGBoost params, ensemble settings, and feature top_k simultaneously.
Objective: maximise 0.4·AUC + 0.6·F1 while penalising train/val AUC gap (overfitting).
Slightly prioritises F1 (60%) over AUC (40%) for better precision-recall balance.

Usage:
    python -m klia.jobs.tune                        # fetch from Neon; auto-applies best params
    python -m klia.jobs.tune --csv data/departures.csv
    python -m klia.jobs.tune --trials 60            # override n_optuna_trials
    python -m klia.jobs.tune --sample 20000         # override sample row count
    python -m klia.jobs.tune --no-apply             # dry run: find params but do NOT write to config.yaml
    python -m klia.jobs.tune --apply-threshold      # also apply the tuned threshold (not recommended
                                                    # until you have verified tuning scores match bootstrap)

After completion, best XGBoost params and top_k are automatically written to config/config.yaml.
Decision threshold is NOT applied by default — bootstrap rediscovers it from honest sequential data.
Then run:
    python -m klia.jobs.update --bootstrap
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import yaml
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from klia.config import load_config
from klia.etl.validate import clean
from klia.features.state import FeatureState
from klia.features.selection import FeatureSelector
from klia.model.incremental import RandomSubspaceEnsemble, batch_prequential, best_threshold
from klia.store.base import open_store

log = logging.getLogger("klia.tune")

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


# ─────────────────────────────── data ────────────────────────────────────────

def _fetch_records(cfg: dict, store, sample_rows: int) -> tuple[list[dict], list[int]]:
    """Fetch and validate rows. Returns raw records and labels — no feature engineering.

    Feature engineering is deferred to _evaluate (per fold) and _compute_probe_features
    (for the SGD selector probe only) to avoid cross-fold leakage in smoothed rate features.
    """
    log.info("fetching up to %d rows for tuning...", sample_rows)
    raw = store.fetch_new_rows(0, sample_rows)
    if raw.empty:
        raise RuntimeError("no rows available in store")

    ok, bad = clean(raw, cfg)
    log.info("validation: %d ok, %d rejected", len(ok), len(bad))
    if ok.empty:
        raise RuntimeError("all rows rejected by validation")

    records = ok.to_dict("records")
    labels  = [int(r["is_delayed"]) for r in records]
    delay_rate = 100.0 * sum(labels) / len(labels) if labels else 0.0
    log.info("loaded %d rows, %.1f%% delayed", len(labels), delay_rate)
    return records, labels


def _compute_probe_features(cfg: dict, records: list[dict],
                             labels: list[int]) -> tuple[list[dict], list[int]]:
    """Compute features sequentially for the SGD feature-selection probe ONLY.

    Used to rank features by importance (FeatureSelector.fit_from_probe).
    NOT used for CV scoring — _evaluate rebuilds features per fold to avoid leakage.
    The sequential pass here is still correct for ranking purposes: feature importance
    ordering is stable even if absolute scores are slightly optimistic.
    """
    state  = FeatureState(cfg)
    X_all: list[dict] = []
    for rec, label in zip(records, labels):
        X_all.append(state.features(rec))   # READ before WRITE — no look-ahead
        state.update(rec, label)
    n_features = len(X_all[0]) if X_all else 0
    log.info("probe features computed: %d rows × %d raw features", len(X_all), n_features)
    return X_all, labels


def _fit_selector(X_probe: list[dict], labels: list[int],
                  top_k: int, probe_rows: int) -> FeatureSelector:
    """Fit a FeatureSelector for the given top_k on the first probe_rows of X_probe.

    Returns a passthrough selector (is_fitted=False) if the probe fails.
    Result is cached per top_k by make_objective's closure — runs once per distinct top_k.
    """
    selector = FeatureSelector()
    ok = selector.fit_from_probe(X_probe[:probe_rows], labels[:probe_rows], top_k=top_k)
    if not ok:
        log.warning("probe failed for top_k=%d; selector will pass all features through", top_k)
    return selector


# ─────────────────────────── model builder ───────────────────────────────────

def _build_model(params: dict, seed: int = 42) -> RandomSubspaceEnsemble:
    """Instantiate rsub_xgb from a flat params dict produced by the Optuna trial."""
    return RandomSubspaceEnsemble(
        base="xgb",
        n_estimators=params["n_estimators"],
        subspace_frac=params["subspace_frac"],
        params={
            "eta":              params["eta"],
            "max_depth":        params["max_depth"],
            "subsample":        params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "min_child_weight": params["min_child_weight"],
            "n_iter":           params["n_iter"],
        },
        seed=seed,
    )


# ─────────────────────────── evaluation ──────────────────────────────────────

def _evaluate(
    records: list[dict], labels: list[int], cfg: dict,
    selector: FeatureSelector, params: dict,
    n_splits: int, batch_size: int,
    gap_penalty: float = 0.3,
    trial: optuna.Trial | None = None,
) -> tuple[float, float, float, float]:
    """Leak-free StratifiedKFold cross-validation.

    For each fold a fresh FeatureState walks ALL records in original (time) order:
      - READ features  → assign to X_tr or X_val by fold membership
      - WRITE to state (regardless of fold membership — mirrors production exactly)

    This ensures each record's smoothed-rate features (airline_rate, route_prev1, etc.)
    only incorporate outcomes from records that PRECEDED it in time, whether those
    records are in the train or val fold. No fold can "see" future outcomes.

    Reports intermediate score after each fold so Optuna's MedianPruner can stop
    unpromising trials after fold 1 without evaluating the remaining folds.

    Returns (mean_val_auc, mean_f1, mean_gap, mean_threshold).
    """
    y_arr = np.array(labels)
    n     = len(records)
    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, f1s, gaps, thresholds = [], [], [], []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(range(n), y_arr)):
        tr_set = set(tr_idx.tolist())

        # ── Rebuild features for this fold with a fresh FeatureState ──────────
        state = FeatureState(cfg)
        X_tr: list[dict] = []
        y_tr: list[int]  = []
        X_val: list[dict] = []
        y_val_list: list[int] = []

        for i, (rec, label) in enumerate(zip(records, labels)):
            feat     = state.features(rec)                          # READ (no leakage)
            feat_sel = selector.filter(feat) if selector.is_fitted else feat
            state.update(rec, label)                                # WRITE

            if i in tr_set:
                X_tr.append(feat_sel)
                y_tr.append(label)
            else:
                X_val.append(feat_sel)
                y_val_list.append(label)

        y_val = np.array(y_val_list)

        # ── Train + score ──────────────────────────────────────────────────────
        model   = _build_model(params, seed=42 + fold_idx)
        p_train = batch_prequential(model, X_tr, y_tr,
                                    batch_size=batch_size, warmup_batches=1)

        if model.n_learned == 0 or y_val.min() == y_val.max():
            aucs.append(0.5); f1s.append(0.0); gaps.append(0.0); thresholds.append(0.5)
        else:
            p_val   = np.atleast_1d(model.predict_proba(X_val, fallback=0.3))
            val_auc = float(roc_auc_score(y_val, p_val))

            fold_thr, f1 = best_threshold(p_val, y_val, default=0.5)
            thresholds.append(fold_thr)

            warmup_n  = batch_size
            scored_y  = np.array(y_tr[warmup_n:])
            scored_p  = p_train[warmup_n:]
            train_auc = (float(roc_auc_score(scored_y, scored_p))
                         if len(scored_y) > 0 and scored_y.min() != scored_y.max()
                         else val_auc)

            gap = max(0.0, train_auc - val_auc)
            log.debug("fold %d: val_auc=%.4f f1=%.4f threshold=%.2f gap=%.4f",
                      fold_idx, val_auc, f1, fold_thr, gap)
            aucs.append(val_auc)
            f1s.append(f1 if f1 > 0.0 else 0.0)
            gaps.append(gap)

        # ── Report running score for fold-level pruning ───────────────────────
        if trial is not None:
            interim = (0.4 * float(np.mean(aucs))
                       + 0.6 * float(np.mean(f1s))
                       - gap_penalty * (float(np.mean(gaps)) if gaps else 0.0))
            trial.report(interim, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(aucs)), float(np.mean(f1s)), float(np.mean(gaps)), float(np.mean(thresholds))


# ─────────────────────────── optuna objective ────────────────────────────────

def make_objective(records: list[dict], labels: list[int], X_probe: list[dict],
                   cfg: dict, tune_cfg: dict):
    """Return the Optuna objective closure.

    Caches FeatureSelector per top_k so the SGD probe runs once per distinct
    top_k value, not once per trial.  Per-fold feature recomputation (in _evaluate)
    uses the raw records + a fresh FeatureState per fold to eliminate leakage.
    """
    probe_rows  = int(cfg["feature_selection"].get("probe_rows", 5_000))
    n_splits    = int(tune_cfg.get("cv_folds",    3))
    batch_size  = int(tune_cfg.get("batch_size", 500))
    gap_penalty = float(tune_cfg.get("gap_penalty", 0.3))

    # top_k → FeatureSelector (cached; avoids re-running the SGD probe per trial)
    _selector_cache: dict[int, FeatureSelector] = {}

    def objective(trial: optuna.Trial) -> float:
        # ── suggest hyperparameters ──────────────────────────────────────────
        top_k = trial.suggest_int("top_k",
                                   tune_cfg.get("top_k_min", 12),
                                   tune_cfg.get("top_k_max", 21))
        if top_k not in _selector_cache:
            _selector_cache[top_k] = _fit_selector(X_probe, labels, top_k, probe_rows)
        selector = _selector_cache[top_k]

        params = {
            "top_k":            top_k,
            "eta":              trial.suggest_float("eta",
                                    tune_cfg.get("eta_min", 0.01),
                                    tune_cfg.get("eta_max", 0.30), log=True),
            "max_depth":        trial.suggest_int("max_depth",
                                    tune_cfg.get("max_depth_min", 3),
                                    tune_cfg.get("max_depth_max", 7)),
            "subsample":        trial.suggest_float("subsample",
                                    tune_cfg.get("subsample_min", 0.60),
                                    tune_cfg.get("subsample_max", 1.00)),
            "colsample_bytree": trial.suggest_float("colsample_bytree",
                                    tune_cfg.get("colsample_min", 0.50),
                                    tune_cfg.get("colsample_max", 1.00)),
            "min_child_weight": trial.suggest_int("min_child_weight",
                                    tune_cfg.get("min_child_weight_min",  1),
                                    tune_cfg.get("min_child_weight_max", 20)),
            "n_iter":           trial.suggest_int("n_iter",
                                    tune_cfg.get("n_iter_min", 10),
                                    tune_cfg.get("n_iter_max", 50)),
            "n_estimators":     trial.suggest_int("n_estimators",
                                    tune_cfg.get("n_estimators_min",  3),
                                    tune_cfg.get("n_estimators_max", 10)),
            "subspace_frac":    trial.suggest_float("subspace_frac",
                                    tune_cfg.get("subspace_frac_min", 0.50),
                                    tune_cfg.get("subspace_frac_max", 0.95)),
        }

        # ── leak-free cross-validate with fold-level pruning ─────────────────
        auc, f1, gap, threshold = _evaluate(
            records, labels, cfg, selector, params,
            n_splits, batch_size, gap_penalty=gap_penalty, trial=trial,
        )

        # ── composite objective ──────────────────────────────────────────────
        score = 0.4 * auc + 0.6 * f1 - gap_penalty * gap

        trial.set_user_attr("auc",       round(auc,       4))
        trial.set_user_attr("f1",        round(f1,        4))
        trial.set_user_attr("gap",       round(gap,       4))
        trial.set_user_attr("threshold", round(threshold, 2))
        log.info(
            "trial %3d | score=%.4f  auc=%.4f  f1=%.4f  gap=%.4f  threshold=%.2f | "
            "top_k=%d eta=%.4f depth=%d sub=%.2f col=%.2f mcw=%d "
            "n_iter=%d n_est=%d sfrac=%.2f",
            trial.number, score, auc, f1, gap, threshold,
            params["top_k"], params["eta"], params["max_depth"],
            params["subsample"], params["colsample_bytree"], params["min_child_weight"],
            params["n_iter"], params["n_estimators"], params["subspace_frac"],
        )
        return score

    return objective


# ─────────────────────── write / apply results ───────────────────────────────

def _write_best_params(best_params: dict, apply: bool = True,
                       apply_threshold: bool = False) -> None:
    """Write best_params to config.yaml.

    Always writes tuning.best_params (audit trail).
    When apply=True: also updates model.params, model.boosting_n_iter,
    and feature_selection.top_k so --bootstrap picks them up immediately.
    Decision threshold is NOT auto-applied unless apply_threshold=True —
    bootstrap rediscovers the threshold from honest sequential data via
    bundle.refresh_threshold(), which is more reliable than CV-derived values.
    """
    with open(_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)

    # ── Audit record (always written) ────────────────────────────────────────
    raw.setdefault("tuning", {})["best_params"] = best_params

    # ── Live config sections ──────────────────────────────────────────────────
    if apply:
        raw["model"]["params"] = {
            "n_estimators":     best_params["n_estimators"],
            "subspace_frac":    round(float(best_params["subspace_frac"]), 4),
            "eta":              round(float(best_params["eta"]), 6),
            "max_depth":        best_params["max_depth"],
            "subsample":        round(float(best_params["subsample"]), 4),
            "colsample_bytree": round(float(best_params["colsample_bytree"]), 4),
            "min_child_weight": best_params["min_child_weight"],
        }
        raw["model"]["boosting_n_iter"]   = best_params["n_iter"]
        raw["feature_selection"]["top_k"] = best_params["top_k"]
        log.info("applied: model.params, model.boosting_n_iter, feature_selection.top_k")

        if apply_threshold:
            raw["model"]["decision_threshold"] = round(float(best_params["threshold"]), 2)
            log.info("applied: model.decision_threshold = %.2f", best_params["threshold"])
        else:
            log.info(
                "decision_threshold NOT applied (bootstrap will rediscover it). "
                "Use --apply-threshold to apply the Optuna value: %.2f",
                best_params.get("threshold", 0.5),
            )

    with open(_CONFIG_PATH, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
    log.info("config.yaml updated")


def _print_apply_instructions(best_params: dict, applied: bool = True,
                               applied_threshold: bool = False) -> None:
    status    = "Auto-applied to config.yaml ✅" if applied else "NOT applied (--no-apply)"
    thr_note  = (
        "✅ Applied"
        if applied_threshold
        else "⚠  NOT applied — bootstrap will rediscover it from honest data"
    )
    print(f"""
──────────────────────────────────────────────────────
  Best params — {status}
──────────────────────────────────────────────────────

  top_k             : {best_params['top_k']}
  eta               : {best_params['eta']:.4f}
  max_depth         : {best_params['max_depth']}
  subsample         : {best_params['subsample']:.2f}
  colsample_bytree  : {best_params['colsample_bytree']:.2f}
  min_child_weight  : {best_params['min_child_weight']}
  n_iter            : {best_params['n_iter']}
  n_estimators      : {best_params['n_estimators']}
  subspace_frac     : {best_params['subspace_frac']:.2f}
  threshold         : {best_params['threshold']:.2f}  ({thr_note})

{"✅ config/config.yaml updated — run bootstrap to retrain:" if applied else "⚠  Apply params manually to config/config.yaml, then run:"}

  python -m klia.jobs.update --bootstrap

──────────────────────────────────────────────────────
""")


# ──────────────────────────────── main ───────────────────────────────────────

def run(n_trials: int | None = None, sample_rows: int | None = None,
        csv: str | None = None, apply: bool = True,
        apply_threshold: bool = False) -> dict:
    cfg      = load_config()
    tune_cfg = cfg.get("tuning", {})
    trials   = n_trials    or cfg.get("eda", {}).get("n_optuna_trials", 30)
    sample   = sample_rows or tune_cfg.get("sample_rows", 15_000)

    store = open_store(cfg, csv)
    store.ensure_schema()

    records, labels = _fetch_records(cfg, store, sample)
    X_probe, _      = _compute_probe_features(cfg, records, labels)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        # Skip pruning for the first 10 trials (need a baseline median).
        # From trial 11 onward, prune if fold-1 score < median of completed trials.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )

    log.info("starting Optuna study: %d trials, sample=%d rows, cv_folds=%d",
             trials, sample, tune_cfg.get("cv_folds", 3))
    t0 = time.time()
    study.optimize(
        make_objective(records, labels, X_probe, cfg, tune_cfg),
        n_trials=trials,
        show_progress_bar=True,
    )

    best     = study.best_trial
    best_thr = best.user_attrs.get("threshold", 0.5)  # named best_thr; avoids shadowing imported best_threshold
    result   = {
        "best_score":     round(best.value, 4),
        "best_auc":       best.user_attrs.get("auc", 0.0),
        "best_f1":        best.user_attrs.get("f1",  0.0),
        "best_gap":       best.user_attrs.get("gap", 0.0),
        "best_threshold": best_thr,
        "best_params":    best.params,
        "n_trials":       trials,
        "seconds":        round(time.time() - t0, 1),
    }

    log.info(
        "best trial #%d: score=%.4f  auc=%.4f  f1=%.4f  gap=%.4f  threshold=%.2f",
        best.number, best.value,
        result["best_auc"], result["best_f1"], result["best_gap"], best_thr,
    )
    best.params["threshold"] = best_thr
    _write_best_params(best.params, apply=apply, apply_threshold=apply_threshold)
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--trials",           type=int,
                    help="number of Optuna trials (default: eda.n_optuna_trials in config.yaml)")
    ap.add_argument("--sample",           type=int,
                    help="rows to fetch from Neon (default: tuning.sample_rows in config.yaml)")
    ap.add_argument("--csv",
                    help="read rows from this CSV instead of Neon")
    ap.add_argument("--no-apply",         action="store_true",
                    help="save best params to tuning.best_params only; "
                         "do NOT update live model/feature_selection sections")
    ap.add_argument("--apply-threshold",  action="store_true",
                    help="also apply the Optuna-tuned threshold to model.decision_threshold "
                         "(off by default — bootstrap rediscovers it from honest sequential data)")
    a = ap.parse_args()

    apply           = not a.no_apply
    apply_threshold = getattr(a, "apply_threshold", False)
    result = run(n_trials=a.trials, sample_rows=a.sample, csv=a.csv,
                 apply=apply, apply_threshold=apply_threshold)

    print(f"\n── Tuning complete ({result['seconds']}s) ──")
    print(f"  Trials run : {result['n_trials']}")
    print(f"  Best score : {result['best_score']:.4f}  "
          f"(AUC={result['best_auc']:.4f}  F1={result['best_f1']:.4f}  "
          f"gap={result['best_gap']:.4f}  threshold={result['best_threshold']:.2f})")
    _print_apply_instructions(result["best_params"], applied=apply,
                               applied_threshold=apply_threshold)


if __name__ == "__main__":
    main()