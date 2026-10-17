"""Incremental update job: fetch only NEW rows, validate, and update the model with ONE
`partial_fit` call on that whole batch (incremental learning, not row-by-row online learning).

    python -m klia.jobs.update                 # normal run (cron / Task Scheduler / GitHub Actions)
    python -m klia.jobs.update --bootstrap     # first run: start from the notebook's trained model
                                                 #  (artifacts/pretrained_bundle.gz) if present,
                                                 #  else a fresh untrained model, then replay history.
                                                 #  Also runs the feature-selection probe to determine
                                                 #  which features the model trains on.
    python -m klia.jobs.update --csv data/departures.csv   # run against a CSV instead of Neon

Gate (opt-in, gate.enabled: true in config.yaml):
    On non-bootstrap runs with an existing incumbent, the update job trains a
    candidate on (1 - val_fraction) of new rows and evaluates both models on the
    chronological held-out remainder.  The candidate is promoted only if it
    strictly improves quality_score = 0.4×AUC + 0.6×F1 without regressing either
    metric beyond metric_tolerance.  On rejection the incumbent is retained and the
    watermark still advances so processed rows are never retrained.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from klia.config import artifacts_dir, load_config
from klia.etl.validate import clean
from klia.features.state import FeatureState
from klia.model.bundle import Bundle, new_bundle
from klia.pipeline import apply_rows, feature_stream
from klia.store.base import open_store
from klia.store.postgres import PostgresStore

log = logging.getLogger("klia.update")


# ─────────────────────────── MLflow helper ────────────────────────────────────

def _log_mlflow(info: dict, version: int, gate_result=None) -> None:
    """Best-effort MLflow logging; silently skipped if mlflow is not installed.

    Supports MLFLOW_TRACKING_URI env var; falls back to local SQLite.
    gate_result: GateResult | None — when present, promotion metrics are logged.
    MLflow failures never corrupt persistence or bypass the gate.
    """
    try:
        import mlflow

        db  = Path(__file__).resolve().parents[2] / "mlflow.db"
        uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{db}")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("klia-flight-delay")

        with mlflow.start_run(run_name=f"update_v{version}"):
            # ── training metrics ──────────────────────────────────────────────
            numeric = {k: v for k, v in info.items() if isinstance(v, (int, float))}
            mlflow.log_metrics({k: float(v) for k, v in numeric.items()})
            mlflow.log_params({"version": version})

            # ── gate metrics (when enabled) ───────────────────────────────────
            if gate_result is not None:
                gate_dict = gate_result.as_dict()
                gate_metrics = {
                    k: float(v) for k, v in gate_dict.items()
                    if isinstance(v, (int, float)) and k != "gate_reason"
                }
                gate_params = {
                    k: str(v) for k, v in gate_dict.items()
                    if k == "gate_reason"
                }
                mlflow.log_metrics(gate_metrics)
                mlflow.log_params(gate_params)
                mlflow.log_params({
                    "gate_inc_version":  info.get("incumbent_version", "n/a"),
                    "gate_cand_version": str(version),
                })

    except Exception as e:
        log.debug("mlflow logging skipped: %s", e)


# ──────────────────────── bootstrap helpers (unchanged) ──────────────────────

def _starting_bundle(cfg: dict, bootstrap: bool) -> Bundle:
    """On --bootstrap, prefer the fully-trained bundle the notebook saved; else fresh."""
    pretrained = artifacts_dir() / "pretrained_bundle.gz"
    if bootstrap and pretrained.exists():
        log.info("bootstrapping from notebook's trained model: %s", pretrained)
        return Bundle.loads(pretrained.read_bytes())
    log.info("starting a fresh, untrained model (%s)", cfg["model"]["name"])
    return new_bundle(cfg)


def _run_feature_selection_probe(bundle: Bundle, store, cfg: dict) -> None:
    """Rank features by importance on a small sample; fit bundle.selector.  Bootstrap only."""
    fs_cfg = cfg.get("feature_selection", {})
    if not fs_cfg.get("enabled", True):
        log.info("feature selection disabled in config; using all features")
        return

    top_k   = int(fs_cfg.get("top_k",      20))
    probe_n = int(fs_cfg.get("probe_rows", 5_000))

    log.info("feature selection probe: sampling up to %d rows…", probe_n)
    probe_raw = store.fetch_new_rows(0, probe_n)
    if probe_raw.empty:
        log.warning("no rows for probe; skipping feature selection")
        return

    probe_ok, _ = clean(probe_raw, cfg)
    if probe_ok.empty:
        log.warning("probe rows all rejected; skipping feature selection")
        return

    probe_state = FeatureState(cfg)
    probe_X, probe_y = feature_stream(probe_ok, probe_state)

    fitted = bundle.selector.fit_from_probe(probe_X, probe_y, top_k=top_k)
    if fitted:
        bundle.meta["feature_selection"] = {
            "enabled":    True,
            "top_k":      top_k,
            "n_selected": len(bundle.selector.selected),
            "selected":   bundle.selector.selected,
            "importances": bundle.selector.importances,
        }
        log.info("feature selection: %d features selected", len(bundle.selector.selected))
    else:
        log.warning("feature selection probe failed; training on all features")


# ──────────────────────────── gate path ──────────────────────────────────────

def _run_gate_path(
    incumbent_bytes: bytes,
    store,
    watermark: int,
    incumbent_version: int,
    cfg: dict,
    max_rows: int | None,
    dry_run: bool,
    t0: float,
) -> dict:
    """Train candidate, run gate, commit winner.  Returns the info dict for run().

    Watermark always advances past every row fetched in this call so that
    processed rows are never retrained on the next run, regardless of outcome.
    """
    from klia.jobs.gate import advance_state_only, run_gate

    gate_cfg    = cfg.get("gate", {})
    sample_rows = int(gate_cfg.get("sample_rows", 10_000))
    val_frac    = float(gate_cfg.get("val_fraction", 0.20))
    min_val     = int(gate_cfg.get("min_val_rows", 200))
    batch_size  = cfg["data"]["batch_size"]

    # ── 1. Fetch all candidate rows in one pass ───────────────────────────────
    cap = min(sample_rows, max_rows) if max_rows else sample_rows
    raw = store.fetch_new_rows(watermark, cap)
    if raw.empty:
        elapsed = round(time.time() - t0, 1)
        return {"status": "no_new_rows", "seconds": elapsed}

    ok, bad   = clean(raw, cfg)
    new_wm    = int(raw["id"].max())   # advance past ALL fetched rows

    totals = {
        "rows_in":       len(raw),
        "rows_ok":       len(ok),
        "rows_rejected": len(bad),
        "rows_learned":  0,
        "drift_events":  0,
    }
    reasons: dict[str, int] = {}
    for r, n in bad["reason"].value_counts().items():
        reasons[r] = int(n)

    # ── 2. Chronological train / val split ────────────────────────────────────
    n_val = max(int(len(ok) * val_frac), min_val) if len(ok) > min_val else 0
    n_val = min(n_val, len(ok) - 1)   # always keep ≥1 train row

    if n_val < min_val:
        # Not enough rows for a reliable gate — fall back to normal training.
        log.info(
            "gate: too few ok rows (%d) for val split (min %d); "
            "training directly on all rows (no gate)", len(ok), min_val,
        )
        bundle = Bundle.loads(incumbent_bytes)
        stats  = apply_rows(bundle, ok, cfg)
        totals["rows_learned"] = stats.get("rows_learned", 0)
        totals["drift_events"] = stats.get("drift_events", 0)
        gate_result = None
    else:
        train_ok = ok.iloc[:-n_val].copy()
        val_ok   = ok.iloc[-n_val:].copy()
        log.info(
            "gate: %d train rows + %d val rows (val_frac=%.2f)",
            len(train_ok), len(val_ok), val_frac,
        )

        # ── 3. Clone incumbent → candidate, train on train_ok ─────────────────
        candidate = Bundle.loads(incumbent_bytes)
        stats     = apply_rows(candidate, train_ok, cfg)
        totals["rows_learned"] = stats.get("rows_learned", 0)
        totals["drift_events"] = stats.get("drift_events", 0)

        # ── 4. Advance eval_incumbent FeatureState through train_ok (no fit) ──
        eval_incumbent = Bundle.loads(incumbent_bytes)
        advance_state_only(eval_incumbent, train_ok)

        # ── 5. Run gate ────────────────────────────────────────────────────────
        gate_result = run_gate(eval_incumbent, candidate, val_ok, cfg)

        if gate_result.promoted:
            # Candidate is promoted; its FeatureState already includes train_ok.
            # Advance it through val_ok too so the committed state is fully current.
            advance_state_only(candidate, val_ok)
            bundle = candidate
            log.info("gate: PROMOTED (Δ=%.4f)", gate_result.score_delta or 0)
        else:
            # Retain incumbent; advance its FeatureState through ALL new rows so
            # the committed state reflects what happened this run.
            retained = Bundle.loads(incumbent_bytes)
            advance_state_only(retained, ok)
            bundle = retained
            log.info("gate: REJECTED — %s; retaining incumbent", gate_result.reason)

    # ── 6. Build info dict ────────────────────────────────────────────────────
    info = {
        **totals,
        "reject_reasons":     reasons,
        "seconds":            round(time.time() - t0, 1),
        "incumbent_version":  incumbent_version,
        **(gate_result.as_dict() if gate_result is not None else {}),
        **{k: stats[k] for k in ("roc_auc", "f1", "log_loss", "threshold")
           if k in stats},
    }
    blob        = bundle.dumps()
    info["bundle_mb"] = round(len(blob) / 1e6, 2)

    if isinstance(store, PostgresStore):
        cap_mb = cfg["registry"].get("max_mb_neon", 400)
        if info["bundle_mb"] > cap_mb:
            raise RuntimeError(
                f"bundle is {info['bundle_mb']} MB, over Neon cap of {cap_mb} MB."
            )

    if dry_run:
        log.info("dry run, not saving: %s", info)
        return {"status": "dry_run", **info}

    version = store.commit(blob, bundle.describe(), new_wm, info)
    log.info("saved model v%s: %s", version, info)
    _log_mlflow(info, version, gate_result)
    return {"status": "ok", "version": version, **info}


# ─────────────────────────────── run ─────────────────────────────────────────

def run(bootstrap: bool = False, csv: str | None = None,
        dry_run: bool = False, max_rows: int | None = None) -> dict:
    cfg   = load_config()
    store = open_store(cfg, csv)
    store.ensure_schema()
    t0    = time.time()

    with store.lock():
        loaded = None if bootstrap else store.load_bundle_bytes()
        if loaded is None:
            if not bootstrap and store.get_watermark() > 0:
                raise RuntimeError(
                    "watermark exists but no model bundle found; re-run with --bootstrap")
            bundle, watermark = _starting_bundle(cfg, bootstrap), 0
        else:
            bundle, watermark = Bundle.loads(loaded[1]), store.get_watermark()
            log.info("loaded bundle v%s, watermark id=%s", loaded[0], watermark)

        # ── Data detection guard ──────────────────────────────────────────────
        if not bootstrap:
            if store.count_new_rows(watermark) == 0:
                elapsed = round(time.time() - t0, 1)
                log.info("no new data, skipped (elapsed=%.1fs)", elapsed)
                return {"status": "no_new_rows", "seconds": elapsed}

        # ── Feature selection probe (bootstrap only) ──────────────────────────
        if bootstrap:
            _run_feature_selection_probe(bundle, store, cfg)

        # ── Gate path (non-bootstrap, gate.enabled=true, incumbent exists) ────
        gate_cfg  = cfg.get("gate", {})
        use_gate  = (
            not bootstrap
            and gate_cfg.get("enabled", False)
            and loaded is not None
        )
        if use_gate:
            return _run_gate_path(
                incumbent_bytes  = loaded[1],
                store            = store,
                watermark        = watermark,
                incumbent_version= loaded[0],
                cfg              = cfg,
                max_rows         = max_rows,
                dry_run          = dry_run,
                t0               = t0,
            )

        # ── Normal training loop (bootstrap or gate disabled) ─────────────────
        totals  = {"rows_in": 0, "rows_ok": 0, "rows_rejected": 0,
                   "rows_learned": 0, "drift_events": 0}
        reasons: dict[str, int] = {}
        batch   = cfg["data"]["batch_size"]
        stats:  dict = {}

        while True:
            raw = store.fetch_new_rows(watermark, batch)
            if raw.empty:
                break
            ok, bad   = clean(raw, cfg)
            watermark = int(raw["id"].max())
            totals["rows_in"]       += len(raw)
            totals["rows_ok"]       += len(ok)
            totals["rows_rejected"] += len(bad)
            for r, n in bad["reason"].value_counts().items():
                reasons[r] = reasons.get(r, 0) + int(n)
            stats = apply_rows(bundle, ok, cfg)
            totals["rows_learned"]  += stats.get("rows_learned", 0)
            totals["drift_events"]  += stats.get("drift_events", 0)
            log.info("processed %d rows (watermark id=%s)", len(raw), watermark)
            if len(raw) < batch or (max_rows and totals["rows_in"] >= max_rows):
                break

        info = {
            **totals,
            "reject_reasons": reasons,
            "seconds": round(time.time() - t0, 1),
            **{k: stats[k] for k in ("roc_auc", "f1", "log_loss", "threshold")
               if k in stats},
        }
        blob        = bundle.dumps()
        info["bundle_mb"] = round(len(blob) / 1e6, 2)

        if isinstance(store, PostgresStore):
            cap = cfg["registry"].get("max_mb_neon", 400)
            if info["bundle_mb"] > cap:
                raise RuntimeError(
                    f"model bundle is {info['bundle_mb']} MB, over Neon cap of {cap} MB.")

        if dry_run:
            log.info("dry run, not saving: %s", info)
            return {"status": "dry_run", **info}

        version = store.commit(blob, bundle.describe(), watermark, info)
        log.info("saved model v%s: %s", version, info)
        _log_mlflow(info, version)
        return {"status": "ok", "version": version, **info}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--csv",       help="read from CSV instead of Neon")
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--max-rows",  type=int)
    a = ap.parse_args()
    run(bootstrap=a.bootstrap, csv=a.csv, dry_run=a.dry_run, max_rows=a.max_rows)


if __name__ == "__main__":
    main()
