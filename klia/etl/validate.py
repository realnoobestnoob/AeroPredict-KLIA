"""Validation and labelling. Raw 'departures' rows in, clean typed rows plus a reject list out.

Handles the mixed 12-hour / 24-hour time strings found in the source data.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import pandas as pd

_TIME_FORMATS = ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p")



def norm(value) -> str:
    """Canonical text key used by training AND serving (so 'airasia ' == 'AIRASIA')."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).upper().split())


@lru_cache(maxsize=4096)
def _parse_time_str(s: str) -> dt.time | None:
    s = s.strip().upper().replace(".", "")
    for ap in ("AM", "PM"):          # '8:05PM' -> '8:05 PM'
        if s.endswith(ap) and not s.endswith(" " + ap):
            s = s[: -len(ap)].strip() + " " + ap
    for fmt in _TIME_FORMATS:
        try:
            return dt.datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    return None


def parse_time(v) -> dt.time | None:
    if v is None or v is pd.NaT or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.time()
    if isinstance(v, dt.datetime):
        return v.time()
    if isinstance(v, dt.time):
        return v
    if isinstance(v, dt.timedelta):
        secs = int(v.total_seconds()) % 86400
        return dt.time(secs // 3600, (secs % 3600) // 60, secs % 60)
    return _parse_time_str(str(v))


def _parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.normalize()
    if s.map(lambda x: isinstance(x, dt.date)).all():
        return pd.to_datetime(s, errors="coerce")
    iso = pd.to_datetime(s, errors="coerce", format="ISO8601")
    bad = iso.isna() & s.notna()
    if bad.any():
        iso[bad] = pd.to_datetime(s[bad], errors="coerce", dayfirst=True)
    return iso.dt.normalize()


def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ok, rejected). `ok` is sorted by scheduled time and carries the label."""
    if df.empty:
        cols = ["id", "airline", "destination", "aircraft", "sched_dt", "delay_min", "is_delayed"]
        return pd.DataFrame(columns=cols), df.assign(reason="")
    threshold = cfg["data"]["delay_threshold_minutes"]
    out = pd.DataFrame({"id": df["id"].values}, index=df.index)

    date = _parse_dates(df["date"])
    sched_t = df["scheduled_departure"].map(parse_time)
    act_t = df["actual_departure"].map(parse_time)

    reason = pd.Series("", index=df.index)
    reason[date.isna()] = "bad date"
    reason[(reason == "") & sched_t.isna()] = "bad scheduled time"
    reason[(reason == "") & act_t.isna()] = "bad actual time"

    out["airline"] = df["airline"].map(norm)
    out["destination"] = df["destination"].map(norm)
    if "aircraft" in df.columns:
        out["aircraft"] = df["aircraft"].map(norm).replace("", "UNKNOWN")
    else:
        out["aircraft"] = "UNKNOWN"
    reason[(reason == "") & ((out["airline"] == "") | (out["destination"] == ""))] = "missing airline/destination"

    ok_mask = reason == ""
    sched = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    act = sched.copy()
    d = date[ok_mask]
    sched[ok_mask] = [dt.datetime.combine(x.date(), t) for x, t in zip(d, sched_t[ok_mask])]
    act[ok_mask] = [dt.datetime.combine(x.date(), t) for x, t in zip(d, act_t[ok_mask])]
    # Only a *large* negative gap means the flight left after midnight.
    # (The old code added a day to ANY negative gap, turning a 5-minute early departure into a 24h delay.)
    gap = (act - sched).dt.total_seconds() / 60
    overnight = ok_mask & (gap < -12 * 60)
    act[overnight] += pd.Timedelta(days=1)
    delay = (act - sched).dt.total_seconds() / 60
    reason[ok_mask & ((delay < -180) | (delay > 1440))] = "implausible delay"

    out["sched_dt"] = sched
    out["delay_min"] = delay
    out["is_delayed"] = (delay >= threshold).fillna(False).astype(int)

    good = reason == ""
    ok = out[good].copy()
    ok = ok.sort_values(["sched_dt", "id"], kind="stable").reset_index(drop=True)
    rejected = df[~good].copy()
    rejected["reason"] = reason[~good]
    return ok, rejected