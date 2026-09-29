"""
Statistics.

A rate with no interval is a rumour. Everything here exists to turn "it worked
62% of the time" into a claim with an honest error bar and something to compare
against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float, float]:
    """
    Wilson score interval. Correct near 0 and 1 and at small n, where the
    textbook normal interval produces bounds outside [0,1] and lies to you.
    """
    if n == 0:
        return (np.nan, np.nan, np.nan)
    z = sps.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def binom_vs(k: int, n: int, p0: float) -> float:
    """Two-sided p-value for the observed rate against a baseline rate."""
    if n == 0 or not np.isfinite(p0):
        return np.nan
    return float(sps.binomtest(int(k), int(n), min(max(p0, 1e-12), 1 - 1e-12),
                               alternative="two-sided").pvalue)


def bootstrap_mean(x: np.ndarray, n_boot: int = 5000, conf: float = 0.95,
                   seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI for mean R. Expectancy is skewed; don't assume normal."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return (float(x.mean()), float(lo), float(hi))


def summarise(events: pd.DataFrame, wins: pd.Series,
              baseline_rate: float | None = None) -> dict:
    n = len(events)
    k = int(wins.sum()) if n else 0
    p, lo, hi = wilson(k, n)
    mean_r, r_lo, r_hi = bootstrap_mean(events["r_fixed"].to_numpy()) if n \
        else (np.nan, np.nan, np.nan)
    out = {
        "n": n, "wins": k, "rate": p, "ci_low": lo, "ci_high": hi,
        "mean_r": mean_r, "mean_r_low": r_lo, "mean_r_high": r_hi,
        "ambiguous": float(events["ambiguous"].mean()) if n else np.nan,
        "median_bars": float(events["bars_held"].median()) if n else np.nan,
    }
    if baseline_rate is not None and np.isfinite(baseline_rate):
        out["baseline_rate"] = baseline_rate
        out["edge"] = p - baseline_rate
        out["p_value"] = binom_vs(k, n, baseline_rate)
    return out


# ---------------------------------------------------------------- baselines

def drift_baseline(m1: pd.DataFrame, events: pd.DataFrame, spec,
                   n_samples: int = 3000, seed: int = 0) -> dict:
    """
    Buy-and-hold drift baseline.

    Takes random entry times across the same period, applies the same
    direction, stop and target sizing, and measures how often that resolves
    positively. This is the "was I just long a market that went up" control.
    """
    from .scanner import score_event
    if events.empty:
        return {"rate": np.nan, "n": 0}

    o = m1["open"].to_numpy(float)
    h = m1["high"].to_numpy(float)
    l = m1["low"].to_numpy(float)
    c = m1["close"].to_numpy(float)
    n_m1 = len(m1)

    rng = np.random.default_rng(seed)
    lo_i = int(m1.index.searchsorted(events["ts"].min()))
    hi_i = int(m1.index.searchsorted(events["ts"].max()))
    hi_i = min(hi_i, n_m1 - spec.scoring.max_holding_bars - 2)
    if hi_i <= lo_i:
        return {"rate": np.nan, "n": 0}

    med_risk = float(events["risk_points"].median())
    med_atr = float(events["atr_at_entry"].median())
    biases = events["bias"].to_numpy()
    idxs = rng.integers(lo_i, hi_i, size=n_samples)

    wins = 0
    used = 0
    for i in idxs:
        bias = int(rng.choice(biases))
        entry = float(o[i])
        half = med_risk / 2.0
        zl, zh = (entry - med_risk, entry) if bias == 1 else (entry, entry + med_risk)
        r = score_event(int(i), bias, zl, zh, med_atr, spec, o, h, l, c, n_m1)
        if r is None:
            continue
        used += 1
        wins += int(r["outcome"] == "win")
    if used == 0:
        return {"rate": np.nan, "n": 0}
    p, lo, hi = wilson(wins, used)
    return {"rate": p, "ci_low": lo, "ci_high": hi, "n": used, "wins": wins}


def placebo_baseline(m1: pd.DataFrame, events: pd.DataFrame, spec,
                     jitter_atr: float = 3.0, seed: int = 0) -> dict:
    """
    Placebo zones.

    Same timestamps, same zone sizes, same scoring — but each zone is shifted
    to a random nearby price that has no claim to being an order block. If the
    placebo scores like the real thing, the level is not what is producing the
    result. This is the control that actually tests the pattern.
    """
    from .scanner import score_event
    if events.empty:
        return {"rate": np.nan, "n": 0}

    o = m1["open"].to_numpy(float)
    h = m1["high"].to_numpy(float)
    l = m1["low"].to_numpy(float)
    c = m1["close"].to_numpy(float)
    n_m1 = len(m1)
    rng = np.random.default_rng(seed)

    wins = used = 0
    idx = m1.index
    horizon = max(spec.scoring.max_holding_bars * 4, 2000)
    for e in events.itertuples(index=False):
        start = int(idx.searchsorted(pd.Timestamp(e.formed_at)))
        if start >= n_m1 - 2:
            continue
        shift = rng.uniform(0.5, jitter_atr) * e.atr_at_entry * rng.choice([-1, 1])
        zl, zh = e.zone_low + shift, e.zone_high + shift

        # The placebo must be *reached* the same way the real zone was. Scoring
        # a level price never traded to would compare the pattern against a
        # fantasy, and the fantasy usually wins.
        stop_at = min(start + horizon, n_m1)
        seg_h, seg_l = h[start:stop_at], l[start:stop_at]
        # Match the approach direction too. A bullish zone is meant to be
        # tested from above; letting the placebo be reached from below compares
        # the pattern against a systematically worse entry and flatters it.
        inside = (seg_h >= zl) & (seg_l <= zh)
        if bool(inside[0]):
            inside[0] = False
        approach = np.empty_like(inside)
        approach[0] = False
        if int(e.bias) == 1:
            approach[1:] = seg_l[:-1] > zh
        else:
            approach[1:] = seg_h[:-1] < zl
        touch = np.flatnonzero(inside & approach)
        if touch.size == 0:
            continue
        i = start + int(touch[0])
        r = score_event(i, int(e.bias), zl, zh, e.atr_at_entry, spec,
                        o, h, l, c, n_m1)
        if r is None:
            continue
        used += 1
        wins += int(r["outcome"] == "win")
    if used == 0:
        return {"rate": np.nan, "n": 0}
    p, lo, hi = wilson(wins, used)
    return {"rate": p, "ci_low": lo, "ci_high": hi, "n": used, "wins": wins}


# ---------------------------------------------------------------- breakdowns

def by_group(events: pd.DataFrame, wins: pd.Series, col: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    df = events.assign(_win=wins.astype(int))
    rows = []
    for key, g in df.groupby(col, dropna=False):
        p, lo, hi = wilson(int(g["_win"].sum()), len(g))
        rows.append({col: key, "n": len(g), "rate": p,
                     "ci_low": lo, "ci_high": hi,
                     "mean_r": float(g["r_fixed"].mean())})
    return pd.DataFrame(rows).sort_values(col).reset_index(drop=True)


def split_is_oos(events: pd.DataFrame, is_start, is_end):
    """Returns (in_sample, out_of_sample). OOS is everything after is_end."""
    if events.empty:
        return events, events
    ts = pd.to_datetime(events["ts"], utc=True)
    a, b = _utc(is_start), _utc(is_end)
    return events[(ts >= a) & (ts <= b)], events[ts > b]


def _utc(x) -> pd.Timestamp:
    """Accepts dates, naive timestamps and tz-aware ones without complaining."""
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
