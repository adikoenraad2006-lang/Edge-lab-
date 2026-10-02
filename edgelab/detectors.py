"""
Zone detectors.

Each returns a DataFrame with one row per zone:

    formed_at   timestamp at which the zone was KNOWN (not drawn)
    zone_low, zone_high
    bias        +1 bullish (expect support), -1 bearish (expect resistance)
    expires_at  timestamp after which the zone is no longer tested
    plus detector-specific context columns used by filters

`formed_at` is deliberately the *close of the confirming bar*, pushed to the
next bar boundary. A zone drawn on a candle ten bars ago was not tradeable ten
bars ago; you learned about it when the confirmation completed. The scanner
only ever tests price action strictly after this timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "formed_at", "zone_low", "zone_high", "bias", "expires_at",
        "zone_height", "zone_height_atr", "displacement_atr", "source_ts",
    ])


def _finish(rows: list[dict], f: pd.DataFrame, max_age_bars: int,
            tf_delta: pd.Timedelta) -> pd.DataFrame:
    if not rows:
        return _empty()
    out = pd.DataFrame(rows)
    out["zone_height"] = out["zone_high"] - out["zone_low"]
    out["expires_at"] = out["formed_at"] + max_age_bars * tf_delta
    out = out[out["zone_height"] > 0]
    return out.reset_index(drop=True)


def order_block(f: pd.DataFrame, displacement_atr: float = 1.5,
                displacement_bars: int = 1, require_swing_break: bool = True,
                swing_lookback: int = 10, zone_from: str = "body",
                max_age_bars: int = 500) -> pd.DataFrame:
    """
    The last opposite-direction candle before a displacement move.

    Bullish: an up-move of >= `displacement_atr` ATR completes; the zone is the
    last down-close candle before that move began. Optionally the move must also
    take out the prior `swing_lookback` bar extreme, which is the "break of
    structure" filter most descriptions of the pattern assume implicitly.
    """
    o, h, l, c = (f[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = f["atr"].to_numpy(float)
    idx = f.index
    tf = _tf_delta(idx)
    db = max(1, int(displacement_bars))
    n = len(f)
    rows = []

    for i in range(swing_lookback + db + 1, n):
        start = i - db + 1
        ref_atr = a[start - 1]
        if not np.isfinite(ref_atr) or ref_atr <= 0:
            continue
        move = c[i] - o[start]
        size = abs(move) / ref_atr
        if size < displacement_atr:
            continue
        bull = move > 0

        if require_swing_break:
            w = slice(max(0, start - swing_lookback), start)
            if bull and not h[i] > h[w].max():
                continue
            if not bull and not l[i] < l[w].min():
                continue

        # last opposite-close candle strictly before the move
        j = start - 1
        found = -1
        stop_at = max(0, start - swing_lookback - 5)
        while j >= stop_at:
            if (bull and c[j] < o[j]) or ((not bull) and c[j] > o[j]):
                found = j
                break
            j -= 1
        if found < 0:
            continue

        if zone_from == "wick":
            lo, hi = l[found], h[found]
        else:
            lo, hi = min(o[found], c[found]), max(o[found], c[found])
        if hi <= lo:
            continue

        rows.append(dict(
            formed_at=idx[i] + tf,          # known at the close of bar i
            zone_low=float(lo), zone_high=float(hi),
            bias=1 if bull else -1,
            zone_height_atr=float((hi - lo) / ref_atr),
            displacement_atr=float(size),
            source_ts=idx[found],
        ))
    return _finish(rows, f, max_age_bars, tf)


def fvg(f: pd.DataFrame, min_gap_atr: float = 0.25,
        max_age_bars: int = 500) -> pd.DataFrame:
    """Three-bar fair value gap: bar i's low above bar i-2's high, or inverse."""
    h, l = f["high"].to_numpy(float), f["low"].to_numpy(float)
    a = f["atr"].to_numpy(float)
    idx = f.index
    tf = _tf_delta(idx)
    rows = []
    for i in range(2, len(f)):
        ref = a[i]
        if not np.isfinite(ref) or ref <= 0:
            continue
        if l[i] > h[i - 2]:
            lo, hi, bias = h[i - 2], l[i], 1
        elif h[i] < l[i - 2]:
            lo, hi, bias = h[i], l[i - 2], -1
        else:
            continue
        if (hi - lo) / ref < min_gap_atr:
            continue
        rows.append(dict(
            formed_at=idx[i] + tf, zone_low=float(lo), zone_high=float(hi),
            bias=bias, zone_height_atr=float((hi - lo) / ref),
            displacement_atr=float((hi - lo) / ref), source_ts=idx[i],
        ))
    return _finish(rows, f, max_age_bars, tf)


def swing_level(f: pd.DataFrame, left: int = 5, right: int = 5,
                zone_pad_atr: float = 0.1,
                max_age_bars: int = 1000) -> pd.DataFrame:
    """Confirmed pivot highs/lows, padded into a zone by a fraction of ATR."""
    from .features import swing_points
    ph, pl = swing_points(f, left, right)
    h, l, a = (f[k].to_numpy(float) for k in ("high", "low", "atr"))
    idx = f.index
    tf = _tf_delta(idx)
    rows = []
    for i in np.flatnonzero(ph.to_numpy() | pl.to_numpy()):
        conf = i + right                    # pivot is only known `right` bars later
        if conf >= len(f):
            continue
        ref = a[conf]
        if not np.isfinite(ref) or ref <= 0:
            continue
        pad = zone_pad_atr * ref
        if ph.iloc[i]:
            lvl, bias = h[i], -1
        else:
            lvl, bias = l[i], 1
        rows.append(dict(
            formed_at=idx[conf] + tf,
            zone_low=float(lvl - pad), zone_high=float(lvl + pad),
            bias=bias, zone_height_atr=float(2 * zone_pad_atr),
            displacement_atr=np.nan, source_ts=idx[i],
        ))
    return _finish(rows, f, max_age_bars, tf)


def prior_session_level(f: pd.DataFrame, which: str = "high",
                        session: str = "RTH", zone_pad_atr: float = 0.1,
                        max_age_bars: int = 1440) -> pd.DataFrame:
    """Prior day's session high / low / close, as a padded zone."""
    idx = f.index
    tf = _tf_delta(idx)
    sub = f[f["session"] == session] if session else f
    if sub.empty:
        return _empty()
    daily = sub.groupby("_date").agg(
        high=("high", "max"), low=("low", "min"),
        close=("close", "last"), open=("open", "first"))
    if which not in daily.columns:
        raise ValueError(f"prior_session_level: which={which!r} not supported")

    # last bar of each day's session (works on every pandas >= 2.0)
    ends = pd.Series(sub.index, index=sub.index).groupby(sub["_date"]).max()
    a = f["atr"]
    rows = []
    dates = list(daily.index)
    for k in range(1, len(dates)):
        lvl = float(daily[which].iloc[k - 1])
        formed = ends.iloc[k - 1] + tf
        pos = a.index.searchsorted(formed)
        if pos >= len(a):
            continue
        # ATR of the session's last bar, which closed at `formed`; the bar at
        # `pos` is still forming then
        ref = float(a.iloc[pos - 1])
        if not np.isfinite(ref) or ref <= 0:
            continue
        pad = zone_pad_atr * ref
        bias = -1 if which in ("high",) else 1
        rows.append(dict(
            formed_at=formed, zone_low=lvl - pad, zone_high=lvl + pad,
            bias=bias, zone_height_atr=float(2 * zone_pad_atr),
            displacement_atr=np.nan, source_ts=ends.iloc[k - 1],
        ))
    return _finish(rows, f, max_age_bars, tf)


def round_number(f: pd.DataFrame, step: float = 100.0,
                 zone_pad_atr: float = 0.1,
                 max_age_bars: int = 1440) -> pd.DataFrame:
    """
    Round-number levels near each session's opening price. Mostly useful as a
    sanity check: if round numbers score as well as your order blocks, the
    order block is not doing the work.
    """
    idx = f.index
    tf = _tf_delta(idx)
    rows = []
    for date, day in f.groupby("_date"):
        px = float(day["open"].iloc[0])
        ref = float(day["atr"].iloc[0])
        if not np.isfinite(ref) or ref <= 0:
            continue
        pad = zone_pad_atr * ref
        base = np.floor(px / step) * step
        for k in (-2, -1, 0, 1, 2):
            lvl = base + k * step
            rows.append(dict(
                formed_at=day.index[0] + tf,
                zone_low=lvl - pad, zone_high=lvl + pad,
                bias=1 if lvl < px else -1,
                zone_height_atr=float(2 * zone_pad_atr),
                displacement_atr=np.nan, source_ts=day.index[0],
            ))
    return _finish(rows, f, max_age_bars, tf)


def _tf_delta(idx: pd.DatetimeIndex) -> pd.Timedelta:
    if len(idx) < 3:
        return pd.Timedelta("1min")
    d = pd.Series(idx).diff().dropna()
    return pd.Timedelta(d.mode().iloc[0]) if len(d) else pd.Timedelta("1min")


REGISTRY = {
    "order_block": order_block,
    "fvg": fvg,
    "swing_level": swing_level,
    "prior_session_level": prior_session_level,
    "round_number": round_number,
}


def detect(f: pd.DataFrame, detector: str, params: dict) -> pd.DataFrame:
    if detector not in REGISTRY:
        raise ValueError(f"unknown detector {detector!r}")
    return REGISTRY[detector](f, **params)
