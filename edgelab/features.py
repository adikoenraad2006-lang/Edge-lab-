"""
Features available to detectors and filter expressions.

Sessions are derived in exchange local time, not UTC, so the boundaries stay
correct across DST changes on both sides of the Atlantic. Getting this wrong is
the single most common way a session study produces a confident wrong answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_EXCHANGE_TZ = "America/New_York"

# (name, start_hhmm, end_hhmm) in exchange local time, evaluated in order.
SESSION_WINDOWS = [
    ("ASIA",   (18, 0), (3, 0)),
    ("LONDON", (3, 0), (9, 30)),
    ("RTH",    (9, 30), (16, 0)),
    ("AFTER",  (16, 0), (18, 0)),
]


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def sessions(idx: pd.DatetimeIndex, tz: str = DEFAULT_EXCHANGE_TZ) -> pd.Series:
    local = idx.tz_convert(tz)
    mins = local.hour * 60 + local.minute
    out = pd.Series("AFTER", index=idx, dtype=object)
    for name, (sh, sm), (eh, em) in SESSION_WINDOWS:
        s, e = sh * 60 + sm, eh * 60 + em
        mask = (mins >= s) & (mins < e) if s < e else (mins >= s) | (mins < e)
        out[mask] = name
    out[local.dayofweek >= 5] = "WEEKEND"
    return out


def swing_points(df: pd.DataFrame, left: int = 5, right: int = 5):
    """
    Confirmed pivots. `right` bars must pass before a pivot is known, so both
    series are shifted to the bar at which the pivot became visible.
    """
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    ph = np.zeros(n, bool)
    pl = np.zeros(n, bool)
    for i in range(left, n - right):
        w_h = h[i - left:i + right + 1]
        w_l = l[i - left:i + right + 1]
        if h[i] == w_h.max() and (w_h == h[i]).sum() == 1:
            ph[i] = True
        if l[i] == w_l.min() and (w_l == l[i]).sum() == 1:
            pl[i] = True
    return pd.Series(ph, index=df.index), pd.Series(pl, index=df.index)


def build(df: pd.DataFrame, exchange_tz: str = DEFAULT_EXCHANGE_TZ,
          atr_n: int = 14) -> pd.DataFrame:
    """Attach the standard feature set to a detection-timeframe frame."""
    out = df.copy()
    out["atr"] = atr(df, atr_n)
    out["session"] = sessions(df.index, exchange_tz)

    local = df.index.tz_convert(exchange_tz)
    out["hour"] = local.hour
    out["minute"] = local.minute
    out["dow"] = local.dayofweek
    out["month"] = local.month
    out["year"] = local.year
    out["_date"] = local.date

    out["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dist_ema50_atr"] = (out["close"] - out["ema50"]) / out["atr"]

    g = out.groupby("_date")
    day_hi = g["high"].cummax()
    day_lo = g["low"].cummin()
    out["day_range_atr"] = (day_hi - day_lo) / out["atr"]

    daily = out.groupby("_date").agg(
        d_high=("high", "max"), d_low=("low", "min"),
        d_close=("close", "last"), d_open=("open", "first"))
    daily["prior_range"] = (daily["d_high"] - daily["d_low"]).shift(1)
    daily["prior_close"] = daily["d_close"].shift(1)
    daily["prior_high"] = daily["d_high"].shift(1)
    daily["prior_low"] = daily["d_low"].shift(1)

    mapped = out["_date"].map(daily["prior_range"])
    out["prior_day_range"] = mapped.to_numpy()
    out["prior_day_range_atr"] = mapped.to_numpy() / out["atr"].to_numpy()
    pc = out["_date"].map(daily["prior_close"]).to_numpy()
    d_open = out["_date"].map(daily["d_open"]).to_numpy()
    out["gap_points"] = d_open - pc
    out["gap_atr"] = (d_open - pc) / out["atr"].to_numpy()
    out["prior_day_high"] = out["_date"].map(daily["prior_high"]).to_numpy()
    out["prior_day_low"] = out["_date"].map(daily["prior_low"]).to_numpy()
    out["prior_day_close"] = pc

    return out


def build_htf(m1: pd.DataFrame, timeframe: str, ema_period: int = 50,
              atr_period: int = 14, slope_bars: int = 3) -> pd.DataFrame:
    """
    Higher-timeframe context, indexed by the moment each bar CLOSED.

    The index is deliberately the close time rather than the open time. A 4h
    bar starting at 08:00 tells you nothing at 09:00 — its direction is not
    settled until 12:00. Indexing by close time means a plain "last row at or
    before now" lookup can never read a bar that had not finished.
    """
    from .data import resample
    htf = resample(m1, timeframe)
    if htf.empty:
        return htf

    out = pd.DataFrame(index=htf.index)
    out["htf_close"] = htf["close"]
    out["htf_open"] = htf["open"]
    out["htf_ema"] = htf["close"].ewm(span=ema_period, adjust=False).mean()
    out["htf_atr"] = atr(htf, atr_period)
    out["htf_range"] = htf["high"] - htf["low"]

    with np.errstate(divide="ignore", invalid="ignore"):
        out["htf_dist_ema_atr"] = (out["htf_close"] - out["htf_ema"]) / out["htf_atr"]
        out["htf_slope_atr"] = (out["htf_ema"].diff(slope_bars) / out["htf_atr"])
        out["htf_range_atr"] = out["htf_range"] / out["htf_atr"]

    out["htf_bias"] = np.where(out["htf_close"] >= out["htf_ema"], 1, -1)
    out["htf_bar_dir"] = np.where(htf["close"] >= htf["open"], 1, -1)

    # Re-index by close time: the bar labelled 08:00 becomes known at 12:00.
    delta = pd.Timedelta(timeframe)
    out.index = out.index + delta
    out.index.name = "closed_at"
    return out
