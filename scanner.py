"""
The scanner: zones -> tests -> scored outcomes.

Two rules this module exists to enforce.

**No look-ahead.** A zone is only tested from `formed_at` onward, and
`formed_at` is when the confirmation completed, not when the source candle
printed. Scoring walks forward one M1 bar at a time and never reads a bar it
would not have had.

**Honest ambiguity.** When stop and target both fall inside the same M1 bar's
range, the data cannot say which came first. Those events are resolved
adversely and counted separately. If they are a large share of your sample, the
headline number is not measuring what you think it is, and the app will say so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import detectors, features
from .filters import evaluate, ExpressionError
from .spec import Spec

# Outcome codes
WIN, LOSS, TIMEOUT = "win", "loss", "timeout"


@dataclass
class ScanResult:
    events: pd.DataFrame
    n_zones: int
    n_tests_raw: int
    n_filtered_out: int
    warnings: list


def run(m1: pd.DataFrame, spec: Spec,
        exchange_tz: str = features.DEFAULT_EXCHANGE_TZ,
        progress=None) -> ScanResult:
    warnings: list[str] = []
    problems = spec.validate()
    if problems:
        raise ValueError("Spec is not runnable:\n  - " + "\n  - ".join(problems))

    from .data import resample
    tf = resample(m1, spec.timeframe)
    f = features.build(tf, exchange_tz=exchange_tz)

    # Higher-timeframe context, if the spec asks for it. Indexed by close time,
    # so the lookup below can only ever land on a bar that had finished.
    htf = None
    htf_ns = None
    if spec.htf.timeframe:
        htf = features.build_htf(m1, spec.htf.timeframe,
                                 spec.htf.ema_period, spec.htf.atr_period,
                                 spec.htf.slope_bars)
        if htf.empty:
            warnings.append(
                f"HTF timeframe {spec.htf.timeframe} produced no bars; "
                "htf_ filters will see neutral values.")
            htf = None
        else:
            htf_ns = htf.index.as_unit("ns").asi8

    zones = detectors.detect(f, spec.zone.detector, spec.zone.resolved())
    if zones.empty:
        return ScanResult(_empty_events(), 0, 0, 0,
                          ["No zones detected. Loosen the detector parameters."])

    if spec.direction == "long":
        zones = zones[zones["bias"] == 1]
    elif spec.direction == "short":
        zones = zones[zones["bias"] == -1]
    zones = zones.reset_index(drop=True)
    if zones.empty:
        return ScanResult(_empty_events(), 0, 0, 0,
                          [f"No zones with direction={spec.direction}."])

    # M1 arrays for the forward walk
    m1 = m1.sort_index()
    # Work in int64 epoch-nanoseconds throughout. Mixing tz-aware Timestamps
    # with numpy datetime64 is a reliable source of silent misalignment, and
    # pandas 3 defaults to microsecond resolution, so the unit is pinned
    # explicitly on both sides of every comparison rather than assumed.
    ts = m1.index.as_unit("ns").asi8
    o = m1["open"].to_numpy(float)
    h = m1["high"].to_numpy(float)
    l = m1["low"].to_numpy(float)
    c = m1["close"].to_numpy(float)
    n_m1 = len(m1)

    # detection-timeframe context, looked up by timestamp at event time
    ctx_idx = f.index.as_unit("ns").asi8
    ctx_cols = ["atr", "session", "hour", "minute", "dow", "month", "year",
                "dist_ema50_atr", "day_range_atr", "prior_day_range_atr",
                "gap_atr"]
    ctx = {k: f[k].to_numpy() for k in ctx_cols if k in f}

    htf_cols = {}
    htf_delta_ns = (pd.Timedelta(spec.htf.timeframe).as_unit("ns").value
                    if spec.htf.timeframe else 0)
    if htf is not None:
        for k in ("htf_bias", "htf_bar_dir", "htf_slope_atr",
                  "htf_dist_ema_atr", "htf_range_atr"):
            htf_cols[k] = htf[k].to_numpy(float)

    tf_delta = pd.Timedelta(spec.timeframe)
    min_gap = spec.trigger.min_bars_after_formation * tf_delta

    rows = []
    n_tests_raw = 0
    n_filtered = 0
    total = len(zones)

    for zi, z in enumerate(zones.itertuples(index=False)):
        if progress is not None and zi % 500 == 0:
            progress(zi / total)

        start_ts = (pd.Timestamp(z.formed_at) + min_gap).as_unit("ns").value
        end_ts = pd.Timestamp(z.expires_at).as_unit("ns").value
        a = int(np.searchsorted(ts, start_ts, "left"))
        b = int(np.searchsorted(ts, end_ts, "right"))
        if a >= n_m1 or b <= a:
            continue

        zl, zh = float(z.zone_low), float(z.zone_high)
        bias = int(z.bias)

        seg_h, seg_l, seg_c = h[a:b], l[a:b], c[a:b]
        if spec.trigger.type == "close_inside":
            hit = (seg_c >= zl) & (seg_c <= zh)
        elif spec.trigger.type == "wick_through":
            hit = (seg_l <= zl) if bias == 1 else (seg_h >= zh)
        else:  # touch
            hit = (seg_h >= zl) & (seg_l <= zh)

        hits = np.flatnonzero(hit)
        if hits.size == 0:
            continue
        if spec.trigger.first_test_only:
            hits = hits[:1]
        else:
            hits = _separate_tests(hits, seg_h, seg_l, zl, zh)

        for k, rel in enumerate(hits, start=1):
            n_tests_raw += 1
            i = a + int(rel)
            ci = int(np.searchsorted(ctx_idx, ts[i], "right")) - 1
            if ci < 0:
                continue
            atr_now = float(ctx["atr"][ci]) if "atr" in ctx else np.nan
            if not np.isfinite(atr_now) or atr_now <= 0:
                continue

            bars_since = ((ts[i] - pd.Timestamp(z.formed_at).as_unit("ns").value)
                          / tf_delta.as_unit("ns").value)

            env = {
                "hour": int(ctx["hour"][ci]), "minute": int(ctx["minute"][ci]),
                "dow": int(ctx["dow"][ci]), "month": int(ctx["month"][ci]),
                "year": int(ctx["year"][ci]),
                "session": str(ctx["session"][ci]),
                "atr": atr_now,
                "zone_height": float(z.zone_height),
                "zone_height_atr": float(z.zone_height_atr),
                "bars_since_zone": float(bars_since),
                "test_number": float(k),
                "displacement_atr": float(z.displacement_atr)
                if np.isfinite(z.displacement_atr) else 0.0,
                "bias": float(bias),
                "dist_ema50_atr": _f(ctx, "dist_ema50_atr", ci),
                "day_range_atr": _f(ctx, "day_range_atr", ci),
                "prior_day_range_atr": _f(ctx, "prior_day_range_atr", ci),
                "gap_atr": _f(ctx, "gap_atr", ci),
            }
            env.update(_htf_env(htf_ns, htf_cols, ts[i], bias, htf_delta_ns))

            keep = True
            for expr in spec.filters:
                try:
                    if not evaluate(expr, env):
                        keep = False
                        break
                except ExpressionError as e:
                    raise ValueError(f"Filter {expr!r} failed: {e}") from None
                except Exception:
                    keep = False
                    break
            if not keep:
                n_filtered += 1
                continue

            scored = score_event(i, bias, zl, zh, atr_now, spec,
                                 o, h, l, c, n_m1)
            if scored is None:
                continue
            rows.append({
                "ts": pd.Timestamp(ts[i], tz="UTC"),
                "formed_at": pd.Timestamp(z.formed_at),
                "bias": bias, "zone_low": zl, "zone_high": zh,
                "test_number": k, **env, **scored,
            })

    if progress is not None:
        progress(1.0)

    ev = pd.DataFrame(rows) if rows else _empty_events()
    if len(ev):
        ev = ev.sort_values("ts").reset_index(drop=True)
        amb = float(ev["ambiguous"].mean())
        if amb > 0.15:
            warnings.append(
                f"{amb:.0%} of events had stop and target inside the same M1 "
                "bar. They were resolved as losses. At this rate the true "
                "number could be materially higher — widen the target, tighten "
                "the stop, or accept that M1 cannot settle this idea."
            )
    return ScanResult(ev, len(zones), n_tests_raw, n_filtered, warnings)


def _htf_env(htf_ns, htf_cols, ts_i, bias, htf_delta_ns) -> dict:
    """
    Values from the last higher-timeframe bar that had CLOSED strictly before
    this event. Returns neutral values when no HTF is configured, so a filter
    referencing htf_ names never explodes mid-scan.
    """
    neutral = {"htf_bias": 0.0, "htf_bar_dir": 0.0, "htf_slope_atr": 0.0,
               "htf_dist_ema_atr": 0.0, "htf_range_atr": 0.0,
               "htf_bar_progress": 0.0, "aligned": 0.0, "aligned_bar": 0.0}
    if htf_ns is None or not len(htf_ns):
        return neutral

    # "left" not "right": a bar closing at exactly this timestamp is not yet
    # usable information at this timestamp.
    j = int(np.searchsorted(htf_ns, ts_i, "left")) - 1
    if j < 0:
        return neutral

    out = {}
    for k, arr in htf_cols.items():
        v = float(arr[j])
        out[k] = v if np.isfinite(v) else 0.0
    # How far into the currently-forming HTF bar this event sits. Normally 0 to
    # 1; larger after a weekend or holiday, because the last closed HTF bar is
    # then hours or days old. Left uncapped on purpose — a stale HTF reference
    # is worth being able to filter on rather than something to paper over.
    out["htf_bar_progress"] = (float(ts_i - htf_ns[j]) / htf_delta_ns
                               if htf_delta_ns else 0.0)
    out["aligned"] = 1.0 if out.get("htf_bias", 0.0) == float(bias) else 0.0
    out["aligned_bar"] = 1.0 if out.get("htf_bar_dir", 0.0) == float(bias) else 0.0
    return {**neutral, **out}


def _f(ctx, key, i):
    if key not in ctx:
        return 0.0
    v = float(ctx[key][i])
    return v if np.isfinite(v) else 0.0


def _separate_tests(hits, seg_h, seg_l, zl, zh):
    """
    Collapse consecutive bars inside the zone into one test. A new test only
    counts once price has fully left the zone and come back.
    """
    out = []
    inside = False
    for r in range(len(seg_h)):
        touching = (seg_h[r] >= zl) and (seg_l[r] <= zh)
        if touching and not inside:
            out.append(r)
            inside = True
        elif not touching:
            inside = False
    return np.array(out, dtype=int)


def score_event(i, bias, zl, zh, atr_now, spec, o, h, l, c, n_m1):
    """Walk M1 forward from the test bar and resolve the trade."""
    sc = spec.scoring
    if sc.entry == "next_open":
        i = i + 1
        if i >= n_m1:
            return None
        entry = float(o[i])
    else:
        entry = zh if bias == 1 else zl          # proximal edge, limit fill

    height = zh - zl
    if sc.stop.mode == "zone_far_edge":
        far = zl if bias == 1 else zh
        stop = far - (sc.stop.value - 1.0) * height * bias
    elif sc.stop.mode == "atr":
        stop = entry - bias * sc.stop.value * atr_now
    else:
        stop = entry - bias * sc.stop.value

    risk = (entry - stop) * bias
    if not np.isfinite(risk) or risk <= 0:
        return None

    if sc.target.mode == "none":
        target = np.inf * bias
    elif sc.target.mode == "r_multiple":
        target = entry + bias * sc.target.value * risk
    elif sc.target.mode == "atr":
        target = entry + bias * sc.target.value * atr_now
    else:
        target = entry + bias * sc.target.value

    end = min(i + sc.max_holding_bars, n_m1)
    mfe = mae = 0.0
    outcome, ambiguous = TIMEOUT, False
    exit_px, bars_held = float(c[end - 1]), end - i
    resolved = False

    # MFE/MAE are tracked across the FULL horizon, not just up to the exit.
    # Truncating them at the stop would make the alternative scoring lenses
    # measure the exit rule instead of the market's response, which is the
    # opposite of what they are for.
    for j in range(i, end):
        hi, lo = h[j], l[j]
        fav = ((hi - entry) if bias == 1 else (entry - lo)) / risk
        adv = ((entry - lo) if bias == 1 else (hi - entry)) / risk
        mfe = max(mfe, fav)
        mae = max(mae, adv)

        if resolved:
            continue

        hit_stop = (lo <= stop) if bias == 1 else (hi >= stop)
        hit_tgt = (np.isfinite(target) and
                   ((hi >= target) if bias == 1 else (lo <= target)))

        if hit_stop and hit_tgt:
            ambiguous = True
            outcome, exit_px, bars_held, resolved = LOSS, float(stop), j - i + 1, True
        elif hit_stop:
            outcome, exit_px, bars_held, resolved = LOSS, float(stop), j - i + 1, True
        elif hit_tgt:
            outcome, exit_px, bars_held, resolved = WIN, float(target), j - i + 1, True

    r_fixed = (exit_px - entry) * bias / risk
    if outcome == TIMEOUT:
        r_fixed = (float(c[end - 1]) - entry) * bias / risk

    return {
        "entry": float(entry), "stop": float(stop),
        "target": float(target) if np.isfinite(target) else np.nan,
        "risk_points": float(risk),
        "outcome": outcome, "ambiguous": bool(ambiguous),
        "r_fixed": float(r_fixed),
        "mfe_r": float(mfe), "mae_r": float(mae),
        "bars_held": int(bars_held),
        "atr_at_entry": float(atr_now),
    }


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "ts", "formed_at", "bias", "zone_low", "zone_high", "test_number",
        "session", "hour", "year", "entry", "stop", "target", "outcome",
        "ambiguous", "r_fixed", "mfe_r", "mae_r", "bars_held",
    ])


# ------------------------------------------------------------------ scoring modes

def rescore(events: pd.DataFrame, mode: str, target_r: float = 2.0) -> pd.Series:
    """
    Re-derive win/loss under a different scoring lens without rescanning.

    'fixed'   the target/stop the spec was scanned with
    'atr'     did MFE reach `target_r` ATR before MAE reached 1 ATR
    'mfe'     did MFE exceed MAE at all (pure directional response)
    """
    if events.empty:
        return pd.Series(dtype=bool)
    if mode == "fixed":
        return events["outcome"].eq(WIN)
    if mode == "mfe":
        return events["mfe_r"] > events["mae_r"]
    if mode == "atr":
        scale = events["risk_points"] / events["atr_at_entry"]
        return (events["mfe_r"] * scale >= target_r) & \
               (events["mae_r"] * scale < 1.0)
    raise ValueError(f"unknown scoring mode {mode!r}")
