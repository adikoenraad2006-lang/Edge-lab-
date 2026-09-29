"""
Self-test / calibration.

The important test is the first one. On a driftless random walk, a setup with a
1R stop and a 2R target must come out near 33%, because that is what geometry
alone produces. If Edge Lab reports 60% on noise, the engine is reading the
future somewhere, and every result it ever gives you is worthless.

Run:  python selftest.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from edgelab import detectors, features, mql5, scanner, stats
from edgelab.data import resample, data_fingerprint, sniff, load_csv
from edgelab.filters import evaluate, check_expression, ExpressionError
from edgelab.library import Library
from edgelab.spec import Spec, Zone, Trigger, Scoring, Stop, Target, Htf

FAILURES = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def synth(n_days=400, drift=0.0, seed=7, start_px=15000.0, vol=1.2):
    """Minute bars, 24x5, from a random walk. No structure to find by design."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-05", periods=n_days * 1440, freq="1min", tz="UTC")
    idx = idx[idx.dayofweek < 5]
    steps = rng.normal(drift, vol, len(idx))
    close = start_px + np.cumsum(steps)
    spread = np.abs(rng.normal(0, vol, len(idx)))
    high = close + spread * rng.uniform(0.2, 1.0, len(idx))
    low = close - spread * rng.uniform(0.2, 1.0, len(idx))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": 1.0}, index=idx)


def base_spec(**kw):
    d = dict(
        name="calibration", timeframe="5min", direction="both",
        zone=Zone("order_block", {"displacement_atr": 1.2,
                                  "require_swing_break": True,
                                  "max_age_bars": 200}),
        trigger=Trigger("touch", first_test_only=True),
        filters=[],
        scoring=Scoring(stop=Stop("zone_far_edge", 1.0),
                        target=Target("r_multiple", 2.0),
                        max_holding_bars=240),
    )
    d.update(kw)
    return Spec(**d)


def main():
    print("\n1. Calibration on a driftless random walk")
    m1 = synth()
    print(f"   {len(m1):,} M1 bars, "
          f"{m1.index[0]:%Y-%m-%d} to {m1.index[-1]:%Y-%m-%d}")

    res = scanner.run(m1, base_spec())
    ev = res.events
    check("zones detected", res.n_zones > 100, f"{res.n_zones} zones")
    check("events scored", len(ev) > 200, f"{len(ev)} events")

    wins = scanner.rescore(ev, "fixed")
    p, lo, hi = stats.wilson(int(wins.sum()), len(ev))
    print(f"   win rate {p:.1%}  (95% CI {lo:.1%}–{hi:.1%})")
    check("2R/1R on noise lands near 1/3", 0.22 <= p <= 0.45,
          f"got {p:.1%}, expected ~33%")

    mean_r = ev["r_fixed"].mean()
    print(f"   mean R {mean_r:+.3f}")
    check("expectancy on noise is near zero", abs(mean_r) < 0.25,
          f"got {mean_r:+.3f}")

    print("\n2. Geometry responds to the target multiple")
    for tmult, lo_b, hi_b in ((1.0, 0.40, 0.62), (3.0, 0.14, 0.36)):
        r = scanner.run(m1, base_spec(
            scoring=Scoring(stop=Stop("zone_far_edge", 1.0),
                            target=Target("r_multiple", tmult),
                            max_holding_bars=240)))
        w = scanner.rescore(r.events, "fixed")
        rate = w.mean() if len(r.events) else np.nan
        print(f"   target {tmult}R -> {rate:.1%} over {len(r.events)} events")
        check(f"{tmult}R rate in plausible band", lo_b <= rate <= hi_b,
              f"got {rate:.1%}")

    print("\n3. No look-ahead: every test is after the zone was confirmed")
    late = (pd.to_datetime(ev["ts"]) <= pd.to_datetime(ev["formed_at"])).sum()
    check("no event precedes its zone's formation", late == 0, f"{late} violations")

    print("\n4. Drift baseline separates a trending market from a flat one")
    up = synth(drift=0.045, seed=11)
    r_up = scanner.run(up, base_spec(direction="long"))
    if len(r_up.events) > 50:
        b = stats.drift_baseline(up, r_up.events, base_spec(direction="long"),
                                 n_samples=600)
        obs = scanner.rescore(r_up.events, "fixed").mean()
        print(f"   trending market: pattern {obs:.1%} vs drift baseline "
              f"{b['rate']:.1%} (n={b['n']})")
        check("drift baseline is measurable", np.isfinite(b["rate"]))
        check("baseline picks up the trend", b["rate"] > 0.30,
              f"got {b['rate']:.1%}")
    else:
        check("trending scan produced events", False, f"{len(r_up.events)}")

    print("\n5. Placebo baseline runs and is comparable")
    pb = stats.placebo_baseline(m1, ev, base_spec())
    print(f"   placebo {pb['rate']:.1%} (n={pb['n']}) vs pattern {p:.1%}")
    check("placebo baseline produced a rate", np.isfinite(pb["rate"]))
    check("on noise, placebo and pattern are close", abs(pb["rate"] - p) < 0.10,
          f"gap {abs(pb['rate'] - p):.1%}")

    print("\n6. Filters")
    check("valid expression accepted", check_expression("session == 'RTH'") is None)
    check("unknown name rejected",
          check_expression("nonsense > 1") is not None)
    blocked = check_expression("__import__('os').system('echo hi')")
    check("code execution blocked", blocked is not None, str(blocked)[:60])
    r_f = scanner.run(m1, base_spec(filters=["session == 'RTH'"]))
    check("filter reduces the sample",
          0 < len(r_f.events) < len(ev),
          f"{len(r_f.events)} of {len(ev)}")
    if len(r_f.events):
        check("filter actually applied",
              set(r_f.events["session"].unique()) == {"RTH"},
              str(set(r_f.events["session"].unique())))

    print("\n7. Higher-timeframe bias filter")
    htf_spec = base_spec(htf=Htf(timeframe="4h", ema_period=50))
    r_htf = scanner.run(m1, htf_spec)
    e2 = r_htf.events
    check("htf scan produced events", len(e2) > 100, f"{len(e2)}")
    check("htf columns present on events",
          all(c in e2 for c in ("htf_bias", "aligned", "htf_bar_progress")))
    check("htf_bias is populated, not neutral",
          set(e2["htf_bias"].unique()) <= {-1.0, 1.0, 0.0}
          and (e2["htf_bias"] != 0).mean() > 0.95,
          f"neutral share {(e2['htf_bias'] == 0).mean():.1%}")
    check("bar progress is non-negative",
          (e2["htf_bar_progress"] >= 0).all())
    check("bar progress is within one bar outside of gaps",
          e2["htf_bar_progress"].between(0, 1).mean() > 0.85,
          f"{e2['htf_bar_progress'].between(0, 1).mean():.1%} within one bar")
    check("gaps are visible rather than clipped",
          e2["htf_bar_progress"].max() > 1.0,
          f"max {e2['htf_bar_progress'].max():.1f} (weekend gap)")

    # THE look-ahead test: recompute htf_bias independently from only those 4h
    # bars that had closed before each event, and demand an exact match.
    htf = features.build_htf(m1, "4h", 50, 14, 3)
    htf_ns = htf.index.as_unit("ns").asi8
    bias_arr = htf["htf_bias"].to_numpy(float)
    mismatches = 0
    for e in e2.itertuples(index=False):
        t = pd.Timestamp(e.ts).as_unit("ns").value
        j = int(np.searchsorted(htf_ns, t, "left")) - 1
        expected = bias_arr[j] if j >= 0 else 0.0
        if float(e.htf_bias) != float(expected):
            mismatches += 1
    check("htf_bias only ever uses closed bars", mismatches == 0,
          f"{mismatches} events read an unclosed bar")

    # Every referenced HTF bar must have closed strictly before the event.
    late = 0
    for e in e2.itertuples(index=False):
        t = pd.Timestamp(e.ts).as_unit("ns").value
        j = int(np.searchsorted(htf_ns, t, "left")) - 1
        if j >= 0 and htf_ns[j] > t:
            late += 1
    check("no HTF bar closes after its event", late == 0, f"{late} violations")

    r_al = scanner.run(m1, base_spec(htf=Htf(timeframe="4h"),
                                     filters=["aligned == 1"]))
    check("aligned filter reduces the sample",
          0 < len(r_al.events) < len(e2), f"{len(r_al.events)} of {len(e2)}")
    if len(r_al.events):
        check("aligned filter kept only aligned events",
              (r_al.events["aligned"] == 1).all())
        check("kept events genuinely match HTF direction",
              (r_al.events["bias"] == r_al.events["htf_bias"]).all())

    bad_htf = base_spec(timeframe="5min", htf=Htf(timeframe="1min"))
    check("HTF shorter than base timeframe is rejected",
          len(bad_htf.validate()) > 0, str(bad_htf.validate())[:70])
    orphan = base_spec(filters=["aligned == 1"])
    check("htf filter without htf.timeframe is rejected",
          len(orphan.validate()) > 0)

    print("\n8. Other detectors run")
    tf = resample(m1, "5min")
    f = features.build(tf)
    for name, params in (("fvg", {}), ("swing_level", {}),
                         ("prior_session_level", {}), ("round_number", {})):
        try:
            z = detectors.detect(f, name, params)
            check(f"{name} produced zones", len(z) > 0, f"{len(z)} zones")
        except Exception as e:
            check(f"{name} ran", False, f"{type(e).__name__}: {e}")

    print("\n9. Rescoring lenses")
    rates = {}
    for mode in ("fixed", "mfe", "atr"):
        w = scanner.rescore(ev, mode, 2.0)
        rates[mode] = float(w.mean())
        check(f"scoring mode {mode}", len(w) == len(ev) and w.dtype == bool,
              f"{w.mean():.1%}")
    # MFE/MAE must span the full horizon, not stop at the exit. If they were
    # truncated, the 'mfe' lens would collapse toward the fixed rate.
    check("mfe lens is not truncated by the fixed exit",
          rates["mfe"] > rates["fixed"] + 0.10,
          f"mfe {rates['mfe']:.1%} vs fixed {rates['fixed']:.1%}")
    check("atr lens produces a usable rate on noise",
          0.05 < rates["atr"] < 0.60, f"{rates['atr']:.1%}")
    long_mfe = float(ev["mfe_r"].max())
    check("some events show excursion beyond the target",
          long_mfe > 2.5, f"max MFE {long_mfe:.1f}R")

    print("\n10. IS/OOS split")
    mid = ev["ts"].iloc[len(ev) // 2]
    a, b2 = stats.split_is_oos(ev, ev["ts"].min(), mid)
    check("split covers the sample", len(a) + len(b2) == len(ev),
          f"{len(a)} + {len(b2)} = {len(ev)}")
    check("holdout is non-empty", len(b2) > 0)

    print("\n11. CSV round trip and timezone handling")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "TEST_2015.csv"
        out = m1.head(5000).copy()
        out.index = out.index.tz_convert("Europe/Amsterdam").tz_localize(None)
        out.to_csv(p, index_label="datetime")
        fmt = sniff(p)
        check("sniffer found the delimiter", fmt.delimiter == ",")
        check("sniffer mapped OHLC",
              all(k in fmt.columns for k in ("open", "high", "low", "close")))
        back = load_csv(p, tz="Europe/Amsterdam")
        check("round trip preserves bar count", len(back) == 5000, str(len(back)))
        diff = abs(back["close"].iloc[0] - m1["close"].iloc[0])
        check("round trip preserves prices", diff < 1e-6, f"diff {diff}")
        check("index is UTC", str(back.index.tz) == "UTC", str(back.index.tz))

        semi = Path(td) / "SEMI_2015.csv"
        semi.write_text(
            "Date;Time;Open;High;Low;Close;Volume\n"
            "2015.01.05;09:00;15000,25;15002,50;14999,00;15001,00;10\n"
            "2015.01.05;09:01;15001,00;15003,00;15000,00;15002,25;12\n")
        f2 = sniff(semi)
        check("European format sniffed", f2.decimal == "," and f2.delimiter == ";",
              f"decimal={f2.decimal!r} delim={f2.delimiter!r}")
        d2 = load_csv(semi, tz="Europe/Amsterdam")
        check("European decimals parsed",
              len(d2) == 2 and abs(d2["open"].iloc[0] - 15000.25) < 1e-9,
              str(d2["open"].tolist() if len(d2) else "empty"))

    print("\n11b. Broker export formats")
    EXPORTS = {
        "MT5 tab + bracket headers":
            "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
            "2015.01.05\t01:00:00\t15000.25\t15002.50\t14999.00\t15001.00\t120\t0\t2\n"
            "2015.01.05\t01:01:00\t15001.00\t15003.00\t15000.00\t15002.25\t98\t0\t2\n",
        "MT5 comma + bracket headers":
            "<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,<VOL>,<SPREAD>\n"
            "2015.01.05,01:00:00,15000.25,15002.50,14999.00,15001.00,120,0,2\n"
            "2015.01.05,01:01:00,15001.00,15003.00,15000.00,15002.25,98,0,2\n",
        "MT5 combined DATETIME":
            "<DATETIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\n"
            "2015.01.05 01:00:00\t15000.25\t15002.50\t14999.00\t15001.00\t120\n"
            "2015.01.05 01:01:00\t15001.00\t15003.00\t15000.00\t15002.25\t98\n",
        "MT4 headerless":
            "2015.01.05,01:00,15000.25,15002.50,14999.00,15001.00,120\n"
            "2015.01.05,01:01,15001.00,15003.00,15000.00,15002.25,98\n",
        "Tickstory generic":
            "DateTime,Open,High,Low,Close,Volume\n"
            "2015-01-05 01:00:00,15000.25,15002.50,14999.00,15001.00,120\n"
            "2015-01-05 01:01:00,15001.00,15003.00,15000.00,15002.25,98\n",
        "European decimals":
            "Date;Time;Open;High;Low;Close;Volume\n"
            "05.01.2015;01:00;15000,25;15002,50;14999,00;15001,00;120\n"
            "05.01.2015;01:01;15001,00;15003,00;15000,00;15002,25;98\n",
        "unknown header names":
            "Bar;T;O1;H1;L1;C1\n"
            "2015-01-05;01:00;15000.25;15002.50;14999.00;15001.00\n"
            "2015-01-05;01:01;15001.00;15003.00;15000.00;15002.25\n",
    }
    with tempfile.TemporaryDirectory() as td:
        for label, content in EXPORTS.items():
            fp = Path(td) / f"{abs(hash(label))}.csv"
            fp.write_text(content)
            try:
                d = load_csv(fp, tz="Europe/Amsterdam")
                ok = len(d) == 2 and abs(d["open"].iloc[0] - 15000.25) < 1e-9
                check(f"parses {label}", ok,
                      f"{len(d)} rows, open={d['open'].iloc[0] if len(d) else None}")
            except Exception as e:
                check(f"parses {label}", False, f"{type(e).__name__}: {e}")

        bad = Path(td) / "bad.csv"
        bad.write_text("alpha,beta\n1,2\n")
        try:
            load_csv(bad)
            check("unparseable file raises", False, "no error raised")
        except ValueError as e:
            check("unparseable file raises a useful error",
                  "header row" in str(e) or "need at least 5" in str(e),
                  str(e)[:60])

        amb = Path(td) / "amb.csv"
        amb.write_text("Date,Time,Open,High,Low,Close\n"
                       "05/01/2015,01:00,1,2,0.5,1.5\n"
                       "06/01/2015,01:01,1,2,0.5,1.5\n")
        check("day-first/month-first ambiguity is flagged",
              sniff(amb).ambiguous_date)

    print("\n12. Library, reveal ledger and trial registry")
    with tempfile.TemporaryDirectory() as td:
        lib = Library(td)
        sp = base_spec()
        summ = stats.summarise(ev, wins)
        sid = lib.save_study(sp, summ, ev, "TEST", data_fingerprint(m1),
                             "2015-01-01", "2018-01-01")
        check("study saved", sid > 0)
        check("study listed", len(lib.list_studies()) == 1)
        check("events reloaded", len(lib.load_events(sid)) == len(ev))
        check("trial registry written", lib.trial_count() == 1)
        fam = "TEST|order_block|both"
        check("reveal count starts at zero", lib.reveal_count(fam) == 0)
        lib.log_reveal(fam, sp.fingerprint(), summ)
        lib.log_reveal(fam, sp.fingerprint(), summ)
        check("reveals accumulate", lib.reveal_count(fam) == 2)

    print("\n13. Spec identity and round trip")
    s1, s2 = base_spec(), base_spec(name="different name", notes="x")
    check("renaming does not change the fingerprint",
          s1.fingerprint() == s2.fingerprint())
    s3 = base_spec(zone=Zone("order_block", {"displacement_atr": 2.0}))
    check("changing a rule changes the fingerprint",
          s1.fingerprint() != s3.fingerprint())
    check("json round trip", Spec.from_json(s1.to_json()).fingerprint()
          == s1.fingerprint())
    bad = base_spec(zone=Zone("order_block", {"not_a_param": 1}))
    check("unknown parameter rejected", len(bad.validate()) > 0,
          str(bad.validate()))
    bad2 = base_spec(filters=["session = 'RTH'"])
    check("bad filter caught by validate", len(bad2.validate()) > 0)

    print("\n14. MQL5 export")
    code = mql5.export(base_spec(), stats.summarise(ev, wins))
    check("export produced code", len(code) > 1500, f"{len(code)} chars")
    check("carries the scaffolding warning", "NOT A FINISHED EA" in code)
    check("has an OnTick", "void OnTick()" in code)
    check("has the detector", "DetectOrderBlock" in code)
    other = mql5.export(base_spec(zone=Zone("fvg", {})))
    check("unsupported detector becomes an explicit stub",
          "TODO" in other and "fvg" in other)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f_ in FAILURES:
            print("   -", f_)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
