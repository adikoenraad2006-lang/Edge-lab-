"""
Edge Lab — run with:  streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from edgelab import charts, data as dataio, mql5, nl, scanner, stats
from edgelab.filters import FEATURE_HELP
from edgelab.library import Library, _family_of, deflated_threshold
from edgelab.spec import Spec, EXAMPLE_SPEC, DETECTORS

st.set_page_config(page_title="Edge Lab", page_icon="◧", layout="wide")

DATE_FLOOR = date(1990, 1, 1)

TZ_CHOICES = ["Europe/Amsterdam", "UTC", "America/New_York", "Europe/London",
              "Etc/GMT-2", "Etc/GMT-3", "Asia/Tokyo", "Australia/Sydney"]

S = st.session_state
S.setdefault("spec", EXAMPLE_SPEC)
S.setdefault("spec_text", EXAMPLE_SPEC.to_json())
S.setdefault("events", None)
S.setdefault("scan_meta", None)
S.setdefault("m1", None)
S.setdefault("baselines", {})
S.setdefault("oos_unlocked", False)


def set_spec(sp: Spec, text: str) -> None:
    """
    Make `sp` the current spec. If its rules differ from the spec the current
    results came from, drop those results: baselines, saving and the holdout
    would otherwise score the new rules against the old scan.
    """
    old = S.get("spec")
    S["spec"], S["spec_text"] = sp, text
    if old is None or old.fingerprint() != sp.fingerprint():
        S["events"] = None
        S["scan_meta"] = None
        S["baselines"] = {}
        S["oos_unlocked"] = False
        S["oos_events"] = None


@st.cache_resource
def get_library(root: str) -> Library:
    return Library(root)


@st.cache_resource(show_spinner=False, max_entries=2)
def load_symbol(folder: str, symbol: str, tz: str, years) -> pd.DataFrame:
    """
    cache_resource, not cache_data: cache_data serialises a copy of the frame,
    which doubles memory for something this size. The frame is only ever read,
    so sharing the object is safe. max_entries=2 stops a session that has
    loaded six symbols from holding all six.
    """
    return dataio.load_folder_symbol(folder, symbol, tz=tz, years=years)


# ============================================================== sidebar
with st.sidebar:
    st.markdown("### Edge Lab")
    lib_root = st.text_input("Library folder", "~/.edgelab")
    lib = get_library(lib_root)

    st.markdown("#### Data")
    folder = st.text_input("CSV folder", "", placeholder="C:/data/m1")
    symbols = []
    if folder and Path(folder).expanduser().exists():
        try:
            symbols = dataio.discover_symbols(Path(folder).expanduser())
        except Exception as e:
            st.error(str(e))
    symbol = st.selectbox("Symbol", symbols) if symbols else st.text_input("Symbol", "")
    tz = st.selectbox("Timestamps are in", TZ_CHOICES, index=0,
                      help="The timezone of the raw CSV timestamps. Wrong value "
                           "here silently ruins every session-based result.")
    exch_tz = st.selectbox("Exchange timezone (sessions)",
                           ["America/New_York", "Europe/London", "UTC",
                            "Asia/Tokyo"], index=0)

    if st.button("Preview format", use_container_width=True):
        try:
            fp = Path(folder).expanduser()
            files = sorted(x for x in fp.glob("*")
                           if x.suffix.lower() in (".csv", ".txt")
                           and x.stem.upper().startswith((symbol or "").upper()))
            if not files:
                st.error(f"No CSV/TXT files starting with {symbol!r} in {fp}")
            else:
                f0 = files[0]
                raw = "".join(open(f0, encoding="utf-8-sig",
                                   errors="replace").readlines()[:3])
                st.code(raw, language=None)
                fm = dataio.sniff(f0)
                st.json({"file": f0.name, "files matched": len(files),
                         "delimiter": repr(fm.delimiter),
                         "decimal": fm.decimal,
                         "header": fm.header,
                         "timestamp format": fm.datetime_format or "(inferred)",
                         "columns": fm.columns,
                         "date+time columns": [fm.date_col, fm.time_col]})
                if fm.ambiguous_date:
                    st.warning("This date format is ambiguous — 05/01 could be "
                               "5 January or 1 May. Day-first is assumed. If "
                               "that is wrong, every result will be wrong.")
                else:
                    st.success("Format detected. Load data should work.")
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")

    yr_lo, yr_hi = st.select_slider(
        "Years to load", options=list(range(2005, date.today().year + 1)),
        value=(2015, date.today().year),
        help="Loading fewer years uses proportionally less memory. Files whose "
             "names contain a year outside this range are skipped before "
             "parsing.")

    if folder and symbol:
        try:
            est = dataio.estimate_memory(Path(folder).expanduser(), symbol)
        except Exception:
            est = {}
        if est:
            st.caption(f"{est['files']} file(s), {est['bytes'] / 1e6:.0f} MB, "
                       f"~{est['est_rows'] / 1e6:.1f}M bars, "
                       f"~{est['est_ram_mb']:.0f} MB once loaded")
            if est["bits"] == 32:
                st.error(
                    "You are running 32-bit Python, which cannot address more "
                    "than about 2 GB no matter how much RAM the machine has. "
                    "That is almost certainly what caused any MemoryError. "
                    "Install 64-bit Python and reinstall the requirements.")
            elif est["est_ram_mb"] > 1500:
                st.warning("Large dataset. Narrow the year range if the load "
                           "runs out of memory.")

    if st.button("Load data", use_container_width=True, type="primary"):
        if not folder or not symbol:
            st.error("Pick a folder and symbol first.")
        else:
            bar = st.progress(0.0, "Parsing…")
            try:
                S["m1"] = load_symbol(str(Path(folder).expanduser()), symbol,
                                      tz, (yr_lo, yr_hi))
                S["events"] = None
                bar.empty()
                st.success(f"{len(S['m1']):,} M1 bars loaded")
            except MemoryError as e:
                bar.empty()
                st.error(str(e))
            except Exception as e:
                bar.empty()
                st.error(f"{type(e).__name__}: {e}")

    m1 = S.get("m1")
    if m1 is not None and len(m1):
        st.caption(f"{m1.index[0]:%Y-%m-%d} → {m1.index[-1]:%Y-%m-%d} UTC  ·  "
                   f"fp `{dataio.data_fingerprint(m1)}`")

    st.markdown("#### In-sample window")

    # The allowed range is deliberately NOT tied to the loaded data. Before a
    # load finishes there is no range to derive, and clamping to "today" made
    # the picker refuse every historical date.
    today = date.today()
    has_data = m1 is not None and len(m1) > 0
    if has_data:
        d0, d1 = m1.index[0].date(), m1.index[-1].date()
    else:
        d0, d1 = DATE_FLOOR, today

    # Seed sensible defaults the first time a given dataset appears, without
    # stomping on a choice the user has already made.
    stamp = f"{symbol}|{len(m1) if has_data else 0}"
    if has_data and S.get("_seeded") != stamp:
        S["is_start_w"] = d0
        S["is_end_w"] = m1.index[int(len(m1) * 0.7)].date()
        S["_seeded"] = stamp

    split_mode = st.radio("Split by", ["Dates", "Percentage"], horizontal=True,
                          key="split_mode")

    if split_mode == "Dates":
        S.setdefault("is_start_w", d0)
        S.setdefault("is_end_w", d1 if not has_data else
                     m1.index[int(len(m1) * 0.7)].date())
        is_start = st.date_input("IS start", min_value=DATE_FLOOR,
                                 max_value=today, key="is_start_w",
                                 format="YYYY-MM-DD")
        is_end = st.date_input("IS end", min_value=DATE_FLOOR,
                               max_value=today, key="is_end_w",
                               format="YYYY-MM-DD")
    else:
        pct = st.slider("In-sample share", 40, 95, 70, 5, key="is_pct",
                        format="%d%%")
        if has_data:
            is_start = d0
            is_end = m1.index[int(len(m1) * pct / 100)].date()
            st.caption(f"IS: {is_start} → {is_end}")
        else:
            is_start, is_end = d0, d1
            st.caption("Load data to see the resulting dates.")

    if is_end <= is_start:
        st.error("IS end must be after IS start.")
    elif has_data:
        if is_start > d1 or is_end < d0:
            st.error(f"That window sits outside your data "
                     f"({d0} → {d1}).")
        else:
            held = m1[m1.index >= pd.Timestamp(is_end, tz="UTC") + pd.Timedelta("1D")]
            if held.empty:
                st.warning("No data left for the holdout. Move IS end earlier.")
            else:
                st.caption(f"Holdout: {len(held):,} bars, "
                           f"{held.index[0].date()} → {held.index[-1].date()}")
    st.caption("Everything after IS end is sealed until you unlock it in the "
               "Holdout tab.")

    st.markdown("#### API key")
    api_key = st.text_input("ANTHROPIC_API_KEY", type="password",
                            help="Only used to translate English into a spec. "
                                 "Leave blank and write the spec by hand.")

    st.divider()
    st.caption(f"Distinct hypotheses tested: **{lib.distinct_trial_count()}**  \n"
               f"Total runs logged: {lib.trial_count()}")


# ============================================================== tabs
tabs = st.tabs(["Idea", "Spec", "Results", "By year", "By session",
                "Excursions", "Events", "Holdout", "Library"])

# ---------------------------------------------------------------- Idea
with tabs[0]:
    st.subheader("Describe the idea")
    st.caption("Plain English. The model only translates it into rules — it "
               "never sees price data and never judges the idea.")
    desc = st.text_area(
        "Idea", height=170,
        placeholder=("When NQ makes a bullish order block on the 5 minute — the "
                     "last red candle before a move up of at least 1.5 ATR that "
                     "breaks the recent high — and price comes back to test it "
                     "for the first time during the New York session, does it "
                     "bounce? Stop below the block, target 2R."))
    c1, c2 = st.columns([1, 3])
    if c1.button("Translate", type="primary", use_container_width=True):
        if not desc.strip():
            st.warning("Write something first.")
        else:
            with st.spinner("Translating…"):
                try:
                    spec, raw = nl.translate(desc, symbol=symbol or "",
                                             api_key=api_key or None)
                    set_spec(spec, raw)
                    st.success("Translated. Check it in the Spec tab before running.")
                except RuntimeError as e:
                    st.error(str(e))

    with st.expander("What the rule language can express"):
        st.markdown("**Detectors**")
        st.json(DETECTORS, expanded=False)
        st.markdown("**Higher-timeframe bias**")
        st.markdown(
            "Set `htf.timeframe` to something longer than `timeframe`, then "
            "filter on `aligned == 1` (setup direction matches the HTF EMA "
            "side) or `aligned_bar == 1` (matches the last closed HTF bar). "
            "Only **closed** HTF bars are ever read — at 10:05 the current 4h "
            "bar is half-formed and using it would leak the future.")
        st.markdown("**Filter variables**")
        st.table(pd.DataFrame(
            [{"name": k, "meaning": v} for k, v in FEATURE_HELP.items()]))

# ---------------------------------------------------------------- Spec
with tabs[1]:
    st.subheader("Rules")
    st.caption("This is what actually runs. Edit freely — the translation is a "
               "draft, not an authority.")
    text = st.text_area("Spec JSON", S["spec_text"], height=440,
                        label_visibility="collapsed")
    c1, c2, c3 = st.columns(3)

    if c1.button("Validate", use_container_width=True):
        try:
            sp = Spec.from_json(text)
            probs = sp.validate()
            if probs:
                st.error("  \n".join(f"• {p}" for p in probs))
            else:
                set_spec(sp, text)
                st.success("Spec is runnable.")
        except json.JSONDecodeError as e:
            st.error(f"Not valid JSON: {e}")

    if c2.button("Reset to example", use_container_width=True):
        set_spec(EXAMPLE_SPEC, EXAMPLE_SPEC.to_json())
        st.rerun()

    run = c3.button("Run scan", type="primary", use_container_width=True)

    if run:
        try:
            sp = Spec.from_json(text)
            set_spec(sp, text)
        except json.JSONDecodeError as e:
            st.error(f"Not valid JSON: {e}")
            sp = None
        if sp is not None:
            if S.get("m1") is None:
                st.error("Load data first.")
            else:
                m1 = S["m1"]
                a = pd.Timestamp(is_start, tz="UTC")
                b = pd.Timestamp(is_end, tz="UTC") + pd.Timedelta("1D")
                is_m1 = m1[(m1.index >= a) & (m1.index < b)]
                bar = st.progress(0.0, "Scanning in-sample…")
                try:
                    res = scanner.run(is_m1, sp, exchange_tz=exch_tz,
                                      progress=lambda p: bar.progress(
                                          min(p, 1.0), "Scanning in-sample…"))
                    S["events"] = res.events
                    S["scan_meta"] = res
                    S["baselines"] = {}
                    S["oos_unlocked"] = False
                    bar.empty()
                    st.success(
                        f"{res.n_zones:,} zones · {res.n_tests_raw:,} tests · "
                        f"{res.n_filtered_out:,} filtered out · "
                        f"{len(res.events):,} scored events")
                    for w in res.warnings:
                        st.warning(w)
                except Exception as e:
                    bar.empty()
                    st.error(f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- helpers
events = S.get("events")
spec = S.get("spec")


def scoring_selector(key: str):
    c1, c2 = st.columns([2, 1])
    mode = c1.radio("Scoring", ["fixed", "atr", "mfe"], horizontal=True,
                    key=f"mode_{key}",
                    help="fixed = the spec's stop/target · atr = target in ATR "
                         "before 1 ATR against · mfe = did it go your way at all")
    tr = c2.number_input("ATR target", 0.5, 10.0, 2.0, 0.5, key=f"tr_{key}") \
        if mode == "atr" else 2.0
    return mode, tr


def wins_for(ev, mode, tr):
    return scanner.rescore(ev, mode, tr)


# ---------------------------------------------------------------- Results
with tabs[2]:
    if events is None:
        st.info("Run a scan from the Spec tab.")
    elif events.empty:
        st.warning("Zero events. Loosen the detector or the filters.")
    else:
        mode, tr = scoring_selector("res")
        wins = wins_for(events, mode, tr)

        base = S["baselines"].get("drift")
        summ = stats.summarise(events, wins,
                               base["rate"] if base else None)

        c = st.columns(4)
        c[0].metric("Win rate", f"{summ['rate']:.1%}",
                    f"n = {summ['n']:,}", delta_color="off")
        c[1].metric("95% interval",
                    f"{summ['ci_low']:.1%} – {summ['ci_high']:.1%}")
        c[2].metric("Mean R", f"{summ['mean_r']:+.3f}",
                    f"{summ['mean_r_low']:+.2f} to {summ['mean_r_high']:+.2f}",
                    delta_color="off")
        c[3].metric("Ambiguous bars", f"{summ['ambiguous']:.1%}")

        st.divider()
        st.markdown("#### Baselines")
        st.caption("A rate on its own means nothing. The question is whether it "
                   "beats something that requires no skill.")
        b1, b2 = st.columns(2)
        if b1.button("Run drift baseline", use_container_width=True):
            with st.spinner("Sampling random entries…"):
                S["baselines"]["drift"] = stats.drift_baseline(
                    S["m1"], events, spec)
            st.rerun()
        if b2.button("Run placebo zones", use_container_width=True):
            with st.spinner("Scoring shifted zones…"):
                S["baselines"]["placebo"] = stats.placebo_baseline(
                    S["m1"], events, spec)
            st.rerun()

        rows = []
        for key, label in (("drift", "Buy-and-hold drift"),
                           ("placebo", "Placebo zones")):
            b = S["baselines"].get(key)
            if b and b.get("n"):
                rows.append({
                    "baseline": label, "rate": b["rate"], "n": b["n"],
                    "your edge": summ["rate"] - b["rate"],
                    "p": stats.binom_vs(summ["wins"], summ["n"], b["rate"]),
                })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(
                df.style.format({"rate": "{:.1%}", "your edge": "{:+.1%}",
                                 "p": "{:.4f}"}),
                use_container_width=True, hide_index=True)
            worst = min(r["your edge"] for r in rows)
            if worst <= 0:
                st.error("The pattern does not beat at least one control. "
                         "That is the answer, and it is a useful one.")
            elif any(r["p"] > 0.05 for r in rows):
                st.warning("The gap over a control is not distinguishable from "
                           "noise at this sample size.")

        n_trials = lib.distinct_trial_count()
        if n_trials > 20:
            st.caption(
                f"You have tested {n_trials} distinct rule sets. With that many "
                f"attempts, the best-looking result from pure noise sits around "
                f"{deflated_threshold(n_trials):.2f} standard deviations above "
                "chance. Judge this number against that, not against 50%.")

        st.plotly_chart(charts.equity(events), use_container_width=True)

        st.divider()
        if st.button("Save study to library", type="primary"):
            sid = lib.save_study(spec, summ, events, symbol or "",
                                 dataio.data_fingerprint(S["m1"]),
                                 is_start, is_end)
            st.success(f"Saved as study #{sid}.")

# ---------------------------------------------------------------- By year
with tabs[3]:
    if events is None or events.empty:
        st.info("Run a scan first.")
    else:
        mode, tr = scoring_selector("yr")
        wins = wins_for(events, mode, tr)
        base = S["baselines"].get("drift")
        by = stats.by_group(events, wins, "year")
        st.plotly_chart(
            charts.rate_bars(by, "year", "Win rate by year",
                             base["rate"] if base else None),
            use_container_width=True)
        st.caption("Grey bars have fewer than 20 events. A rate that was strong "
                   "before 2020 and average since is a dead pattern with a "
                   "flattering average.")
        st.dataframe(by.style.format(
            {"rate": "{:.1%}", "ci_low": "{:.1%}", "ci_high": "{:.1%}",
             "mean_r": "{:+.3f}"}), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- By session
with tabs[4]:
    if events is None or events.empty:
        st.info("Run a scan first.")
    else:
        mode, tr = scoring_selector("ss")
        wins = wins_for(events, mode, tr)
        base = S["baselines"].get("drift")
        b = base["rate"] if base else None
        st.plotly_chart(
            charts.rate_bars(stats.by_group(events, wins, "session"),
                             "session", "Win rate by session", b),
            use_container_width=True)
        st.plotly_chart(
            charts.rate_bars(stats.by_group(events, wins, "hour"),
                             "hour", "Win rate by hour (exchange time)", b),
            use_container_width=True)
        st.caption("If one hour carries the whole result, you have found a "
                   "session effect wearing an order block costume — or a "
                   "small-sample accident.")

        if "aligned" in events.columns and events["htf_bias"].abs().sum() > 0:
            st.divider()
            st.markdown("#### Higher-timeframe alignment")
            al = events.assign(
                htf=np.where(events["aligned"] == 1, "with HTF", "against HTF"))
            st.plotly_chart(
                charts.rate_bars(stats.by_group(al, wins, "htf"), "htf",
                                 "Win rate with and against the higher timeframe",
                                 b),
                use_container_width=True)
            st.caption("Run this before adding an alignment filter. If the two "
                       "bars overlap, the filter is throwing away half your "
                       "sample to buy nothing.")

# ---------------------------------------------------------------- Excursions
with tabs[5]:
    if events is None or events.empty:
        st.info("Run a scan first.")
    else:
        st.plotly_chart(charts.mfe_mae(events), use_container_width=True)
        st.plotly_chart(charts.mfe_vs_mae(events), use_container_width=True)
        q = events[["mfe_r", "mae_r", "bars_held"]].describe(
            percentiles=[.25, .5, .75, .9])
        st.dataframe(q.style.format("{:.2f}"), use_container_width=True)
        st.caption("If median MFE sits well below your target, the target is "
                   "the problem, not the entry.")

# ---------------------------------------------------------------- Events
with tabs[6]:
    if events is None or events.empty:
        st.info("Run a scan first.")
    else:
        st.markdown("#### Event log")
        show = events.drop(columns=[c for c in ("formed_at",) if c in events])
        st.dataframe(show, use_container_width=True, height=300)
        st.download_button("Download event log (CSV)",
                           events.to_csv(index=False).encode(),
                           file_name=f"edgelab_events_{spec.fingerprint()}.csv",
                           mime="text/csv")

        st.markdown("#### Inspect one")
        filt = st.selectbox("Show", ["all", "wins", "losses", "ambiguous only"])
        pool = events
        if filt == "wins":
            pool = events[events["outcome"] == "win"]
        elif filt == "losses":
            pool = events[events["outcome"] == "loss"]
        elif filt == "ambiguous only":
            pool = events[events["ambiguous"]]
        if pool.empty:
            st.info("Nothing matches.")
        else:
            i = st.slider("Event", 0, len(pool) - 1, 0)
            st.plotly_chart(charts.event_overlay(S["m1"], pool.iloc[i]),
                            use_container_width=True)
            st.caption("Read twenty of these before you trust any percentage. "
                       "It is the fastest way to find a rule that is not doing "
                       "what you meant.")

# ---------------------------------------------------------------- Holdout
with tabs[7]:
    st.subheader("Out-of-sample")
    if events is None or spec is None:
        st.info("Run an in-sample scan first.")
    elif S.get("m1") is None:
        st.info("Load data first.")
    else:
        family = _family_of(spec)
        looks = lib.reveal_count(family)
        st.metric("Times you have unlocked the holdout for this idea", looks)
        if looks >= 3:
            st.error(
                f"You have looked {looks} times. After the first look, the "
                "holdout stops being out-of-sample — you are now tuning on it. "
                "Consider retiring this idea or reserving fresh data.")
        elif looks >= 1:
            st.warning("Already looked once. A second look after a parameter "
                       "change is no longer a clean test.")

        st.caption("This is one-way. Unlock only when you have stopped changing "
                   "the spec.")
        if st.button("Unlock holdout", type="primary"):
            m1 = S["m1"]
            oos_m1 = m1[m1.index >= pd.Timestamp(is_end, tz="UTC") + pd.Timedelta("1D")]
            if oos_m1.empty:
                st.error("No data after the IS window.")
            else:
                bar = st.progress(0.0, "Scanning holdout…")
                res = scanner.run(oos_m1, spec, exchange_tz=exch_tz,
                                  progress=lambda p: bar.progress(min(p, 1.0)))
                bar.empty()
                S["oos_events"] = res.events
                S["oos_unlocked"] = True
                mode = S.get("mode_res", "fixed")
                tr = S.get("tr_res", 2.0)
                oos_summ = stats.summarise(res.events,
                                           wins_for(res.events, mode, tr))
                lib.log_reveal(family, spec.fingerprint(), oos_summ)
                st.rerun()

        if S.get("oos_unlocked") and S.get("oos_events") is not None:
            mode = S.get("mode_res", "fixed")
            tr = S.get("tr_res", 2.0)
            is_s = stats.summarise(events, wins_for(events, mode, tr))
            oos = S["oos_events"]
            oos_s = stats.summarise(oos, wins_for(oos, mode, tr)) if len(oos) \
                else {"n": 0}
            st.plotly_chart(charts.is_oos(is_s, oos_s), use_container_width=True)
            if oos_s.get("n"):
                drop = is_s["rate"] - oos_s["rate"]
                overlap = not (oos_s["ci_high"] < is_s["ci_low"] or
                               oos_s["ci_low"] > is_s["ci_high"])
                st.write(
                    f"In sample **{is_s['rate']:.1%}** (n={is_s['n']:,}) · "
                    f"holdout **{oos_s['rate']:.1%}** (n={oos_s['n']:,}) · "
                    f"change **{-drop:+.1%}**")
                if overlap:
                    st.success("The intervals overlap. The holdout does not "
                               "contradict the in-sample result.")
                else:
                    st.error("The intervals do not overlap. The in-sample "
                             "result did not survive.")

        st.divider()
        hist = lib.reveal_history(family)
        if len(hist):
            st.markdown("#### Reveal history for this idea")
            st.dataframe(hist[["revealed_at", "n_events", "rate"]],
                         use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Library
with tabs[8]:
    st.subheader("Saved studies")
    df = lib.list_studies()
    if df.empty:
        st.info("Nothing saved yet.")
    else:
        st.dataframe(
            df.style.format({"rate": "{:.1%}", "ci_low": "{:.1%}",
                             "ci_high": "{:.1%}", "mean_r": "{:+.3f}",
                             "baseline_rate": "{:.1%}", "p_value": "{:.4f}",
                             "ambiguous": "{:.1%}"}),
            use_container_width=True, hide_index=True, height=320)

        ids = df["id"].tolist()
        c1, c2, c3 = st.columns(3)
        sid = c1.selectbox("Study", ids)
        if c2.button("Load spec", use_container_width=True):
            rec = lib.get_study(int(sid))
            set_spec(Spec.from_json(rec["spec_json"]), rec["spec_json"])
            st.success(f"Loaded spec from #{sid}. Open the Spec tab.")
        if c3.button("Delete", use_container_width=True):
            lib.delete_study(int(sid))
            st.rerun()

        rec = lib.get_study(int(sid))
        if rec:
            st.markdown("#### MQL5 skeleton")
            st.caption("Scaffolding for MT5 — detector and level arithmetic "
                       "transcribed, execution and risk left as marked TODOs.")
            try:
                code = mql5.export(Spec.from_json(rec["spec_json"]),
                                   json.loads(rec["summary_json"] or "{}"))
            except ValueError as e:
                code = None
                st.warning(f"No MQL5 export: {e}")
            if code:
                st.download_button("Download .mq5", code.encode(),
                                   file_name=f"EdgeLab_{rec['fingerprint']}.mq5")
                with st.expander("Preview"):
                    st.code(code, language="cpp")

            ev = lib.load_events(int(sid))
            if len(ev):
                st.download_button("Download event log (CSV)",
                                   ev.to_csv(index=False).encode(),
                                   file_name=f"events_study_{sid}.csv",
                                   mime="text/csv")
