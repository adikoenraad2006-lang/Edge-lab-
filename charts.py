"""Charts. Every rate chart carries its confidence interval — a bar without one
invites you to read a 12-event bucket as a finding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ACCENT = "#2f6f4e"
MUTED = "#9aa5a0"
WARN = "#b4553a"


def rate_bars(df: pd.DataFrame, x: str, title: str,
              baseline: float | None = None, min_n: int = 20) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=f"{title} — no events")
        return fig
    colors = [ACCENT if n >= min_n else MUTED for n in df["n"]]
    fig.add_bar(
        x=df[x].astype(str), y=df["rate"], marker_color=colors,
        error_y=dict(type="data", symmetric=False,
                     array=(df["ci_high"] - df["rate"]),
                     arrayminus=(df["rate"] - df["ci_low"]),
                     color="#55605a", thickness=1.2),
        customdata=np.stack([df["n"], df["ci_low"], df["ci_high"]], -1),
        hovertemplate=("%{x}<br>rate %{y:.1%}<br>n=%{customdata[0]}"
                       "<br>95%% CI %{customdata[1]:.1%}–%{customdata[2]:.1%}"
                       "<extra></extra>"),
        name="observed",
    )
    if baseline is not None and np.isfinite(baseline):
        fig.add_hline(y=baseline, line_dash="dash", line_color=WARN,
                      annotation_text=f"baseline {baseline:.1%}",
                      annotation_position="top left")
    fig.update_layout(
        title=title, yaxis_title="win rate", yaxis_tickformat=".0%",
        showlegend=False, height=420, bargap=0.25,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def mfe_mae(events: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if events.empty:
        return fig
    fig.add_histogram(x=events["mfe_r"], name="MFE (R)", opacity=0.65,
                      marker_color=ACCENT, nbinsx=60)
    fig.add_histogram(x=events["mae_r"], name="MAE (R)", opacity=0.65,
                      marker_color=WARN, nbinsx=60)
    fig.update_layout(
        barmode="overlay", title="Excursion distribution",
        xaxis_title="R multiple reached before exit", yaxis_title="events",
        height=420, margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def mfe_vs_mae(events: pd.DataFrame) -> go.Figure:
    """
    Where the mass sits tells you whether the target is placed sensibly. A cloud
    hugging the diagonal means price wanders both ways and the pattern is not
    directional.
    """
    fig = go.Figure()
    if events.empty:
        return fig
    win = events["outcome"] == "win"
    for mask, name, col in ((win, "win", ACCENT), (~win, "not win", MUTED)):
        s = events[mask]
        fig.add_scatter(x=s["mae_r"], y=s["mfe_r"], mode="markers", name=name,
                        marker=dict(size=5, color=col, opacity=0.55))
    m = float(max(events["mfe_r"].max(), events["mae_r"].max(), 1))
    fig.add_scatter(x=[0, m], y=[0, m], mode="lines", name="MFE = MAE",
                    line=dict(dash="dot", color="#55605a"))
    fig.update_layout(title="MFE against MAE, per event",
                      xaxis_title="MAE (R)", yaxis_title="MFE (R)",
                      height=460, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def equity(events: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if events.empty:
        return fig
    e = events.sort_values("ts")
    fig.add_scatter(x=e["ts"], y=e["r_fixed"].cumsum(), mode="lines",
                    line=dict(color=ACCENT, width=1.6), name="cumulative R")
    fig.update_layout(title="Cumulative R over the sample (no costs applied)",
                      yaxis_title="R", height=380,
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def event_overlay(m1: pd.DataFrame, event: pd.Series,
                  before: int = 120, after: int = 240) -> go.Figure:
    """One event, with its zone, entry, stop and target drawn on M1."""
    ts = pd.Timestamp(event["ts"])
    i = int(m1.index.searchsorted(ts))
    a, b = max(0, i - before), min(len(m1), i + after)
    s = m1.iloc[a:b]
    fig = go.Figure(go.Candlestick(
        x=s.index, open=s["open"], high=s["high"], low=s["low"], close=s["close"],
        increasing_line_color="#3f7d5f", decreasing_line_color="#a8563f",
        name="price"))
    fig.add_hrect(y0=event["zone_low"], y1=event["zone_high"],
                  fillcolor=ACCENT, opacity=0.14, line_width=0)
    for key, colour, dash in (("entry", "#2f6f4e", "solid"),
                              ("stop", WARN, "dash"),
                              ("target", "#3a6fb4", "dash")):
        v = event.get(key)
        if v is not None and np.isfinite(v):
            fig.add_hline(y=float(v), line_color=colour, line_dash=dash,
                          annotation_text=key, annotation_position="right")
    fig.add_vline(x=ts, line_color="#55605a", line_dash="dot")
    fig.update_layout(
        title=(f"{ts:%Y-%m-%d %H:%M} UTC · "
               f"{'long' if event['bias'] == 1 else 'short'} · "
               f"{event['outcome']}"
               f"{' (ambiguous bar)' if event.get('ambiguous') else ''}"),
        xaxis_rangeslider_visible=False, height=520,
        margin=dict(l=40, r=20, t=60, b=40))
    return fig


def is_oos(is_summary: dict, oos_summary: dict) -> go.Figure:
    fig = go.Figure()
    labels, rates, los, his = [], [], [], []
    for name, s in (("in sample", is_summary), ("out of sample", oos_summary)):
        if not s or not s.get("n"):
            continue
        labels.append(f"{name}<br>n={s['n']}")
        rates.append(s["rate"])
        los.append(s["rate"] - s["ci_low"])
        his.append(s["ci_high"] - s["rate"])
    if not labels:
        return fig
    fig.add_bar(x=labels, y=rates, marker_color=[ACCENT, "#3a6fb4"][:len(labels)],
                error_y=dict(type="data", symmetric=False, array=his,
                             arrayminus=los, color="#55605a"))
    fig.update_layout(title="In sample against holdout", yaxis_tickformat=".0%",
                      yaxis_range=[0, 1], height=400, showlegend=False,
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig
