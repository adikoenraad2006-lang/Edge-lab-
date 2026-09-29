"""
The rule spec: a trade idea expressed as data, not code.

Everything downstream (scanner, scoring, MQL5 export, the study library) reads
this one object. A spec plus a data hash fully determines a result, which is
what makes a study reproducible six months later.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Literal


# --------------------------------------------------------------------------
# Detectors: what creates the zone/level we are going to test
# --------------------------------------------------------------------------
# Every detector returns rows of (formed_at, zone_low, zone_high, bias).
# `formed_at` is the timestamp at which the zone is KNOWN, not the timestamp of
# the candle it is drawn from. This distinction is the whole ballgame for
# look-ahead bias: an order block is drawn on a candle that happened N bars ago,
# but you could not have known it was an order block until the displacement
# completed. The scanner refuses to test any touch before `formed_at`.

DETECTORS = {
    "order_block": {
        "displacement_atr": 1.5,      # move size that qualifies as displacement
        "displacement_bars": 1,        # bars the move may take
        "require_swing_break": True,   # must the move break a prior swing?
        "swing_lookback": 10,
        "zone_from": "body",           # "body" | "wick"
        "max_age_bars": 500,           # zone expires after this many bars
    },
    "fvg": {
        "min_gap_atr": 0.25,
        "max_age_bars": 500,
    },
    "swing_level": {
        "left": 5,
        "right": 5,
        "zone_pad_atr": 0.1,
        "max_age_bars": 1000,
    },
    "prior_session_level": {
        "which": "high",               # high | low | close | open
        "session": "RTH",
        "zone_pad_atr": 0.1,
        "max_age_bars": 1440,
    },
    "round_number": {
        "step": 100.0,
        "zone_pad_atr": 0.1,
        "max_age_bars": 1440,
    },
}


@dataclass
class Zone:
    detector: str = "order_block"
    params: dict[str, Any] = field(default_factory=dict)

    def resolved(self) -> dict[str, Any]:
        if self.detector not in DETECTORS:
            raise ValueError(
                f"unknown detector {self.detector!r}; "
                f"available: {sorted(DETECTORS)}"
            )
        merged = dict(DETECTORS[self.detector])
        merged.update(self.params or {})
        return merged


@dataclass
class Htf:
    """
    Higher-timeframe context.

    Only CLOSED higher-timeframe bars are ever read. At 10:05 the current 4h
    bar is half-formed and its direction is not yet knowable; using it would
    leak the future into the filter and inflate every result.
    """
    timeframe: str = ""          # "" disables HTF entirely
    ema_period: int = 50
    atr_period: int = 14
    slope_bars: int = 3          # lookback for the EMA slope measure


@dataclass
class Trigger:
    """What counts as 'testing' the zone."""
    type: Literal["touch", "close_inside", "wick_through"] = "touch"
    first_test_only: bool = True      # count re-tests separately, never together
    min_bars_after_formation: int = 1


@dataclass
class Stop:
    mode: Literal["zone_far_edge", "atr", "fixed_points"] = "zone_far_edge"
    value: float = 1.0                # multiplier (zone_far_edge/atr) or points


@dataclass
class Target:
    mode: Literal["r_multiple", "atr", "fixed_points", "none"] = "r_multiple"
    value: float = 2.0


@dataclass
class Scoring:
    stop: Stop = field(default_factory=Stop)
    target: Target = field(default_factory=Target)
    max_holding_bars: int = 240       # in base (M1) bars
    entry: Literal["zone_proximal", "next_open"] = "zone_proximal"


@dataclass
class Spec:
    name: str = "untitled study"
    notes: str = ""
    symbol: str = ""
    timeframe: str = "5min"           # detection timeframe; scoring is always M1
    direction: Literal["long", "short", "both"] = "both"
    zone: Zone = field(default_factory=Zone)
    htf: Htf = field(default_factory=Htf)
    trigger: Trigger = field(default_factory=Trigger)
    filters: list[str] = field(default_factory=list)   # safe expressions
    scoring: Scoring = field(default_factory=Scoring)

    # ---------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "Spec":
        d = dict(d or {})
        zone = d.pop("zone", {}) or {}
        htf = d.pop("htf", {}) or {}
        trig = d.pop("trigger", {}) or {}
        scor = d.pop("scoring", {}) or {}
        stop = (scor.pop("stop", {}) or {}) if isinstance(scor, dict) else {}
        targ = (scor.pop("target", {}) or {}) if isinstance(scor, dict) else {}
        known = {f for f in cls.__dataclass_fields__}
        d = {k: v for k, v in d.items() if k in known}
        return cls(
            zone=Zone(**{k: v for k, v in zone.items()
                         if k in Zone.__dataclass_fields__}),
            htf=Htf(**{k: v for k, v in htf.items()
                       if k in Htf.__dataclass_fields__}),
            trigger=Trigger(**{k: v for k, v in trig.items()
                               if k in Trigger.__dataclass_fields__}),
            scoring=Scoring(
                stop=Stop(**{k: v for k, v in stop.items()
                             if k in Stop.__dataclass_fields__}),
                target=Target(**{k: v for k, v in targ.items()
                                 if k in Target.__dataclass_fields__}),
                **{k: v for k, v in scor.items()
                   if k in Scoring.__dataclass_fields__},
            ),
            **d,
        )

    @classmethod
    def from_json(cls, s: str) -> "Spec":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------ identity
    def fingerprint(self) -> str:
        """
        Stable hash of the *rules only* — name and notes excluded, so renaming a
        study does not make it look like a fresh hypothesis to the registry.
        """
        d = self.to_dict()
        d.pop("name", None)
        d.pop("notes", None)
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def validate(self) -> list[str]:
        """Returns a list of human-readable problems. Empty means good to run."""
        problems = []
        if self.zone.detector not in DETECTORS:
            problems.append(f"Unknown detector: {self.zone.detector}")
        else:
            allowed = set(DETECTORS[self.zone.detector])
            for k in (self.zone.params or {}):
                if k not in allowed:
                    problems.append(
                        f"Detector {self.zone.detector} has no parameter {k!r}. "
                        f"Valid: {sorted(allowed)}"
                    )
        if self.direction not in ("long", "short", "both"):
            problems.append(f"direction must be long/short/both, got {self.direction}")
        if self.scoring.max_holding_bars < 1:
            problems.append("max_holding_bars must be >= 1")
        if self.scoring.stop.value <= 0:
            problems.append("stop value must be > 0")
        if self.scoring.target.mode != "none" and self.scoring.target.value <= 0:
            problems.append("target value must be > 0")
        try:
            import pandas as pd
            pd.Timedelta(self.timeframe)
        except Exception:
            problems.append(f"timeframe {self.timeframe!r} is not a pandas offset")
        if self.htf.timeframe:
            try:
                import pandas as pd
                if pd.Timedelta(self.htf.timeframe) <= pd.Timedelta(self.timeframe):
                    problems.append(
                        f"htf.timeframe ({self.htf.timeframe}) must be longer "
                        f"than timeframe ({self.timeframe}); otherwise it is "
                        "not a higher timeframe.")
            except Exception:
                problems.append(
                    f"htf.timeframe {self.htf.timeframe!r} is not a pandas offset")
        elif any("htf_" in f or "aligned" in f for f in self.filters):
            problems.append(
                "A filter uses an htf_ variable but htf.timeframe is empty. "
                "Set it, e.g. \"htf\": {\"timeframe\": \"4h\"}.")
        from .filters import check_expression
        for f in self.filters:
            err = check_expression(f)
            if err:
                problems.append(f"Filter {f!r}: {err}")
        return problems


EXAMPLE_SPEC = Spec(
    name="NQ bullish order block, first test",
    notes="Baseline version of the idea before any tuning.",
    timeframe="5min",
    direction="long",
    zone=Zone(detector="order_block", params={
        "displacement_atr": 1.5,
        "require_swing_break": True,
        "zone_from": "body",
    }),
    htf=Htf(timeframe="4h", ema_period=50),
    trigger=Trigger(type="touch", first_test_only=True),
    filters=["session == 'RTH'", "aligned == 1"],
    scoring=Scoring(
        stop=Stop(mode="zone_far_edge", value=1.0),
        target=Target(mode="r_multiple", value=2.0),
        max_holding_bars=240,
    ),
)
