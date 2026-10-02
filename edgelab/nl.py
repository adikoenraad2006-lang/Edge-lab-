"""
Natural language -> rule spec.

The model's only job is translation. It never sees price data, never scores
anything, and never decides whether an idea is good. It converts your English
into the JSON the deterministic engine runs, and you review the result before
anything executes. If the API is unavailable you can still write the spec by
hand; nothing downstream depends on this module.
"""

from __future__ import annotations

import json
import os
import re

from .spec import Spec, DETECTORS
from .filters import FEATURE_HELP

MODEL = "claude-sonnet-5"


def _system_prompt() -> str:
    det = json.dumps(DETECTORS, indent=2)
    feats = "\n".join(f"  {k}: {v}" for k, v in FEATURE_HELP.items())
    return f"""You translate a trader's plain-English description of a setup into a strict JSON rule spec. You output JSON and nothing else — no prose, no markdown fences.

Schema:
{{
  "name": string,
  "notes": string,
  "direction": "long" | "short" | "both",
  "timeframe": pandas offset string, e.g. "1min", "5min", "15min", "1h",
  "zone": {{ "detector": <one of below>, "params": {{...}} }},
  "htf": {{ "timeframe": pandas offset longer than "timeframe", or "" to disable,
           "ema_period": int, "atr_period": int, "slope_bars": int }},
  "trigger": {{ "type": "touch"|"close_inside"|"wick_through",
               "first_test_only": bool, "min_bars_after_formation": int }},
  "filters": [ <boolean expression strings> ],
  "scoring": {{
     "stop":   {{ "mode": "zone_far_edge"|"atr"|"fixed_points", "value": number }},
     "target": {{ "mode": "r_multiple"|"atr"|"fixed_points"|"none", "value": number }},
     "max_holding_bars": int,   // in M1 bars
     "entry": "zone_proximal"|"next_open"
  }}
}}

Available detectors and their parameters (defaults shown). Use ONLY these keys:
{det}

Filter expressions are boolean Python over these names only:
{feats}
Strings must be quoted, e.g. session == 'RTH'. Allowed functions: abs, min, max, round, int, float, bool.

Rules:
- Choose the detector that best matches the description. If the trader describes something none of these capture, pick the closest and say so in "notes".
- Do not invent parameter names. Anything not in the list above must be expressed as a filter, or noted as unsupported in "notes".
- If the trader did not specify something, use a plain default and record the assumption in "notes". Never silently guess a filter they did not ask for.
- max_holding_bars is in M1 bars regardless of the detection timeframe. 240 = four hours.
- "reacted positively" with no stated target means target r_multiple 2.0, stop at the zone's far edge.
- Higher-timeframe bias: when the trader mentions a bigger-picture direction ("only longs when the 4h is bullish", "with the daily trend", "in line with the higher timeframe"), set htf.timeframe and express the condition as a filter using the htf_ variables. "aligned == 1" means the setup direction matches the HTF EMA side; "aligned_bar == 1" matches the last closed HTF bar's direction. Leave htf.timeframe as "" when no higher-timeframe condition was described, and never add one they did not ask for.
- htf.timeframe MUST be longer than timeframe.
- Keep "notes" short and factual: what you assumed, what you could not express."""


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def translate(description: str, symbol: str = "",
              api_key: str | None = None, model: str = MODEL) -> tuple[Spec, str]:
    """
    Returns (spec, raw_json). Raises RuntimeError with a readable message on
    failure so the UI can fall back to manual editing.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "No ANTHROPIC_API_KEY set. Either export it, paste it in the "
            "sidebar, or write the spec by hand in the editor."
        )
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic") from None

    client = anthropic.Anthropic(api_key=key)
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=2000,
            system=_system_prompt(),
            messages=[{"role": "user", "content": description}],
        )
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}") from None

    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    raw = _strip_fences(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Model did not return valid JSON ({e}). Raw output:\n{text[:800]}"
        ) from None

    data.setdefault("symbol", symbol)
    spec = Spec.from_dict(data)
    problems = spec.validate()
    if problems:
        raise RuntimeError(
            "Translated spec has problems — fix them in the editor before "
            "running:\n  - " + "\n  - ".join(problems) +
            f"\n\nSpec:\n{spec.to_json()}"
        )
    return spec, spec.to_json()


def _strip_fences(t: str) -> str:
    t = t.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        return m.group(1).strip()
    a, b = t.find("{"), t.rfind("}")
    return t[a:b + 1] if a >= 0 and b > a else t
