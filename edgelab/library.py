"""
The study library.

Two jobs beyond plain storage.

**The OOS ledger.** Every time you unlock the out-of-sample window for a spec
family, it is recorded. Look eleven times while nudging a parameter and the
twelfth result is not out-of-sample any more. The app will not stop you, but it
will tell you what you have done.

**The shared trial registry.** Writes to the same `trials.csv` used by mt5scan,
so hypotheses tested here count toward the same lifetime total. That count is
what deflates your significance thresholds. Keeping two separate counters is
the same as not counting.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL,
    symbol TEXT,
    fingerprint TEXT NOT NULL,
    family TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    data_fingerprint TEXT,
    is_start TEXT, is_end TEXT,
    n_events INTEGER, rate REAL, ci_low REAL, ci_high REAL,
    mean_r REAL, baseline_rate REAL, p_value REAL,
    ambiguous REAL,
    summary_json TEXT,
    events_path TEXT
);
CREATE TABLE IF NOT EXISTS oos_reveals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    revealed_at TEXT NOT NULL,
    family TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    study_id INTEGER,
    n_events INTEGER, rate REAL, ci_low REAL, ci_high REAL
);
CREATE INDEX IF NOT EXISTS ix_studies_fp ON studies(fingerprint);
CREATE INDEX IF NOT EXISTS ix_reveals_family ON oos_reveals(family);
"""


class Library:
    def __init__(self, root: str | Path = "~/.edgelab"):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "events").mkdir(exist_ok=True)
        self.db_path = self.root / "studies.db"
        self.trials_csv = self.root / "trials.csv"
        with self._conn() as cx:
            cx.executescript(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    # -------------------------------------------------------------- studies
    def save_study(self, spec, summary: dict, events: pd.DataFrame,
                   symbol: str, data_fp: str, is_start, is_end,
                   family: str | None = None) -> int:
        family = family or _family_of(spec)
        now = datetime.now(timezone.utc).isoformat()
        ev_path = ""
        with self._conn() as cx:
            cur = cx.execute(
                """INSERT INTO studies (created_at,name,symbol,fingerprint,family,
                   spec_json,data_fingerprint,is_start,is_end,n_events,rate,
                   ci_low,ci_high,mean_r,baseline_rate,p_value,ambiguous,
                   summary_json,events_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now, spec.name, symbol, spec.fingerprint(), family,
                 spec.to_json(), data_fp, str(is_start), str(is_end),
                 summary.get("n"), summary.get("rate"), summary.get("ci_low"),
                 summary.get("ci_high"), summary.get("mean_r"),
                 summary.get("baseline_rate"), summary.get("p_value"),
                 summary.get("ambiguous"), json.dumps(summary, default=str), ""))
            sid = cur.lastrowid
            if len(events):
                ev_path = str(self.root / "events" / f"study_{sid}.parquet")
                try:
                    events.to_parquet(ev_path)
                except Exception:
                    ev_path = str(self.root / "events" / f"study_{sid}.csv")
                    events.to_csv(ev_path, index=False)
                cx.execute("UPDATE studies SET events_path=? WHERE id=?",
                           (ev_path, sid))
        self._log_trial(spec, symbol, summary, family)
        return sid

    def list_studies(self) -> pd.DataFrame:
        with self._conn() as cx:
            return pd.read_sql_query(
                "SELECT id,created_at,name,symbol,family,fingerprint,n_events,"
                "rate,ci_low,ci_high,mean_r,baseline_rate,p_value,ambiguous "
                "FROM studies ORDER BY id DESC", cx)

    def get_study(self, sid: int) -> dict | None:
        with self._conn() as cx:
            cur = cx.execute("SELECT * FROM studies WHERE id=?", (sid,))
            cols = [d[0] for d in cur.description]
            row = cur.fetchone()
        return dict(zip(cols, row)) if row else None

    def load_events(self, sid: int) -> pd.DataFrame:
        rec = self.get_study(sid)
        if not rec or not rec.get("events_path"):
            return pd.DataFrame()
        p = Path(rec["events_path"])
        if not p.exists():
            return pd.DataFrame()
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    def delete_study(self, sid: int):
        rec = self.get_study(sid)
        if rec and rec.get("events_path"):
            Path(rec["events_path"]).unlink(missing_ok=True)
        with self._conn() as cx:
            cx.execute("DELETE FROM studies WHERE id=?", (sid,))

    # ------------------------------------------------------------ oos ledger
    def reveal_count(self, family: str) -> int:
        with self._conn() as cx:
            (n,) = cx.execute(
                "SELECT COUNT(*) FROM oos_reveals WHERE family=?",
                (family,)).fetchone()
        return int(n)

    def log_reveal(self, family: str, fingerprint: str, summary: dict,
                   study_id: int | None = None):
        with self._conn() as cx:
            cx.execute(
                """INSERT INTO oos_reveals (revealed_at,family,fingerprint,
                   study_id,n_events,rate,ci_low,ci_high)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (datetime.now(timezone.utc).isoformat(), family, fingerprint,
                 study_id, summary.get("n"), summary.get("rate"),
                 summary.get("ci_low"), summary.get("ci_high")))

    def reveal_history(self, family: str) -> pd.DataFrame:
        with self._conn() as cx:
            return pd.read_sql_query(
                "SELECT * FROM oos_reveals WHERE family=? ORDER BY id",
                cx, params=(family,))

    # --------------------------------------------------------- trial registry
    def _log_trial(self, spec, symbol: str, summary: dict, family: str):
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "edgelab",
            "symbol": symbol,
            "hypothesis": spec.name,
            "fingerprint": spec.fingerprint(),
            "family": family,
            "n": summary.get("n"),
            "rate": summary.get("rate"),
            "mean_r": summary.get("mean_r"),
            "p_value": summary.get("p_value"),
        }
        header = not self.trials_csv.exists()
        pd.DataFrame([row]).to_csv(self.trials_csv, mode="a",
                                   header=header, index=False)

    def trial_count(self) -> int:
        if not self.trials_csv.exists():
            return 0
        try:
            return len(pd.read_csv(self.trials_csv))
        except Exception:
            return 0

    def distinct_trial_count(self) -> int:
        """Distinct rule fingerprints — re-running the same spec is not a new test."""
        if not self.trials_csv.exists():
            return 0
        try:
            df = pd.read_csv(self.trials_csv)
            return int(df["fingerprint"].nunique()) if "fingerprint" in df else len(df)
        except Exception:
            return 0


def _family_of(spec) -> str:
    """
    Groups specs that are the same idea with different knobs. Parameter tweaks
    stay in one family, so the OOS ledger counts the looks you actually took at
    the holdout for that idea.
    """
    return f"{spec.symbol or '*'}|{spec.zone.detector}|{spec.direction}"


def deflated_threshold(n_trials: int) -> float:
    """
    Expected maximum win-rate deviation from pure noise given how many
    hypotheses have been tried. Rough guide, in the spirit of the deflated
    Sharpe ratio: with enough attempts, an impressive number is the expected
    output of randomness rather than evidence against it.
    """
    import numpy as np
    n = max(1, n_trials)
    return float(np.sqrt(2 * np.log(n)))
