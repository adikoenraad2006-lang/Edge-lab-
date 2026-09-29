"""
Data loading.

Sniffs arbitrary M1 CSV exports (MT5, Tickstory, Dukascopy, generic OHLCV)
rather than requiring one layout, shows you what it detected, and caches the
parsed result as Parquet so the second run is instant.

Timezone is never guessed silently. You confirm it once per folder and it is
stored alongside the cache, because a wrong timezone quietly destroys every
session-based conclusion the app will produce.
"""

from __future__ import annotations

import csv
import hashlib
import struct
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIRNAME = ".edgelab_cache"

# All names below are already normalised: lowercase, punctuation stripped.
_DATETIME_NAMES = ["datetime", "timestamp", "dt", "gmttime", "localtime",
                   "opentime", "bartime", "servertime"]
_DATE_NAMES = ["date"]
_TIME_NAMES = ["time"]
_OHLC = {
    "open": ["open", "o", "bidopen", "openbid", "openprice"],
    "high": ["high", "h", "bidhigh", "highbid", "highprice"],
    "low": ["low", "l", "bidlow", "lowbid", "lowprice"],
    "close": ["close", "c", "bidclose", "closebid", "closeprice", "last"],
    "volume": ["volume", "vol", "v", "tickvol", "tickvolume", "realvolume",
               "bidvolume", "volumefrom"],
}


@dataclass
class CsvFormat:
    delimiter: str
    decimal: str
    has_header: bool
    columns: dict          # role -> column index
    datetime_format: str | None
    date_col: int | None   # when date and time are separate columns
    time_col: int | None
    sample: list           # first few parsed rows, for the UI preview
    header: list = None    # raw header cells, or None when headerless
    ambiguous_date: bool = False   # dd/mm vs mm/dd cannot be told apart

    def to_dict(self):
        return asdict(self)


def _norm(s: str) -> str:
    """
    MT5 writes headers as <DATE> <TIME> <OPEN> <TICKVOL>. Stripping the angle
    brackets along with all other punctuation lets one lookup table cover MT5,
    MT4, Tickstory, Dukascopy and generic exports.
    """
    return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())


def sniff(path: str | Path, n_lines: int = 50) -> CsvFormat:
    """Work out how to read this file. Raises with a clear message if it can't."""
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        head = [fh.readline() for _ in range(n_lines)]
    head = [ln for ln in head if ln.strip()]
    if not head:
        raise ValueError(f"{path.name} is empty")

    try:
        delim = csv.Sniffer().sniff(
            "".join(head[:10]), delimiters=",;\t|"
        ).delimiter
    except csv.Error:
        counts = {d: head[-1].count(d) for d in ",;\t|"}
        delim = max(counts, key=counts.get)
        if counts[delim] == 0:
            raise ValueError(
                f"{path.name}: could not find a column separator. "
                "Expected comma, semicolon, tab or pipe."
            ) from None

    first = [c.strip() for c in head[0].rstrip("\n").split(delim)]
    has_header = not _looks_numeric_or_date(first[0])

    body = head[1:] if has_header else head
    if not body:
        raise ValueError(f"{path.name}: header only, no data rows")
    sample_rows = [r.rstrip("\n").split(delim) for r in body[:5]]

    # Decimal separator: if a field contains a comma inside a numeric-looking
    # token and the delimiter is not a comma, it is a European decimal.
    decimal = "."
    if delim != "," and any(
        re.fullmatch(r"-?\d+,\d+", c.strip()) for r in sample_rows for c in r
    ):
        decimal = ","

    columns: dict[str, int] = {}
    date_col = time_col = None

    if has_header:
        norm = [_norm(c) for c in first]
        for role, names in _OHLC.items():
            for i, c in enumerate(norm):
                if c in names:
                    columns[role] = i
                    break

        # Timestamp: a single combined column wins; otherwise a DATE + TIME
        # pair, which is what MT5's bar export writes.
        for i, c in enumerate(norm):
            if c in _DATETIME_NAMES:
                columns["datetime"] = i
                break
        if "datetime" not in columns:
            di = next((i for i, c in enumerate(norm) if c in _DATE_NAMES), None)
            ti = next((i for i, c in enumerate(norm) if c in _TIME_NAMES), None)
            if di is not None and ti is not None:
                date_col, time_col = di, ti
            elif di is not None:
                columns["datetime"] = di
            elif ti is not None:
                columns["datetime"] = ti

        # Last resort: the header used names we do not know. Fall back on shape.
        if not all(k in columns for k in ("open", "high", "low", "close")) \
                or ("datetime" not in columns and date_col is None):
            guessed = _guess_positional(sample_rows[0])
            if guessed:
                g_cols, g_date, g_time = guessed
                for k, v in g_cols.items():
                    columns.setdefault(k, v)
                if "datetime" not in columns and date_col is None:
                    date_col, time_col = g_date, g_time
    else:
        guessed = _guess_positional(sample_rows[0])
        if not guessed:
            raise ValueError(
                f"{path.name}: found {len(sample_rows[0])} columns and no "
                f"header; need at least 5 (timestamp + OHLC). "
                f"First row: {sample_rows[0]}"
            )
        columns, date_col, time_col = guessed

    missing = [k for k in ("open", "high", "low", "close") if k not in columns]
    no_ts = "datetime" not in columns and date_col is None
    if missing or no_ts:
        want = missing + (["timestamp"] if no_ts else [])
        raise ValueError(
            f"{path.name}: could not identify {', '.join(want)}.\n"
            f"  delimiter: {delim!r}   decimal: {decimal!r}\n"
            f"  header row:    {first}\n"
            f"  first data row: {sample_rows[0]}\n"
            f"  mapped so far: {columns}\n"
            "Either rename the columns to open/high/low/close plus "
            "date+time (or datetime), or send these two lines to get the "
            "sniffer extended."
        )

    if date_col is not None:
        ts_sample = [f"{r[date_col]} {r[time_col]}" if time_col is not None
                     else r[date_col] for r in sample_rows]
    else:
        ts_sample = [r[columns["datetime"]] for r in sample_rows]
    dt_fmt = _guess_datetime_format(ts_sample)

    return CsvFormat(
        delimiter=delim, decimal=decimal, has_header=has_header,
        columns=columns, datetime_format=dt_fmt,
        date_col=date_col, time_col=time_col,
        sample=sample_rows[:3],
        header=first if has_header else None,
        ambiguous_date=_date_is_ambiguous(ts_sample, dt_fmt),
    )


def _date_is_ambiguous(samples: list, fmt: str | None) -> bool:
    """
    True when day-first and month-first both parse every sample. 05/01/2015 is
    either 5 January or 1 May and nothing in the file says which; the loader
    assumes day-first, and the UI says so rather than hiding the coin flip.
    """
    if not fmt or not any(sep in fmt for sep in ("/", ".", "-")):
        return False
    if fmt.startswith("%Y"):
        # year-first (ISO, MT5's 2015.01.05) is always year-month-day
        return False
    swapped = (fmt.replace("%d", "\x00").replace("%m", "%d")
                  .replace("\x00", "%m"))
    if swapped == fmt:
        return False
    for cand in (fmt, swapped):
        try:
            for s_ in samples:
                pd.to_datetime(s_.strip(), format=cand)
        except (ValueError, TypeError):
            return False
    return True


def _guess_positional(row: list):
    """
    Map columns by shape when the names are unhelpful. Covers the two layouts
    that account for nearly everything: date,time,O,H,L,C[,V] and
    datetime,O,H,L,C[,V]. Trailing extras (MT5's spread and real volume) are
    ignored rather than treated as an error.
    """
    cols: dict[str, int] = {}
    if len(row) >= 6 and _looks_date(row[0]) and _looks_time(row[1]):
        cols.update(open=2, high=3, low=4, close=5)
        if len(row) > 6:
            cols["volume"] = 6
        return cols, 0, 1
    if len(row) >= 5:
        cols["datetime"] = 0
        cols.update(open=1, high=2, low=3, close=4)
        if len(row) > 5:
            cols["volume"] = 5
        return cols, None, None
    return None


def _looks_numeric_or_date(s: str) -> bool:
    s = s.strip().strip('"')
    if re.fullmatch(r"-?\d+([.,]\d+)?", s):
        return True
    return bool(re.match(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}", s) or
                re.match(r"\d{1,2}[-./]\d{1,2}[-./]\d{4}", s))


def _looks_date(s: str) -> bool:
    return bool(re.match(r"^\s*\d{4}[-./]\d{1,2}[-./]\d{1,2}\s*$", s) or
                re.match(r"^\s*\d{1,2}[-./]\d{1,2}[-./]\d{4}\s*$", s))


def _looks_time(s: str) -> bool:
    return bool(re.match(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$", s))


_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y.%m.%d",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M",
    "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    "%Y%m%d %H:%M:%S", "%Y%m%d%H%M%S",
]


def _guess_datetime_format(samples: list[str]) -> str | None:
    for fmt in _FORMATS:
        try:
            for s in samples:
                pd.to_datetime(s.strip(), format=fmt)
            return fmt
        except (ValueError, TypeError):
            continue
    return None  # fall back to pandas inference


def _localize(idx: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """
    DST-safe localisation. Autumn gives duplicated wall-clock hours and spring
    gives non-existent ones; both raise by default and would kill a ten-year
    load on a single bar.
    """
    if tz in (None, "", "UTC"):
        return idx.tz_localize("UTC")
    return idx.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")


def load_csv(path: str | Path, tz: str = "UTC",
             fmt: CsvFormat | None = None,
             chunksize: int = 400_000,
             progress=None) -> pd.DataFrame:
    """
    Load one M1 CSV into a tz-aware, UTC-indexed OHLCV frame.

    Read in chunks, and let pandas parse the numeric columns natively rather
    than pulling the whole file in as strings first. Ten years of M1 is several
    million rows; as Python string objects that is well over a gigabyte before
    a single number exists, which is how you get a MemoryError on a 1 MiB
    allocation. Each chunk is converted to float64 and datetime immediately, so
    peak memory stays near the size of one chunk rather than the whole file.
    """
    path = Path(path)
    fmt = fmt or sniff(path)
    usecols = sorted({*fmt.columns.values(),
                      *(c for c in (fmt.date_col, fmt.time_col) if c is not None)})
    pos = {orig: i for i, orig in enumerate(usecols)}
    ts_cols = {fmt.date_col, fmt.time_col, fmt.columns.get("datetime")}
    ts_cols.discard(None)

    # Timestamp columns must stay text; everything else pandas can parse as a
    # number directly, including European decimals via the `decimal` argument.
    dtype = {c: str for c in usecols if c in ts_cols}

    reader = pd.read_csv(
        path, sep=fmt.delimiter, header=0 if fmt.has_header else None,
        usecols=usecols, decimal=fmt.decimal, engine="c",
        dtype=dtype, encoding="utf-8-sig", on_bad_lines="skip",
        chunksize=chunksize, memory_map=False,
    )

    parts = []
    rows_seen = 0
    for chunk in reader:
        chunk.columns = range(len(chunk.columns))

        idx = _parse_timestamps(chunk, pos, fmt)

        piece = pd.DataFrame(index=pd.DatetimeIndex(idx))
        for role in ("open", "high", "low", "close", "volume"):
            if role not in fmt.columns:
                continue
            col = chunk[pos[fmt.columns[role]]]
            if col.dtype == object:
                # Only reached when the column had stray text in it; the
                # `decimal` argument already handled the normal comma case.
                col = pd.to_numeric(
                    col.astype(str).str.replace(" ", "", regex=False)
                       .str.replace(",", ".", regex=False), errors="coerce")
            # .to_numpy() is load-bearing: assigning a RangeIndex Series onto a
            # DatetimeIndex frame aligns on the index and yields all-NaN
            # silently, which reads downstream as "the file was empty".
            piece[role] = pd.to_numeric(col, errors="coerce").to_numpy(
                dtype="float64", na_value=np.nan)
        if "volume" not in piece.columns:
            piece["volume"] = np.float32(np.nan)
        else:
            piece["volume"] = piece["volume"].astype("float32")

        piece = piece[~piece.index.isna()]
        piece = piece.dropna(subset=["open", "high", "low", "close"])
        if len(piece):
            parts.append(piece)

        rows_seen += len(chunk)
        if progress is not None:
            progress(rows_seen)
        del chunk, idx

    if not parts:
        return _empty_frame()

    out = pd.concat(parts, copy=False)
    parts.clear()
    if out.index.duplicated().any():
        out = out[~out.index.duplicated(keep="first")]
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()

    out.index = _localize(pd.DatetimeIndex(out.index), tz)
    out = out[~out.index.isna()]
    out.index = out.index.tz_convert("UTC")
    out.index.name = "ts"

    bad = (out["high"] < out["low"]) | (out["high"] < out["open"]) | \
          (out["high"] < out["close"]) | (out["low"] > out["open"]) | \
          (out["low"] > out["close"])
    if bad.any():
        out = out[~bad]
    return out


def _parse_timestamps(chunk, pos, fmt) -> pd.DatetimeIndex:
    """
    Parse a chunk's timestamps as cheaply as possible.

    With separate DATE and TIME columns, the two are parsed independently and
    added. A date column over ten years has ~2,500 distinct values and a time
    column 1,440, so pandas' parse cache does almost all the work. Gluing them
    into one string first would instead create millions of unique strings,
    which is both slower and the single largest memory spike in the load.
    """
    if fmt.date_col is not None and fmt.time_col is not None:
        dfmt = (fmt.datetime_format.split(" ")[0]
                if fmt.datetime_format else None)
        dates = pd.to_datetime(chunk[pos[fmt.date_col]].str.strip(),
                               format=dfmt, errors="coerce", cache=True)
        times = pd.to_timedelta(
            chunk[pos[fmt.time_col]].str.strip().where(
                lambda x: x.str.count(":") == 2, other=None
            ).fillna(chunk[pos[fmt.time_col]].str.strip() + ":00"),
            errors="coerce")
        return pd.DatetimeIndex(dates + times)

    col = (fmt.date_col if fmt.date_col is not None
           else fmt.columns["datetime"])
    ts = chunk[pos[col]].str.strip()
    return pd.DatetimeIndex(
        pd.to_datetime(ts, format=fmt.datetime_format, errors="coerce",
                       cache=True)
        if fmt.datetime_format else
        pd.to_datetime(ts, errors="coerce", cache=True))


def _empty_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame({c: pd.Series(dtype="float64")
                         for c in ("open", "high", "low", "close", "volume")},
                        index=idx)


def _cache_key(path: Path, tz: str) -> str:
    st = path.stat()
    raw = f"{path.name}|{st.st_size}|{int(st.st_mtime)}|{tz}|v2"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load_folder_symbol(folder: str | Path, symbol: str, tz: str = "UTC",
                       use_cache: bool = True, progress=None,
                       years: tuple | None = None) -> pd.DataFrame:
    """
    Load every CSV in `folder` whose stem starts with `symbol`, concatenated.
    Handles the common case of one file per year.
    """
    folder = Path(folder)
    files = sorted(p for p in folder.glob("*")
                   if p.suffix.lower() in (".csv", ".txt")
                   and p.stem.upper().startswith(symbol.upper()))
    if not files:
        raise FileNotFoundError(f"no CSV files for {symbol!r} in {folder}")

    # When files are named per year, skip the ones outside the requested range
    # before parsing rather than after. Halving the load is the cheapest
    # possible fix for a machine that cannot hold the whole decade.
    if years:
        lo, hi = years
        keep = []
        for f in files:
            found = re.findall(r"(19|20)\d{2}", f.stem)
            yrs = [int(f.stem[m.start():m.start() + 4])
                   for m in re.finditer(r"(?:19|20)\d{2}", f.stem)]
            if not yrs or any(lo <= y <= hi for y in yrs):
                keep.append(f)     # unnamed years are kept, then trimmed below
        files = keep or files

    cache_dir = folder / CACHE_DIRNAME
    try:
        cache_dir.mkdir(exist_ok=True)
    except OSError:
        use_cache = False   # read-only location; parse every time instead

    parts = []
    for n, f in enumerate(files, 1):
        if progress is not None:
            progress(n / len(files), f.name)
        cache = cache_dir / f"{f.stem}_{_cache_key(f, tz)}.parquet"
        if use_cache and cache.exists():
            parts.append(pd.read_parquet(cache))
            continue
        try:
            df = load_csv(f, tz=tz)
        except MemoryError:
            raise MemoryError(
                f"Ran out of memory parsing {f.name} "
                f"({f.stat().st_size / 1e9:.1f} GB).\n"
                f"Python is {8 * struct.calcsize('P')}-bit"
                + (" — a 32-bit build caps out near 2 GB no matter how much "
                   "RAM the machine has. Reinstall 64-bit Python."
                   if struct.calcsize("P") == 4 else
                   ". Close other applications, or split the file by year and "
                   "load one at a time; each year is cached separately.")
            ) from None
        if use_cache and len(df):
            try:
                df.to_parquet(cache, compression="snappy")
            except Exception:
                pass  # cache is an optimisation, never a hard requirement
        parts.append(df)

    out = pd.concat(parts, copy=False) if len(parts) > 1 else parts[0]
    parts.clear()
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    if out.index.duplicated().any():
        out = out[~out.index.duplicated(keep="first")]
    return out


def discover_symbols(folder: str | Path) -> list[str]:
    """
    Guess the symbol list from filenames. Strips trailing year/period suffixes
    so EURUSD_2019.csv and EURUSD_2020.csv collapse into one symbol.
    """
    folder = Path(folder)
    stems = [p.stem for p in folder.glob("*")
             if p.suffix.lower() in (".csv", ".txt")]
    syms = set()
    for s in stems:
        s = re.split(r"[_\-. ]", s)[0]
        s = re.sub(r"(M1|M5|M15|H1|D1)$", "", s, flags=re.I)
        if s:
            syms.add(s.upper())
    return sorted(syms)


def resample(m1: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """M1 -> detection timeframe. Bars are left-labelled and left-closed."""
    if pd.Timedelta(timeframe) <= pd.Timedelta("1min"):
        return m1.copy()
    out = m1.resample(timeframe, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    })
    return out.dropna(subset=["open", "high", "low", "close"])


def data_fingerprint(df: pd.DataFrame) -> str:
    """Identifies the dataset a study ran against, so results stay traceable."""
    if df.empty:
        return "empty"
    raw = (f"{len(df)}|{df.index[0].isoformat()}|{df.index[-1].isoformat()}|"
           f"{float(df['close'].iloc[0]):.6f}|{float(df['close'].iloc[-1]):.6f}")
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def estimate_memory(folder: str | Path, symbol: str) -> dict:
    """
    Rough sizing before a load. Bar count is estimated from file size and the
    average line length of a small sample, which is close enough to warn on.
    """
    folder = Path(folder)
    files = sorted(p for p in folder.glob("*")
                   if p.suffix.lower() in (".csv", ".txt")
                   and p.stem.upper().startswith(symbol.upper()))
    if not files:
        return {}
    total = sum(p.stat().st_size for p in files)
    with open(files[0], "r", encoding="utf-8-sig", errors="replace") as fh:
        lines = [fh.readline() for _ in range(200)]
    lines = [x for x in lines if x.strip()]
    avg = max(1.0, sum(len(x) for x in lines) / max(1, len(lines)))
    rows = int(total / avg)
    return {
        "files": len(files),
        "bytes": total,
        "est_rows": rows,
        # 5 float columns + a datetime index, all 8 bytes wide
        "est_ram_mb": rows * 6 * 8 / 1e6,
        "bits": 8 * struct.calcsize("P"),
    }
