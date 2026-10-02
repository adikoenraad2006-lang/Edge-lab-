# Edge Lab

Describe a trade idea in English, turn it into mechanical rules, and measure how
often it actually worked — with a confidence interval, a control group, and a
sealed holdout.

It does not tell you what to trade. It tells you whether an observation is worth
the cost of a full MT5 tick backtest.

## Install

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens in your browser. Everything runs locally; price data never leaves your
machine. The Anthropic API key is optional and is only used to translate English
into a rule spec — you can write the spec by hand and never call the API.

## Layout

```
app.py          Streamlit UI (run from this folder)
selftest.py     engine checks
diagnose.py     memory diagnostic for large CSV folders
edgelab/        the engine package: data, detectors, features, filters,
                scanner, stats, spec, library, nl, mql5, charts
```

Always run the commands from the repo root so `import edgelab` resolves.

## First run

1. **Sidebar** — point at your CSV folder, pick a symbol, confirm the timezone
   of the raw timestamps, set the in-sample end date.
2. **Idea tab** — describe the setup in plain English, hit Translate.
3. **Spec tab** — read what it produced. Fix anything wrong. Run scan.
4. **Results** — win rate, interval, and the baselines. Run both baselines
   before believing anything.
5. **Holdout** — only when you have stopped changing the spec.

## Verify it before you trust it

```bash
python selftest.py
```

The first check is the one that matters: on a driftless random walk, a 1R stop
with a 2R target must land near 33%, because geometry alone produces that. It
currently reports 36%. If a change ever makes that number jump to 60%, the
engine has started reading the future and every result it gives you is worthless.

## Data

Any M1 CSV. The loader sniffs the delimiter, decimal separator, header and
timestamp format. Formats covered by regression tests:

- MT5 bar export, tab or comma separated, with `<DATE> <TIME> <OPEN>` style
  bracket headers, including the trailing `<VOL>` and `<SPREAD>` columns
- MT5 export with a single combined `<DATETIME>` column
- MT4 headerless `date,time,O,H,L,C,V`
- Tickstory and other plain `DateTime,Open,High,Low,Close,Volume` exports
- European `;` separated with `,` decimals
- Unrecognised header names, via a positional fallback

One file per year is fine — files whose names start with the symbol are
concatenated. Use **Preview format** in the sidebar before a slow load: it
shows the raw first lines and exactly what was detected. If a format still
fails, the error prints the header, the first data row and what it managed to
map, which is enough to extend the sniffer.

Day-first versus month-first dates are genuinely undecidable from the file
alone (05/01/2015 is either 5 January or 1 May). Day-first is assumed and the
preview warns you when it applies.

Parsed files are cached as Parquet in a `.edgelab_cache` folder beside the CSVs,
so only the first load is slow.

**Memory.** The loader streams the file in chunks and parses numbers natively
rather than reading everything as text first, so peak memory stays close to the
size of the finished frame. A full ten years of M1 (3.7M bars, a 235 MB CSV)
loads in about 28 seconds and peaks around 600 MB including the interpreter.

If you hit a `MemoryError`, check the bit-width reported in the sidebar first.
**32-bit Python cannot address more than about 2 GB regardless of installed
RAM**, and that is the usual cause. Otherwise narrow the year range — files
whose names contain a year outside the range are skipped before parsing, so
halving the range roughly halves the memory.

**Timezone is the one thing you must get right.** Tickstory exports in the
timezone you selected at export time, usually Europe/Amsterdam. Choose wrong and
every session result is confidently wrong while looking completely normal.

## When a load runs out of memory

```bash
python diagnose.py "C:/path/to/your/csv/folder"
```

Reports Python's bitness, free RAM, and how much each symbol actually needs.

A `MemoryError` on a *small* allocation (a megabyte or so) almost always means
32-bit Python, which cannot address more than about 2 GB regardless of how much
RAM is installed. The loader is chunked and parses numerically, so genuine
exhaustion is rare on a 64-bit build with a normal amount of RAM.

If it is a real shortage, narrow the **Years to load** slider — it skips files
outside the range before parsing rather than after. The parsed Parquet cache is
much smaller than the CSV, so the second load of the same range is far cheaper.

## What the rule language covers

**Detectors:** `order_block`, `fvg`, `swing_level`, `prior_session_level`,
`round_number`.

**Triggers:** first touch, close inside, wick through. Re-tests are counted
separately from first tests, never pooled.

**Filters:** any boolean expression over session, hour, day of week, year, ATR,
zone height in ATR, bars since formation, test number, displacement size,
distance from EMA50, day range, prior day range, gap. Evaluated with a
restricted AST, so a spec cannot execute code.

**Higher-timeframe bias:** set `htf.timeframe` to anything longer than the
detection timeframe, then filter on `aligned`, `aligned_bar`, `htf_bias`,
`htf_slope_atr`, `htf_dist_ema_atr`, `htf_range_atr` or `htf_bar_progress`.

```json
"timeframe": "5min",
"htf": { "timeframe": "4h", "ema_period": 50 },
"filters": ["session == 'RTH'", "aligned == 1"]
```

Only **closed** HTF bars are read. The HTF frame is indexed by each bar's close
time rather than its open time, so a "last row at or before now" lookup cannot
land on a bar that had not finished. At 10:05 the 08:00 4h bar has not closed
and is invisible; the 04:00 one is what you get. The self-test recomputes
`htf_bias` independently and demands an exact match on every event.

`htf_bar_progress` goes above 1.0 after a weekend or holiday, because the last
closed HTF bar is then days old. That is left uncapped deliberately — filter on
`htf_bar_progress < 1.5` if you want to exclude a stale reference.

Check the alignment breakdown in the By session tab *before* adding an
alignment filter. If the with-HTF and against-HTF intervals overlap, the filter
halves your sample and buys nothing.

**Scoring:** stop at the zone's far edge, an ATR multiple, or fixed points.
Target as an R multiple, an ATR multiple, fixed points, or none. Three lenses
are computed on every run and switchable without rescanning: the spec's own
stop/target, an ATR-based target, and pure MFE-versus-MAE.

## Design decisions worth knowing

**Look-ahead is structurally prevented.** A zone carries `formed_at` — the
moment the confirmation completed, not the timestamp of the candle it was drawn
on. An order block is drawn on a candle from ten bars ago, but you could not
have known it was one until the displacement finished. The scanner refuses to
test any touch before that. The self-test asserts zero violations.

**Ambiguous bars are counted, not hidden.** When stop and target both fall
inside one M1 bar's range, the data cannot say which came first. Those are
resolved as losses and reported as a percentage. Above 15%, the app warns you:
at that rate the headline number is not measuring what you think it is.

**MFE and MAE span the full horizon**, not just up to the exit. Truncating them
at the stop would make the alternative lenses measure your exit rule rather than
the market's response.

**Two controls, and the second one is the real test.** Drift asks "did this beat
being long a market that went up." Placebo zones ask "would any random level
have done the same" — same timestamps, same zone sizes, same approach direction,
arbitrary price. On synthetic noise the placebo lands within 4% of the pattern,
which is what a correctly-calibrated control should do.

**The holdout is sealed and every unlock is logged.** After the first look it
stops being out-of-sample. Look eleven times while nudging an ATR multiple and
the twelfth result is not a test of anything. The app tells you the count.

**The trial registry is shared with `mt5scan`.** Every study writes to the same
`trials.csv`, so hypotheses tested here count toward the same lifetime total
that deflates your significance thresholds. Two separate counters is the same as
not counting.

## MQL5 export

Produces scaffolding: inputs matching the spec, the detector transcribed, and
the stop/target arithmetic. Position sizing, spread handling, session guards,
prop-firm drawdown rules and order management are marked TODO. Only
`order_block` has a real MQL5 translation; other detectors emit an explicit stub
rather than pretending.

Anything that claims to turn a statistical spec into a live EA without human
work is skipping the part where money is lost.

## Known limitations

- **No costs anywhere.** Every number is gross. A CFD spread will eat part of
  any edge, and the shorter the holding period the more it eats.
- **M1 resolution caps what is answerable.** Sub-minute sequencing is
  unknowable; that is what the ambiguous count exists to surface.
- **Thin zones distort R multiples.** A zone 0.05 ATR tall produces enormous R
  numbers from tiny moves. Filter with `zone_height_atr > 0.2` unless you mean
  to include them.
- **Detection is on resampled bars.** Resampled M5 from M1 is not identical to
  broker-native M5 bars; boundaries can differ.
- **HTF alignment is not implemented in the MQL5 export.** The filter list is
  printed as a comment in the generated file; you must code it yourself.
- **A surviving study is a reason to backtest properly, not a strategy.**
