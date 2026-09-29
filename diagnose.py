"""
Memory diagnostic.

Run this when a load fails with MemoryError:

    python diagnose.py "C:/path/to/your/csv/folder"

It reports the three things that decide whether your data fits: Python's
bitness, available RAM, and how big the files actually are.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} PB"


def main(folder: str | None):
    bits = 8 * struct.calcsize("P")
    print("=" * 62)
    print("PYTHON")
    print(f"  version    : {sys.version.split()[0]}")
    print(f"  bitness    : {bits}-bit")
    print(f"  executable : {sys.executable}")
    if bits == 32:
        print()
        print("  >>> THIS IS YOUR PROBLEM. <<<")
        print("  A 32-bit Python cannot address more than about 2 GB, however")
        print("  much RAM the machine has. Ten years of M1 will not fit.")
        print("  Install 64-bit Python from python.org (the installer is")
        print("  labelled 'Windows installer (64-bit)'), then reinstall the")
        print("  requirements and try again.")

    print()
    print("MEMORY")
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"  total      : {human(vm.total)}")
        print(f"  available  : {human(vm.available)}")
        print(f"  in use     : {vm.percent:.0f}%")
        avail = vm.available
    except ImportError:
        avail = None
        print("  (pip install psutil for RAM figures)")
        if os.name == "nt":
            print("  Meanwhile: Task Manager -> Performance -> Memory")

    print()
    print("LIBRARIES")
    for mod in ("pandas", "numpy", "pyarrow"):
        try:
            m = __import__(mod)
            print(f"  {mod:<10}: {m.__version__}")
        except ImportError:
            print(f"  {mod:<10}: NOT INSTALLED"
                  + ("  <-- needed for the Parquet cache" if mod == "pyarrow"
                     else ""))

    if not folder:
        print()
        print("Pass your CSV folder to check file sizes:")
        print('  python diagnose.py "C:/path/to/csv/folder"')
        return

    print()
    print("DATA")
    fp = Path(folder).expanduser()
    if not fp.exists():
        print(f"  folder not found: {fp}")
        return

    files = sorted(p for p in fp.glob("*")
                   if p.suffix.lower() in (".csv", ".txt"))
    if not files:
        print(f"  no CSV/TXT files in {fp}")
        return

    by_symbol: dict[str, list] = {}
    for f in files:
        sym = f.stem.split("_")[0].split("-")[0].upper()
        by_symbol.setdefault(sym, []).append(f)

    print(f"  folder     : {fp}")
    print(f"  files      : {len(files)}")
    print()
    print(f"  {'symbol':<14}{'files':>7}{'on disk':>14}{'est. rows':>14}{'est. RAM':>12}")
    print("  " + "-" * 59)

    worst = 0
    for sym, fs in sorted(by_symbol.items()):
        size = sum(f.stat().st_size for f in fs)
        rows = size / 45          # ~45 bytes per M1 row of CSV text
        ram = rows * 48 * 2       # float64 OHLC + index, doubled at concat
        worst = max(worst, ram)
        print(f"  {sym:<14}{len(fs):>7}{human(size):>14}{rows:>14,.0f}{human(ram):>12}")

    print()
    biggest = max(files, key=lambda f: f.stat().st_size)
    print(f"  largest file: {biggest.name} ({human(biggest.stat().st_size)})")

    print()
    print("VERDICT")
    if bits == 32:
        print("  Install 64-bit Python. Nothing else will help.")
    elif avail is not None and worst > avail * 0.7:
        print(f"  The largest symbol needs roughly {human(worst)} but only")
        print(f"  {human(avail)} is available. Options, cheapest first:")
        print("    - close other applications (browsers are the usual culprit)")
        print("    - load a shorter history: set the start year in the sidebar")
        print("    - split the big file into one file per year; each year is")
        print("      cached separately and the cache is far smaller than CSV")
    elif avail is not None:
        print(f"  Roughly {human(worst)} needed against {human(avail)} available.")
        print("  This should fit. If it still fails, the file may have a")
        print("  malformed section that is being read as text — run the")
        print("  Preview format button in the sidebar and check the output.")
    else:
        print(f"  Largest symbol needs roughly {human(worst)}. Compare that")
        print("  against your free RAM in Task Manager.")
    print("=" * 62)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
