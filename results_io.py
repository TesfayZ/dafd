"""CSV output helper shared by every benchmark.

Set DAFD_QUICK=1 for a reduced-size smoke run; quick outputs go to Results/quick/ and never
overwrite the full results.
"""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUICK = os.environ.get("DAFD_QUICK", "0") == "1"
RESULTS = ROOT / "Results" / ("quick" if QUICK else "")


def save_csv(name, rows, fieldnames=None):
    """Write a list of dicts to Results/<name>. Columns are the union of keys, in first-seen order."""
    rows = list(rows)
    if not rows:
        print(f"  (no rows for {name})")
        return None
    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(dict.fromkeys(k for r in rows for k in r))
    path = RESULTS / name
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {path.relative_to(ROOT)} ({len(rows)} rows)")
    return path


def n_trials(full, quick):
    """Pick the full or the reduced trial count."""
    return quick if QUICK else full
