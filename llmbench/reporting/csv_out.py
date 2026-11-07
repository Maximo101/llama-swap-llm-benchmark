"""CSV output."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

from ..scoring import FIELDS

def _rounded(r): return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}


def write_csv(results, path, append=False):
    new = not (append and Path(path).exists())
    if not new:
        # The column set grew (eval columns): re-write an older history file under the new header first,
        # otherwise the appended rows would sit under the wrong columns. A .bak copy is kept.
        with open(path, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            old_fields = rd.fieldnames or []
            old_rows = list(rd) if old_fields != FIELDS else None
        if old_rows is not None:
            shutil.copy2(path, str(path) + ".bak")
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(old_rows)
    with open(path, "a" if append else "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(_rounded(r) for r in results)
