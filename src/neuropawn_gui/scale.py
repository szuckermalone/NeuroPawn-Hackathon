"""Correct the Knight board's microvolt scale for BrainFlow releases that predate fix #867.

BrainFlow <= 5.23.0 converts Knight ADC counts with ``4 / 32767 / 12 * 1e6`` (10.1728 uV
per count) and stores the result as float32. Upstream later added a ``/ 79.57`` term
(PR #867, 2026-10-21), giving 0.12785 uV per count. Data from the old release is 79.57x
too large.

The correction is exact, not approximate. Each old-scale value is an integer ADC count
times 10.1728 (rounded to float32), so we recover the integer count with ``rint`` and
multiply by the corrected factor. The result is bit-identical to what a fixed BrainFlow
would have produced.

The check runs on every chunk instead of trusting the installed version, so the app keeps
working after a BrainFlow upgrade. If the values are already on the corrected scale they
pass through untouched. If they fit neither scale they also pass through untouched, and
the chunk is counted as "unknown" so the problem is visible.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from .boards import ADC_MAX_COUNTS, KNIGHT_LEGACY_UV_PER_COUNT, KNIGHT_UV_PER_COUNT

LEGACY = "legacy"    # pre-#867 scale -> corrected
CORRECT = "correct"  # already on the fixed scale
EMPTY = "empty"      # all zeros (identical under both scales)
UNKNOWN = "unknown"  # fits neither; passed through unchanged

_LEGACY_TOL = 0.02   # in ADC counts; float32 rounding of |v|<=333k uV is < 0.003 counts
_CORRECT_TOL = 1e-4


def classify(values: np.ndarray) -> str:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v) & (v != 0.0)]
    if v.size == 0:
        return EMPTY
    r = v / KNIGHT_LEGACY_UV_PER_COUNT
    if np.all(np.abs(r - np.rint(r)) < _LEGACY_TOL) and np.all(np.abs(np.rint(r)) <= ADC_MAX_COUNTS + 1):
        return LEGACY
    r = v / KNIGHT_UV_PER_COUNT
    if np.all(np.abs(r - np.rint(r)) < _CORRECT_TOL):
        return CORRECT
    return UNKNOWN


def legacy_to_corrected(values: np.ndarray) -> np.ndarray:
    counts = np.rint(np.asarray(values, dtype=np.float64) / KNIGHT_LEGACY_UV_PER_COUNT)
    return counts * KNIGHT_UV_PER_COUNT


class KnightScaleCorrector:
    """Applied to every chunk of Knight EEG before it's recorded or displayed."""

    def __init__(self):
        self.verdicts: Counter = Counter()
        self.last: str | None = None

    def apply(self, eeg: np.ndarray) -> tuple[np.ndarray, str]:
        verdict = classify(eeg)
        self.verdicts[verdict] += 1
        self.last = verdict
        if verdict == LEGACY:
            return legacy_to_corrected(eeg), verdict
        return eeg, verdict


def describe(brainflow_version: str | None) -> dict:
    return {
        "mode": "auto (checked on every chunk)",
        "brainflow_version": brainflow_version,
        "legacy_uv_per_count": KNIGHT_LEGACY_UV_PER_COUNT,
        "corrected_uv_per_count": KNIGHT_UV_PER_COUNT,
        "note": "BrainFlow <= 5.23.0 reports Knight EEG 79.57x too large (fixed upstream in #867). "
                "Chunks detected as legacy are converted exactly: counts = rint(v / legacy), "
                "uV = counts * corrected. Chunks already on the corrected scale are untouched.",
    }


def fix_csv(path: Path) -> Path:
    """Write ``<name>_scalefixed.csv`` (+ .json) with corrected EEG columns. The original is untouched.

    Only the ``chN_uV`` cells change; every other cell is copied byte-for-byte.
    """
    path = Path(path)
    out = path.with_name(path.stem + "_scalefixed.csv")
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    eeg_idx = [header.index(f"ch{i}_uV") for i in range(1, 9)]
    vals = np.array([[float(r[i]) for i in eeg_idx] for r in rows])
    verdict = classify(vals)
    if verdict != LEGACY:
        raise ValueError(f"{path.name}: EEG values are not on the legacy scale (verdict: {verdict}); nothing to fix")
    fixed = legacy_to_corrected(vals).tolist()
    for r, fr in zip(rows, fixed):
        for k, i in enumerate(eeg_idx):
            r[i] = repr(fr[k])
    tmp = out.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, out)

    sidecar = path.with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    meta.setdefault("board", {})["uv_per_adc_count"] = KNIGHT_UV_PER_COUNT
    meta["board"]["scale_correction"] = {**describe(None), "applied_offline_from": path.name}
    meta["csv_file"] = out.name
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Fix the Knight uV scale in recordings made with BrainFlow <= 5.23.0")
    ap.add_argument("csv", nargs="+", type=Path)
    args = ap.parse_args(argv)
    rc = 0
    for p in args.csv:
        try:
            print(f"{p.name} -> {fix_csv(p).name}")
        except (ValueError, FileExistsError) as e:
            print(f"{p.name}: skipped ({e})")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
