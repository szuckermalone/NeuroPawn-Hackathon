import csv
import json
from pathlib import Path

import numpy as np
import pytest

from neuropawn_gui import scale
from neuropawn_gui.acquisition import AcquisitionEngine
from neuropawn_gui.boards import (
    KNIGHT_ID, KNIGHT_LEGACY_UV_PER_COUNT, KNIGHT_UV_PER_COUNT, ChannelLayout,
)
from neuropawn_gui.integrity import SequenceTracker
from neuropawn_gui.recorder import CsvRecorder
from neuropawn_gui.ringbuffer import RingBuffer

REAL = Path(__file__).resolve().parents[1] / "data/recordings/neuropawn_knight_2026-11-07_16-31-49.csv"


def _legacy(counts):
    # What BrainFlow 5.23.0 emits: counts * 10.1728, stored as float32.
    return (counts * KNIGHT_LEGACY_UV_PER_COUNT).astype(np.float32).astype(np.float64)


def test_factor_is_79_57():
    assert KNIGHT_LEGACY_UV_PER_COUNT / KNIGHT_UV_PER_COUNT == pytest.approx(79.57, rel=1e-12)


@pytest.mark.parametrize("n", [1, 5, 200])
def test_legacy_detected_and_corrected_exactly(n):
    counts = np.random.default_rng(n).integers(-32767, 32768, size=(8, n))
    v = _legacy(counts)
    assert scale.classify(v) == scale.LEGACY
    fixed = scale.legacy_to_corrected(v)
    assert np.array_equal(fixed, counts * KNIGHT_UV_PER_COUNT)  # bit-identical to fixed BrainFlow


def test_corrected_scale_passes_through():
    counts = np.random.default_rng(1).integers(-32767, 32768, size=(8, 50))
    v = counts * KNIGHT_UV_PER_COUNT
    c = scale.KnightScaleCorrector()
    out, verdict = c.apply(v)
    assert verdict == scale.CORRECT and out is v


def test_unknown_and_empty():
    assert scale.classify(np.zeros((8, 10))) == scale.EMPTY
    noise = np.random.default_rng(2).normal(0, 50, (8, 10))
    c = scale.KnightScaleCorrector()
    out, verdict = c.apply(noise)
    assert verdict == scale.UNKNOWN and out is noise and c.verdicts[scale.UNKNOWN] == 1


def test_engine_records_corrected_values(tmp_path):
    """Knight path through the engine (no hardware): legacy chunk in -> corrected uV in CSV."""
    layout = ChannelLayout.for_board(KNIGHT_ID)
    eng = AcquisitionEngine()
    eng.layout = layout
    eng.display = RingBuffer(len(layout.display_rows), 1000)
    eng._scale = scale.KnightScaleCorrector()
    eng._recorder = CsvRecorder(tmp_path / "k.csv", layout)
    counts = np.random.default_rng(3).integers(-3000, 3000, size=(8, 40))
    data = np.zeros((layout.num_rows, 40))
    data[layout.package_row] = np.arange(40)
    data[layout.eeg_rows] = _legacy(counts)
    data[layout.timestamp_row] = 1e9 + np.arange(40) / 125
    raw_copy = data.copy()
    eng._process(data, SequenceTracker())
    summary = eng._recorder.close()
    assert np.array_equal(data, raw_copy), "caller's chunk must not be modified"
    with open(tmp_path / "k.csv") as f:
        rows = list(csv.DictReader(f))
    got = np.array([[float(r[f"ch{i}_uV"]) for i in range(1, 9)] for r in rows]).T
    assert np.array_equal(got, counts * KNIGHT_UV_PER_COUNT)
    assert summary["scale_check_chunks"] == {"legacy": 1}
    block, _ = eng.display.latest(40)
    assert np.array_equal(block[:8], counts * KNIGHT_UV_PER_COUNT)


def test_fix_csv_changes_only_eeg_cells(tmp_path):
    layout = ChannelLayout.for_board(KNIGHT_ID)
    rec = CsvRecorder(tmp_path / "old.csv", layout)
    counts = np.random.default_rng(4).integers(-2000, 2000, size=(8, 30))
    data = np.zeros((layout.num_rows, 30))
    data[layout.package_row] = np.arange(30)
    data[layout.eeg_rows] = _legacy(counts)
    data[layout.timestamp_row] = 1.7e9 + np.arange(30) * 0.008
    rec.write_chunk(data, np.zeros(30, dtype=int))
    rec.close()
    out = scale.fix_csv(tmp_path / "old.csv")
    a = list(csv.reader(open(tmp_path / "old.csv")))
    b = list(csv.reader(open(out)))
    hdr = a[0]
    eeg_idx = {hdr.index(f"ch{i}_uV") for i in range(1, 9)}
    for ra, rb in zip(a[1:], b[1:]):
        for i, (ca, cb) in enumerate(zip(ra, rb)):
            if i not in eeg_idx:
                assert ca == cb
    got = np.array([[float(r[i]) for i in sorted(eeg_idx)] for r in b[1:]]).T
    assert np.array_equal(got, counts * KNIGHT_UV_PER_COUNT)
    assert json.loads(out.with_suffix(".json").read_text())["board"]["scale_correction"]
    with pytest.raises(FileExistsError):
        scale.fix_csv(tmp_path / "old.csv")


@pytest.mark.skipif(not REAL.exists(), reason="real recording not present")
def test_real_recording_is_legacy_in_every_sample():
    d = np.loadtxt(REAL, delimiter=",", skiprows=1)
    eeg = d[:, 4:12].T
    assert all(scale.classify(eeg[:, i:i + 1]) == scale.LEGACY for i in range(0, eeg.shape[1], 7))
    fixed = scale.legacy_to_corrected(eeg)
    assert np.allclose(fixed, eeg / 79.57, atol=1e-4)
    assert np.abs(fixed).max() < 400  # realistic EEG range once corrected
