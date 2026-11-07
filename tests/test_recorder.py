import csv
import json

import numpy as np
import pytest

from neuropawn_gui.boards import KNIGHT_ID, KNIGHT_IMU_ID, KNIGHT_UV_PER_COUNT, ChannelLayout
from neuropawn_gui.integrity import SequenceTracker
from neuropawn_gui.recorder import CsvRecorder, sidecar_path


def _fake_knight_chunk(layout, n, start_pkg=0, t0=1_762_500_000.0, rng=None):
    rng = rng or np.random.default_rng(0)
    data = np.zeros((layout.num_rows, n))
    data[layout.package_row] = (np.arange(n) + start_pkg) % 256
    # Real Knight values are int16 counts * scale; reproduce that exactly.
    counts = rng.integers(-32767, 32768, size=(8, n))
    data[layout.eeg_rows] = counts * KNIGHT_UV_PER_COUNT
    data[layout.extra_rows[0]] = rng.integers(0, 256, n)
    data[layout.extra_rows[1]] = rng.integers(0, 256, n)
    if layout.has_imu:
        data[layout.extra_rows[2:]] = rng.standard_normal((9, n)).astype(np.float32)
    data[layout.timestamp_row] = t0 + np.cumsum(rng.uniform(0.001, 0.02, n))
    data[layout.marker_row, n // 2] = 3.0
    return data, counts


@pytest.mark.parametrize("board_id", [KNIGHT_ID, KNIGHT_IMU_ID])
def test_values_round_trip_bit_exact(tmp_path, board_id):
    layout = ChannelLayout.for_board(board_id)
    path = tmp_path / "rec.csv"
    rec = CsvRecorder(path, layout, {"serial_port": "/dev/test"})
    tracker = SequenceTracker()
    chunks = []
    for k in range(5):
        data, counts = _fake_knight_chunk(layout, 37, start_pkg=k * 37, rng=np.random.default_rng(k))
        rec.write_chunk(data, tracker.process(data[layout.package_row]))
        chunks.append((data, counts))
    summary = rec.close()

    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    assert header == layout.csv_columns()
    assert len(body) == 5 * 37 == summary["rows"]

    src = np.concatenate([c[0] for c in chunks], axis=1)
    counts = np.concatenate([c[1] for c in chunks], axis=1)
    col = {name: i for i, name in enumerate(header)}
    for j, row in enumerate(body):
        assert int(row[col["sample_index"]]) == j
        assert float(row[col["timestamp_unix"]]) == src[layout.timestamp_row, j]
        assert int(row[col["package_num"]]) == src[layout.package_row, j]
        assert int(row[col["gap_before"]]) == 0
        for ch in range(8):
            v = float(row[col[f"ch{ch + 1}_uV"]])
            assert v == src[layout.eeg_rows[ch], j]  # bit-exact float64
            # ...and the original integer ADC count is recoverable exactly.
            assert round(v / KNIGHT_UV_PER_COUNT) == counts[ch, j]
        for name, r in zip(layout.extra_names, layout.extra_rows):
            assert float(row[col[name]]) == src[r, j]
        assert float(row[col["marker"]]) == src[layout.marker_row, j]

    meta = json.loads(sidecar_path(path).read_text())
    assert meta["clean_shutdown"] is True
    assert meta["summary"]["rows"] == 185
    assert meta["summary"]["markers"] == 5
    assert meta["board"]["sampling_rate_hz"] == 125
    assert meta["board"]["uv_per_adc_count"] == KNIGHT_UV_PER_COUNT
    assert meta["column_order"] == header


def test_gap_flags_written(tmp_path):
    layout = ChannelLayout.for_board(KNIGHT_ID)
    rec = CsvRecorder(tmp_path / "g.csv", layout)
    tracker = SequenceTracker()
    data, _ = _fake_knight_chunk(layout, 10)
    data[layout.package_row] = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]
    rec.write_chunk(data, tracker.process(data[layout.package_row]))
    s = rec.close()
    assert s["gap_events"] == 1 and s["missing_packets"] == 2
    with open(tmp_path / "g.csv") as f:
        gaps = [int(r["gap_before"]) for r in csv.DictReader(f)]
    assert gaps == [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]


def test_refuses_to_overwrite(tmp_path):
    layout = ChannelLayout.for_board(KNIGHT_ID)
    p = tmp_path / "x.csv"
    p.write_text("precious")
    with pytest.raises(FileExistsError):
        CsvRecorder(p, layout)
    assert p.read_text() == "precious"


def test_unclean_shutdown_is_visible(tmp_path):
    layout = ChannelLayout.for_board(KNIGHT_ID)
    rec = CsvRecorder(tmp_path / "u.csv", layout)
    meta = json.loads(sidecar_path(tmp_path / "u.csv").read_text())
    assert meta["clean_shutdown"] is False  # until close() runs
    rec.close()
