"""End-to-end: real BrainFlow (synthetic board) -> engine -> CSV, compared value-for-value."""

import csv
import json
import time

import numpy as np

from neuropawn_gui.acquisition import AcquisitionEngine, State
from neuropawn_gui.boards import BOARDS_BY_KEY
from neuropawn_gui.recorder import sidecar_path


def _wait_for(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_csv_matches_brainflow_output_exactly(tmp_path):
    eng = AcquisitionEngine()
    captured = []
    eng.chunk_listeners.append(captured.append)
    eng.connect(BOARDS_BY_KEY["synthetic"])
    assert _wait_for(lambda: eng.stats().state is State.STREAMING)

    path = tmp_path / "synthetic.csv"
    eng.start_recording(path)
    assert _wait_for(lambda: eng.stats().recording_path is not None)
    rec_start_chunk = len(captured)  # recording starts at the chunk after this point... see below
    eng.insert_marker(7)
    time.sleep(2.0)
    eng.disconnect()  # stops while recording: tail must be drained into the file
    assert eng.join(10)
    st = eng.stats()
    assert st.state is State.IDLE, st.message

    layout = eng.layout
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    file_data = np.array(body, dtype=np.float64)
    n = len(body)
    assert n > 1.5 * layout.sampling_rate

    # Find where the recording begins in the captured stream and compare everything after it.
    all_src = np.concatenate(captured, axis=1)
    ts_col = header.index("timestamp_unix")
    start = int(np.where(all_src[layout.timestamp_row] == file_data[0, ts_col])[0][0])
    # Recording must begin exactly at a chunk boundary (never mid-chunk) ...
    boundaries = set(np.cumsum([0] + [c.shape[1] for c in captured]).tolist())
    assert start in boundaries
    assert rec_start_chunk >= 1
    # ... and contain every sample after it.
    src = all_src[:, start:]
    assert src.shape[1] == n, "file must contain every sample received after recording began, and nothing else"

    eeg_cols = [header.index(f"ch{i}_uV") for i in range(1, 9)]
    assert np.array_equal(file_data[:, eeg_cols].T, src[layout.eeg_rows])
    assert np.array_equal(file_data[:, ts_col], src[layout.timestamp_row])
    assert np.array_equal(file_data[:, header.index("package_num")], src[layout.package_row])
    assert np.array_equal(file_data[:, header.index("sample_index")], np.arange(n))
    assert (file_data[:, header.index("gap_before")] == 0).all()
    assert (file_data[:, header.index("marker")] == 7).sum() == 1

    meta = json.loads(sidecar_path(path).read_text())
    assert meta["clean_shutdown"] is True
    assert meta["summary"]["rows"] == n
    assert meta["summary"]["missing_packets"] == 0
    assert st.last_recording_summary["rows"] == n


def test_display_buffer_is_a_copy(tmp_path):
    """Mutating what the display reads must not affect what gets recorded."""
    eng = AcquisitionEngine()
    eng.connect(BOARDS_BY_KEY["synthetic"])
    assert _wait_for(lambda: eng.stats().state is State.STREAMING)
    eng.start_recording(tmp_path / "c.csv")
    time.sleep(0.6)
    block, _ = eng.display.latest(100)
    block[:] = 0.0  # vandalize the display copy
    time.sleep(0.4)
    eng.disconnect()
    assert eng.join(10)
    data = np.loadtxt(tmp_path / "c.csv", delimiter=",", skiprows=1)
    assert np.abs(data[:, 4:12]).sum() > 0


def test_configuration_phase_blocks_recording(monkeypatch):
    """Knight path: chon/rldadd are sent per channel; recording is refused until setup completes."""
    import pytest
    from neuropawn_gui import acquisition
    from neuropawn_gui.boards import BoardSpec, SYNTHETIC_ID

    for name in ("CONFIG_SETTLE_S", "CONFIG_PRE_S", "CONFIG_BETWEEN_S", "CONFIG_POST_S"):
        monkeypatch.setattr(acquisition, name, 0.05)
    sent = []
    from brainflow.board_shim import BoardShim
    monkeypatch.setattr(BoardShim, "config_board", lambda self, cmd: sent.append(cmd) or "")

    eng = AcquisitionEngine()
    eng.connect(BoardSpec("cfg", "synthetic+config", SYNTHETIC_ID, False, True))
    assert _wait_for(lambda: eng.stats().state is State.CONFIGURING)
    with pytest.raises(RuntimeError):
        eng.start_recording("/tmp/should-not-exist.csv")
    assert _wait_for(lambda: eng.stats().state is State.STREAMING, timeout=15)
    eng.disconnect()
    assert eng.join(10)
    assert sent == [c for pair in [(f"chon_{i}_12", f"rldadd_{i}") for i in range(1, 9)] for c in pair]


def test_bad_port_reports_error():
    eng = AcquisitionEngine()
    eng.connect(BOARDS_BY_KEY["knight"], "/dev/cu.does-not-exist")
    assert eng.join(15)
    st = eng.stats()
    assert st.state is State.ERROR
    assert "does-not-exist" in st.error
