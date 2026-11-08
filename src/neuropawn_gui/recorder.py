"""Lossless CSV recorder.

Guarantees:
* Every value is written exactly as BrainFlow delivered it. Floats are written
  with Python's repr (shortest round-trip form), so ``float(cell)`` gives back
  the identical float64. No filtering, rounding, or resampling happens here.
* Rows are written in the order received, one row per sample, with a
  per-file ``sample_index`` and the ``gap_before`` continuity flag.
* The file is flushed after every chunk and fsynced at least once per second,
  so a crash or power loss costs at most ~1 s of data.
* A ``.json`` sidecar describes the recording (board, scale, commands sent,
  software versions). It's written at start and rewritten with an integrity
  summary on clean close (``clean_shutdown: true``).
"""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

from . import __version__
from .boards import ChannelLayout

FSYNC_INTERVAL_S = 1.0


def _int_if_integral(v: float):
    return int(v) if float(v).is_integer() else v


def _pkg_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def sidecar_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".json")


class CsvRecorder:
    def __init__(self, path: Path, layout: ChannelLayout, session_info: dict | None = None):
        self.path = Path(path)
        self.layout = layout
        if self.path.exists():
            raise FileExistsError(f"Refusing to overwrite existing recording: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "x", newline="", encoding="utf-8")
        self._writer = csv.writer(self._f, lineterminator="\n")
        self._writer.writerow(layout.csv_columns())
        self.rows_written = 0
        self.gap_events = 0
        self.missing_packets = 0
        self.repeat_events = 0
        self.backstep_events = 0
        self.marker_count = 0
        self.scale_verdicts: dict[str, int] = {}
        self.first_timestamp: float | None = None
        self.last_timestamp: float | None = None
        self._last_fsync = time.monotonic()
        self.started_at = datetime.now(timezone.utc).astimezone()
        self.closed = False
        self._meta = self._build_metadata(session_info or {})
        self._write_sidecar()
        self._f.flush()

    # ------------------------------------------------------------------------
    def _build_metadata(self, session_info: dict) -> dict:
        L = self.layout
        return {
            "format": "neuropawn-gui-csv/1",
            "csv_file": self.path.name,
            "recording_started": self.started_at.isoformat(),
            "recording_ended": None,
            "clean_shutdown": False,
            "board": {
                "name": L.board_name,
                "brainflow_board_id": L.board_id,
                "sampling_rate_hz": L.sampling_rate,
                "eeg_channels": 8,
                "gain": 12 if L.uv_per_count else None,
                "uv_per_adc_count": L.uv_per_count,
                "full_scale_uv": L.full_scale_uv,
                **session_info,
            },
            "columns": {
                "sample_index": "0-based row counter within this file (one row per sample received).",
                "timestamp_unix": "Host time (s since Unix epoch) when BrainFlow parsed the packet. "
                                  "Packets arrive over USB in bursts, so use sample_index / sampling_rate_hz "
                                  "for evenly spaced sample timing.",
                "package_num": "Board's 8-bit packet counter (0-255, wraps).",
                "gap_before": "Packets missing immediately before this row according to package_num "
                              "(0 = contiguous, -1 = counter repeated, -2 = counter went backwards: "
                              "a stale/re-sent packet from a board fault).",
                "chN_uV": "EEG channel N in microvolts, unfiltered. For Knight boards, values from BrainFlow "
                          "<= 5.23.0 are rescaled exactly to the corrected scale (see board.scale_correction); "
                          "otherwise exactly as returned by BrainFlow.",
                "loff_statp/loff_statn": "Raw lead-off status bytes from the board (bit k = channel k+1).",
                "accel/gyro/mag_*": "IMU values as returned by BrainFlow (Knight IMU board only).",
                "marker": "Event marker inserted from the GUI (0 = none).",
            },
            "column_order": L.csv_columns(),
            "software": {
                "neuropawn_gui": __version__,
                "brainflow": _pkg_version("brainflow"),
                "numpy": _pkg_version("numpy"),
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "summary": None,
        }

    def _write_sidecar(self) -> None:
        tmp = sidecar_path(self.path).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._meta, indent=2), encoding="utf-8")
        os.replace(tmp, sidecar_path(self.path))

    # ------------------------------------------------------------------------
    def write_chunk(self, data: np.ndarray, gap_before: np.ndarray, scale_verdict: str | None = None) -> int:
        """Append every sample (column) of a BrainFlow data matrix. Returns rows written.

        ``scale_verdict`` (Knight only) records how scale.py classified this chunk; the
        counts go into the summary so a reader can confirm every chunk was handled.
        """
        if self.closed:
            raise RuntimeError("Recorder is closed")
        L = self.layout
        n = int(data.shape[1])
        if n == 0:
            return 0
        if gap_before.shape[0] != n:
            raise ValueError("gap_before length does not match chunk length")

        ts = data[L.timestamp_row].tolist()
        pkg = data[L.package_row].tolist()
        eeg = data[L.eeg_rows].T.tolist()
        extras = data[L.extra_rows].T.tolist() if L.extra_rows else [[] for _ in range(n)]
        imu_start = 2 if L.has_leadoff else 0  # loff bytes are integers; IMU stays float
        marker = data[L.marker_row].tolist()
        gaps = gap_before.tolist()

        base = self.rows_written
        rows = []
        for j in range(n):
            ex = extras[j]
            ex_out = [_int_if_integral(v) for v in ex[:imu_start]] + ex[imu_start:]
            rows.append([base + j, ts[j], int(pkg[j]), int(gaps[j]), *eeg[j], *ex_out, _int_if_integral(marker[j])])
        self._writer.writerows(rows)
        self._f.flush()

        now = time.monotonic()
        if now - self._last_fsync >= FSYNC_INTERVAL_S:
            os.fsync(self._f.fileno())
            self._last_fsync = now

        self.rows_written += n
        g = np.asarray(gap_before)
        self.gap_events += int((g > 0).sum())
        self.missing_packets += int(g[g > 0].sum())
        self.repeat_events += int((g == -1).sum())
        self.backstep_events += int((g == -2).sum())
        self.marker_count += int(np.count_nonzero(data[L.marker_row]))
        if scale_verdict is not None:
            self.scale_verdicts[scale_verdict] = self.scale_verdicts.get(scale_verdict, 0) + 1
        if self.first_timestamp is None:
            self.first_timestamp = ts[0]
        self.last_timestamp = ts[-1]
        return n

    def annotate(self, key: str, value) -> None:
        """Add or replace a field in the sidecar's ``board`` section (e.g. the linked game session)."""
        self._meta["board"][key] = value
        self._write_sidecar()

    def close(self) -> dict:
        """Flush, fsync, close, and finalize the sidecar. Returns the summary."""
        if self.closed:
            return self._meta["summary"]
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        self.closed = True
        fs = self.layout.sampling_rate
        summary = {
            "rows": self.rows_written,
            "duration_s_by_sample_count": self.rows_written / fs if fs else None,
            "duration_s_by_host_clock": (
                self.last_timestamp - self.first_timestamp
                if self.first_timestamp is not None and self.last_timestamp is not None else None
            ),
            "effective_sampling_rate_hz": (
                (self.rows_written + self.missing_packets - 1) / (self.last_timestamp - self.first_timestamp)
                if self.first_timestamp is not None and self.last_timestamp and self.last_timestamp > self.first_timestamp
                else None
            ),
            "gap_events": self.gap_events,
            "missing_packets": self.missing_packets,
            "counter_repeats": self.repeat_events,
            "counter_backsteps": self.backstep_events,
            "markers": self.marker_count,
            "scale_check_chunks": self.scale_verdicts or None,
            "first_timestamp_unix": self.first_timestamp,
            "last_timestamp_unix": self.last_timestamp,
        }
        self._meta["summary"] = summary
        self._meta["recording_ended"] = datetime.now(timezone.utc).astimezone().isoformat()
        self._meta["clean_shutdown"] = True
        self._write_sidecar()
        return summary
