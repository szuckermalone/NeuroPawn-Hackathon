"""Viewer for saved recordings: full-length stacked time series with pan/zoom."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QLabel, QMainWindow, QVBoxLayout, QWidget

from . import theme
from .dsp import DISPLAY_FILTERS, NOTCH_OPTIONS, apply_display_filter
from .recorder import sidecar_path
from .scale import LEGACY, classify, legacy_to_corrected
from .widgets import Panel, _style_plot


@dataclass
class Recording:
    path: Path
    columns: list[str]
    data: np.ndarray  # (n_samples, n_columns)
    meta: dict | None
    fs: float
    scale_note: str | None = None

    def col(self, name: str) -> np.ndarray:
        return self.data[:, self.columns.index(name)]

    @property
    def eeg(self) -> np.ndarray:
        return np.vstack([self.col(f"ch{i}_uV") for i in range(1, 9)])

    @property
    def time_s(self) -> np.ndarray:
        """Sample-clock time, advanced across any detected packet gaps."""
        gaps = np.clip(self.col("gap_before"), 0, None)
        return (self.col("sample_index") + np.cumsum(gaps)) / self.fs


def load_recording(path: Path) -> Recording:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    required = {"sample_index", "timestamp_unix", "gap_before", *(f"ch{i}_uV" for i in range(1, 9))}
    missing = required - set(header)
    if missing:
        raise ValueError(f"Not a NeuroPawn GUI recording (missing columns: {', '.join(sorted(missing))})")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    meta = None
    sc = sidecar_path(path)
    if sc.exists():
        meta = json.loads(sc.read_text(encoding="utf-8"))
    fs = None
    if meta:
        fs = meta.get("board", {}).get("sampling_rate_hz")
    if not fs:
        ts = data[:, header.index("timestamp_unix")]
        fs = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 125.0

    # Recordings made before the scale fix: correct in memory (the file is not modified).
    note = None
    eeg_idx = [header.index(f"ch{i}_uV") for i in range(1, 9)]
    already = bool(meta and meta.get("board", {}).get("scale_correction"))
    is_knight = bool(meta and meta.get("board", {}).get("name") in ("Knight", "KnightIMU"))
    if not already and (is_knight or meta is None) and classify(data[:, eeg_idx]) == LEGACY:
        data[:, eeg_idx] = legacy_to_corrected(data[:, eeg_idx])
        note = "µV scale corrected on load (recorded with BrainFlow ≤ 5.23.0, values were 79.57× too large)"
    return Recording(path, header, data, meta, float(fs), note)


class ReviewWindow(QMainWindow):
    def __init__(self, path: Path):
        super().__init__()
        self.rec = load_recording(path)
        self.setWindowTitle(f"Recording: {self.rec.path.name}")
        self.resize(1300, 860)

        root = QWidget(objectName="Workspace")
        v = QVBoxLayout(root)
        v.setContentsMargins(6, 6, 6, 6)
        panel = Panel(self.rec.path.name)
        self.filter_combo = QComboBox()
        for f in DISPLAY_FILTERS:
            self.filter_combo.addItem(f.label, f)
        self.notch_combo = QComboBox()
        for label, hz in NOTCH_OPTIONS:
            self.notch_combo.addItem(label, hz)
        panel.add_title_widget("Filter", self.filter_combo)
        panel.add_title_widget(None, self.notch_combo)

        self.info = QLabel(self._info_text())
        self.info.setStyleSheet(f"color: {theme.MUTED}; padding: 2px 4px;")
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        panel.body_layout.addWidget(self.info)

        self.glw = pg.GraphicsLayoutWidget()
        panel.body_layout.addWidget(self.glw, 1)
        v.addWidget(panel)
        self.setCentralWidget(root)

        self.t = self.rec.time_s
        markers = self.rec.col("marker") if "marker" in self.rec.columns else np.zeros(len(self.t))
        mk_idx = np.nonzero(markers)[0]
        self.curves = []
        first = None
        for i in range(8):
            p = self.glw.addPlot(row=i, col=0)
            _style_plot(p)
            p.setMouseEnabled(x=True, y=False)
            p.getAxis("left").setWidth(52)
            p.setLabel("left", f"Ch{i + 1}", color=theme.CHANNEL_COLORS[i])
            c = p.plot(pen=pg.mkPen(theme.CHANNEL_COLORS[i], width=1))
            c.setDownsampling(auto=True, method="peak")
            c.setClipToView(True)
            self.curves.append(c)
            for j in mk_idx:
                p.addItem(pg.InfiniteLine(self.t[j], angle=90, pen=pg.mkPen("#111", style=Qt.PenStyle.DashLine),
                                          label=f"{markers[j]:g}" if i == 0 else None,
                                          labelOpts={"position": 0.9, "color": "#111"}))
            if first is None:
                first = p
            else:
                p.setXLink(first)
            if i < 7:
                p.getAxis("bottom").setStyle(showValues=False)
            else:
                p.setLabel("bottom", "Time (s, sample clock)")
        self._redraw()
        if len(self.t):
            first.setXRange(self.t[0], min(self.t[-1], self.t[0] + 10), padding=0)
        self.filter_combo.currentIndexChanged.connect(self._redraw)
        self.notch_combo.currentIndexChanged.connect(self._redraw)

    def _info_text(self) -> str:
        r = self.rec
        n = len(r.data)
        gaps = r.col("gap_before")
        missing = int(gaps[gaps > 0].sum())
        parts = [f"{n:,} samples", f"{n / r.fs:,.1f} s at {r.fs:g} Hz", f"{missing} missing packets"]
        if r.meta:
            parts.append(f"started {r.meta.get('recording_started', '?')}")
            parts.append(f"board {r.meta.get('board', {}).get('name', '?')}")
            if not r.meta.get("clean_shutdown"):
                parts.append("⚠ not closed cleanly (app crashed or was killed); data up to the last flush is intact")
        else:
            parts.append("no .json sidecar found")
        if r.scale_note:
            parts.append("⚠ " + r.scale_note)
        return "   ·   ".join(parts) + "      (scroll/drag to pan & zoom; filters affect display only)"

    def _redraw(self, *_):
        y = apply_display_filter(self.rec.eeg, self.rec.fs, self.filter_combo.currentData(),
                                 self.notch_combo.currentData())
        for i, c in enumerate(self.curves):
            c.setData(self.t, y[i])
            vb = c.getViewBox()
            if vb is not None:
                lo, hi = np.percentile(y[i], [0.5, 99.5]) if y.shape[1] else (-1, 1)
                pad = max(5.0, 0.15 * (hi - lo))
                vb.setYRange(lo - pad, hi + pad, padding=0)
