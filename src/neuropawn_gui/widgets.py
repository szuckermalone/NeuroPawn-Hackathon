"""Display panels. Everything here works on copies of the data; nothing feeds back to the recording."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSizePolicy,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import theme
from .boards import NUM_EEG, ChannelLayout
from .dsp import BANDS, DISPLAY_FILTERS, NOTCH_OPTIONS

pg.setConfigOptions(antialias=True, background="w", foreground="#333")

QUALITY_COLORS = {"good": theme.GREEN, "fair": theme.AMBER, "poor": theme.RED}

WINDOW_OPTIONS = [1, 3, 5, 10, 20]
SCALE_OPTIONS = ["Auto", 10, 25, 50, 100, 200, 500, 1000, 4200]


class Panel(QFrame):
    """White widget panel with an OpenBCI-style title bar."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        bar = QFrame(objectName="PanelTitleBar")
        self.title_layout = QHBoxLayout(bar)
        self.title_layout.setContentsMargins(10, 4, 8, 4)
        self.title_label = QLabel(title, objectName="PanelTitle")
        self.title_layout.addWidget(self.title_label)
        self.title_layout.addStretch(1)
        outer.addWidget(bar)
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(self.body, 1)

    def add_title_widget(self, label: str | None, widget: QWidget) -> None:
        if label:
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {theme.MUTED};")
            self.title_layout.addWidget(lbl)
        self.title_layout.addWidget(widget)


def _style_plot(p: pg.PlotItem) -> None:
    p.setMenuEnabled(False)
    p.hideButtons()
    p.showGrid(x=True, y=True, alpha=0.18)
    for ax in ("left", "bottom"):
        p.getAxis(ax).setPen(pg.mkPen("#9AA3AD"))
        p.getAxis(ax).setTextPen(pg.mkPen("#4A545E"))


# ----------------------------------------------------------------------------- time series
class ChannelRow(QWidget):
    toggled = pyqtSignal(int, bool)

    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx
        self.color = theme.CHANNEL_COLORS[idx]
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        self.button = QPushButton(str(idx + 1), objectName="ChannelButton")
        self.button.setCheckable(True)
        self.button.setChecked(True)
        self.button.setToolTip(f"Turn channel {idx + 1} off/on: hides it and excludes it from the drowsiness detector "
                               "(the recording always keeps all channels)")
        self.button.toggled.connect(self._on_toggle)
        self._paint_button(True)
        h.addWidget(self.button, 0, Qt.AlignmentFlag.AlignVCenter)

        self.badge = QLabel("…", objectName="QualityBadge")
        self.badge.setFixedWidth(46)
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setToolTip("Connection quality (0–100), from the last 4 s of signal")
        h.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignVCenter)

        self.plot = pg.PlotWidget()
        self.plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot.setMinimumHeight(40)
        pi = self.plot.getPlotItem()
        _style_plot(pi)
        pi.setMouseEnabled(x=False, y=False)
        pi.getAxis("left").setWidth(46)
        pi.getAxis("bottom").setStyle(showValues=False)
        pi.getAxis("bottom").setHeight(4)
        self.artifact_curve = pi.plot(pen=None, fillLevel=0, brush=pg.mkBrush(224, 56, 45, 45))
        self.artifact_curve.setZValue(-10)
        self.curve = pi.plot(pen=pg.mkPen(self.color, width=1.2))
        self.marker_curve = pi.plot(pen=pg.mkPen("#111", width=1, style=Qt.PenStyle.DashLine), connect="pairs")
        h.addWidget(self.plot, 1)

        self.readout = QLabel("—\n", objectName="Readout")
        self.readout.setFixedWidth(104)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(self.readout)

    def _paint_button(self, on: bool) -> None:
        bg = self.color if on else "#C5CAD0"
        self.button.setStyleSheet(f"QPushButton#ChannelButton {{ background: {bg}; }}")

    def _on_toggle(self, on: bool) -> None:
        self._paint_button(on)
        self.curve.setVisible(on)
        self.toggled.emit(self.idx, on)

    @property
    def visible(self) -> bool:
        return self.button.isChecked()

    def set_quality(self, q) -> None:
        if q is None:
            self.badge.setText("…")
            self.badge.setStyleSheet(f"color: {theme.MUTED}; background: {theme.PANEL_TITLE}; border-radius: 9px;")
            self.badge.setToolTip("Measuring… (needs 2 s of signal)")
            return
        bg = QUALITY_COLORS[q.status]
        self.badge.setText(f"{q.symbol} {q.score}")
        self.badge.setStyleSheet(f"color: white; background: {bg}; border-radius: 9px; font-weight: 700; padding: 2px 0;")
        m = q.metrics
        self.badge.setToolTip(
            f"Ch{q.channel}: {q.status.upper()} ({q.score}/100)" + (f", {q.reason}" if q.reason else "") +
            f"\n60 Hz hum: {m['line_uv_rthz']:.2f} µV/√Hz (good < 2)"
            f"\nNoise 1–40 Hz: {m['std_uv']:.1f} µV (good 2–40)"
            f"\nDrift: {m['drift_uv']:.0f} µV p-p (good < 600)")

    def show_x_axis(self, show: bool) -> None:
        ax = self.plot.getPlotItem().getAxis("bottom")
        ax.setStyle(showValues=show)
        ax.setHeight(28 if show else 4)
        self.plot.getPlotItem().setLabel("bottom", "Time (s)" if show else None)


class TimeSeriesPanel(Panel):
    channel_visibility_changed = pyqtSignal()

    def __init__(self):
        super().__init__("Time Series")
        self.window_combo = QComboBox()
        for w in WINDOW_OPTIONS:
            self.window_combo.addItem(f"{w} s", w)
        self.window_combo.setCurrentIndex(WINDOW_OPTIONS.index(5))
        self.scale_combo = QComboBox()
        for s in SCALE_OPTIONS:
            self.scale_combo.addItem("Auto" if s == "Auto" else f"±{s} µV", s)
        self.scale_combo.setCurrentIndex(0)
        self.filter_combo = QComboBox()
        for f in DISPLAY_FILTERS:
            self.filter_combo.addItem(f.label, f)
        self.notch_combo = QComboBox()
        for label, hz in NOTCH_OPTIONS:
            self.notch_combo.addItem(label, hz)
        self.car_button = QPushButton("CAR")
        self.car_button.setCheckable(True)
        self.car_button.setToolTip(
            "Common average reference: subtract the mean of channels 3–8 from every channel to remove "
            "shared noise. Ch1/2 (blink channels) are left out of the average. Display only.")
        self.car_button.setStyleSheet(
            "QPushButton { padding: 2px 8px; } QPushButton:checked { background: #1F456E; color: white; }")
        self.add_title_widget("Window", self.window_combo)
        self.add_title_widget("Scale", self.scale_combo)
        self.add_title_widget("Filter", self.filter_combo)
        self.add_title_widget(None, self.notch_combo)
        self.add_title_widget(None, self.car_button)
        for c in (self.window_combo, self.scale_combo, self.filter_combo, self.notch_combo):
            c.setToolTip("Display only. Recordings always contain the unfiltered values from the board.")
        self.filter_combo.setToolTip(
            "Display only. 'Clean' = 1–40 Hz + notch, with blinks (ch1/2 > 100 µV) and artifacts "
            "(other channels > 150 µV) shaded red. Recordings are never filtered.")

        self.rows: list[ChannelRow] = []
        self.body_layout.setSpacing(2)
        for i in range(NUM_EEG):
            row = ChannelRow(i)
            row.toggled.connect(lambda *_: self.channel_visibility_changed.emit())
            self.rows.append(row)
            self.body_layout.addWidget(row, 1)
        self.rows[-1].show_x_axis(True)
        for r in self.rows[:-1]:
            r.plot.setXLink(self.rows[-1].plot)
        self.full_scale_uv: float | None = None
        self.set_window()
        self.window_combo.currentIndexChanged.connect(self.set_window)

    # settings ---------------------------------------------------------------
    @property
    def window_s(self) -> int:
        return int(self.window_combo.currentData())

    @property
    def display_filter(self):
        return self.filter_combo.currentData()

    @property
    def notch_hz(self):
        return self.notch_combo.currentData()

    def visible_channels(self) -> list[int]:
        return [r.idx for r in self.rows if r.visible]

    def set_window(self, *_):
        for r in self.rows:
            r.plot.setXRange(-self.window_s, 0, padding=0)

    @property
    def car_enabled(self) -> bool:
        return self.car_button.isChecked()

    def clear(self):
        for r in self.rows:
            r.curve.setData([], [])
            r.marker_curve.setData([], [])
            r.artifact_curve.setData([], [])
            r.readout.setText("—\n")
            r.set_quality(None)

    def set_quality(self, report) -> None:
        for r in self.rows:
            r.set_quality(None if report is None else report.channels[r.idx])

    # drawing ----------------------------------------------------------------
    def update_plots(self, shown: np.ndarray, raw: np.ndarray, markers: np.ndarray, fs: float,
                     artifact: np.ndarray | None = None) -> None:
        """shown: filtered (8, n); raw: unfiltered (8, n) as recorded; markers: (n,);
        artifact: optional (n,) bool mask to shade."""
        n = shown.shape[1]
        if n == 0:
            return
        x = (np.arange(n) - (n - 1)) / fs
        scale = self.scale_combo.currentData()
        is_raw = self.display_filter.kind == "raw"
        mk = np.nonzero(markers)[0]
        rms_n = min(n, int(fs))
        for r in self.rows:
            y = shown[r.idx]
            r.curve.setData(x, y)
            if scale == "Auto":
                lo, hi = float(np.min(y)), float(np.max(y))
                if is_raw:
                    pad = max(5.0, 0.1 * (hi - lo))
                    lo, hi = lo - pad, hi + pad
                else:
                    m = max(10.0, 1.1 * max(abs(lo), abs(hi)))
                    lo, hi = -m, m
            else:
                c = float(np.mean(y)) if is_raw else 0.0
                lo, hi = c - float(scale), c + float(scale)
            r.plot.setYRange(lo, hi, padding=0)
            if artifact is not None and artifact.any():
                r.artifact_curve.setFillLevel(lo)
                r.artifact_curve.setData(x, np.where(artifact, hi, lo))
            else:
                r.artifact_curve.setData([], [])
            if mk.size:
                xs = np.repeat(x[mk], 2)
                ys = np.tile([lo, hi], mk.size)
                r.marker_curve.setData(xs, ys)
            else:
                r.marker_curve.setData([], [])

            seg = y[-rms_n:]
            rms = float(np.sqrt(np.mean((seg - seg.mean()) ** 2)))
            status, color = "", theme.MUTED
            if self.full_scale_uv:
                peak = float(np.max(np.abs(raw[r.idx, -rms_n:]))) / self.full_scale_uv
                if peak >= 0.99:
                    status, color = "RAILED", theme.RED
                elif peak >= 0.90:
                    status, color = f"near rail {peak * 100:.0f}%", theme.AMBER
            r.readout.setText(f"{rms:8.1f} µVrms\n{status}")
            r.readout.setStyleSheet(f"color: {color if status else theme.MUTED};")


# ----------------------------------------------------------------------------- FFT
class FFTPanel(Panel):
    def __init__(self):
        super().__init__("FFT Plot")
        self.max_hz_combo = QComboBox()
        for hz in (20, 40, 60):
            self.max_hz_combo.addItem(f"0–{hz} Hz", hz)
        self.max_hz_combo.setCurrentIndex(2)
        self.add_title_widget("Range", self.max_hz_combo)
        self.plot = pg.PlotWidget()
        pi = self.plot.getPlotItem()
        _style_plot(pi)
        pi.setLogMode(x=False, y=True)
        pi.setLabel("left", "Amplitude (µV/√Hz)")
        pi.setLabel("bottom", "Frequency (Hz)")
        pi.setMouseEnabled(x=False, y=False)
        self.curves = [pi.plot(pen=pg.mkPen(c, width=1.4)) for c in theme.CHANNEL_COLORS]
        self.body_layout.addWidget(self.plot)
        self.max_hz_combo.currentIndexChanged.connect(self._apply_range)
        self._nyquist = 62.5
        self._apply_range()

    def _apply_range(self, *_):
        self.plot.setXRange(0, min(self.max_hz_combo.currentData(), self._nyquist), padding=0.01)

    def set_sampling_rate(self, fs: float):
        self._nyquist = fs / 2
        self._apply_range()

    def update_spectrum(self, f: np.ndarray, asd: np.ndarray, visible: list[int]):
        for i, c in enumerate(self.curves):
            if i in visible:
                c.setData(f, np.maximum(asd[i], 1e-4))
                c.setVisible(True)
            else:
                c.setVisible(False)

    def clear(self):
        for c in self.curves:
            c.setData([], [])


# ----------------------------------------------------------------------------- band power
class BandPowerWidget(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        self.plot = pg.PlotWidget()
        pi = self.plot.getPlotItem()
        _style_plot(pi)
        pi.setMouseEnabled(x=False, y=False)
        pi.setLabel("left", "Relative power (%)")
        pi.setYRange(0, 100, padding=0)
        pi.setXRange(-0.6, len(BANDS) - 0.4, padding=0)
        ticks = [(i, f"{name}\n{lo:g}–{hi:g} Hz") for i, (name, lo, hi) in enumerate(BANDS)]
        pi.getAxis("bottom").setTicks([ticks])
        pi.getAxis("bottom").setHeight(44)
        self.bars = pg.BarGraphItem(
            x=np.arange(len(BANDS)), height=np.zeros(len(BANDS)), width=0.65,
            brushes=[pg.mkBrush(c) for c in theme.BAND_COLORS], pens=[pg.mkPen(None)] * len(BANDS),
        )
        pi.addItem(self.bars)
        self.labels = []
        for i in range(len(BANDS)):
            t = pg.TextItem("", color="#333", anchor=(0.5, 1.0))
            t.setPos(i, 0)
            pi.addItem(t)
            self.labels.append(t)
        v.addWidget(self.plot)
        note = QLabel("Mean across visible channels, from the displayed (filtered) signal, last 4 s.")
        note.setStyleSheet(f"color: {theme.MUTED}; font-size: 11px;")
        v.addWidget(note)

    def update_bands(self, rel: np.ndarray):
        self.bars.setOpts(height=rel)
        for i, t in enumerate(self.labels):
            t.setText(f"{rel[i]:.0f}%")
            t.setPos(i, rel[i] + 7)

    def clear(self):
        self.update_bands(np.zeros(len(BANDS)))


# ----------------------------------------------------------------------------- live numeric values
class LiveValuesWidget(QWidget):
    COLS = ["Ch", "Latest µV", "Mean µV (1 s)", "Min µV (1 s)", "Max µV (1 s)", "Std µV (1 s)", "Lead-off P/N"]

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        self.table = QTableWidget(NUM_EEG, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for i in range(NUM_EEG):
            item = QTableWidgetItem(f" {i + 1} ")
            item.setForeground(QColor("white"))
            item.setBackground(QColor(theme.CHANNEL_COLORS[i]))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 0, item)
            for c in range(1, len(self.COLS)):
                it = QTableWidgetItem("—")
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, c, it)
        v.addWidget(self.table, 1)
        self.footer = QLabel("")
        self.footer.setStyleSheet(f"color: {theme.MUTED}; font-size: 11px;")
        self.footer.setWordWrap(True)
        v.addWidget(self.footer)

    def update_values(self, raw: np.ndarray, fs: float, loff: tuple[int, int] | None, pkg_info: str):
        n = min(raw.shape[1], int(fs))
        if n == 0:
            return
        seg = raw[:, -n:]
        latest = raw[:, -1]
        stats = (latest, seg.mean(axis=1), seg.min(axis=1), seg.max(axis=1), seg.std(axis=1))
        for i in range(NUM_EEG):
            for c, arr in enumerate(stats, start=1):
                self.table.item(i, c).setText(f"{arr[i]:,.2f}")
            if loff is None:
                txt = "n/a"
            else:
                p, nn = (loff[0] >> i) & 1, (loff[1] >> i) & 1
                txt = f"{p}/{nn}"
            self.table.item(i, 6).setText(txt)
        self.footer.setText(
            "These are the unfiltered values the CSV records (shown here to 2 decimals; the CSV keeps full "
            f"float64 precision). {pkg_info}"
        )

    def clear(self):
        for i in range(NUM_EEG):
            for c in range(1, len(self.COLS)):
                self.table.item(i, c).setText("—")
        self.footer.setText("")


# ----------------------------------------------------------------------------- signal check
class SignalCheckWidget(QWidget):
    COLS = ["Ch", "Status", "Score", "Issue", "60 Hz µV/√Hz", "Noise µV", "Drift µV"]
    TIPS = {
        "60 Hz hum": "Press the electrode firmly against the scalp, part the hair, add a little paste. "
                     "Check that the ear clip (RLD/COMM) is snug.",
        "too noisy": "Sit still and relax your jaw. Re-seat the electrode if it stays noisy.",
        "drifting": "Give it 10–20 s to settle after adjusting. Avoid tugging the cables.",
        "flat": "Channel may still be configuring, or the electrode lead is unplugged. Check the pin "
                "is on the positive rail.",
        "railed": "No contact. Re-seat the electrode, check the lead, and check the ear clip.",
        "copies": "Fix the channel with the lowest score first; one loose electrode can contaminate the rest.",
    }

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        from PyQt6.QtWidgets import QCheckBox
        self.ignore_hum_cb = QCheckBox("Ignore 60 Hz hum (it's filtered out before analysis; recordings stay raw)")
        self.ignore_hum_cb.setChecked(True)
        self.ignore_hum_cb.setToolTip("When on, channels are judged on flat/railed/drift/noise only. Turn off to use the "
                                      "hum column for fixing electrode contact.")
        v.addWidget(self.ignore_hum_cb)
        self.headline = QLabel("Start a session to check electrode contact.")
        self.headline.setStyleSheet("font-size: 15px; font-weight: 700;")
        v.addWidget(self.headline)
        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(f"color: white; background: {theme.RED}; padding: 6px; border-radius: 3px;")
        self.warning.hide()
        v.addWidget(self.warning)
        self.table = QTableWidget(NUM_EEG, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for i in range(NUM_EEG):
            item = QTableWidgetItem(f" {i + 1} ")
            item.setForeground(QColor("white"))
            item.setBackground(QColor(theme.CHANNEL_COLORS[i]))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 0, item)
            for c in range(1, len(self.COLS)):
                self.table.setItem(i, c, QTableWidgetItem("—"))
        v.addWidget(self.table, 1)
        self.tips = QLabel("")
        self.tips.setWordWrap(True)
        self.tips.setStyleSheet(f"color: {theme.MUTED}; font-size: 11px;")
        v.addWidget(self.tips)

    def update_report(self, report) -> None:
        if report is None:
            self.headline.setText("Measuring… (needs 2 s of signal)")
            self.headline.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {theme.MUTED};")
            return
        n_good, n = report.n_good, len(report.channels)
        color = theme.GREEN if n_good == n and not report.warning else theme.AMBER if n_good >= n - 2 else theme.RED
        verdict = "Ready to record" if n_good == n and not report.warning else "Fix the flagged channels before recording"
        self.headline.setText(f"{n_good}/{n} channels good · {verdict}")
        self.headline.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {color};")
        self.warning.setVisible(bool(report.warning))
        self.warning.setText(report.warning or "")
        tips = []
        for i, q in enumerate(report.channels):
            st = self.table.item(i, 1)
            st.setText(f"{q.symbol}  {q.status}")
            st.setForeground(QColor(QUALITY_COLORS[q.status]))
            self.table.item(i, 2).setText(str(q.score))
            self.table.item(i, 3).setText(q.reason or "OK")
            m = q.metrics
            self.table.item(i, 4).setText(f"{m['line_uv_rthz']:.2f}")
            self.table.item(i, 5).setText(f"{m['std_uv']:.1f}")
            self.table.item(i, 6).setText(f"{m['drift_uv']:.0f}")
            for key, tip in self.TIPS.items():
                if q.reason.startswith(key) and tip not in tips:
                    tips.append(tip)
        self.tips.setText("Tips: " + " ".join(tips) if tips else
                          "Scores use the last 4 s: 60 Hz hum, noise level, drift, flat/railed signal. "
                          "Hover a channel badge for details.")

    def clear(self) -> None:
        self.headline.setText("Start a session to check electrode contact.")
        self.headline.setStyleSheet("font-size: 15px; font-weight: 700;")
        self.warning.hide()
        self.tips.setText("")
        for i in range(NUM_EEG):
            for c in range(1, len(self.COLS)):
                self.table.item(i, c).setText("—")


# ----------------------------------------------------------------------------- drowsiness
class DrowsinessWidget(QWidget):
    """Score, state, ratio readouts, history plot, and alert controls."""

    recalibrate_requested = pyqtSignal()
    test_alert_requested = pyqtSignal()
    launch_game_requested = pyqtSignal(float, bool)

    def __init__(self):
        from PyQt6.QtWidgets import QCheckBox, QProgressBar, QSpinBox
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 4, 6, 4)

        top = QHBoxLayout()
        self.score_label = QLabel("—")
        self.score_label.setFixedWidth(118)
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet("font-size: 40px; font-weight: 800; border-radius: 8px; "
                                       f"background: {theme.PANEL_TITLE}; color: {theme.MUTED};")
        self.score_label.setToolTip("Drowsiness score: % of the last 30 s showing the pre-sleep pattern "
                                    "(theta replacing alpha). Alert at 50.")
        top.addWidget(self.score_label)
        info = QVBoxLayout()
        self.state_label = QLabel("Start a session")
        self.state_label.setStyleSheet("font-size: 16px; font-weight: 700;")
        info.addWidget(self.state_label)
        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet(f"color: {theme.MUTED}; font-family: Menlo, monospace; font-size: 11px;")
        info.addWidget(self.detail_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Calibrating… stay awake: eyes open, then closed  %p%")
        self.progress.setFixedHeight(16)
        info.addWidget(self.progress)
        top.addLayout(info, 1)
        v.addLayout(top)

        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet(f"color: white; background: {theme.RED}; padding: 6px; border-radius: 3px; font-weight: 700;")
        self.banner.setMinimumHeight(48)
        self.banner.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.banner.hide()
        v.addWidget(self.banner)

        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(170)  # the tab scrolls on small screens instead of squeezing this
        pi = self.plot.getPlotItem()
        _style_plot(pi)
        pi.setMouseEnabled(x=False, y=False)
        pi.setLabel("left", "Score")
        pi.setLabel("bottom", "Time (s)")
        pi.setYRange(0, 100, padding=0.02)
        pi.addItem(pg.InfiniteLine(50, angle=0, pen=pg.mkPen(theme.RED, width=1, style=Qt.PenStyle.DashLine),
                                   label="alert 50", labelOpts={"position": 0.05, "color": theme.RED}))
        self.score_curve = pi.plot(pen=pg.mkPen(theme.RED, width=2), fillLevel=0, brush=pg.mkBrush(224, 56, 45, 40))
        self.ratio_view = pg.ViewBox()
        pi.showAxis("right")
        pi.scene().addItem(self.ratio_view)
        pi.getAxis("right").linkToView(self.ratio_view)
        pi.getAxis("right").setLabel("θ/α ratio", color=theme.CHANNEL_COLORS[2])
        self.ratio_view.setXLink(pi)
        self.ratio_view.setYRange(0, 4)
        self.ratio_curve = pg.PlotDataItem(pen=pg.mkPen(theme.CHANNEL_COLORS[2], width=1.2))
        self.ratio_view.addItem(self.ratio_curve)
        self.ratio_view.addItem(pg.InfiniteLine(1.0, angle=0, pen=pg.mkPen(theme.CHANNEL_COLORS[2], style=Qt.PenStyle.DotLine)))
        # behaviour from the driving game: lapses (red) and lane departures (amber) along the top
        self.events_scatter = pg.ScatterPlotItem(size=10, pen=None, symbol="t")
        pi.addItem(self.events_scatter)
        self._behaviour: list[tuple[float, str]] = []
        pi.vb.sigResized.connect(lambda: self.ratio_view.setGeometry(pi.vb.sceneBoundingRect()))
        v.addWidget(self.plot, 1)

        row = QHBoxLayout()
        self.alerts_cb = QCheckBox("Send alerts")
        self.alerts_cb.setChecked(True)
        self.speak_cb = QCheckBox("Speak alert")
        row.addWidget(self.alerts_cb)
        row.addWidget(self.speak_cb)
        row.addStretch(1)
        self.recal_btn = QPushButton("Recalibrate (60 s)")
        self.recal_btn.setToolTip("Re-measure this person's awake baseline. Stay awake: ~30 s eyes open, ~30 s eyes closed.")
        self.recal_btn.clicked.connect(self.recalibrate_requested.emit)
        self.test_btn = QPushButton("Test alert")
        self.test_btn.clicked.connect(self.test_alert_requested.emit)
        row.addWidget(self.recal_btn)
        row.addWidget(self.test_btn)
        v.insertLayout(1, row)  # alert controls right under the score, always visible
        game = QHBoxLayout()
        title = QLabel("Night Drive")
        title.setStyleSheet("font-weight: 700;")
        game.addWidget(title)
        self.game_btn = QPushButton("Launch driving task")
        self.game_btn.setToolTip("Opens the monotonous driving task. This GUI keeps the headset and interrupts the "
                                 "drive when it detects the pre-sleep pattern. Recording starts automatically.")
        self.game_btn.clicked.connect(lambda: self.launch_game_requested.emit(float(self.minutes_spin.value()),
                                                                              self.fullscreen_cb.isChecked()))
        self.minutes_spin = QSpinBox()
        self.minutes_spin.setRange(1, 180)
        self.minutes_spin.setValue(30)
        self.minutes_spin.setSuffix(" min")
        self.fullscreen_cb = QCheckBox("Fullscreen")
        game.addWidget(self.game_btn)
        game.addWidget(self.minutes_spin)
        game.addWidget(self.fullscreen_cb)
        game.addStretch(1)
        v.addLayout(game)
        self.game_status = QLabel("Not running.")
        self.game_status.setStyleSheet(f"color: {theme.MUTED}; font-size: 11px;")
        self.game_status.setWordWrap(True)
        v.addWidget(self.game_status)

        note = QLabel("Research prototype, not a safety device. Fires when theta has replaced alpha and risen ≥2× over "
                      "baseline for >50% of 30 s (sleep-onset criterion) or for ~8 s straight (microsleep). Eyes-open "
                      "alpha blocking, blinking, and attention-related theta don't count. ▲ red = lapse, ▲ amber = "
                      "lane departure in the driving task.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.MUTED}; font-size: 10px;")
        v.addWidget(note)
        self.clear()

    def clear(self) -> None:
        self.score_label.setText("—")
        self.score_label.setStyleSheet("font-size: 40px; font-weight: 800; border-radius: 8px; "
                                       f"background: {theme.PANEL_TITLE}; color: {theme.MUTED};")
        self.state_label.setText("Start a session")
        self.detail_label.setText("")
        self.progress.setValue(0)
        self.progress.hide()
        self.banner.hide()
        self.score_curve.setData([], [])
        self.ratio_curve.setData([], [])
        self._behaviour = []
        self.events_scatter.setData([], [])

    def add_behaviour_event(self, t: float, kind: str) -> None:
        self._behaviour.append((t, kind))

    def show_alert(self, text: str) -> None:
        self.banner.setText(text)
        self.banner.show()

    def update_state(self, s, t0: float) -> None:
        cal = s.phase == "calibrating"
        self.progress.setVisible(cal)
        if cal:
            self.progress.setValue(int(s.calibration_progress * 100))
        if s.score is None:
            self.score_label.setText("—")
            color, fg = theme.PANEL_TITLE, theme.MUTED
        else:
            self.score_label.setText(f"{s.score:.0f}")
            color = theme.RED if s.score >= 50 else theme.AMBER if s.score >= 25 else theme.GREEN
            fg = "white"
        self.score_label.setStyleSheet(f"font-size: 40px; font-weight: 800; border-radius: 8px; background: {color}; color: {fg};")
        self.state_label.setText(s.label)
        parts = []
        if s.ratio is not None:
            parts.append(f"θ/α {s.ratio:5.2f} (≥1.0)")
        if s.theta_rel is not None:
            parts.append(f"θ {s.theta_rel:4.1f}× base (≥2.0)")
        if s.beta_rel is not None:
            parts.append(f"β {s.beta_rel:4.1f}× base (≤1.5)")
        self.detail_label.setText("   ".join(parts))
        if s.history:
            h = np.array([(t, np.nan if sc is None else sc, np.nan if r is None else r) for t, sc, r in s.history])
            x = h[:, 0] - t0
            self.score_curve.setData(x, np.nan_to_num(h[:, 1], nan=0.0), connect="finite")
            self.ratio_curve.setData(x, np.clip(h[:, 2], 0, 4), connect="finite")
        if self._behaviour:
            colors = {"lapse": theme.RED, "lane_departure": theme.AMBER}
            self.events_scatter.setData(
                [{"pos": (tb - t0, 96 if k == "lapse" else 90), "brush": pg.mkBrush(colors.get(k, theme.MUTED))}
                 for tb, k in self._behaviour])


# ----------------------------------------------------------------------------- IMU
class IMUWidget(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        self.note = QLabel("IMU data is only available with the Knight IMU board.")
        self.note.setStyleSheet(f"color: {theme.MUTED};")
        v.addWidget(self.note)
        self.plots = {}
        self.curves = {}
        for key, title in (("accel", "Accelerometer"), ("gyro", "Gyroscope")):
            pw = pg.PlotWidget()
            pi = pw.getPlotItem()
            _style_plot(pi)
            pi.setTitle(title, size="9pt")
            pi.addLegend(offset=(5, 5), labelTextSize="8pt")
            pi.setMouseEnabled(x=False, y=False)
            self.curves[key] = [pi.plot(pen=pg.mkPen(c, width=1.2), name=f"{key}_{ax}")
                                for c, ax in zip(theme.IMU_COLORS, "xyz")]
            self.plots[key] = pw
            v.addWidget(pw, 1)
        self.set_available(False)

    def set_available(self, ok: bool):
        self.note.setVisible(not ok)
        for pw in self.plots.values():
            pw.setVisible(ok)

    def update_imu(self, layout: ChannelLayout, block: np.ndarray, fs: float, window_s: int):
        n = min(block.shape[1], int(window_s * fs))
        if n == 0:
            return
        x = (np.arange(n) - (n - 1)) / fs
        for key in ("accel", "gyro"):
            for curve, ax in zip(self.curves[key], "xyz"):
                idx = layout.display_extra_index(f"{key}_{ax}")
                curve.setData(x, block[idx, -n:])
            self.plots[key].setXRange(-window_s, 0, padding=0)

    def clear(self):
        for cs in self.curves.values():
            for c in cs:
                c.setData([], [])
