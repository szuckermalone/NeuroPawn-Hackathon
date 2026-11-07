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
        self.button.setToolTip(f"Show/hide channel {idx + 1} (display only; recording always keeps all channels)")
        self.button.toggled.connect(self._on_toggle)
        self._paint_button(True)
        h.addWidget(self.button, 0, Qt.AlignmentFlag.AlignVCenter)

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
