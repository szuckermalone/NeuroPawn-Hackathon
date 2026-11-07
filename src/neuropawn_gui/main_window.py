"""Main window: session controls, live plots, recording."""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSplitter, QStatusBar, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from . import cleaning, scale, theme
from .acquisition import AcquisitionEngine, State
from .boards import BOARDS, BOARDS_BY_KEY, NUM_EEG, list_serial_ports
from .dsp import amplitude_spectrum, apply_display_filter, relative_band_powers
from .widgets import BandPowerWidget, FFTPanel, IMUWidget, LiveValuesWidget, Panel, TimeSeriesPanel

PLOT_INTERVAL_MS = 33
SLOW_INTERVAL_MS = 250
FILTER_PAD_S = 2  # extra history so zero-phase filtering has settled at the visible window edge
SPECTRUM_S = 4


def _fmt_duration(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


class MainWindow(QMainWindow):
    def __init__(self, recordings_dir: Path, board_key: str | None = None, port: str | None = None):
        super().__init__()
        self.setWindowTitle("NeuroPawn EEG")
        self.resize(1440, 900)
        self.recordings_dir = Path(recordings_dir)
        self.engine = AcquisitionEngine()
        self._last_total = -1
        self._shown_error_serial = 0
        self._review_windows: list = []
        self._session_layout = None

        self._build_ui()
        self._refresh_ports(select=port)
        if board_key:
            self.board_combo.setCurrentIndex(max(0, self.board_combo.findData(board_key)))
        self._on_board_changed()

        self._plot_timer = QTimer(self, interval=PLOT_INTERVAL_MS, timeout=self._update_plots)
        self._slow_timer = QTimer(self, interval=SLOW_INTERVAL_MS, timeout=self._update_slow)
        self._plot_timer.start()
        self._slow_timer.start()
        self._update_slow()

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self):
        root = QWidget(objectName="Workspace")
        v = QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._build_top_bar())

        work = QWidget(objectName="Workspace")
        wl = QHBoxLayout(work)
        wl.setContentsMargins(6, 6, 6, 6)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(6)
        self.ts_panel = TimeSeriesPanel()
        split.addWidget(self.ts_panel)

        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(6)
        self.fft_panel = FFTPanel()
        right.addWidget(self.fft_panel)
        tabs_panel = Panel("Analysis")
        self.tabs = QTabWidget()
        self.band_widget = BandPowerWidget()
        self.values_widget = LiveValuesWidget()
        self.imu_widget = IMUWidget()
        self.tabs.addTab(self.band_widget, "Band Power")
        self.tabs.addTab(self.values_widget, "Live Values")
        self.tabs.addTab(self.imu_widget, "IMU")
        tabs_panel.body_layout.addWidget(self.tabs)
        right.addWidget(tabs_panel)
        right.setSizes([420, 420])
        split.addWidget(right)
        split.setSizes([880, 540])
        wl.addWidget(split)
        v.addWidget(work, 1)
        self.setCentralWidget(root)

        sb = QStatusBar()
        self.sb_rate = QLabel("—")
        self.sb_samples = QLabel("")
        self.sb_integrity = QLabel("")
        self.sb_rec = QLabel("")
        self.sb_scale = QLabel("")
        for w in (self.sb_rate, self.sb_samples, self.sb_integrity, self.sb_scale):
            sb.addWidget(w)
        sb.addPermanentWidget(self.sb_rec)
        self.setStatusBar(sb)

        self.ts_panel.channel_visibility_changed.connect(lambda: setattr(self, "_last_total", -1))
        for c in (self.ts_panel.filter_combo, self.ts_panel.notch_combo, self.ts_panel.scale_combo,
                  self.ts_panel.window_combo):
            c.currentIndexChanged.connect(lambda *_: setattr(self, "_last_total", -1))
        self.ts_panel.car_button.toggled.connect(lambda *_: setattr(self, "_last_total", -1))

        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._toggle_recording)
        QShortcut(QKeySequence("M"), self, activated=lambda: self._insert_marker(1))
        for k in range(1, 10):
            QShortcut(QKeySequence(str(k)), self, activated=lambda k=k: self._insert_marker(k))

    def _build_top_bar(self) -> QFrame:
        bar = QFrame(objectName="TopBar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 8, 14, 8)
        h.setSpacing(8)
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.addWidget(QLabel("NEUROPAWN EEG", objectName="AppTitle"))
        title_box.addWidget(QLabel("Knight board · 8 ch · BrainFlow", objectName="AppSubtitle"))
        h.addLayout(title_box)
        h.addSpacing(18)

        h.addWidget(QLabel("Board"))
        self.board_combo = QComboBox()
        for spec in BOARDS:
            self.board_combo.addItem(spec.label, spec.key)
        self.board_combo.currentIndexChanged.connect(self._on_board_changed)
        h.addWidget(self.board_combo)

        self.port_label = QLabel("Port")
        h.addWidget(self.port_label)
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setMinimumWidth(230)
        self.port_combo.setToolTip("Serial port of the Knight board (type one in if it isn't listed)")
        h.addWidget(self.port_combo)
        self.refresh_btn = QToolButton(text="↻")
        self.refresh_btn.setStyleSheet("font-size: 15px; font-weight: 700; padding: 1px 7px;")
        self.refresh_btn.setToolTip("Rescan serial ports")
        self.refresh_btn.clicked.connect(lambda: self._refresh_ports())
        h.addWidget(self.refresh_btn)

        self.session_btn = QPushButton("Start Session", objectName="StartButton")
        self.session_btn.setMinimumWidth(124)
        self.session_btn.clicked.connect(self._toggle_session)
        h.addWidget(self.session_btn)

        self.state_label = QLabel("", objectName="StateLabel")
        self.state_label.setMinimumWidth(240)
        h.addWidget(self.state_label, 1)

        self.record_btn = QPushButton("●  Start Recording", objectName="RecordButton")
        self.record_btn.setMinimumWidth(150)
        self.record_btn.setToolTip("Write every sample, unfiltered, to a CSV in the recordings folder (Ctrl+R)")
        self.record_btn.clicked.connect(self._toggle_recording)
        h.addWidget(self.record_btn)

        self.marker_btn = QPushButton("Marker")
        self.marker_btn.setToolTip("Insert an event marker into the data stream (keys M or 1–9)")
        self.marker_btn.clicked.connect(lambda: self._insert_marker(1))
        h.addWidget(self.marker_btn)

        self.open_btn = QPushButton("Open Recording…")
        self.open_btn.clicked.connect(self._open_recording)
        h.addWidget(self.open_btn)
        self.folder_btn = QPushButton("Folder")
        self.folder_btn.setToolTip("Open the recordings folder")
        self.folder_btn.clicked.connect(self._open_folder)
        h.addWidget(self.folder_btn)
        return bar

    # ------------------------------------------------------------------ controls
    def _current_spec(self):
        return BOARDS_BY_KEY[self.board_combo.currentData()]

    def _on_board_changed(self, *_):
        needs = self._current_spec().needs_port
        running = self.engine.is_running()
        self.port_combo.setEnabled(needs and not running)
        self.refresh_btn.setEnabled(needs and not running)

    def _refresh_ports(self, select: str | None = None):
        current = select or self.port_combo.currentText()
        self.port_combo.clear()
        for dev, desc in list_serial_ports():
            self.port_combo.addItem(dev, dev)
            self.port_combo.setItemData(self.port_combo.count() - 1, desc, Qt.ItemDataRole.ToolTipRole)
        if current:
            i = self.port_combo.findText(current)
            if i >= 0:
                self.port_combo.setCurrentIndex(i)
            else:
                self.port_combo.setEditText(current)

    def _toggle_session(self):
        if self.engine.is_running():
            self.engine.disconnect()
            self.session_btn.setEnabled(False)
            return
        spec = self._current_spec()
        port = self.port_combo.currentText().strip()
        if spec.needs_port and not port:
            QMessageBox.warning(self, "No port selected",
                                "Plug in the Knight board over USB-C, press ⟳ to rescan, and pick its port "
                                "(on a Mac it looks like /dev/cu.usbserial-… or /dev/cu.usbmodem…).")
            return
        self.engine.connect(spec, port)
        layout = self.engine.layout
        self._session_layout = layout
        self._last_total = -1
        self.ts_panel.full_scale_uv = layout.full_scale_uv
        self.ts_panel.clear()
        self.fft_panel.clear()
        self.fft_panel.set_sampling_rate(layout.sampling_rate)
        self.band_widget.clear()
        self.values_widget.clear()
        self.imu_widget.clear()
        self.imu_widget.set_available(layout.has_imu)
        self.board_combo.setEnabled(False)
        self._on_board_changed()

    def _toggle_recording(self):
        st = self.engine.stats()
        if st.recording_path is not None:
            self.engine.stop_recording()
            return
        if st.state is not State.STREAMING:
            return
        spec = self._current_spec()
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = self.recordings_dir / f"neuropawn_{spec.key}_{stamp}.csv"
        try:
            self.recordings_dir.mkdir(parents=True, exist_ok=True)
            self.engine.start_recording(path)
        except (RuntimeError, OSError) as e:
            QMessageBox.warning(self, "Can't start recording", str(e))

    def _insert_marker(self, value: int):
        if self.engine.stats().state is not State.STREAMING:
            return
        if self.engine.insert_marker(value):
            self.statusBar().showMessage(f"Marker {value} inserted at {datetime.now():%H:%M:%S.%f}"[:-3], 3000)

    def _open_recording(self):
        from .review import ReviewWindow
        start = str(self.recordings_dir if self.recordings_dir.exists() else Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Open recording", start, "CSV recordings (*.csv)")
        if not path:
            return
        try:
            w = ReviewWindow(Path(path))
        except Exception as e:  # malformed file
            QMessageBox.warning(self, "Couldn't open recording", f"{path}\n\n{e}")
            return
        w.show()
        self._review_windows.append(w)

    def _open_folder(self):
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(self.recordings_dir)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.recordings_dir)))

    # ------------------------------------------------------------------ periodic updates
    def _update_plots(self):
        layout, disp = self._session_layout, self.engine.display
        if layout is None or disp is None:
            return
        fs = layout.sampling_rate
        n_win = int(self.ts_panel.window_s * fs)
        block, total = disp.latest(n_win + FILTER_PAD_S * fs)
        if total == self._last_total or block.shape[1] < 2:
            return
        self._last_total = total
        raw = block[:NUM_EEG]
        shown = self._display_signal(raw, fs)
        markers = block[layout.display_marker_index]
        artifact = None
        if self.ts_panel.display_filter.flag_artifacts:
            artifact = cleaning.artifact_mask(shown, fs)[-n_win:]
        self.ts_panel.update_plots(shown[:, -n_win:], raw[:, -n_win:], markers[-n_win:], fs, artifact)

    def _display_signal(self, raw: np.ndarray, fs: float) -> np.ndarray:
        """Filtered (and optionally re-referenced) copy for display. Never touches the recording."""
        flt = self.ts_panel.display_filter
        shown = apply_display_filter(raw, fs, flt, self.ts_panel.notch_hz)
        if self.ts_panel.car_enabled and flt.kind != "raw":
            shown = cleaning.common_average(shown)
        return shown

    def _update_slow(self):
        st = self.engine.stats()
        running = self.engine.is_running()
        layout, disp = self._session_layout, self.engine.display

        # --- buttons / state label
        if running:
            stopping = st.state is State.STOPPING
            self.session_btn.setText("Stopping…" if stopping else "Stop Session")
            self.session_btn.setObjectName("StopButton")
            self.session_btn.setEnabled(not stopping)
        else:
            self.session_btn.setText("Start Session")
            self.session_btn.setObjectName("StartButton")
            self.session_btn.setEnabled(True)
            self.board_combo.setEnabled(True)
        self.session_btn.style().unpolish(self.session_btn)
        self.session_btn.style().polish(self.session_btn)
        self._on_board_changed()

        recording = st.recording_path is not None
        self.record_btn.setEnabled(recording or st.state is State.STREAMING)
        self.record_btn.setText("■  Stop Recording" if recording else "●  Start Recording")
        self.record_btn.setProperty("recording", "true" if recording else "false")
        self.record_btn.style().unpolish(self.record_btn)
        self.record_btn.style().polish(self.record_btn)
        self.marker_btn.setEnabled(st.state is State.STREAMING)

        color = {State.STREAMING: "#7EE2A8", State.CONFIGURING: "#FFD479", State.CONNECTING: "#FFD479",
                 State.STOPPING: "#FFD479", State.ERROR: "#FF9A92"}.get(st.state, "#C9D6E3")
        text = st.message or st.state.value
        if st.state is State.CONFIGURING and st.config_total:
            text = f"Configuring channels {st.config_done}/{st.config_total} (gain 12 + RLD)…"
        if st.state is State.STREAMING and st.seconds_since_data > 2.0:
            text, color = f"No data for {st.seconds_since_data:.0f} s, check the USB cable / board power", "#FF9A92"
        if st.state is State.ERROR:
            text = "Error (see dialog)"
        self.state_label.setText(f"●  {text}")
        self.state_label.setStyleSheet(f"color: {color};")

        if st.error and st.error_serial != self._shown_error_serial:
            self._shown_error_serial = st.error_serial
            QMessageBox.critical(self, "NeuroPawn EEG", st.error)

        # --- status bar
        if layout is not None and (running or st.samples_received):
            self.sb_rate.setText(f"Sample rate: {st.measured_rate_hz:6.1f} Hz (nominal {layout.sampling_rate})")
            self.sb_samples.setText(f"Samples: {st.samples_received:,}")
            ok = st.missing_packets == 0 and st.counter_repeats == 0
            self.sb_integrity.setText(
                "Continuity: OK (no dropped packets)" if ok else
                f"Continuity: {st.missing_packets:,} packets missing in {st.gap_events} gaps"
                + (f", {st.counter_repeats} repeats" if st.counter_repeats else ""))
            self.sb_integrity.setStyleSheet(f"color: {'#9FE3B8' if ok else '#FFB4AD'};")
            if st.scale_unknown_chunks:
                self.sb_scale.setText(f"µV scale: UNVERIFIED ({st.scale_unknown_chunks} chunks fit no known scale)")
                self.sb_scale.setStyleSheet("color: #FFB4AD;")
            elif st.scale_verdict == scale.LEGACY:
                self.sb_scale.setText("µV scale: corrected (BrainFlow ≤5.23 bug, ÷79.57)")
                self.sb_scale.setStyleSheet("color: #A9C1DA;")
            elif st.scale_verdict == scale.CORRECT:
                self.sb_scale.setText("µV scale: OK")
                self.sb_scale.setStyleSheet("color: #9FE3B8;")
            else:
                self.sb_scale.setText("")
        if recording:
            dur = time.monotonic() - (st.recording_started_monotonic or time.monotonic())
            self.sb_rec.setText(f"● REC {_fmt_duration(dur)}  ·  {st.recording_path.name}  ·  {st.recording_rows:,} rows")
            self.sb_rec.setStyleSheet("color: #FFB4AD; font-weight: 700;")
        elif st.last_recording_path is not None and st.last_recording_summary:
            s = st.last_recording_summary
            self.sb_rec.setText(f"Saved {st.last_recording_path.name}  ·  {s['rows']:,} rows  ·  "
                                f"{s['missing_packets']} missing")
            self.sb_rec.setStyleSheet("color: #DCE6F0;")
        else:
            self.sb_rec.setText(f"Recordings → {self.recordings_dir}")
            self.sb_rec.setStyleSheet("color: #A9C1DA;")

        # --- analysis panels (4 Hz)
        if layout is None or disp is None:
            return
        fs = layout.sampling_rate
        block, total = disp.latest(SPECTRUM_S * fs + FILTER_PAD_S * fs)
        if block.shape[1] < fs:
            return
        raw = block[:NUM_EEG]
        visible = self.ts_panel.visible_channels()
        shown = self._display_signal(raw, fs)
        spec = amplitude_spectrum(shown[:, -SPECTRUM_S * fs:], fs)
        if spec is not None:
            f, asd = spec
            self.fft_panel.update_spectrum(f, asd, visible)
            if visible:
                rel = relative_band_powers(f, asd[visible]).mean(axis=0)
                self.band_widget.update_bands(rel)
        loff = None
        if layout.has_leadoff:
            loff = (int(block[layout.display_extra_index("loff_statp"), -1]),
                    int(block[layout.display_extra_index("loff_statn"), -1]))
        self.values_widget.update_values(raw, fs, loff, f"Total samples received: {total:,}.")
        if layout.has_imu:
            self.imu_widget.update_imu(layout, block, fs, self.ts_panel.window_s)

    # ------------------------------------------------------------------ shutdown
    def closeEvent(self, event):
        st = self.engine.stats()
        if st.recording_path is not None:
            ans = QMessageBox.question(
                self, "Recording in progress",
                "A recording is in progress. Stop it, save the file, and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if self.engine.is_running():
            self.engine.disconnect()
            self.engine.join(timeout=15)
        for w in self._review_windows:
            w.close()
        event.accept()
