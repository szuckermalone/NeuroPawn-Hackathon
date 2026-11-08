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
    QApplication, QComboBox, QScrollArea, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSplitter, QStatusBar, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from . import cleaning, drowsiness, notify, quality, scale, theme
from .link import GAME_PORT, GUI_PORT, MARKERS, LinkEndpoint
from .acquisition import AcquisitionEngine, State
from .boards import BOARDS, BOARDS_BY_KEY, NUM_EEG, list_serial_ports
from .dsp import amplitude_spectrum, apparent_line_hz, apply_display_filter, relative_band_powers
from .widgets import (
    QUALITY_COLORS, BandPowerWidget, DrowsinessWidget, FFTPanel, IMUWidget, LiveValuesWidget, Panel,
    SignalCheckWidget,
    TimeSeriesPanel,
)

PLOT_INTERVAL_MS = 33
DROWSINESS_MARKER = 90  # written into the recording's marker column when an alert fires
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
        self._quality_tracker = quality.QualityTracker()
        self._quality: quality.QualityReport | None = None
        self._fs_true: float | None = None
        self._drowsy: drowsiness.DrowsinessMonitor | None = None
        self._drowsy_t0 = 0.0
        self.alert_log: list[dict] = []
        # link to the driving game (separate process); the GUI keeps the headset
        self.game_proc: subprocess.Popen | None = None
        self.game: dict = {"seen": -1e9, "session": None, "state": None, "stats": {}, "acks": []}
        self._last_game_status = -1e9
        try:
            self.link: LinkEndpoint | None = LinkEndpoint(GUI_PORT, GAME_PORT)
            self.link_error = None
        except OSError as e:  # e.g. a second GUI already holds the port
            self.link, self.link_error = None, str(e)

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
        self.check_widget = SignalCheckWidget()
        self.band_widget = BandPowerWidget()
        self.values_widget = LiveValuesWidget()
        self.imu_widget = IMUWidget()
        self.drowsy_widget = DrowsinessWidget()
        self.drowsy_widget.recalibrate_requested.connect(self._recalibrate_drowsiness)
        self.drowsy_widget.test_alert_requested.connect(lambda: self._fire_drowsiness_alert(test=True))
        self.drowsy_widget.launch_game_requested.connect(self._launch_game)
        self.tabs.addTab(self.check_widget, "Signal Check")
        self._drowsy_tab = QScrollArea()
        self._drowsy_tab.setWidgetResizable(True)
        self._drowsy_tab.setFrameShape(QFrame.Shape.NoFrame)
        self._drowsy_tab.setWidget(self.drowsy_widget)
        self.tabs.addTab(self._drowsy_tab, "Drowsiness")
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
        self.ts_panel.channel_visibility_changed.connect(self._note_channel_exclusion)
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

        self.quality_label = QLabel("")
        self.quality_label.setToolTip("Electrode connection check. See the Signal Check tab for details.")
        self.quality_label.mousePressEvent = lambda _e: self.tabs.setCurrentWidget(self.check_widget)
        h.addWidget(self.quality_label)

        self.drowsy_label = QLabel("")
        self.drowsy_label.setToolTip("Drowsiness score (0–100). See the Drowsiness tab.")
        self.drowsy_label.mousePressEvent = lambda _e: self.tabs.setCurrentWidget(self._drowsy_tab)
        h.addWidget(self.drowsy_label)

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
        self.check_widget.clear()
        self._quality_tracker.reset()
        self._quality = None
        self._fs_true = None
        self._drowsy = drowsiness.DrowsinessMonitor(layout.sampling_rate)
        self._drowsy_t0 = time.monotonic()
        self.drowsy_widget.clear()
        self.drowsy_label.setText("")
        self.tabs.setCurrentWidget(self.check_widget)
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
            self.engine.start_recording(path, {"signal_check_at_start": quality.snapshot(self._quality),
                                               "drowsiness_detector": self._drowsiness_config(),
                                               "game_markers": MARKERS,
                                               "night_drive_session": self.game.get("session") if self._game_connected else None})
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
        self._poll_game_link()
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

    def _line_hz(self, mains_hz: float | None) -> float | None:
        """Mains frequency as it appears in the data (board clock runs ~0.5% fast)."""
        if not mains_hz or self._session_layout is None:
            return mains_hz
        return apparent_line_hz(mains_hz, self._session_layout.sampling_rate, self._fs_true)

    def _display_signal(self, raw: np.ndarray, fs: float) -> np.ndarray:
        """Filtered (and optionally re-referenced) copy for display. Never touches the recording."""
        flt = self.ts_panel.display_filter
        shown = apply_display_filter(raw, fs, flt, self._line_hz(self.ts_panel.notch_hz))
        if self.ts_panel.car_enabled and flt.kind != "raw":
            shown = cleaning.common_average(shown)
        return shown

    def _update_slow(self):
        if self.game_proc is not None and self.game_proc.poll() is not None and not self._game_connected:
            self.game_proc = None
            self._update_game_status_label()
        st = self.engine.stats()
        if st.fs_true:
            self._fs_true = st.fs_true
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
            notch = self.ts_panel.notch_hz
            if notch and self._fs_true:
                self.sb_rate.setText(self.sb_rate.text() + f" · board clock {self._fs_true:.2f} Hz → notch tuned to "
                                     f"{self._line_hz(notch):.2f} Hz")
            elif notch:
                self.sb_rate.setText(self.sb_rate.text() + " · tuning notch to board clock…")
            self.sb_samples.setText(f"Samples: {st.samples_received:,}")
            ok = st.missing_packets == 0 and st.counter_repeats == 0 and st.counter_backsteps == 0
            if st.stream_fault:
                self.sb_integrity.setText(f"⚠ STREAM FAULT: {st.stream_fault}. Unplug the headset USB 5 s, then Start Session")
                self.sb_integrity.setStyleSheet("color: #FF6B60; font-weight: 700;")
            else:
                parts = []
                if st.missing_packets:
                    parts.append(f"{st.missing_packets:,} missing in {st.gap_events} gaps")
                if st.counter_repeats:
                    parts.append(f"{st.counter_repeats:,} repeats")
                if st.counter_backsteps:
                    parts.append(f"{st.counter_backsteps:,} out of order")
                self.sb_integrity.setText("Continuity: OK (no dropped packets)" if ok else
                                          "Stream OK now · earlier: " + ", ".join(parts))
                self.sb_integrity.setStyleSheet(f"color: {'#9FE3B8' if ok else '#FFD479'};")
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
        if st.stream_fault:  # the data is broken: don't score electrodes or drowsiness from it
            self._show_stream_fault(st)
        else:
            self._update_quality(raw, fs, layout)
            if st.state is State.STREAMING:
                self._update_drowsiness(raw)
        loff = None
        if layout.has_leadoff:
            loff = (int(block[layout.display_extra_index("loff_statp"), -1]),
                    int(block[layout.display_extra_index("loff_statn"), -1]))
        self.values_widget.update_values(raw, fs, loff, f"Total samples received: {total:,}.")
        if layout.has_imu:
            self.imu_widget.update_imu(layout, block, fs, self.ts_panel.window_s)

    def _show_stream_fault(self, st) -> None:
        self.quality_label.setText(" ✗ STREAM FAULT ")
        self.quality_label.setStyleSheet("color: white; background: #C8382F; border-radius: 4px; padding: 3px 6px; font-weight: 700;")
        self.check_widget.headline.setText("Board stream fault: electrode scores paused")
        self.check_widget.headline.setStyleSheet("font-size: 15px; font-weight: 700; color: #C8382F;")
        self.check_widget.warning.setText(
            f"The headset is sending a broken data stream ({st.stream_fault}). This is a board/USB problem, not "
            "electrode contact. Stop Session, unplug the headset's USB cable for 5 seconds, plug it back in, "
            "and Start Session again.")
        self.check_widget.warning.show()
        if st.stream_fault_serial != getattr(self, "_shown_fault_serial", 0):
            self._shown_fault_serial = st.stream_fault_serial
            QMessageBox.warning(self, "Headset stream fault",
                                f"The headset is sending a broken data stream: {st.stream_fault}.\n\n"
                                "This isn't electrode contact. Stop Session, unplug the headset's USB cable for "
                                "5 seconds, plug it back in, and Start Session again.")

    def _update_quality(self, raw: np.ndarray, fs: float, layout) -> None:
        if not self.engine.is_running():
            return
        line_hz = self._line_hz(self.ts_panel.notch_hz or 60.0)
        report = quality.assess(raw, fs, line_hz=line_hz, full_scale_uv=layout.full_scale_uv,
                                ignore_hum=self.check_widget.ignore_hum_cb.isChecked())
        self._quality = None if report is None else self._quality_tracker.update(report, time.monotonic())
        q = self._quality
        self.ts_panel.set_quality(q)
        self.check_widget.update_report(q)
        if q is None:
            self.quality_label.setText("Signal: measuring…")
            self.quality_label.setStyleSheet("color: #C9D6E3; font-weight: 600;")
            return
        n_good, n = q.n_good, len(q.channels)
        status = quality.GOOD if n_good == n and not q.warning else quality.FAIR if n_good >= n - 2 else quality.POOR
        mark = {"good": "✓", "fair": "!", "poor": "✗"}[status]
        self.quality_label.setText(f" {mark} Signal {n_good}/{n} good ")
        self.quality_label.setStyleSheet(
            f"color: white; background: {QUALITY_COLORS[status]}; border-radius: 4px; padding: 3px 6px; font-weight: 700;")
        self.quality_label.setToolTip(q.warning or "Click for details (Signal Check tab)")

    # ------------------------------------------------------------------ drowsiness
    def _drowsiness_config(self) -> dict | None:
        m = self._drowsy
        if m is None:
            return None
        return {
            "baseline_theta_uv2": m.baseline_theta, "baseline_beta_uv2": m.baseline_beta,
            "ratio_threshold": m.ratio_threshold, "theta_rise": m.theta_rise, "beta_max": m.beta_max,
            "alert_score": m.alert_score, "epoch_s": drowsiness.EPOCH_S, "window_s": drowsiness.WINDOW_S,
            "alert_marker_value": DROWSINESS_MARKER,
            "quality_ignores_hum": self.check_widget.ignore_hum_cb.isChecked(),
            "excluded_channels": [r.idx + 1 for r in self.ts_panel.rows if not r.visible],
        }

    def _note_channel_exclusion(self) -> None:
        off = [r.idx + 1 for r in self.ts_panel.rows if not r.visible]
        self.statusBar().showMessage(
            ("Excluded from the drowsiness detector: " + ", ".join(f"Ch{c}" for c in off)) if off
            else "All channels available to the drowsiness detector", 6000)
        if self.engine.stats().recording_path is not None:  # keep the recording's metadata in sync
            self.engine.annotate_recording("drowsiness_detector", self._drowsiness_config())

    def _update_drowsiness(self, raw: np.ndarray) -> None:
        m = self._drowsy
        if m is None or self._quality is None:
            return
        usable = np.array([c.status != quality.POOR for c in self._quality.channels])
        usable &= np.array([r.visible for r in self.ts_panel.rows])  # channel buttons switched off = excluded
        s = m.update(raw, time.monotonic(), usable)
        if s is None:
            return
        self.drowsy_widget.update_state(s, self._drowsy_t0)
        if s.score is None:
            txt, bg = ("Drowsiness: calibrating…" if s.phase == "calibrating" else "Drowsiness: —"), "#4A6A8C"
        else:
            txt = f"Drowsiness {s.score:.0f}" + ("  PRE-SLEEP" if s.score >= m.alert_score else "")
            bg = theme.RED if s.score >= m.alert_score else theme.AMBER if s.score >= drowsiness.REARM_SCORE else theme.GREEN
        self.drowsy_label.setText(f" {txt} ")
        self.drowsy_label.setStyleSheet(f"color: white; background: {bg}; border-radius: 4px; padding: 3px 6px; font-weight: 700;")
        if s.alert:
            self._fire_drowsiness_alert(score=s.score, reason=s.alert_reason)

    def _fire_drowsiness_alert(self, score: float | None = None, test: bool = False, reason: str = "") -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        enabled = self.drowsy_widget.alerts_cb.isChecked()
        to_game = enabled and self._game_connected and self.link is not None
        if to_game:  # interrupts the drive with a full-screen alarm
            self.link.send("alert", score=None if score is None else round(score, 1), test=test, reason=reason)
        if test:
            sent = {} if to_game else notify.drowsiness_alert(score or 50, speak_aloud=self.drowsy_widget.speak_cb.isChecked())
            where = "to Night Drive" if to_game else (", ".join(k for k, v in sent.items() if v) or "no channels available")
            self.statusBar().showMessage(f"Test alert sent ({where})", 5000)
            return
        marked = self.engine.insert_marker(DROWSINESS_MARKER)
        why = "microsleep pattern (~8 s)" if reason == "microsleep" else "pre-sleep pattern"
        self.alert_log.append({"time": stamp, "score": score, "reason": reason, "marker_inserted": marked,
                               "sent_to_game": to_game})
        self.drowsy_widget.show_alert(f"⚠ {stamp}: {why} (score {0 if score is None else score:.0f}). "
                                      + ("Drive interrupted" if to_game else "Alert sent" if enabled else "Alerts are off")
                                      + (", marker 90 written to the recording." if marked else "."))
        if enabled and not to_game:  # the game has its own alarm; otherwise notify on the desktop
            notify.drowsiness_alert(score or 0, speak_aloud=self.drowsy_widget.speak_cb.isChecked())
            QApplication.beep()
        elif enabled and self.drowsy_widget.speak_cb.isChecked():
            notify.speak("Warning. Drowsiness detected. Please take a break.")
        self.tabs.setCurrentWidget(self._drowsy_tab)

    # ------------------------------------------------------------------ driving game link
    @property
    def _game_connected(self) -> bool:
        return time.monotonic() - self.game["seen"] < 3.0

    def _launch_game(self, minutes: float, fullscreen: bool) -> None:
        if self.game_proc is not None and self.game_proc.poll() is None:
            self.statusBar().showMessage("Night Drive is already running.", 4000)
            return
        if self.link is None:
            QMessageBox.warning(self, "Night Drive", f"The game link couldn't start ({self.link_error}). "
                                "Is another copy of the GUI open?")
            return
        st = self.engine.stats()
        if st.state is State.STREAMING and st.recording_path is None:
            self._toggle_recording()  # the whole drive should be recorded
        elif st.state is not State.STREAMING:
            self.statusBar().showMessage("Headset isn't streaming: Night Drive will run without drowsiness detection.", 8000)
        cmd = [sys.executable, str(Path(__file__).resolve().parents[2] / "run_game.py"), "--minutes", f"{minutes:g}"]
        if fullscreen:
            cmd.append("--fullscreen")
        try:
            self.game_proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[2]))
        except OSError as e:
            QMessageBox.warning(self, "Night Drive", f"Couldn't start the game: {e}")
            return
        self.drowsy_widget.game_status.setText("Starting Night Drive…")
        self.tabs.setCurrentWidget(self._drowsy_tab)

    def _poll_game_link(self) -> None:
        if self.link is None:
            return
        for msg in self.link.poll():
            self._handle_game_message(msg)
        now = time.monotonic()
        if now - self._last_game_status >= 1.0 and (self._game_connected or self.game_proc is not None):
            self._send_game_status()
            self._last_game_status = now

    def _send_game_status(self) -> None:
        st = self.engine.stats()
        m, q = self._drowsy, self._quality
        last = m.history[-1][1] if m is not None and m.history else None
        self.link.send("status", streaming=st.state is State.STREAMING, recording=st.recording_path is not None,
                       signal=q.summary() if q is not None else None,
                       phase=None if m is None else ("monitoring" if m.calibrated else "calibrating"),
                       score=None if last is None else round(last))

    def _handle_game_message(self, msg: dict) -> None:
        kind = msg.get("type")
        g = self.game
        was_connected = self._game_connected
        g["seen"] = time.monotonic()
        if kind == "hello":
            g["session"] = msg.get("session")
            self._send_game_status()
            if self.engine.stats().recording_path is not None:  # link this EEG file to the game's log folder
                self.engine.annotate_recording("night_drive_session", g["session"])
            self._ensure_recording_for_game()
            if not was_connected:
                self.statusBar().showMessage(f"Night Drive connected ({g['session']})", 5000)
        elif kind == "state":
            g["state"] = msg.get("state")
            if g["state"] in ("calibration_eyes_open", "countdown", "driving"):
                self._ensure_recording_for_game()
        elif kind == "heartbeat":
            g["state"], g["stats"] = msg.get("state"), msg.get("stats") or {}
        elif kind == "calibrate":
            self._recalibrate_drowsiness()
        elif kind == "alert_ack":
            g["acks"].append(msg.get("latency"))
            self.engine.insert_marker(MARKERS["alert_ack"])
            if not msg.get("test"):
                self.drowsy_widget.show_alert(self.drowsy_widget.banner.text()
                                              + f"  Driver acknowledged after {msg.get('latency', 0):.1f} s.")
        elif kind == "event":
            name = msg.get("event", "")
            if name == "drift_onset":
                name = "drift_left" if msg.get("dir", 1) < 0 else "drift_right"
            if name in MARKERS:
                self.engine.insert_marker(MARKERS[name])
            if name in ("lapse", "lane_departure"):
                self.drowsy_widget.add_behaviour_event(time.monotonic(), name)
        elif kind == "bye":
            g["seen"] = -1e9
            g["state"] = "closed"
            self.statusBar().showMessage("Night Drive closed. The recording is still running; stop it when you're done.", 8000)
        self._update_game_status_label()

    def _ensure_recording_for_game(self) -> None:
        """A drive without an EEG recording is wasted: start recording automatically
        whenever the game connects or starts while the headset is streaming."""
        st = self.engine.stats()
        if st.state is State.STREAMING and st.recording_path is None:
            self._toggle_recording()
            self.statusBar().showMessage("Night Drive is running, so recording started automatically.", 8000)

    def _update_game_status_label(self) -> None:
        g, s = self.game, self.game.get("stats") or {}
        if not self._game_connected:
            running = self.game_proc is not None and self.game_proc.poll() is None
            text = "Starting… waiting for the game to connect." if running else (
                "Closed." if g.get("state") == "closed" else "Not running.")
        else:
            rt = s.get("median_rt_s")
            text = (f"Connected · {g.get('state')} · {s.get('driving_s', 0) / 60:.1f} min driven · {s.get('drifts', 0)} drifts · "
                    f"median RT {'—' if rt is None else f'{rt:.2f} s'} · {s.get('lapses', 0)} lapses · "
                    f"{s.get('lane_departures', 0)} lane departures · SDLP {s.get('sdlp_m') or '—'} m")
            if g["acks"]:
                text += f" · alert acks {', '.join(f'{a:.1f}s' for a in g['acks'] if a is not None)}"
        self.drowsy_widget.game_status.setText(text)

    def _recalibrate_drowsiness(self) -> None:
        if self._drowsy is not None:
            self._drowsy.reset_calibration()
            self.drowsy_widget.banner.hide()
            self.statusBar().showMessage("Recalibrating drowsiness baseline: stay awake, ~30 s eyes open then ~30 s eyes closed", 8000)

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
