"""Acquisition engine: owns the BrainFlow session on a background thread.

Data path (one direction only):

    board ring buffer --get_board_data()--> chunk --> [CSV recorder]   (exact values)
                                                  --> [display buffer] (copy, for plots)

``get_board_data()`` *removes* what it returns from BrainFlow's buffer, so every
sample is read exactly once. The display never touches the recorded values, and
filters are applied only to copies in the GUI. BrainFlow's own buffer holds an
hour of Knight data, so even if this process stalls nothing is lost.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowError, BrainFlowInputParams

from . import scale
from .boards import KNIGHT_ID, KNIGHT_IMU_ID, NUM_EEG, BoardSpec, ChannelLayout, channel_setup_commands
from .integrity import SequenceTracker
from .recorder import CsvRecorder, _pkg_version
from .ringbuffer import RingBuffer

POLL_INTERVAL_S = 0.04
BRAINFLOW_BUFFER_SAMPLES = 450_000  # ~1 h at 125 Hz
DISPLAY_SECONDS = 60
STALL_WARNING_S = 2.0
RATE_WINDOW_S = 5.0

# Timing from NeuroPawn's reference code: the firmware needs time to apply each
# register write. Configuring 8 channels takes ~16 s.
CONFIG_SETTLE_S = 2.0
CONFIG_PRE_S = 0.5
CONFIG_BETWEEN_S = 1.0
CONFIG_POST_S = 0.5


class State(enum.Enum):
    IDLE = "Idle"
    CONNECTING = "Connecting"
    CONFIGURING = "Configuring channels"
    STREAMING = "Streaming"
    STOPPING = "Stopping"
    ERROR = "Error"


@dataclasses.dataclass
class Stats:
    state: State = State.IDLE
    message: str = ""
    error: str | None = None
    error_serial: int = 0
    samples_received: int = 0
    gap_events: int = 0
    missing_packets: int = 0
    counter_repeats: int = 0
    measured_rate_hz: float = 0.0
    seconds_since_data: float = 0.0
    config_done: int = 0
    config_total: int = 0
    config_warning: str | None = None
    recording_path: Path | None = None
    recording_rows: int = 0
    recording_started_monotonic: float | None = None
    last_recording_summary: dict | None = None
    last_recording_path: Path | None = None
    scale_verdict: str | None = None  # last chunk: legacy (corrected) / correct / empty / unknown
    scale_unknown_chunks: int = 0


class AcquisitionEngine:
    def __init__(self):
        self._lock = threading.Lock()  # protects _stats
        self._board_lock = threading.Lock()  # serializes our calls into BoardShim
        self._stats = Stats()
        self._cmds: queue.Queue = queue.Queue()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._cfg_thread: threading.Thread | None = None
        self._board: BoardShim | None = None
        self._recorder: CsvRecorder | None = None
        self._rate_hist: collections.deque = collections.deque()
        self._last_data_monotonic = 0.0
        self._sent_commands: list[str] = []
        self.spec: BoardSpec | None = None
        self.port: str = ""
        self.layout: ChannelLayout | None = None
        self.display: RingBuffer | None = None
        self._scale: scale.KnightScaleCorrector | None = None
        # Test hook: callables invoked with each raw chunk exactly as BrainFlow returned it
        # (before the Knight scale correction).
        self.chunk_listeners: list[Callable[[np.ndarray], None]] = []

    # ------------------------------------------------------------------ public API
    def stats(self) -> Stats:
        with self._lock:
            s = dataclasses.replace(self._stats)
        if s.state in (State.STREAMING, State.CONFIGURING) and self._last_data_monotonic:
            s.seconds_since_data = time.monotonic() - self._last_data_monotonic
        return s

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def connect(self, spec: BoardSpec, port: str = "") -> None:
        if self.is_running():
            raise RuntimeError("A session is already running")
        BoardShim.disable_board_logger()
        self.spec = spec
        self.port = port
        self.layout = ChannelLayout.for_board(spec.board_id)
        self.display = RingBuffer(len(self.layout.display_rows), self.layout.sampling_rate * DISPLAY_SECONDS)
        self._scale = scale.KnightScaleCorrector() if spec.board_id in (KNIGHT_ID, KNIGHT_IMU_ID) else None
        self._stop_evt.clear()
        self._cmds = queue.Queue()
        self._rate_hist.clear()
        self._last_data_monotonic = 0.0
        self._sent_commands = []
        with self._lock:
            keep = (self._stats.last_recording_summary, self._stats.last_recording_path, self._stats.error_serial)
            self._stats = Stats(state=State.CONNECTING, message=f"Opening {port or spec.label}…")
            self._stats.last_recording_summary, self._stats.last_recording_path, self._stats.error_serial = keep
        self._thread = threading.Thread(target=self._run, name="acquisition", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        """Ask the session to stop. Pending data is drained and any recording finalized."""
        self._stop_evt.set()

    def join(self, timeout: float | None = None) -> bool:
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    def start_recording(self, path: Path) -> None:
        """Begin writing at the next chunk boundary. Only allowed once channels are configured."""
        if self.stats().state is not State.STREAMING:
            raise RuntimeError("Recording is only available while streaming (after channel setup).")
        self._cmds.put(("start_rec", Path(path)))

    def stop_recording(self) -> None:
        self._cmds.put(("stop_rec", None))

    def insert_marker(self, value: float) -> bool:
        if value == 0:
            raise ValueError("Marker value 0 is reserved for 'no marker'")
        with self._board_lock:
            board = self._board
            if board is None:
                return False
            try:
                board.insert_marker(float(value))
                return True
            except BrainFlowError:
                return False

    # ------------------------------------------------------------------ internals
    def _set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self._stats, k, v)

    def _fail(self, msg: str) -> None:
        with self._lock:
            self._stats.state = State.ERROR
            self._stats.error = msg
            self._stats.error_serial += 1
            self._stats.message = msg

    def _run(self) -> None:
        spec, layout = self.spec, self.layout
        assert spec is not None and layout is not None
        params = BrainFlowInputParams()
        if spec.needs_port:
            params.serial_port = self.port
        board = BoardShim(spec.board_id, params)
        prepared = False
        tracker = SequenceTracker()
        try:
            try:
                board.prepare_session()
                prepared = True
                board.start_stream(BRAINFLOW_BUFFER_SAMPLES)
            except BrainFlowError as e:
                hint = ""
                if spec.needs_port:
                    hint = (" Check the USB cable, that the port is right, and that the NeuroPawn EXG "
                            "Visualizer (or any other program) is closed, because only one program can use the port.")
                self._fail(f"Could not start the board on {self.port or spec.label}: {e}.{hint}")
                return
            with self._board_lock:
                self._board = board

            if spec.configure_channels:
                cmds = channel_setup_commands()
                self._set(state=State.CONFIGURING, config_done=0, config_total=len(cmds),
                          message="Configuring channels (gain 12 + RLD)…")
                self._cfg_thread = threading.Thread(target=self._configure, args=(board, cmds), daemon=True)
                self._cfg_thread.start()
            else:
                self._set(state=State.STREAMING, message="Streaming")

            self._last_data_monotonic = time.monotonic()
            while not self._stop_evt.is_set():
                t0 = time.monotonic()
                self._handle_commands()
                with self._board_lock:
                    data = board.get_board_data()
                if data.shape[1]:
                    self._process(data, tracker)
                time.sleep(max(0.0, POLL_INTERVAL_S - (time.monotonic() - t0)))

            # ---- orderly shutdown: stop, drain the remainder, close the file
            self._set(state=State.STOPPING, message="Stopping…")
            if self._cfg_thread is not None:
                self._cfg_thread.join(timeout=5)
            with self._board_lock:
                board.stop_stream()
                data = board.get_board_data()
            if data.shape[1]:
                self._process(data, tracker)
            self._handle_commands()
            self._close_recorder()
            self._set(state=State.IDLE, message="Session stopped")
        except Exception as e:  # unexpected: never lose the file silently
            try:
                with self._board_lock:
                    leftover = board.get_board_data() if prepared else np.empty((0, 0))
                if leftover.size:
                    self._process(leftover, tracker)
            except Exception:
                pass
            self._close_recorder()
            self._fail(f"Acquisition stopped unexpectedly: {e!r}")
        finally:
            with self._board_lock:
                self._board = None
            if prepared:
                try:
                    board.release_session()
                except BrainFlowError:
                    pass

    def _configure(self, board: BoardShim, cmds: list[tuple[str, str]]) -> None:
        wait = self._stop_evt.wait
        if wait(CONFIG_SETTLE_S):
            return
        for i, (on_cmd, rld_cmd) in enumerate(cmds, start=1):
            for delay, cmd in ((CONFIG_PRE_S, on_cmd), (CONFIG_BETWEEN_S, rld_cmd)):
                if wait(delay):
                    return
                try:
                    with self._board_lock:
                        board.config_board(cmd)
                    self._sent_commands.append(cmd)
                except BrainFlowError as e:
                    self._set(config_warning=f"'{cmd}' failed: {e}")
            if wait(CONFIG_POST_S):
                return
            self._set(config_done=i, message=f"Configuring channels… {i}/{len(cmds)}")
        with self._lock:
            if self._stats.state is State.CONFIGURING:
                self._stats.state = State.STREAMING
                warn = self._stats.config_warning
                self._stats.message = "Streaming" + (f" (warning: {warn})" if warn else "")

    def _handle_commands(self) -> None:
        while True:
            try:
                cmd, arg = self._cmds.get_nowait()
            except queue.Empty:
                return
            if cmd == "start_rec" and self._recorder is None:
                try:
                    info = {"serial_port": self.port or None,
                            "board_key": self.spec.key if self.spec else None,
                            "config_commands_sent": list(self._sent_commands)}
                    if self._scale is not None:
                        info["scale_correction"] = scale.describe(_pkg_version("brainflow"))
                    self._recorder = CsvRecorder(arg, self.layout, info)
                    self._set(recording_path=arg, recording_rows=0,
                              recording_started_monotonic=time.monotonic())
                except OSError as e:
                    self._recorder = None
                    self._fail(f"Could not create recording file: {e}")
                    with self._lock:
                        self._stats.state = State.STREAMING
            elif cmd == "stop_rec":
                self._close_recorder()

    def _close_recorder(self) -> None:
        rec = self._recorder
        if rec is None:
            return
        self._recorder = None
        try:
            summary = rec.close()
        except OSError as e:
            summary = None
            self._fail(f"Error finalizing recording {rec.path.name}: {e}")
        self._set(recording_path=None, recording_started_monotonic=None,
                  last_recording_summary=summary, last_recording_path=rec.path)

    def _process(self, data: np.ndarray, tracker: SequenceTracker) -> None:
        layout = self.layout
        for cb in self.chunk_listeners:
            cb(data.copy())
        gaps = tracker.process(data[layout.package_row])

        verdict = None
        if self._scale is not None:
            fixed, verdict = self._scale.apply(data[layout.eeg_rows])
            if verdict == scale.LEGACY:
                data = data.copy()
                data[layout.eeg_rows] = fixed

        if self._recorder is not None:
            try:
                self._recorder.write_chunk(data, gaps, scale_verdict=verdict)
            except OSError as e:
                path = self._recorder.path
                self._close_recorder()
                self._fail(f"Recording stopped, write to {path.name} failed: {e}")
                with self._lock:
                    self._stats.state = State.STREAMING

        self.display.append(data[layout.display_rows])

        now = time.monotonic()
        self._last_data_monotonic = now
        n = data.shape[1]
        ts_last = float(data[layout.timestamp_row, -1])
        with self._lock:
            s = self._stats
            s.samples_received += n
            total = s.samples_received
        self._rate_hist.append((ts_last, total))
        while self._rate_hist and ts_last - self._rate_hist[0][0] > RATE_WINDOW_S:
            self._rate_hist.popleft()
        rate = 0.0
        if len(self._rate_hist) >= 2:
            (t_a, n_a), (t_b, n_b) = self._rate_hist[0], self._rate_hist[-1]
            if t_b > t_a:
                rate = (n_b - n_a) / (t_b - t_a)
        with self._lock:
            s = self._stats
            s.gap_events = tracker.gap_events
            s.missing_packets = tracker.missing_packets
            s.counter_repeats = tracker.repeat_events
            s.measured_rate_hz = rate
            if verdict is not None:
                s.scale_verdict = verdict
                s.scale_unknown_chunks = self._scale.verdicts[scale.UNKNOWN]
            if self._recorder is not None:
                s.recording_rows = self._recorder.rows_written


__all__ = ["AcquisitionEngine", "State", "Stats", "NUM_EEG"]
