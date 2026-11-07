"""Board definitions, channel layout, and serial-port discovery.

Facts about the Knight board come from BrainFlow's driver
(src/board_controller/neuropawn/knight*.cpp) and NeuroPawn's example repos:

* 8 EXG channels, 125 Hz, int16 samples.
* BrainFlow converts counts to microvolts with a fixed gain of 12:
      uV = 4 / (2**15 - 1) / 12 / 79.57 * 1e6 * counts
  so channels must be switched on with gain 12 (``chon_{ch}_12``) for the
  microvolt values to be correct.
* The package counter is a single byte (wraps at 256).
* The timestamp is the host clock when BrainFlow parsed the packet.
* "other" channels are LOFF STATP / LOFF STATN, plus 9 IMU floats on the IMU
  variant (ax, ay, az, gx, gy, gz, mx, my, mz).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from brainflow.board_shim import BoardIds, BoardShim

NUM_EEG = 8
GAIN = 12
ADC_MAX_COUNTS = 32767
# Identical to KnightBase::eeg_scale in BrainFlow master (after fix #867, 2026-10-21).
KNIGHT_UV_PER_COUNT = 4.0 / (32768.0 - 1.0) / 12.0 / 79.57 * 1_000_000.0  # ~0.12785 uV
KNIGHT_FULL_SCALE_UV = ADC_MAX_COUNTS * KNIGHT_UV_PER_COUNT  # ~4189 uV
# BrainFlow <= 5.23.0 (the current pip release) lacks the /79.57 term, so it reports
# Knight values 79.57x too large, in float32. See scale.py, which corrects this.
KNIGHT_LEGACY_UV_PER_COUNT = 4.0 / (32768.0 - 1.0) / 12.0 * 1_000_000.0  # ~10.1728 "uV"

IMU_NAMES = [
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
]

KNIGHT_ID = BoardIds.NEUROPAWN_KNIGHT_BOARD.value
KNIGHT_IMU_ID = BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU.value
SYNTHETIC_ID = BoardIds.SYNTHETIC_BOARD.value


@dataclass(frozen=True)
class BoardSpec:
    key: str
    label: str
    board_id: int
    needs_port: bool
    configure_channels: bool  # send chon/rldadd commands after streaming starts


BOARDS: list[BoardSpec] = [
    BoardSpec("knight", "NeuroPawn Knight (8 ch)", KNIGHT_ID, True, True),
    BoardSpec("knight_imu", "NeuroPawn Knight IMU (8 ch + IMU)", KNIGHT_IMU_ID, True, True),
    BoardSpec("synthetic", "Synthetic demo (no hardware)", SYNTHETIC_ID, False, False),
]
BOARDS_BY_KEY = {b.key: b for b in BOARDS}


def channel_setup_commands(num_channels: int = NUM_EEG, gain: int = GAIN) -> list[tuple[str, str]]:
    """(chon, rldadd) command pairs, exactly as NeuroPawn's examples send them."""
    return [(f"chon_{ch}_{gain}", f"rldadd_{ch}") for ch in range(1, num_channels + 1)]


@dataclass
class ChannelLayout:
    """Where each signal lives in BrainFlow's data matrix, and how it maps to CSV columns."""

    board_id: int
    board_name: str
    sampling_rate: int
    num_rows: int
    package_row: int
    timestamp_row: int
    marker_row: int
    eeg_rows: list[int]
    extra_names: list[str] = field(default_factory=list)
    extra_rows: list[int] = field(default_factory=list)
    uv_per_count: float | None = None
    full_scale_uv: float | None = None

    @classmethod
    def for_board(cls, board_id: int) -> "ChannelLayout":
        d = BoardShim.get_board_descr(board_id)
        eeg_rows = list(d["eeg_channels"][:NUM_EEG])
        if len(eeg_rows) != NUM_EEG:
            raise ValueError(f"Board {board_id} has {len(eeg_rows)} EEG channels, expected {NUM_EEG}")
        extra_names: list[str] = []
        extra_rows: list[int] = []
        uv_per_count = full_scale = None
        if board_id in (KNIGHT_ID, KNIGHT_IMU_ID):
            other = list(d.get("other_channels", []))
            extra_names = ["loff_statp", "loff_statn"]
            if board_id == KNIGHT_IMU_ID:
                extra_names += IMU_NAMES
            if len(other) != len(extra_names):
                raise ValueError(
                    f"BrainFlow reports {len(other)} auxiliary rows for this board, "
                    f"expected {len(extra_names)}; BrainFlow version may be incompatible."
                )
            extra_rows = other
            uv_per_count = KNIGHT_UV_PER_COUNT
            full_scale = KNIGHT_FULL_SCALE_UV
        return cls(
            board_id=board_id,
            board_name=d["name"],
            sampling_rate=int(d["sampling_rate"]),
            num_rows=int(d["num_rows"]),
            package_row=int(d["package_num_channel"]),
            timestamp_row=int(d["timestamp_channel"]),
            marker_row=int(d["marker_channel"]),
            eeg_rows=eeg_rows,
            extra_names=extra_names,
            extra_rows=extra_rows,
            uv_per_count=uv_per_count,
            full_scale_uv=full_scale,
        )

    # ---- CSV -----------------------------------------------------------------
    @property
    def eeg_column_names(self) -> list[str]:
        return [f"ch{i}_uV" for i in range(1, NUM_EEG + 1)]

    def csv_columns(self) -> list[str]:
        return [
            "sample_index",
            "timestamp_unix",
            "package_num",
            "gap_before",
            *self.eeg_column_names,
            *self.extra_names,
            "marker",
        ]

    # ---- display ring buffer ---------------------------------------------------
    @property
    def display_rows(self) -> list[int]:
        """Rows copied into the display buffer: EEG, extras, marker (in that order)."""
        return [*self.eeg_rows, *self.extra_rows, self.marker_row]

    @property
    def display_marker_index(self) -> int:
        return NUM_EEG + len(self.extra_rows)

    def display_extra_index(self, name: str) -> int | None:
        if name not in self.extra_names:
            return None
        return NUM_EEG + self.extra_names.index(name)

    @property
    def has_imu(self) -> bool:
        return "accel_x" in self.extra_names

    @property
    def has_leadoff(self) -> bool:
        return "loff_statp" in self.extra_names


def list_serial_ports() -> list[tuple[str, str]]:
    """Return (device, description), most likely USB adapters first."""
    try:
        from serial.tools import list_ports
    except ImportError:  # pragma: no cover
        return []
    ports = []
    for p in list_ports.comports():
        dev = p.device
        if sys.platform == "darwin":
            if dev.startswith("/dev/tty."):
                continue  # use the callout (cu.*) devices on macOS
            if any(s in dev for s in ("Bluetooth", "debug-console", "wlan-debug")):
                continue
        ports.append((dev, p.description or ""))

    def rank(item: tuple[str, str]) -> int:
        dev, desc = item
        text = (dev + " " + desc).lower()
        return 0 if any(k in text for k in ("usbserial", "usbmodem", "wchusb", "ch34", "cp210", "ftdi", "arduino", "ttyusb", "ttyacm")) else 1

    return sorted(ports, key=rank)
