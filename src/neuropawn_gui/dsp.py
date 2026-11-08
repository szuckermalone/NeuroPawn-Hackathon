"""Display-only signal processing. Inputs are never modified; recordings are never filtered."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import integrate, signal


@dataclass(frozen=True)
class DisplayFilter:
    label: str
    kind: str  # "raw" | "dc" | "band"
    band: tuple[float, float] | None = None
    flag_artifacts: bool = False  # shade blinks / artifacts (cleaning.artifact_mask)


DISPLAY_FILTERS = [
    DisplayFilter("Clean 1–40 Hz + artifact flags", "band", (1.0, 40.0), flag_artifacts=True),
    DisplayFilter("Bandpass 1–50 Hz", "band", (1.0, 50.0)),
    DisplayFilter("Bandpass 5–50 Hz", "band", (5.0, 50.0)),
    DisplayFilter("Bandpass 7–13 Hz (alpha)", "band", (7.0, 13.0)),
    DisplayFilter("Bandpass 13–30 Hz (beta)", "band", (13.0, 30.0)),
    DisplayFilter("DC removed only", "dc"),
    DisplayFilter("Raw (as recorded)", "raw"),
]
NOTCH_OPTIONS = [("Notch 60 Hz", 60.0), ("Notch 50 Hz", 50.0), ("No notch", None)]

BANDS = [
    ("Delta", 1.0, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta", 13.0, 30.0),
    ("Gamma", 30.0, 50.0),
]


@lru_cache(maxsize=64)
def _bandpass_sos(lo: float, hi: float, fs: float, order: int = 4):
    hi = min(hi, 0.45 * fs)
    return signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def apparent_line_hz(mains_hz: float, fs_nominal: float, fs_true: float | None) -> float:
    """Where mains hum actually lands in data labeled with the nominal sample rate.

    BrainFlow labels Knight data as 125 Hz, but the board's clock runs ~0.5% fast
    (125.64-125.71 Hz measured on our recordings). Every frequency therefore appears
    ~0.5% low, and 60 Hz hum shows up at ~59.68 Hz. A notch at exactly 60.00 Hz misses
    it: 21 dB rejection offline / 10 dB live, versus 42-51 / 28-33 dB when centered
    correctly (measured on our real recordings).
    """
    if not fs_true or fs_true <= 0:
        return mains_hz
    return mains_hz * fs_nominal / fs_true


@lru_cache(maxsize=64)
def _notch_sos(f0: float, fs: float, q: float = 30.0):
    if f0 >= fs / 2:
        return None
    b, a = signal.iirnotch(f0, q, fs=fs)
    return signal.tf2sos(b, a)


def apply_display_filter(x: np.ndarray, fs: float, flt: DisplayFilter, notch_hz: float | None) -> np.ndarray:
    """Zero-phase filter a (channels, samples) array. Returns a new array."""
    y = np.array(x, dtype=np.float64, copy=True)
    if flt.kind == "raw" or y.shape[-1] == 0:
        return y
    y -= y.mean(axis=-1, keepdims=True)
    sections = []
    if notch_hz:
        sos = _notch_sos(round(float(notch_hz), 2), float(fs))
        if sos is not None:
            sections.append(sos)
    if flt.kind == "band" and flt.band:
        sections.append(_bandpass_sos(flt.band[0], flt.band[1], float(fs)))
    if not sections:
        return y
    sos = np.vstack(sections)
    if y.shape[-1] <= 3 * (2 * len(sos) + 1):
        return y
    return signal.sosfiltfilt(sos, y, axis=-1)


def amplitude_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Welch amplitude spectral density (uV/sqrt(Hz)) per channel, 0.5 Hz resolution."""
    n = x.shape[-1]
    if n < int(fs):
        return None
    nperseg = min(n, int(2 * fs))
    f, p = signal.welch(x, fs=fs, nperseg=nperseg, axis=-1, detrend="constant")
    return f, np.sqrt(p)


def relative_band_powers(f: np.ndarray, asd: np.ndarray) -> np.ndarray:
    """Relative power (%) of each band in ``BANDS``, per channel: (channels, bands)."""
    psd = asd ** 2
    powers = []
    for _, lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        powers.append(integrate.trapezoid(psd[..., m], f[m], axis=-1) if m.sum() > 1 else np.zeros(psd.shape[:-1]))
    p = np.stack(powers, axis=-1)
    tot = p.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(tot > 0, 100.0 * p / tot, 0.0)
    return rel
