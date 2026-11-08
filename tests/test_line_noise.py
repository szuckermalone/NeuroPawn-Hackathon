"""Mains-hum handling: the notch must sit where the hum actually lands in the data."""

import time
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from neuropawn_gui.acquisition import ClockFit
from neuropawn_gui.dsp import DISPLAY_FILTERS, apparent_line_hz, apply_display_filter

REC = Path(__file__).resolve().parents[1] / "data/recordings/neuropawn_knight_2026-11-07_17-29-49.csv"
DC_ONLY = next(f for f in DISPLAY_FILTERS if f.kind == "dc")


def test_apparent_line_frequency():
    assert apparent_line_hz(60, 125, 125.641) == pytest.approx(59.694, abs=1e-3)
    assert apparent_line_hz(60, 125, None) == 60
    assert apparent_line_hz(50, 125, 125.0) == 50


def test_clockfit_recovers_rate_from_bursty_timestamps():
    rng = np.random.default_rng(0)
    fs_true, n = 125.66, 125 * 60
    t_ideal = 1.7e9 + np.arange(n) / fs_true
    # USB bursts: samples are parsed in batches of ~5, plus scheduling jitter.
    ts = np.repeat(t_ideal[4::5], 5)[:n] + rng.uniform(0, 0.004, n)
    cf = ClockFit()
    for i in range(0, n, 7):
        cf.add(ts[i:i + 7], np.zeros(min(7, n - i), dtype=int))
    assert cf.rate() == pytest.approx(fs_true, rel=3e-4)
    assert cf.span_s() == pytest.approx(n / fs_true, abs=0.05)


def test_clockfit_counts_dropped_packets():
    fs_true = 125.66
    idx = np.r_[0:500, 520:1500]  # 20 packets lost
    ts = 1.7e9 + idx / fs_true
    gaps = np.zeros(idx.size, dtype=int)
    gaps[500] = 20
    cf = ClockFit()
    cf.add(ts, gaps)
    assert cf.rate() == pytest.approx(fs_true, rel=1e-9)


def test_retuned_notch_beats_fixed_60_on_synthetic_hum():
    fs, fs_true = 125.0, 125.66
    t = np.arange(int(30 * fs)) / fs_true          # real sample instants
    hum = 50 * np.sin(2 * np.pi * 60 * t)            # 60 Hz in real time
    x = hum[None, :]

    def left(f0):
        y = apply_display_filter(x, fs, DC_ONLY, f0)[0, 250:-250]
        return np.sqrt(np.mean((y - y.mean()) ** 2))
    fixed, tuned = left(60.0), left(apparent_line_hz(60, fs, fs_true))
    assert fixed > 2.0          # off-center notch leaves several uV of a 50 uV hum
    assert tuned < 0.1 * fixed  # centered notch removes >90% of what's left


@pytest.mark.skipif(not REC.exists(), reason="recording absent")
def test_real_recording_clock_and_hum_rejection():
    from neuropawn_gui.review import load_recording
    r = load_recording(REC)
    assert r.fs_true == pytest.approx(125.66, abs=0.05)
    f0 = apparent_line_hz(60, r.fs, r.fs_true)
    x = r.eeg[7:8]  # Ch8: ~490 uV rms of hum

    def hum_rms(y):
        sos = signal.butter(2, [f0 - 0.4, f0 + 0.4], "bandpass", fs=r.fs, output="sos")
        return np.sqrt(np.mean(signal.sosfiltfilt(sos, y, axis=-1)[:, 400:-400] ** 2))
    before = hum_rms(x - x.mean())
    fixed = hum_rms(apply_display_filter(x, r.fs, DC_ONLY, 60.0))
    tuned = hum_rms(apply_display_filter(x, r.fs, DC_ONLY, f0))
    assert 20 * np.log10(before / fixed) < 25      # the old notch: ~21 dB
    assert 20 * np.log10(before / tuned) > 40      # retuned: > 40 dB


def test_engine_reports_true_rate(monkeypatch):
    from neuropawn_gui import acquisition
    from neuropawn_gui.acquisition import AcquisitionEngine, State
    from neuropawn_gui.boards import BOARDS_BY_KEY
    monkeypatch.setattr(acquisition, "FS_TRUE_MIN_S", 1.5)
    eng = AcquisitionEngine()
    eng.connect(BOARDS_BY_KEY["synthetic"])
    end = time.monotonic() + 10
    while time.monotonic() < end and eng.stats().fs_true is None:
        time.sleep(0.05)
    st = eng.stats()
    eng.disconnect()
    eng.join(10)
    assert st.state is State.STREAMING
    assert st.fs_true == pytest.approx(250, rel=0.02)


def test_clockfit_ignores_repeated_counters():
    """Last session: 11% of packets repeated their counter and the old fit said 129 Hz."""
    rng = np.random.default_rng(3)
    fs_true, n = 125.66, 125 * 120
    t = 1.7e9 + np.arange(n) / fs_true
    dup = rng.random(n) < 0.11                         # duplicate packets: same instant, counter repeats
    ts = np.repeat(t, np.where(dup, 2, 1))
    gaps = np.concatenate([[0, -1] if d else [0] for d in dup])
    cf = ClockFit()
    for i in range(0, ts.size, 9):
        cf.add(ts[i:i + 9], gaps[i:i + 9])
    assert cf.rate() == pytest.approx(fs_true, rel=1e-3)


def test_engine_rejects_implausible_clock(monkeypatch):
    from neuropawn_gui import acquisition
    from neuropawn_gui.acquisition import AcquisitionEngine, State
    from neuropawn_gui.boards import BOARDS_BY_KEY
    monkeypatch.setattr(acquisition, "FS_TRUE_MIN_S", 1.0)
    monkeypatch.setattr(acquisition, "FS_TRUE_MAX_DEVIATION", 1e-9)  # nothing is plausible
    eng = AcquisitionEngine()
    eng.connect(BOARDS_BY_KEY["synthetic"])
    end = time.monotonic() + 6
    while time.monotonic() < end and eng.stats().samples_received < 250 * 3:
        time.sleep(0.05)
    st = eng.stats()
    eng.disconnect()
    eng.join(10)
    assert st.state is State.STREAMING and st.fs_true is None
