from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from neuropawn_gui.cleaning import (
    RealtimeCleaner, artifact_mask, clean_offline, common_average,
)

FS = 125.0
REAL = Path(__file__).resolve().parents[1] / "data/recordings/neuropawn_knight_2026-11-07_16-31-49.csv"


def _synthetic(secs=20, seed=0, blinks=(5.0, 11.0, 16.0)):
    rng = np.random.default_rng(seed)
    t = np.arange(int(secs * FS)) / FS
    x = rng.normal(0, 5, (8, t.size))
    x += 10 * np.sin(2 * np.pi * 10 * t)            # alpha
    x += 30 * np.sin(2 * np.pi * 60 * t)            # mains hum
    x += 80 * np.sin(2 * np.pi * 0.2 * t)           # drift
    x += np.arange(8)[:, None] * 50.0               # DC offsets
    blink = np.zeros(t.size)
    for b in blinks:
        blink += 200 * np.exp(-0.5 * ((t - b) / 0.08) ** 2)
    leak = np.array([1.0, 1.0, 0.12, 0.10, 0.05, 0.04, 0.01, 0.03])
    x += leak[:, None] * blink
    return t, x, blink, leak


def _amp_at(x, f0):
    f, p = signal.welch(x, fs=FS, nperseg=int(4 * FS), axis=-1)
    return np.sqrt(p[..., np.argmin(abs(f - f0))])


def test_realtime_chunked_equals_whole():
    _, x, _, _ = _synthetic()
    whole, m_whole = RealtimeCleaner(FS).process(x)
    rt = RealtimeCleaner(FS)
    parts, masks, i = [], [], 0
    for n in np.random.default_rng(1).integers(1, 17, 2000):
        if i >= x.shape[1]:
            break
        y, m = rt.process(x[:, i:i + n])
        parts.append(y)
        masks.append(m)
        i += n
    assert np.allclose(np.concatenate(parts, axis=1), whole, atol=1e-9)
    assert np.array_equal(np.concatenate(masks), m_whole)


def test_realtime_removes_hum_drift_and_dc():
    _, x, _, _ = _synthetic()
    y, _ = RealtimeCleaner(FS).process(x)
    settle = int(3 * FS)
    assert (_amp_at(y[:, settle:], 60) < 0.02 * _amp_at(x[:, settle:], 60)).all()
    assert np.abs(y[:, settle:].mean(axis=1)).max() < 3.0
    a10_in, a10_out = _amp_at(x[:, settle:], 10), _amp_at(y[:, settle:], 10)
    assert (a10_out > 0.7 * a10_in).all()  # alpha preserved


def test_realtime_flags_blinks_causally():
    t, x, _, _ = _synthetic()
    _, m = RealtimeCleaner(FS).process(x)
    for b in (5.0, 11.0, 16.0):
        # Causal filters delay the waveform, so the flag lands shortly after the true peak.
        j = int(b * FS)
        assert m[j: j + int(0.2 * FS)].any(), f"blink at {b}s not flagged within 200 ms"
    assert m.mean() < 0.15


def test_offline_detects_and_regresses_blinks():
    _, x, blink, leak = _synthetic()
    res = clean_offline(x, FS)
    times = sorted(res.blink_samples / FS)
    assert np.allclose(times, [5.0, 11.0, 16.0], atol=0.05)
    assert np.allclose(res.regression_coefs[2:], leak[2:], atol=0.03)
    for b in (5.0, 11.0, 16.0):
        j = int(b * FS)
        assert np.abs(res.data[2:6, j]).max() < 20  # leakage on ch3-6 largely gone (was 8-24 uV + noise)


def test_car_excludes_frontal_from_mean():
    x = np.zeros((8, 10))
    x[0] = 1000.0  # big blink on ch1 only
    y = common_average(x)
    assert np.allclose(y[2:], 0.0), "blink on ch1 must not leak into other channels via CAR"


@pytest.mark.parametrize("n", [1, 5, 13, 40, 400])
def test_artifact_mask_length_matches_input(n):
    """Regression: short windows (stream start) must not return a kernel-length mask."""
    y = np.zeros((8, n))
    y[0, n // 2] = 500
    m = artifact_mask(y, FS)
    assert m.shape == (n,) and m[n // 2]


def test_artifact_mask_thresholds():
    y = np.zeros((8, 500))
    y[0, 100] = 120  # blink above 100 on frontal
    y[4, 300] = 120  # below 150 on non-frontal: not an artifact
    m = artifact_mask(y, FS)
    assert m[100] and not m[300]


@pytest.mark.skipif(not REAL.exists(), reason="real recording not present")
def test_real_recording():
    from neuropawn_gui.review import load_recording
    rec = load_recording(REAL)
    assert rec.scale_note  # legacy scale corrected on load
    res = clean_offline(rec.eeg, rec.fs)
    times = res.blink_samples / rec.fs
    for expected in (18.1, 26.3, 29.8):
        assert np.min(np.abs(times - expected)) < 0.3
    assert (np.array(res.report["line_60hz_asd_after"]) < 0.01).all()
    assert res.report["epochs_bad"] <= 2
    std_after = np.array(res.report["std_uv_after"])
    assert (std_after[2:] < 10).all()
