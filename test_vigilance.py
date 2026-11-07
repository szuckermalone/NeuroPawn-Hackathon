"""Simple deterministic tests for experimental vigilance index."""
import numpy as np
from vigilance import VigilanceMonitor
from signals import extract


def test_baseline_and_alert():
    rng = np.random.default_rng(22)
    monitor = VigilanceMonitor(calibration_windows=8, smoothing=2, persistence=2)
    for _ in range(8):
        outcome = monitor.update(rng.normal([2.0, 3.0, 3.0], .12), "ok")
    assert outcome["state"] == "BASELINE READY"
    assert not outcome["alert"]
    for _ in range(3):
        outcome = monitor.update([4.3, 5.3, 1.3], "ok")
    assert outcome["alert"] and outcome["state"] == "ELEVATED EEG CHANGE"
    outcome = monitor.update(None, "timestamp gap")
    assert outcome["state"] == "SIGNAL UNAVAILABLE" and not outcome["alert"]


def test_signal_features():
    fs = 125
    t = np.arange(fs*8)/fs
    rng = np.random.default_rng(11)
    eeg = rng.normal(0, 2, (len(t),8))
    eeg += 6 * np.sin(2*np.pi*10*t)[:,None]
    feature, quality = extract(t,eeg,8,fs)
    assert quality == "ok"
    assert len(feature) == 3
    assert extract(t,eeg,10,fs)[1] == "insufficient or stale data"


if __name__ == "__main__":
    test_baseline_and_alert()
    test_signal_features()
    print("PASS: baseline calibration, persistence, signal-quality gating, EEG feature extraction")
