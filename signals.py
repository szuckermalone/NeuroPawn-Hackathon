"""Eight-channel features. Input samples are in microvolts and Unix seconds."""
import numpy as np
from scipy.signal import welch

CHANNELS = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
FEATURES = ["frontal_theta", "posterior_alpha", "central_beta"]


def extract(t, eeg, end, fs):
    start = np.searchsorted(t, end - 8)
    stop = np.searchsorted(t, end, side="left")
    x = eeg[start:stop]
    times = t[start:stop]
    if len(x) < 0.95 * 8 * fs or end - times[-1] > 0.25:
        return None, "insufficient or stale data"
    dt = np.diff(times)
    if np.any(dt < 0) or np.max(dt) > 0.25:
        return None, "timestamp gap"
    if not np.isfinite(x).all() or np.min(np.std(x, axis=0)) < 0.1:
        return None, "flat or invalid channel"
    if np.max(np.ptp(x, axis=0)) > 500:
        return None, "large excursion"
    freq, power = welch(x, fs=fs, nperseg=2*fs, axis=0)
    values = []
    for lo, hi, channels in [(4, 8, [0, 1]), (8, 13, [4, 5, 6, 7]), (13, 30, [2, 3])]:
        band = power[(freq >= lo) & (freq < hi)][:, channels]
        values.append(float(np.log(max(1e-12, band.sum(axis=0).mean() * (freq[1]-freq[0])))))
    return values, "ok"
