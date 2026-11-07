"""Experimental, baseline-relative EEG vigilance change indicator.

Not a calibrated probability of drowsiness and not for driving decisions.
Designed to operate on the existing 3 log-band-power features.
"""
from collections import deque
import numpy as np


class VigilanceMonitor:
    """Reference-normalised, quality-gated band-power trend monitor.

    Calibration uses accepted feature windows only. Inference does not update the
    reference. Higher theta and alpha, lower beta are *illustrative* changes,
    not sufficient evidence for fatigue. Smoothing and persistence reduce flicker.
    """
    def __init__(self, calibration_windows=8, smoothing=3, persistence=3):
        if calibration_windows < 3 or smoothing < 1 or persistence < 1:
            raise ValueError("Invalid calibration, smoothing or persistence")
        self.calibration_windows = calibration_windows
        self.reference = []
        self.center = None
        self.scale = None
        self.recent = deque(maxlen=smoothing)
        self.persistence = persistence
        self.high_count = 0
        self.last_state = "CALIBRATING"

    def update(self, features, quality="ok"):
        out = dict(state=self.last_state, score=None, alert=False,
                   calibration=f"{len(self.reference)}/{self.calibration_windows}",
                   signal_quality=quality, note="Experimental index, not a fatigue diagnosis")
        if quality != "ok" or features is None:
            self.high_count = 0
            self.last_state = "SIGNAL UNAVAILABLE"
            out["state"] = self.last_state
            return out
        x = np.asarray(features, dtype=float)
        if x.shape != (3,) or not np.isfinite(x).all():
            self.high_count = 0
            self.last_state = "SIGNAL UNAVAILABLE"
            out["state"] = self.last_state
            out["signal_quality"] = "invalid features"
            return out
        if self.center is None:
            self.reference.append(x.copy())
            out["calibration"] = f"{len(self.reference)}/{self.calibration_windows}"
            if len(self.reference) == self.calibration_windows:
                baseline = np.asarray(self.reference)
                self.center = np.median(baseline, axis=0)
                # Robust dispersion, with a conservative floor for small samples.
                self.scale = np.maximum(1.4826 * np.median(np.abs(baseline - self.center), axis=0), 0.3)
                self.last_state = "BASELINE READY"
                out["state"] = self.last_state
            return out
        z = np.clip((x-self.center)/self.scale, -4.0, 4.0)
        # Hypothesis-based composite index; weights are NOT learned or validated.
        raw = float(0.5*z[0] + 0.3*z[1] - 0.2*z[2])
        self.recent.append(raw)
        score = float(np.mean(self.recent))
        if score >= 1.5:
            self.high_count += 1
        else:
            self.high_count = 0
        state = "ELEVATED EEG CHANGE" if self.high_count >= self.persistence else "MONITORING"
        self.last_state = state
        out.update(state=state, score=round(score, 3),
                   alert=state == "ELEVATED EEG CHANGE", calibration="complete")
        return out
