"""Chronological calibration, scoring and event-by-event replay."""
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


class Predictor:
    def __init__(self, calibration):
        self.n = calibration
        self.history = []
        self.training = []
        self.models = None

    def predict(self, elapsed, features):
        recent = float(np.median(self.history[-5:])) if self.history else 0.
        base = [elapsed, recent]
        if self.models is None and len(self.training) >= self.n:
            x, y = zip(*self.training)
            self.models = []
            for width in (2, 5):
                model = make_pipeline(StandardScaler(), Ridge(alpha=10))
                model.fit(np.asarray(x)[:, :width], np.log(y))
                self.models.append(model)
            self.median = float(np.median(self.history))
        result = dict(phase="test" if self.models else "calibration", median_ms=None,
                      behaviour_ms=None, combined_ms=None)
        if self.models:
            result["median_ms"] = self.median
            result["behaviour_ms"] = float(np.exp(self.models[0].predict([base])[0]))
            if features is not None:
                result["combined_ms"] = float(np.exp(self.models[1].predict([base+features])[0]))
        return base, result

    def observe(self, base, features, rt):
        if rt is not None and np.isfinite(rt) and rt > 0:
            self.history.append(rt)
            if features is not None and self.models is None:
                self.training.append((base + features, rt))


def records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summary(events):
    tests = [e for e in events if e["phase"] == "test"]
    matched = [e for e in tests if e["combined_ms"] is not None and e["rt_ms"] is not None]
    out = dict(total=len(events), test=len(tests), paired=len(matched),
               test_misses=sum(e["rt_ms"] is None for e in tests),
               eeg_available=sum(e["combined_ms"] is not None for e in tests))
    for key in ("median_ms", "behaviour_ms", "combined_ms"):
        out[key+"_mae"] = round(float(np.mean([abs(e[key]-e["rt_ms"]) for e in matched])), 1) if matched else None
    return out
