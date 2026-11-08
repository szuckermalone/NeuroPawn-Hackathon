"""Electrode connection check: a 0-100 score and a ✓ / ! / ✗ per channel.

The Knight firmware doesn't report impedance or lead-off status (the LOFF bytes are
always 0), so contact quality is judged from the signal itself over the last few
seconds. Each channel gets several sub-scores, each tied to a failure we saw on
real recordings:

* line   60 Hz hum. A loose electrode picks up mains hum.
         Good contact: < 2 uV/sqrt(Hz). The floating Ch8 measured ~26.
* noise  1-40 Hz amplitude. EEG is ~5-40 uV. Much larger means a loose
         electrode, movement, or muscle.
* flat   nearly no signal: channel switched off, unplugged, or shorted.
* drift  large slow wander or DC jumps (electrode settling, cable tugging).
* rail   signal pinned near the ADC limit (about ±4190 uV).

The channel score is the *worst* sub-score, so one real problem can't be hidden by
the others looking fine. The reason shown is that worst sub-score.

Separately, if several channels are near-copies of each other (r >= 0.95), the
report carries a warning. That's what happens when a floating electrode in the
right-leg-drive loop injects its noise into every other channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

WINDOW_S = 4.0
MIN_WINDOW_S = 2.0
GOOD, FAIR, POOR = "good", "fair", "poor"
SYMBOL = {GOOD: "✓", FAIR: "!", POOR: "✗"}

# (good_at_or_below, bad_at_or_above) for "lower is better" metrics.
LINE_UV = (2.0, 8.0)          # uV/sqrt(Hz) of mains peak above the nearby noise floor
NOISE_UV = (40.0, 100.0)      # std of the 1-40 Hz signal
DRIFT_UV = (600.0, 2000.0)    # peak-to-peak of the 0-1 Hz component over the window
FLAT_UV = (0.5, 2.0)          # std of 1-40 Hz: <=0.5 dead, >=2 alive
RAIL_FRACTION = (0.90, 0.99)  # of full scale
IDENTICAL_R = 0.95
COPY_PARTNERS = 2


@dataclass
class ChannelQuality:
    channel: int          # 1-based
    score: int            # 0-100
    status: str           # good / fair / poor
    reason: str           # human-readable, the worst sub-score ("" when good)
    metrics: dict = field(default_factory=dict)

    @property
    def symbol(self) -> str:
        return SYMBOL[self.status]


@dataclass
class QualityReport:
    channels: list[ChannelQuality]
    identical: list[int]            # 1-based channels that are near-copies of each other
    warning: str | None

    @property
    def n_good(self) -> int:
        return sum(c.status == GOOD for c in self.channels)

    def summary(self) -> str:
        return f"{self.n_good}/{len(self.channels)} channels good"


def _ramp(v: float, good: float, bad: float) -> float:
    """100 at/below ``good``, 0 at/above ``bad``, linear in between."""
    if v <= good:
        return 100.0
    if v >= bad:
        return 0.0
    return 100.0 * (bad - v) / (bad - good)


def _status(score: float) -> str:
    return GOOD if score >= 80 else FAIR if score >= 50 else POOR


_SOS_CACHE: dict = {}


def _sos(kind: str, fs: float):
    key = (kind, round(fs, 3))
    if key not in _SOS_CACHE:
        if kind == "band":
            _SOS_CACHE[key] = signal.butter(4, [1.0, min(40.0, 0.45 * fs)], "bandpass", fs=fs, output="sos")
        else:
            _SOS_CACHE[key] = signal.butter(2, 1.0, "lowpass", fs=fs, output="sos")
    return _SOS_CACHE[key]


def assess(raw: np.ndarray, fs: float, line_hz: float | None = 60.0,
           full_scale_uv: float | None = None, ignore_hum: bool = False) -> QualityReport | None:
    """Score each channel from the most recent ``WINDOW_S`` seconds of unfiltered uV.

    Returns None until at least ``MIN_WINDOW_S`` of data is available.
    """
    n_win = int(WINDOW_S * fs)
    x = np.asarray(raw, dtype=np.float64)[:, -n_win:]
    if x.shape[1] < int(MIN_WINDOW_S * fs):
        return None
    xc = x - x.mean(axis=1, keepdims=True)
    band = signal.sosfiltfilt(_sos("band", fs), xc, axis=-1)
    slow = signal.sosfiltfilt(_sos("slow", fs), xc, axis=-1)
    std = band.std(axis=1)
    drift = np.ptp(slow, axis=1)

    line = np.full(x.shape[0], np.nan)
    line_ratio = np.full(x.shape[0], np.inf)
    if line_hz and line_hz < fs / 2 - 1:
        # Hum is a narrow peak. Measure how far it rises above nearby frequencies, so
        # broadband noise (which is high everywhere) isn't mistaken for hum.
        f, p = signal.welch(xc, fs=fs, nperseg=min(x.shape[1], int(2 * fs)), axis=-1)
        peak = p[:, (f >= line_hz - 1.5) & (f <= line_hz + 1.5)].max(axis=1)
        nb = ((f >= line_hz - 10) & (f <= line_hz - 3)) | ((f >= line_hz + 3) & (f <= line_hz + 10))
        floor = np.median(p[:, nb], axis=1) if nb.any() else np.zeros_like(peak)
        # With 4 s of data the max of a few random bins can sit ~3x above the median, so
        # only the part of the peak above 4x the floor counts. Real hum is far above
        # that (~300-500x on our recordings).
        line = np.sqrt(np.maximum(peak - 4.0 * floor, 0.0))
        line_ratio = peak / np.maximum(floor, 1e-12)

    channels = []
    for i in range(x.shape[0]):
        # Order matters on ties: the first, most specific explanation wins
        # (a channel pinned at the rail is also "flat", but "railed" says more).
        subs = {}
        if full_scale_uv:
            frac = float(np.abs(x[i]).max()) / full_scale_uv
            subs["railed (no contact)"] = _ramp(frac, *RAIL_FRACTION)
        # ignore_hum: the notch and the analyses' 1-40 Hz filters remove mains hum, so a channel that
        # is otherwise clean stays usable. Hum is still measured and reported in the metrics.
        if not ignore_hum:
            subs["60 Hz hum (check contact)"] = _ramp(line[i], *LINE_UV) if np.isfinite(line[i]) else 100.0
        subs["too noisy (loose or moving)"] = _ramp(std[i], *NOISE_UV)
        subs["drifting (electrode settling)"] = _ramp(drift[i], *DRIFT_UV)
        subs["flat (off or unplugged?)"] = 100.0 - _ramp(std[i], *FLAT_UV)
        score = min(subs.values())
        worst = [k for k, v in subs.items() if v == score]
        # On a tie, only blame hum if it's a genuine narrow peak (>= 20x the nearby floor).
        # Otherwise a very noisy channel would often read as "hum".
        if len(worst) > 1 and line_ratio[i] < 20:
            worst = [k for k in worst if not k.startswith("60 Hz")]
        reason = worst[0]
        st = _status(score)
        channels.append(ChannelQuality(
            channel=i + 1, score=int(round(score)), status=st, reason="" if st == GOOD else reason,
            metrics={"line_uv_rthz": float(line[i]), "std_uv": float(std[i]), "drift_uv": float(drift[i])},
        ))

    # Near-identical channels: RLD contamination from a floating electrode.
    alive = [i for i in range(x.shape[0]) if std[i] > FLAT_UV[1]]
    identical: list[int] = []
    warning = None
    if len(alive) >= 3:
        r = np.corrcoef(band[alive])
        # A channel counts as a "copy" only if it matches at least COPY_PARTNERS others. Neighbouring
        # electrodes legitimately correlate in pairs (F3/F4 share blinks, P3/O1 share alpha); the RLD
        # contamination this catches made 6 channels near-identical to each other.
        partners = (np.abs(r) >= IDENTICAL_R).sum(axis=1) - 1
        hits = {alive[a] for a in range(len(alive)) if partners[a] >= COPY_PARTNERS}
        if len(hits) >= 3:
            identical = sorted(h + 1 for h in hits)
            worst = min(channels, key=lambda c: c.score)
            for i in hits:
                c = channels[i]
                if c.score > 60:
                    c.score, c.status, c.reason = 60, FAIR, "copies other channels (shared noise)"
            names = ", ".join(f"Ch{c}" for c in identical)
            warning = (f"{names} are near-copies of each other. A loose electrode in the RLD loop is "
                       f"usually spreading its noise to the others. Fix the worst channel first "
                       f"(Ch{worst.channel}: {worst.reason or 'lowest score'}), or switch it off.")
    return QualityReport(channels, identical, warning)


class QualityTracker:
    """Smooths successive reports for display, so badges don't flicker.

    Scores are an exponential moving average. A channel's reason is the most recent
    problem seen while it's below "good". The copies warning stays up for
    ``warning_hold_s`` after it was last detected.
    """

    def __init__(self, alpha: float = 0.35, warning_hold_s: float = 5.0):
        self.alpha = alpha
        self.warning_hold_s = warning_hold_s
        self.reset()

    def reset(self) -> None:
        self._ema: np.ndarray | None = None
        self._reasons: list[str] = []
        self._warning: str | None = None
        self._identical: list[int] = []
        self._warning_t = -np.inf

    def update(self, report: QualityReport, now: float) -> QualityReport:
        scores = np.array([c.score for c in report.channels], dtype=float)
        if self._ema is None or self._ema.shape != scores.shape:
            self._ema = scores
            self._reasons = [c.reason for c in report.channels]
        else:
            self._ema = self.alpha * scores + (1 - self.alpha) * self._ema
        out = []
        for i, (c, s) in enumerate(zip(report.channels, self._ema)):
            if c.reason:
                self._reasons[i] = c.reason
            st = _status(s)
            out.append(ChannelQuality(c.channel, int(round(s)), st, "" if st == GOOD else self._reasons[i], c.metrics))
        if report.warning:
            self._warning, self._identical, self._warning_t = report.warning, report.identical, now
        held = (now - self._warning_t) <= self.warning_hold_s
        return QualityReport(out, self._identical if held else [], self._warning if held else None)


def snapshot(report: QualityReport | None) -> dict | None:
    """JSON-friendly form for a recording's sidecar."""
    if report is None:
        return None
    return {
        "summary": report.summary(),
        "warning": report.warning,
        "channels": [
            {"channel": c.channel, "score": c.score, "status": c.status, "reason": c.reason,
             **{k: round(v, 3) for k, v in c.metrics.items()}}
            for c in report.channels
        ],
    }
