"""Pre-sleep (drowsiness) detector built on the theta/alpha ratio.

What "pre-sleep" means here
---------------------------
Sleep-onset scoring (AASM manual; Hori et al., 1994, sleep-onset stages) marks the
wake -> N1 transition, in people who produce alpha, as the posterior **alpha rhythm being
replaced by low-amplitude 4-7 Hz (theta) activity for more than half of a 30 s
epoch**. Sleepiness also correlates with less alpha and more central/frontal theta in
resting EEG (Strijkstra et al., 2003, Neurosci Lett 340:17-20), and with lower beta
in fatigue studies (Lal & Craig, 2002; Jap et al., 2009).

So theta must *replace* alpha. A low-alpha ratio alone isn't enough
-----------------------------------------------------------------------
On our own Knight recordings the theta/alpha power ratio is 1.3-2.1 in alert,
eyes-OPEN wakefulness (vision suppresses alpha) and 0.08-0.23 in relaxed eyes-closed
wakefulness. A ratio threshold alone would fire whenever someone opens their eyes.
A 4 s window therefore counts as pre-sleep only if **all** of these hold:

  1. theta/alpha >= RATIO_THRESHOLD (1.0)   alpha no longer dominates
  2. theta >= THETA_RISE x personal baseline   theta has actually increased
  3. beta  <= BETA_MAX  x personal baseline    not the theta of focused attention
                                               (frontal-midline theta comes with
                                               maintained beta; drowsiness lowers beta)
  4. no blink on F3/F4 in the window           blinking = eyes open and awake
  5. clean signal on the channels used         no artifacts, quality check not poor

Theta comes from C3/C4 (central). Frontal F3/F4 is excluded because blinks and eye
movements put large slow waves there. Alpha comes from P3/P4/O1/O2 (where alpha
lives; in our eyes-closed session parietal/occipital alpha rose 10-11x, frontal 1.6x).

Score and alert
---------------
score = % of valid windows in the last 30 s that meet the pre-sleep pattern (0-100).
Following the ">50 % of a 30 s epoch" rule, an alert fires at score >= ALERT_SCORE (50)
and re-arms once the score falls below REARM_SCORE and the cooldown has passed.

This is a research prototype, not a validated sleepiness detector or a safety device.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

THETA, ALPHA, BETA = (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)
FRONTAL = (0, 1)           # F3, F4  (blink detection only)
THETA_CH = (2, 3)          # C3, C4
ALPHA_CH = (4, 5, 6, 7)    # P3, P4, O1, O2

WINDOW_S = 4.0
STEP_S = 2.0
EPOCH_S = 30.0
CALIBRATION_S = 60.0
MIN_CAL_WINDOWS = 10
MIN_VALID_FRACTION = 0.5

RATIO_THRESHOLD = 1.0
THETA_RISE = 2.0
BETA_MAX = 1.5
BLINK_UV = 100.0
ARTIFACT_UV = 200.0      # strong eyes-closed alpha reached 158 uV on C3 in our data
ALERT_SCORE = 50.0
REARM_SCORE = 25.0
COOLDOWN_S = 60.0
# Driver mode: a microsleep (3-15 s of sleep-like EEG with the eyes closing; Boyle et al. 2008;
# Poudel et al. 2014) is dangerous long before a 30 s epoch is half full. Three consecutive
# pre-sleep windows (4 s windows every 2 s, so ~8 s of continuous pattern) also trigger an alert.
# 0 disables the rule.
MICROSLEEP_WINDOWS = 3

DROWSY, AWAKE, BLINK, INVALID = "pre-sleep", "awake", "blinking", "invalid"


@dataclass
class WindowResult:
    t: float
    label: str
    ratio: float | None = None          # theta / alpha
    theta_rel: float | None = None      # theta / baseline theta
    beta_rel: float | None = None
    why: str = ""
    theta_abs: float | None = None      # uV^2, used for calibration
    beta_abs: float | None = None


@dataclass
class DrowsinessState:
    phase: str                     # "calibrating" | "monitoring"
    score: float | None            # 0-100, None when not enough valid data
    label: str                     # short human-readable state
    ratio: float | None
    theta_rel: float | None
    beta_rel: float | None
    alert: bool                    # True only on the update where an alert fires
    calibration_progress: float    # 0..1
    history: list = field(default_factory=list)
    alert_reason: str = ""         # "sustained" (score >= 50) or "microsleep" (consecutive windows)
    consecutive: int = 0           # current run of consecutive pre-sleep windows


def band_powers(seg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel theta, alpha, beta power (uV^2) of a (channels, samples) window."""
    f, p = signal.welch(seg, fs=fs, nperseg=min(seg.shape[1], int(2 * fs)), axis=-1, detrend="linear")
    df = f[1] - f[0]

    def bp(lo, hi):
        return p[:, (f >= lo) & (f < hi)].sum(axis=1) * df
    return bp(*THETA), bp(*ALPHA), bp(*BETA)


class DrowsinessMonitor:
    def __init__(self, fs: float, ratio_threshold: float = RATIO_THRESHOLD, theta_rise: float = THETA_RISE,
                 beta_max: float = BETA_MAX, alert_score: float = ALERT_SCORE,
                 microsleep_windows: int = MICROSLEEP_WINDOWS):
        self.fs = float(fs)
        self.microsleep_windows = microsleep_windows
        self.ratio_threshold = ratio_threshold
        self.theta_rise = theta_rise
        self.beta_max = beta_max
        self.alert_score = alert_score
        self._sos = signal.butter(4, [1.0, 40.0], "bandpass", fs=self.fs, output="sos")
        n_epoch = int(round(EPOCH_S / STEP_S))
        self._recent: collections.deque[WindowResult] = collections.deque(maxlen=n_epoch)
        self.history: collections.deque[tuple[float, float | None, float | None]] = collections.deque(maxlen=600)
        self.last_window: WindowResult | None = None
        self.reset_calibration()
        self._armed = True
        self._last_alert_t = -np.inf
        self._last_t = -np.inf

    # -------------------------------------------------------------- calibration
    def reset_calibration(self) -> None:
        self._cal: list[tuple[float, float]] = []
        self._cal_start: float | None = None
        self.baseline_theta: float | None = None
        self.baseline_beta: float | None = None
        self._recent.clear()

    @property
    def calibrated(self) -> bool:
        return self.baseline_theta is not None

    def set_baseline(self, theta: float, beta: float) -> None:
        self.baseline_theta, self.baseline_beta = float(theta), float(beta)

    # -------------------------------------------------------------- analysis
    def classify(self, raw: np.ndarray, t: float, usable: np.ndarray | None = None) -> WindowResult:
        """Label one window of unfiltered uV (8, n). ``usable``: bool per channel (quality OK)."""
        usable = np.ones(raw.shape[0], dtype=bool) if usable is None else np.asarray(usable, dtype=bool)
        th_ch = [c for c in THETA_CH if usable[c]]
        al_ch = [c for c in ALPHA_CH if usable[c]]
        if not th_ch or not al_ch:
            return WindowResult(t, INVALID, why="no usable central/posterior channels")
        x = raw - raw.mean(axis=1, keepdims=True)
        y = signal.sosfiltfilt(self._sos, x, axis=-1)
        used = th_ch + al_ch
        if np.abs(y[used]).max() > ARTIFACT_UV:
            return WindowResult(t, INVALID, why="artifact")
        theta, alpha, beta = band_powers(y, self.fs)
        th, al, be = theta[th_ch].mean(), alpha[al_ch].mean(), beta[th_ch].mean()
        ratio = th / max(al, 1e-12)
        fr = [c for c in FRONTAL if usable[c]]
        blink = bool(fr) and np.abs(y[fr]).max() > BLINK_UV
        theta_rel = th / self.baseline_theta if self.baseline_theta else None
        beta_rel = be / self.baseline_beta if self.baseline_beta else None
        res = WindowResult(t, AWAKE, ratio, theta_rel, beta_rel, theta_abs=th, beta_abs=be)
        if blink:
            res.label, res.why = BLINK, "blinking (eyes open)"
        elif theta_rel is not None and ratio >= self.ratio_threshold and theta_rel >= self.theta_rise \
                and (beta_rel is None or beta_rel <= self.beta_max):
            res.label = DROWSY
        else:
            reasons = []
            if ratio < self.ratio_threshold:
                reasons.append("alpha dominant")
            if theta_rel is not None and theta_rel < self.theta_rise:
                reasons.append("theta at baseline")
            if beta_rel is not None and beta_rel > self.beta_max:
                reasons.append("beta elevated (attention)")
            res.why = ", ".join(reasons)
        return res

    def update(self, raw: np.ndarray, t: float, usable: np.ndarray | None = None) -> DrowsinessState | None:
        """Feed the most recent >= WINDOW_S of raw uV. Evaluates at most every STEP_S seconds."""
        n = int(WINDOW_S * self.fs)
        if raw.shape[1] < n or t - self._last_t < STEP_S - 1e-6:
            return None
        self._last_t = t
        w = self.classify(raw[:, -n:], t, usable)
        self.last_window = w

        if not self.calibrated:
            # Baseline = median theta/beta of clean awake windows (blinking windows included:
            # theta comes from central channels, and eyes-open blinking is a normal awake state).
            if self._cal_start is None:
                self._cal_start = t
            if w.label in (AWAKE, BLINK) and w.theta_abs is not None:
                self._cal.append((w.theta_abs, w.beta_abs))
            elapsed = t - self._cal_start
            if elapsed >= CALIBRATION_S and len(self._cal) >= MIN_CAL_WINDOWS:
                arr = np.array(self._cal)
                self.set_baseline(np.median(arr[:, 0]), np.median(arr[:, 1]))
            else:
                progress = min(0.99, elapsed / CALIBRATION_S)
                label = "Calibrating baseline" if w.label != INVALID else "Calibrating (waiting for clean signal)"
                return DrowsinessState("calibrating", None, label, w.ratio, None, None, False, progress,
                                       list(self.history))

        self._recent.append(w)
        valid = [r for r in self._recent if r.label != INVALID]
        score = None
        if len(valid) >= MIN_VALID_FRACTION * self._recent.maxlen:
            score = 100.0 * sum(r.label == DROWSY for r in valid) / len(valid)
        self.history.append((t, score, w.ratio))

        consecutive = 0
        for r in reversed(self._recent):
            if r.label != DROWSY:
                break
            consecutive += 1
        sustained = score is not None and score >= self.alert_score
        micro = bool(self.microsleep_windows) and consecutive >= self.microsleep_windows

        alert, reason = False, ""
        if self._armed and (sustained or micro) and t - self._last_alert_t >= COOLDOWN_S:
            alert, self._armed, self._last_alert_t = True, False, t
            reason = "sustained" if sustained else "microsleep"
        elif not self._armed and (score is None or score < REARM_SCORE) and consecutive == 0:
            self._armed = True

        if micro and not sustained:
            label = "MICROSLEEP pattern"
        elif score is None:
            label = "Signal unavailable" if w.label == INVALID else "Collecting 30 s…"
        elif score >= self.alert_score:
            label = "PRE-SLEEP pattern"
        elif score >= REARM_SCORE:
            label = "Drowsy signs"
        elif w.label == BLINK:
            label = "Awake (eyes open)"
        elif w.ratio is not None and w.ratio < 0.5:
            label = "Awake, relaxed (alpha)"
        else:
            label = "Awake"
        return DrowsinessState("monitoring", score, label, w.ratio, w.theta_rel, w.beta_rel, alert, 1.0,
                               list(self.history), reason, consecutive)
