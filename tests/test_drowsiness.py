from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from neuropawn_gui import drowsiness as D

FS = 125.0
REC = Path(__file__).resolve().parents[1] / "data/recordings"


def _background(n, seed):
    """Independent pink-ish EEG background, ~10 uV, on 8 channels."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, (8, n))
    b, a = signal.butter(1, 3.0, "lowpass", fs=FS)
    return 6 * signal.lfilter(b, a, w, axis=1) * 3 + rng.normal(0, 3, (8, n))


def _rhythm(n, freq, amp, seed):
    """Narrow-band rhythm with wandering phase/amplitude (not a pure sinusoid)."""
    rng = np.random.default_rng(seed)
    x = signal.sosfiltfilt(signal.butter(4, [freq - 1.5, freq + 1.5], "bandpass", fs=FS, output="sos"),
                           rng.normal(0, 1, n))
    return amp * x / x.std()


def session(minutes_awake_closed=1.5, pre_sleep_after=None, seconds=180, seed=0, alpha_only_drop=False,
            attention=False, blinks_every=None):
    """Eyes-closed relaxed wakefulness (posterior alpha), optionally transitioning to pre-sleep."""
    n = int(seconds * FS)
    t = np.arange(n) / FS
    x = _background(n, seed)
    alpha = _rhythm(n, 10.5, 18.0, seed + 1)
    theta = _rhythm(n, 6.0, 5.0, seed + 2)
    beta = _rhythm(n, 20.0, 3.0, seed + 3)
    ramp = np.zeros(n) if pre_sleep_after is None else np.clip((t - pre_sleep_after) / 20.0, 0, 1)
    if alpha_only_drop:
        a_gain, th_gain, b_gain = 1 - 0.9 * ramp, np.ones(n), np.ones(n)
    elif attention:
        a_gain, th_gain, b_gain = 1 - 0.9 * ramp, 1 + 1.5 * ramp, 1 + 1.0 * ramp
    else:
        a_gain, th_gain, b_gain = 1 - 0.85 * ramp, 1 + 1.6 * ramp, 1 - 0.3 * ramp
    for ch in (4, 5, 6, 7):
        x[ch] += a_gain * alpha
    for ch in range(8):
        x[ch] += th_gain * theta * (1.0 if ch in (2, 3) else 0.7) + b_gain * beta
    if blinks_every:
        for tb in np.arange(2.0, seconds, blinks_every):
            x[:2] += 200 * np.exp(-0.5 * ((t - tb) / 0.08) ** 2)
    return x + np.arange(8)[:, None] * 30.0


def run(x, monitor, usable=None, step=0.5):
    """Feed ``x`` like the GUI does (4 Hz). Time continues from the monitor's last update,
    so calibration and the test signal form one continuous session."""
    t0 = monitor._last_t + D.STEP_S if np.isfinite(monitor._last_t) else 0.0
    out = []
    for end in range(int(4 * FS), x.shape[1] + 1, int(step * FS)):
        s = monitor.update(x[:, :end], t0 + end / FS, usable)
        if s is not None:
            out.append((end / FS, s))  # time relative to the start of x
    return out


def calibrated(seed=0):
    m = D.DrowsinessMonitor(FS)
    run(session(seconds=70, seed=seed), m)
    assert m.calibrated
    return m


def scores(out):
    return [s.score for _, s in out if s.score is not None]


def test_calibration_needs_time_and_clean_windows():
    m = D.DrowsinessMonitor(FS)
    out = run(session(seconds=40), m)
    assert not m.calibrated and all(s.phase == "calibrating" for _, s in out)
    run(session(seconds=70), m)
    assert m.calibrated


def test_relaxed_eyes_closed_never_alerts():
    m = calibrated()
    out = run(session(seconds=180, seed=5), m)
    assert not any(s.alert for _, s in out) and max(scores(out)) < D.REARM_SCORE


def test_pre_sleep_transition_alerts_once_after_onset():
    m = calibrated()
    out = run(session(seconds=180, seed=6, pre_sleep_after=60), m)
    alerts = [t for t, s in out if s.alert]
    assert len(alerts) == 1, alerts
    assert 60 < alerts[0] < 60 + 20 + 30, alerts  # within the 30 s epoch after the ramp completes
    assert max(scores(out)) >= 80


@pytest.mark.parametrize("kind", ["alpha_only_drop", "attention"])
def test_normal_attenuation_and_attention_do_not_alert(kind):
    m = calibrated()
    out = run(session(seconds=180, seed=7, pre_sleep_after=60, **{kind: True}), m)
    assert not any(s.alert for _, s in out), kind
    assert max(scores(out)) < D.ALERT_SCORE


def test_blinking_means_awake():
    m = calibrated()
    out = run(session(seconds=120, seed=8, pre_sleep_after=30, blinks_every=3.0), m)
    assert not any(s.alert for _, s in out)
    assert m.last_window.label == D.BLINK


def test_poor_channels_are_excluded():
    m = calibrated()
    usable = np.ones(8, dtype=bool)
    usable[[2, 3]] = False  # no central channels -> cannot judge theta
    out = run(session(seconds=60, seed=9), m, usable=usable)
    assert all(s.score is None for _, s in out)
    assert m.last_window.label == D.INVALID


def test_cooldown_and_rearm():
    m = calibrated()
    a = session(seconds=150, seed=10, pre_sleep_after=10)
    b = session(seconds=90, seed=11)  # back to relaxed wakefulness
    c = session(seconds=150, seed=12, pre_sleep_after=10)
    x = np.concatenate([a, b, c], axis=1)
    out = run(x, m)
    alerts = [t for t, s in out if s.alert]
    assert len(alerts) == 2, alerts  # one per episode, not one per update
    assert alerts[1] - alerts[0] > D.COOLDOWN_S


def _real(name):
    from neuropawn_gui.review import load_recording
    return load_recording(REC / name)


@pytest.mark.skipif(not (REC / "neuropawn_knight_2026-11-07_17-29-49.csv").exists(), reason="recording absent")
def test_real_awake_sessions_never_alert(monkeypatch):
    monkeypatch.setattr(D, "CALIBRATION_S", 40.0)
    usable = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)  # O1/O2 flagged poor by the signal check
    for name in ("neuropawn_knight_2026-11-07_17-29-49.csv", "neuropawn_knight_2026-11-07_16-31-49.csv"):
        r = _real(name)
        cal = D.DrowsinessMonitor(r.fs)
        run(r.eeg, cal, usable)
        m = D.DrowsinessMonitor(r.fs)
        m.set_baseline(cal.baseline_theta, cal.baseline_beta)
        out = run(r.eeg, m, usable)
        assert not any(s.alert for _, s in out), name
        assert max(scores(out), default=0) == 0, name


@pytest.mark.skipif(not (REC / "neuropawn_knight_2026-11-07_17-29-49.csv").exists(), reason="recording absent")
def test_real_eyes_open_ratio_is_above_one_but_not_drowsy(monkeypatch):
    """The reason a bare theta/alpha threshold fails: eyes-open wakefulness already has ratio > 1."""
    r = _real("neuropawn_knight_2026-11-07_17-29-49.csv")
    m = D.DrowsinessMonitor(r.fs)
    seg = r.eeg[:, int(10 * r.fs):int(14 * r.fs)]
    w = m.classify(seg, 14.0, np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool))
    assert w.ratio > 1.0
    closed = r.eeg[:, int(40 * r.fs):int(44 * r.fs)]
    assert m.classify(closed, 44.0, np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)).ratio < 0.5


# ------------------------------------------------------------------ driver mode: microsleep rule
def _scripted(labels, microsleep_windows=D.MICROSLEEP_WINDOWS):
    """Run the alert logic on a scripted sequence of window labels (one per 2 s step)."""
    m = D.DrowsinessMonitor(FS, microsleep_windows=microsleep_windows)
    m.set_baseline(10.0, 10.0)
    seq = iter(labels)
    m.classify = lambda raw, t, usable=None: D.WindowResult(t, next(seq), 1.5, 2.5, 1.0)
    raw = np.zeros((8, int(D.WINDOW_S * FS)))
    return [m.update(raw, 100.0 + D.STEP_S * k) for k in range(len(labels))]


def test_microsleep_needs_three_consecutive_windows():
    A, S = D.AWAKE, D.DROWSY
    out = _scripted([A] * 10 + [S, S] + [A] * 5)      # ~6 s blip
    assert not any(s.alert for s in out)
    out = _scripted([A] * 10 + [S, S, S] + [A] * 5)   # ~8 s of pattern
    alerts = [k for k, s in enumerate(out) if s.alert]
    assert alerts == [12] and out[12].alert_reason == "microsleep" and out[12].label == "MICROSLEEP pattern"


def test_microsleep_rule_can_be_disabled():
    A, S = D.AWAKE, D.DROWSY
    out = _scripted([A] * 10 + [S] * 4 + [A] * 5, microsleep_windows=0)
    assert not any(s.alert for s in out)              # 4 of 15 windows < 50 %: no sustained alert either


def test_microsleep_alerts_sooner_than_sustained_rule():
    def first_alert(micro):
        m = calibrated()
        m.microsleep_windows = micro
        out = run(session(seconds=150, seed=13, pre_sleep_after=40), m)
        return next((t, s.alert_reason) for t, s in out if s.alert)
    t_micro, why_micro = first_alert(D.MICROSLEEP_WINDOWS)
    t_sust, why_sust = first_alert(0)
    assert why_micro == "microsleep" and why_sust == "sustained"
    assert t_micro < t_sust - 5
