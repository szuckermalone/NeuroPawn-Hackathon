from pathlib import Path

import numpy as np
import pytest

from neuropawn_gui.boards import KNIGHT_FULL_SCALE_UV
from neuropawn_gui.quality import GOOD, POOR, QualityTracker, assess, snapshot

FS = 125.0
REC = Path(__file__).resolve().parents[1] / "data/recordings"


def _eeg(seed=0, secs=4.0, amp=10.0):
    """Independent, EEG-like 1/f-ish noise on 8 channels with DC offsets."""
    rng = np.random.default_rng(seed)
    n = int(secs * FS)
    x = np.cumsum(rng.normal(0, 1, (8, n)), axis=1)
    x -= np.linspace(x[:, :1], x[:, -1:], n, axis=1).squeeze(-1)  # remove random-walk trend
    x = x / x.std(axis=1, keepdims=True) * amp * 0.5 + rng.normal(0, amp * 0.8, (8, n))
    return x + np.arange(8)[:, None] * 40.0


def test_clean_signal_is_all_good():
    q = assess(_eeg(), FS, 60, KNIGHT_FULL_SCALE_UV)
    assert [c.status for c in q.channels] == [GOOD] * 8, [(c.score, c.reason) for c in q.channels]
    assert q.warning is None and q.n_good == 8


@pytest.mark.parametrize("problem,reason_prefix", [
    ("hum", "60 Hz hum"), ("flat", "flat"), ("noisy", "too noisy"), ("railed", "railed"), ("drift", "drifting"),
])
def test_each_failure_is_caught_on_its_channel(problem, reason_prefix):
    x = _eeg(seed=1)
    t = np.arange(x.shape[1]) / FS
    if problem == "hum":
        x[7] += 25 * np.sin(2 * np.pi * 60 * t)
    elif problem == "flat":
        x[7] = 3.0 + np.random.default_rng(2).normal(0, 0.05, x.shape[1])
    elif problem == "noisy":
        x[7] += np.random.default_rng(3).normal(0, 150, x.shape[1])
    elif problem == "railed":
        x[7] = np.clip(x[7] * 400, -KNIGHT_FULL_SCALE_UV, KNIGHT_FULL_SCALE_UV)
    elif problem == "drift":
        x[7] += 2500 * t / t[-1]
    q = assess(x, FS, 60, KNIGHT_FULL_SCALE_UV)
    bad = q.channels[7]
    assert bad.status == POOR and bad.reason.startswith(reason_prefix), (bad.score, bad.reason)
    assert all(c.status == GOOD for c in q.channels[:7]), [(c.score, c.reason) for c in q.channels[:7]]


def test_copies_warning_names_the_worst_channel():
    x = _eeg(seed=4)
    rng = np.random.default_rng(5)
    shared = rng.normal(0, 20, x.shape[1])
    x[:6] = shared + rng.normal(0, 1, (6, x.shape[1]))  # ch1-6 are copies
    t = np.arange(x.shape[1]) / FS
    x[7] += 30 * np.sin(2 * np.pi * 60 * t)              # ch8 floating
    q = assess(x, FS, 60, KNIGHT_FULL_SCALE_UV)
    assert q.identical == [1, 2, 3, 4, 5, 6]
    assert "Ch8" in q.warning
    assert all(q.channels[i].status != GOOD for i in range(6))


def test_needs_two_seconds():
    assert assess(_eeg(secs=1.5), FS) is None


def test_tracker_smooths_and_holds_warning():
    x = _eeg(seed=6)
    good = assess(x, FS, 60, KNIGHT_FULL_SCALE_UV)
    x2 = x.copy()
    x2[0] += 40 * np.sin(2 * np.pi * 60 * np.arange(x.shape[1]) / FS)
    bad = assess(x2, FS, 60, KNIGHT_FULL_SCALE_UV)
    tr = QualityTracker(alpha=0.35, warning_hold_s=5)
    tr.update(good, 0.0)
    one_bad = tr.update(bad, 0.25)
    assert one_bad.channels[0].score > bad.channels[0].score  # one bad reading doesn't slam it to 0
    for k in range(10):
        r = tr.update(bad, 0.5 + k * 0.25)
    assert r.channels[0].status == POOR and r.channels[0].reason.startswith("60 Hz")
    snap = snapshot(r)
    assert snap["channels"][0]["status"] == POOR and "summary" in snap


def _windows(path):
    from neuropawn_gui.review import load_recording
    r = load_recording(path)
    out = []
    for end in range(int(4 * r.fs), r.eeg.shape[1], int(2 * r.fs)):
        out.append(assess(r.eeg[:, :end], r.fs, 60, KNIGHT_FULL_SCALE_UV))
    return out


@pytest.mark.skipif(not (REC / "neuropawn_knight_2026-11-07_17-09-29.csv").exists(), reason="recording absent")
def test_real_session_with_floating_ch8():
    w = _windows(REC / "neuropawn_knight_2026-11-07_17-09-29.csv")
    assert all(q.channels[7].status == POOR for q in w)               # floating O2 always flagged
    assert sum(bool(q.warning) for q in w) >= len(w) // 3             # contamination warning shows up
    assert all("Ch8" in q.warning for q in w if q.warning)


@pytest.mark.skipif(not (REC / "neuropawn_knight_2026-11-07_16-31-49.csv").exists(), reason="recording absent")
def test_real_session_mostly_good():
    w = _windows(REC / "neuropawn_knight_2026-11-07_16-31-49.csv")
    for ch in (0, 2, 3, 4, 5, 6):
        assert all(q.channels[ch].status == GOOD for q in w)
    assert all(q.channels[1].status != GOOD for q in w)  # ch2 had strong 60 Hz hum
    assert not any(q.warning for q in w)


def test_ignore_hum_keeps_otherwise_clean_channels_usable():
    x = _eeg(seed=21)
    t = np.arange(x.shape[1]) / FS
    x += 40 * np.sin(2 * np.pi * 60 * t)                 # strong common-mode hum on every channel
    x[7] = 3.0 + np.random.default_rng(22).normal(0, 0.05, x.shape[1])   # ch8 genuinely flat
    strict = assess(x, FS, 60, KNIGHT_FULL_SCALE_UV)
    assert all(c.status == POOR and c.reason.startswith("60 Hz") for c in strict.channels[:7])
    tolerant = assess(x, FS, 60, KNIGHT_FULL_SCALE_UV, ignore_hum=True)
    assert all(c.status == GOOD for c in tolerant.channels[:7]), [(c.score, c.reason) for c in tolerant.channels]
    assert tolerant.channels[7].status == POOR and tolerant.channels[7].reason.startswith("flat")
    assert tolerant.channels[0].metrics["line_uv_rthz"] > 8   # hum still measured and reported


def test_correlated_neighbour_pairs_are_not_copies():
    """F3/F4 share blinks and P3/O1 share alpha: two natural pairs, not RLD contamination."""
    x = _eeg(seed=31)
    rng = np.random.default_rng(32)
    shared_front, shared_back = rng.normal(0, 15, x.shape[1]), rng.normal(0, 10, x.shape[1])
    x[0] += shared_front * 3
    x[1] += shared_front * 3          # ch1-ch2 correlated
    x[4] += shared_back * 3
    x[6] += shared_back * 3           # ch5-ch7 correlated
    q = assess(x, FS, 60, KNIGHT_FULL_SCALE_UV, ignore_hum=True)
    assert q.warning is None and all(c.status == GOOD for c in q.channels), [(c.score, c.reason) for c in q.channels]
