"""Slide figure for one Night Drive session: EEG drowsiness measures + driving performance over time.

Usage:
    PYTHONPATH=src python notebooks/make_drive_figure.py data/game_sessions/night_drive_<stamp> [--recording <eeg.csv>]

If no --recording is given, the EEG recording linked to the session (its JSON's
`night_drive_session`) is found automatically in data/recordings. Without one, only the
driving panels are drawn. The EEG measures are recomputed by replaying the raw recording
through the same detector the GUI runs live (it's deterministic): the theta/alpha ratio
against its 1.0 limit, theta vs. the personal baseline against 2x, and the 0-100 score.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from neuropawn_gui import drowsiness as D  # noqa: E402
from neuropawn_gui.boards import KNIGHT_FULL_SCALE_UV  # noqa: E402
from neuropawn_gui.link import MARKERS  # noqa: E402
from neuropawn_gui.quality import POOR, QualityTracker, assess  # noqa: E402
from neuropawn_gui.review import load_recording  # noqa: E402

BLUE, ORANGE, RED, AMBER, INK, MUTED = "#36579E", "#E0602D", "#C8382F", "#C98A00", "#1B1F23", "#6B7580"


def load_game(session: Path):
    ev = list(csv.DictReader(open(session / "events.csv")))
    tel = list(csv.DictReader(open(session / "telemetry.csv")))
    states = {r["detail"]: float(r["unix_time"]) for r in ev if r["event"] == "state"}
    drive_start_unix = states.get("driving")
    onset = {r["trial"]: float(r["drive_time_s"]) for r in ev if r["event"] == "drift_onset"}
    g = {
        "drive_start_unix": drive_start_unix,
        "states": states,
        "rt": np.array([(onset[r["trial"]], float(r["rt_s"])) for r in ev if r["event"] == "response" and r["trial"] in onset]),
        "lapse": np.array([float(r["drive_time_s"]) for r in ev if r["event"] == "lapse"]),
        "departure": np.array([float(r["drive_time_s"]) for r in ev if r["event"] == "lane_departure"]),
        "alerts": np.array([float(r["drive_time_s"]) for r in ev if r["event"] == "alert_received"]),
        "t": np.array([float(r["drive_time_s"]) for r in tel]),
        "x": np.array([float(r["x_m"]) for r in tel]),
    }
    return g


def find_recording(session: Path) -> Path | None:
    for js in sorted((REPO / "data" / "recordings").glob("*.json"), reverse=True):
        try:
            if json.loads(js.read_text()).get("board", {}).get("night_drive_session") == session.name:
                return js.with_suffix(".csv")
        except (OSError, json.JSONDecodeError):
            continue
    return None


def replay_eeg(rec_path: Path):
    """Run the recording through the detector like the GUI does (quality-gated channels,
    recalibration when the game's baseline starts). Returns per-window results."""
    rec = load_recording(rec_path)
    fs, eeg = rec.fs, rec.eeg
    det = ((rec.meta or {}).get("board", {}).get("drowsiness_detector") or {})
    ignore_hum = bool(det.get("quality_ignores_hum", False))  # replay exactly what the GUI did live
    excluded = [c - 1 for c in det.get("excluded_channels") or []]
    ts = rec.col("timestamp_unix")
    marker = rec.col("marker")
    cal_idx = np.nonzero(marker == MARKERS["calibration_eyes_open"])[0]
    mon = D.DrowsinessMonitor(fs)
    qt = QualityTracker()
    out = []
    step = int(fs / 2)
    for end in range(int(D.WINDOW_S * fs), eeg.shape[1] + 1, step):
        if cal_idx.size and end - step < cal_idx[0] <= end:
            mon.reset_calibration()
        q = assess(eeg[:, :end], fs, 60.0, KNIGHT_FULL_SCALE_UV, ignore_hum=ignore_hum)
        q = qt.update(q, end / fs) if q else None
        usable = None if q is None else np.array([c.status != POOR for c in q.channels])
        if usable is not None and excluded:
            usable[excluded] = False
        s = mon.update(eeg[:, max(0, end - int(D.WINDOW_S * fs)):end], end / fs, usable)
        if s is not None and mon.last_window is not None:
            out.append((ts[end - 1], mon.last_window, s.score, usable))
    if not mon.calibrated:
        return None
    rows = []
    for t_unix, w, score, usable in out:  # theta vs. the final baseline, for every window
        rows.append((t_unix, w.ratio, None if w.theta_abs is None else w.theta_abs / mon.baseline_theta, score, w.label))
    return {"rows": rows, "baseline": (mon.baseline_theta, mon.baseline_beta), "markers": (ts, marker)}


def rolling_median(t, y, width_s):
    out = np.full(len(y), np.nan)
    for i, ti in enumerate(t):
        m = (t > ti - width_s) & (t <= ti) & np.isfinite(y)
        if m.sum() >= 3:
            out[i] = np.median(y[m])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--recording", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    g = load_game(args.session)
    if g["drive_start_unix"] is None:
        sys.exit("This session never reached the driving phase.")
    rec_path = args.recording or find_recording(args.session)
    eeg = replay_eeg(rec_path) if rec_path else None

    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#9AA3AD"})
    n_panels = 5 if eeg else 2  # EEG: theta/alpha, theta vs baseline, score; driving: reaction time, SDLP
    fig, axes = plt.subplots(n_panels, 1, figsize=(13.33, 10.0 if eeg else 6.0), dpi=200, sharex=True,
                             gridspec_kw={"hspace": 0.28, "left": 0.07, "right": 0.98, "top": 0.88,
                                          "bottom": 0.07 if eeg else 0.11})
    to_min = 1 / 60.0
    end_min = g["t"].max() * to_min if g["t"].size else 1
    title = ("EEG drowsiness markers and driving performance during a monotonous drive" if eeg
             else "Driving performance declined over a monotonous night drive")
    fig.suptitle(title, x=0.07, ha="left", fontsize=19, fontweight="bold", color=INK)
    sub = f"Night Drive session {args.session.name.replace('night_drive_', '')}"
    fig.text(0.07, 0.905 if eeg else 0.895, sub + ("" if eeg else " · EEG for this drive wasn't recorded (driving data only)"),
             fontsize=11, color=MUTED)

    k = 0
    if eeg:
        t0 = g["drive_start_unix"]
        R = np.array([(r[0], r[1] if r[1] is not None else np.nan, r[2] if r[2] is not None else np.nan,
                       r[3] if r[3] is not None else np.nan) for r in eeg["rows"]])
        tm = (R[:, 0] - t0) * to_min
        st = g["states"]
        shade = [(st.get("calibration_eyes_open"), st.get("calibration_eyes_closed"), "#E8EEF6", "eyes open"),
                 (st.get("calibration_eyes_closed"), st.get("countdown"), "#EFE3F5", "eyes closed")]
        for ax_i, (col, lim, lab, logy) in enumerate([(1, 1.0, "θ/α ratio", True), (2, 2.0, "θ vs. baseline (×)", False)]):
            ax = axes[ax_i]
            y = R[:, col]
            ax.plot(tm, y, color="#B9C2CC", lw=0.8)
            ax.plot(tm, rolling_median(R[:, 0], y, 60), color=ORANGE if col == 2 else BLUE, lw=2.4, label="1-min median")
            ax.axhline(lim, color=RED, ls="--", lw=1.2)
            ax.text(tm.min(), lim * (1.08 if logy else 1.04), f"pre-sleep limit {lim:g}", color=RED, fontsize=10)
            for a, b, c, lbl in shade:
                if a and b:
                    ax.axvspan((a - t0) * to_min, (b - t0) * to_min, color=c, lw=0)
            if logy:
                ax.set_yscale("log")
            ax.set_ylabel(lab)
        axes[0].text(((st.get("calibration_eyes_closed", t0) + st.get("countdown", t0)) / 2 - t0) * to_min,
                     axes[0].get_ylim()[0] * 1.3, "eyes closed\n(awake)", ha="center", fontsize=9, color="#7C4B8D")
        ax = axes[2]
        ax.fill_between(tm, np.nan_to_num(R[:, 3]), color=RED, alpha=0.25, lw=0)
        ax.plot(tm, R[:, 3], color=RED, lw=1.6)
        ax.axhline(50, color=RED, ls="--", lw=1)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Drowsiness score")
        for a in g["alerts"]:
            ax.axvline(a * to_min, color=RED, lw=2)
        k = 3

    ax = axes[k]
    if g["rt"].size:
        ax.scatter(g["rt"][:, 0] * to_min, g["rt"][:, 1], s=14, color="#9DAFC6", label="each drift")
        ax.plot(g["rt"][:, 0] * to_min, rolling_median(g["rt"][:, 0], g["rt"][:, 1], 120), color=BLUE, lw=2.4,
                label="2-min median")
    for i, lt in enumerate(g["lapse"]):
        ax.axvline(lt * to_min, color=RED, lw=1.5, ls=":", label="lapse (no correction in 4 s)" if i == 0 else None)
    ax.set_ylabel("Reaction time (s)")
    ax.set_ylim(0, min(6, max(2.5, np.nanpercentile(g["rt"][:, 1], 98) * 1.15)) if g["rt"].size else 3)
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=10)

    ax = axes[k + 1]
    edges = np.arange(0, g["t"].max() + 60, 60)
    sd = [g["x"][(g["t"] >= a) & (g["t"] < a + 60)].std() if ((g["t"] >= a) & (g["t"] < a + 60)).sum() > 100 else np.nan
          for a in edges[:-1]]
    ax.bar((edges[:-1] + 30) * to_min, sd, width=0.8, color="#9DAFC6", label="weaving per minute (SDLP)")
    if g["departure"].size:
        ax.scatter(g["departure"] * to_min, np.full(g["departure"].size, np.nanmax(sd) * 1.08), marker="v", s=40,
                   color=AMBER, label="lane departure", zorder=3)
    ax.set_ylabel("SDLP (m)")
    ax.set_xlabel("Minutes into the drive")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=10)
    for a in axes:
        a.set_xlim(min(-1.2 if eeg else 0, a.get_xlim()[0]), end_min + 0.2)

    out = args.out or (REPO / "presentation" / "figures" / f"{args.session.name}{'' if eeg else '_driving'}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print("saved", out, "| EEG:", rec_path if eeg else "none")


if __name__ == "__main__":
    main()
