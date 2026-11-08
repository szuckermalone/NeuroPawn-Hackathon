"""Exploratory reanalysis of the final Night Drive run: many EEG drowsiness metrics vs. driving behaviour.

Every metric listed in METRICS is reported (no cherry-picking). Per-minute values are medians of
4 s windows (2 s step); artifact windows are dropped. Correlations use Spearman's rho, with a
partial correlation controlling for time-on-task, because anything that drifts with time will
correlate with a behaviour that also drifts with time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal, stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "notebooks"))

from neuropawn_gui.review import load_recording  # noqa: E402
import make_drive_figure as F  # noqa: E402

SESSION = REPO / "data/game_sessions/night_drive_2026-11-07_20-26-32"
RECORDING = REPO / "data/recordings/neuropawn_knight_2026-11-07_20-38-03.csv"
FRONT, CENT, POST = [0, 1], [2, 3], [4, 5, 6]          # F3 F4 | C3 C4 | P3 P4 O1 (O2 excluded: poor)
ART_UV = 200.0

METRICS = {
    "θ/α (detector)": "theta(C3,C4) / alpha(P3,P4,O1)",
    "(θ+α)/β": "Eoh 2005; Jap 2009: largest fatigue increase",
    "α/β": "Jap 2009",
    "θ/β": "Jap 2009",
    "(θ+α)/(α+β)": "Jap 2009",
    "log α posterior": "alpha power, P3/P4/O1",
    "log θ central": "theta power, C3/C4",
    "log β central": "beta power, C3/C4 (falls with fatigue)",
    "relative α": "alpha / 1-30 Hz, posterior",
    "relative θ": "theta / 1-30 Hz, central",
    "α peak freq": "posterior alpha peak (slows with drowsiness)",
    "aperiodic slope": "1/f exponent 2-40 Hz (steeper = sleepier)",
    "blink rate": "frontal F3/F4 blinks per minute",
    "blink duration": "frontal blink half-width (longer = sleepier)",
    "slow eye mvmt": "frontal 0.2-1 Hz power (SEMs with drowsiness)",
}


def band(P, f, lo, hi):
    m = (f >= lo) & (f < hi)
    return np.trapezoid(P[..., m], f[m], axis=-1)


def window_metrics(seg, fs):
    f, P = signal.welch(seg, fs=fs, nperseg=int(2 * fs), axis=-1)
    th = band(P[CENT], f, 4, 8).mean()
    al = band(P[POST], f, 8, 13).mean()
    be = band(P[CENT], f, 13, 30).mean()
    tot_p = band(P[POST], f, 1, 30).mean()
    tot_c = band(P[CENT], f, 1, 30).mean()
    pp = P[POST].mean(0)
    m = (f >= 7) & (f <= 13)
    peak = f[m][np.argmax(pp[m])]
    fit = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    slope = -np.polyfit(np.log10(f[fit]), np.log10(P[CENT + POST].mean(0)[fit]), 1)[0]
    return {
        "θ/α (detector)": th / al, "(θ+α)/β": (th + al) / be, "α/β": al / be, "θ/β": th / be,
        "(θ+α)/(α+β)": (th + al) / (al + be), "log α posterior": np.log10(al), "log θ central": np.log10(th),
        "log β central": np.log10(be), "relative α": al / tot_p, "relative θ": th / tot_c,
        "α peak freq": peak, "aperiodic slope": slope,
    }


def blinks(raw_front, fs):
    """Blinks on the frontal channels: peaks of the 0.5-10 Hz signal, with half-prominence width."""
    x = signal.sosfiltfilt(signal.butter(2, [0.5, 10], "bandpass", fs=fs, output="sos"), raw_front.mean(0))
    mad = np.median(np.abs(x - np.median(x))) * 1.4826
    pk, props = signal.find_peaks(np.abs(x), height=max(60.0, 5 * mad), distance=int(0.3 * fs), prominence=40)
    w = signal.peak_widths(np.abs(x), pk, rel_height=0.5)[0] / fs
    return pk, w


def analyse():
    rec = load_recording(RECORDING)
    fs = rec.fs
    eeg = rec.eeg
    ts = rec.col("timestamp_unix")
    g = F.load_game(SESSION)
    t0 = g["drive_start_unix"]
    t = ts - t0
    x = eeg - eeg.mean(1, keepdims=True)
    sos = signal.butter(4, [1, 40], "bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x, axis=1)
    slow = signal.sosfiltfilt(signal.butter(2, [0.2, 1.0], "bandpass", fs=fs, output="sos"), x[FRONT].mean(0))

    # 4 s windows, 2 s step, over baseline + drive
    win, step = int(4 * fs), int(2 * fs)
    rows = []
    for s in range(0, y.shape[1] - win, step):
        tc = t[s + win // 2]
        if tc < -90 or tc > 1205:
            continue
        seg = y[:, s:s + win]
        if np.abs(seg[CENT + POST]).max() > ART_UV:
            continue
        r = window_metrics(seg, fs)
        r["slow eye mvmt"] = float(np.log10(np.mean(slow[s:s + win] ** 2)))
        r["t"] = tc
        rows.append(r)
    pk, bw = blinks(x[FRONT], fs)
    bt = t[pk]

    mins = np.arange(20)
    per_min = {}
    for k in METRICS:
        if k == "blink rate":
            per_min[k] = np.array([np.sum((bt >= m * 60) & (bt < (m + 1) * 60)) for m in mins], float)
        elif k == "blink duration":
            per_min[k] = np.array([np.median(bw[(bt >= m * 60) & (bt < (m + 1) * 60)]) if np.any((bt >= m * 60) & (bt < (m + 1) * 60)) else np.nan for m in mins])
        else:
            per_min[k] = np.array([np.nanmedian([r[k] for r in rows if m * 60 <= r["t"] < (m + 1) * 60] or [np.nan]) for m in mins])
    rt = g["rt"]
    rt_min = np.array([np.median(rt[(rt[:, 0] >= m * 60) & (rt[:, 0] < (m + 1) * 60), 1]) for m in mins])
    sdlp = np.array([g["x"][(g["t"] >= m * 60) & (g["t"] < (m + 1) * 60)].std() for m in mins])
    dep_min = np.array([np.sum((g["departure"] >= m * 60) & (g["departure"] < (m + 1) * 60)) for m in mins], float)

    def partial_spearman(a, b, c):
        ra, rb, rc = (stats.rankdata(v) for v in (a, b, c))
        res = lambda u: u - np.polyval(np.polyfit(rc, u, 1), rc)
        return stats.pearsonr(res(ra), res(rb))

    results = []
    for k, desc in METRICS.items():
        v = per_min[k]
        ok = np.isfinite(v) & np.isfinite(rt_min)
        r_time = stats.spearmanr(mins[ok], v[ok])
        r_rt = stats.spearmanr(v[ok], rt_min[ok])
        r_sd = stats.spearmanr(v[ok], sdlp[ok])
        pr = partial_spearman(v[ok], rt_min[ok], mins[ok].astype(float))
        results.append(dict(metric=k, desc=desc, rho_time=r_time.statistic, p_time=r_time.pvalue,
                            rho_rt=r_rt.statistic, p_rt=r_rt.pvalue, rho_sdlp=r_sd.statistic, p_sdlp=r_sd.pvalue,
                            rho_rt_partial=pr.statistic, p_rt_partial=pr.pvalue))

    # trial level: EEG in the 6 s before each drift onset vs that drift's reaction time
    onset_rt = rt
    trial = []
    for t_on, rtv in onset_rt:
        s_idx = np.searchsorted(t, t_on - 6.0)
        e_idx = np.searchsorted(t, t_on)
        seg = y[:, s_idx:e_idx]
        if seg.shape[1] < int(5 * fs) or np.abs(seg[CENT + POST]).max() > ART_UV:
            continue
        r = window_metrics(seg, fs)
        r["rt"] = rtv
        r["t"] = t_on
        trial.append(r)
    trial_res = {}
    for k in [k for k in METRICS if k not in ("blink rate", "blink duration", "slow eye mvmt")]:
        a = np.array([r[k] for r in trial])
        b = np.array([r["rt"] for r in trial])
        trial_res[k] = stats.spearmanr(a, b)

    # baseline (eyes open vs closed) for this run
    st = g["states"]
    eo = [r for r in rows if st["calibration_eyes_open"] - t0 <= r["t"] < st["calibration_eyes_closed"] - t0]
    ec = [r for r in rows if st["calibration_eyes_closed"] - t0 <= r["t"] < st["countdown"] - t0]
    base = {"eyes_open_theta_alpha": float(np.median([r["θ/α (detector)"] for r in eo])),
            "eyes_closed_theta_alpha": float(np.median([r["θ/α (detector)"] for r in ec])),
            "eyes_open_alpha": float(np.median([10 ** r["log α posterior"] for r in eo])),
            "eyes_closed_alpha": float(np.median([10 ** r["log α posterior"] for r in ec]))}

    return dict(rows=rows, per_min=per_min, mins=mins, rt_min=rt_min, sdlp=sdlp, dep_min=dep_min,
                results=results, trial=trial, trial_res=trial_res, blink_t=bt, blink_w=bw, base=base, game=g,
                n_windows_drive=sum(1 for r in rows if 0 <= r["t"] <= 1200))


if __name__ == "__main__":
    A = analyse()
    n = len(A["results"])
    print(f"{len(A['rows'])} clean windows ({A['n_windows_drive']} during the drive); Bonferroni alpha = 0.05/{n} = {0.05/n:.4f}")
    print(f"baseline: θ/α eyes open {A['base']['eyes_open_theta_alpha']:.2f} -> closed {A['base']['eyes_closed_theta_alpha']:.2f}; "
          f"alpha x{A['base']['eyes_closed_alpha']/A['base']['eyes_open_alpha']:.1f}")
    print(f"{'metric':18s} {'vs time':>14s} {'vs RT':>14s} {'vs SDLP':>14s} {'vs RT | time':>16s}")
    for r in sorted(A["results"], key=lambda r: r["p_rt"]):
        f = lambda rho, p: f"{rho:+.2f} (p{p:.3f})"
        print(f"{r['metric']:18s} {f(r['rho_time'], r['p_time']):>14s} {f(r['rho_rt'], r['p_rt']):>14s} {f(r['rho_sdlp'], r['p_sdlp']):>14s} {f(r['rho_rt_partial'], r['p_rt_partial']):>16s}")
    print(f"\ntrial level (6 s before each drift vs its RT, n={len(A['trial'])}):")
    for k, r in sorted(A["trial_res"].items(), key=lambda kv: kv[1].pvalue):
        print(f"  {k:18s} rho {r.statistic:+.2f}  p {r.pvalue:.3f}")
    print("\nper-minute blink rate:", A["per_min"]["blink rate"].astype(int).tolist())
    print("per-minute blink dur  :", np.round(A["per_min"]["blink duration"], 3).tolist())
