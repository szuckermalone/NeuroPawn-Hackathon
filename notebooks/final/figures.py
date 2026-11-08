"""All presentation figures, generated from the real recordings. Run:  PYTHONPATH=src python notebooks/final/figures.py"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy import signal, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "notebooks"))

import metrics as M  # noqa: E402
from neuropawn_gui import drowsiness as D  # noqa: E402
from neuropawn_gui.boards import KNIGHT_FULL_SCALE_UV  # noqa: E402
from neuropawn_gui.dsp import apparent_line_hz  # noqa: E402
from neuropawn_gui.quality import POOR, QualityTracker, assess  # noqa: E402
from neuropawn_gui.review import load_recording  # noqa: E402

OUT = REPO / "presentation" / "figures" / "final"
OUT.mkdir(parents=True, exist_ok=True)
BG, INK, MUTED, GRID = "#F6F5F1", "#14213D", "#5B6577", "#D9DCE1"
BLUE, ORANGE, RED, AMBER, GREY = "#2F5DA8", "#E0602D", "#C8382F", "#C98A00", "#9AA5B4"
plt.rcParams.update({
    "font.family": "Helvetica Neue", "font.size": 15, "axes.facecolor": BG, "figure.facecolor": BG,
    "savefig.facecolor": BG, "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", name)


A = M.analyse()
g = A["game"]
rec_final = load_recording(M.RECORDING)
fs = rec_final.fs
t0 = g["drive_start_unix"]

# ------------------------------------------------------------------ 1. stream integrity
raw_bin = Path("/private/tmp/claude-501/-Users-sebastian-Documents-Projects-NeuroPawn-Hackathon/"
               "9e18bac0-514e-4fab-ad03-14f683e50ee6/scratchpad/raw_serial.bin")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
pkg = rec_final.col("package_num")[:600]
ax[0].plot(np.arange(pkg.size) / fs, pkg, color=BLUE, lw=1.6)
ax[0].set(title="Healthy: counter climbs 0–255, 125.6 packets/s", xlabel="Seconds", ylabel="Packet counter", ylim=(-10, 270))
if raw_bin.exists():
    b = raw_bin.read_bytes()
    fr, i = [], 0
    while i + 21 <= len(b) and len(fr) < 600:
        if b[i] == 0xA0 and b[i + 20] == 0xC0:
            fr.append(b[i + 1])
            i += 21
        else:
            i += 1
    fr = fr[:40]
    ax[1].step(np.arange(len(fr)) / 3024.5 * 1000, fr, where="post", color=RED, lw=1.8)
    ax[1].plot(np.arange(len(fr)) / 3024.5 * 1000, fr, "o", color=RED, ms=4)
ax[1].set(title="Stuck board: counter flips 235 / 236, 3,024 packets/s, all channels 0",
          xlabel="Milliseconds (first 40 packets)", ylim=(233.5, 237.5), yticks=[234, 235, 236, 237])
fig.tight_layout()
save(fig, "integrity_counter.png")

# ------------------------------------------------------------------ 2. hum: tuned notch on the final run
from neuropawn_gui.dsp import DISPLAY_FILTERS, apply_display_filter  # noqa: E402
dc = next(f for f in DISPLAY_FILTERS if f.kind == "dc")
x = rec_final.eeg[2:7] - rec_final.eeg[2:7].mean(1, keepdims=True)
seg = slice(int(120 * fs), int(420 * fs))
f0 = apparent_line_hz(60, fs, rec_final.fs_true)


def hum_rms(y):
    s = signal.sosfiltfilt(signal.butter(2, [f0 - .5, f0 + .5], "bandpass", fs=fs, output="sos"), y, axis=-1)
    return np.sqrt((s[:, 400:-400] ** 2).mean(-1))


before = hum_rms(x[:, seg])
fixed_off = 20 * np.log10(before / hum_rms(apply_display_filter(x[:, seg], fs, dc, 60.0)))
tuned_off = 20 * np.log10(before / hum_rms(apply_display_filter(x[:, seg], fs, dc, f0)))
b_n, a_n = signal.iirnotch(60.0, 30, fs=fs)
b_t, a_t = signal.iirnotch(f0, 30, fs=fs)
sos_f, sos_t = signal.tf2sos(b_n, a_n), signal.tf2sos(b_t, a_t)
fixed_live = 20 * np.log10(before / hum_rms(signal.sosfilt(sos_f, x[:, seg], axis=-1)))
tuned_live = 20 * np.log10(before / hum_rms(signal.sosfilt(sos_t, x[:, seg], axis=-1)))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
ff, P = signal.welch(x[:, seg], fs=fs, nperseg=int(8 * fs), axis=-1)
ax[0].semilogy(ff, np.sqrt(P.mean(0)), color=INK, lw=1.4)
ax[0].axvline(60, color=GREY, ls="--", lw=1.2)
ax[0].axvline(f0, color=ORANGE, lw=2)
ax[0].text(60.2, ax[0].get_ylim()[1] * 0.3, "where a 60 Hz\nnotch aims", color=MUTED, fontsize=12)
ax[0].text(f0 - 0.25, ax[0].get_ylim()[1] * 0.3, f"where the hum is\n{f0:.2f} Hz", color=ORANGE, fontsize=12, ha="right")
ax[0].set(xlim=(56, 62), title="Board clock is 125.66 Hz, not 125: hum lands at 59.7 Hz", xlabel="Frequency (Hz)", ylabel="µV/√Hz (C3–O1)")
lab = ["Offline\n(zero-phase)", "Live\n(causal)"]
xs = np.arange(2)
ax[1].bar(xs - 0.18, [np.median(fixed_off), np.median(fixed_live)], 0.34, color=GREY, label="notch at 60.00 Hz")
ax[1].bar(xs + 0.18, [np.median(tuned_off), np.median(tuned_live)], 0.34, color=ORANGE, label=f"tuned to {f0:.2f} Hz")
for xi, v in zip(list(xs - 0.18) + list(xs + 0.18), [np.median(fixed_off), np.median(fixed_live), np.median(tuned_off), np.median(tuned_live)]):
    ax[1].text(xi, v + 1, f"{v:.0f} dB", ha="center", fontsize=13, fontweight="bold", color=INK)
ax[1].set(xticks=xs, xticklabels=lab, ylabel="Hum removed (dB)", title="Final run, Ch3–Ch7")
ax[1].legend(loc="upper right", fontsize=12)
fig.tight_layout()
save(fig, "hum_notch.png")
hum_stats = dict(fixed_off=np.median(fixed_off), tuned_off=np.median(tuned_off), fixed_live=np.median(fixed_live),
                 tuned_live=np.median(tuned_live), f0=f0, hum_uv=float(np.median(before)))

# ------------------------------------------------------------------ 3. alpha reactivity, final-run baseline
y = signal.sosfiltfilt(signal.butter(4, [1, 40], "bandpass", fs=fs, output="sos"),
                       rec_final.eeg - rec_final.eeg.mean(1, keepdims=True), axis=1)
tt = rec_final.col("timestamp_unix") - t0
st = g["states"]
eo = (tt >= st["calibration_eyes_open"] - t0 + 2) & (tt < st["calibration_eyes_closed"] - t0)
ec = (tt >= st["calibration_eyes_closed"] - t0 + 2) & (tt < st["countdown"] - t0)
fo, Po = signal.welch(y[M.POST][:, eo], fs=fs, nperseg=int(4 * fs), axis=-1)
_, Pc = signal.welch(y[M.POST][:, ec], fs=fs, nperseg=int(4 * fs), axis=-1)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.axvspan(8, 13, color="#F3E3BF", lw=0)
ax.semilogy(fo, np.sqrt(Po.mean(0)), color=BLUE, lw=2.2, label="eyes open")
ax.semilogy(fo, np.sqrt(Pc.mean(0)), color=ORANGE, lw=2.6, label="eyes closed")
pk = np.argmax(np.sqrt(Pc.mean(0)) * ((fo >= 8) & (fo <= 13)))
ax.annotate(f"{fo[pk]:.1f} Hz alpha", (fo[pk], np.sqrt(Pc.mean(0))[pk]), xytext=(fo[pk] + 5, np.sqrt(Pc.mean(0))[pk]),
            color=ORANGE, fontweight="bold", arrowprops=dict(arrowstyle="-", color=ORANGE))
ax.set(xlim=(1, 30), xlabel="Frequency (Hz)", ylabel="µV/√Hz (P3, P4, O1)",
       title=f"Final run baseline: alpha ×{A['base']['eyes_closed_alpha'] / A['base']['eyes_open_alpha']:.0f} with eyes closed")
ax.legend(loc="upper right")
fig.tight_layout()
save(fig, "alpha_final_baseline.png")

# ------------------------------------------------------------------ 4. detector validation (real awake data + simulations)
D.CALIBRATION_S = 40.0
r3 = load_recording(REPO / "data/recordings/neuropawn_knight_2026-11-07_17-29-49.csv")
e3, f3 = r3.eeg, r3.fs


def band_comp(z, lo, hi):
    return signal.sosfiltfilt(signal.butter(4, [lo, hi], "bandpass", fs=f3, output="sos"), z, axis=-1)


def mirror(seg_, reps):
    return np.concatenate([seg_ if k % 2 == 0 else seg_[:, ::-1] for k in range(reps)], axis=1)


def run(z, mon):
    qt = QualityTracker()
    out = []
    for end in range(int(4 * f3), z.shape[1] + 1, int(f3 / 2)):
        q = assess(z[:, :end], f3, 59.68, KNIGHT_FULL_SCALE_UV)
        q = qt.update(q, end / f3) if q else None
        us = None if q is None else np.array([c.status != POOR for c in q.channels])
        s = mon.update(z[:, :end], end / f3, us)
        if s is not None:
            out.append((end / f3, s.score if s.score is not None else np.nan, s.alert))
    return np.array(out)


cal = D.DrowsinessMonitor(f3)
run(e3, cal)
base = (cal.baseline_theta, cal.baseline_beta)


def fresh():
    m = D.DrowsinessMonitor(f3)
    m.set_baseline(*base)
    return m


closed = mirror(e3[:, int(32 * f3):int(54 * f3)], 5)
opn = mirror(e3[:, int(2 * f3):int(28 * f3)], 4)
a_, th_, be_ = band_comp(closed, 8, 13), band_comp(closed, 4, 8), band_comp(closed, 13, 30)
ramp = np.clip((np.arange(closed.shape[1]) / f3 - 30) / 20, 0, 1)
cases = [
    ("Real: eyes open, alert", opn, GREY),
    ("Real: eyes closed, relaxed", closed, "#7C8BA6"),
    ("Sim: alpha blocked (normal attenuation)", closed - 0.9 * a_, "#A7B4C9"),
    ("Sim: focused attention (θ×2.5, β×2)", closed - 0.9 * a_ + 1.5 * th_ + 1.0 * be_, BLUE),
    ("Sim: pre-sleep at 30 s (α −85%, θ×2)", closed - ramp * 0.85 * a_ + ramp * 1.0 * th_ - ramp * 0.3 * be_, ORANGE),
]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
summary = []
for name, z, col in cases:
    res = run(z, fresh())
    alerts = res[res[:, 2] == 1, 0]
    summary.append((name, np.nanmax(res[:, 1]), alerts[0] if alerts.size else None))
    if "pre-sleep" in name:
        ax[0].axvspan(30, 50, color="#FBE6DA", lw=0)
        ax[0].text(40, 6, "pattern\nfades in", ha="center", color=ORANGE, fontsize=12)
        ax[0].plot(res[:, 0], res[:, 1], color=ORANGE, lw=3)
        for tA in alerts:
            ax[0].axvline(tA, color=RED, lw=2)
            ax[0].text(tA + 1.5, 90, f"microsleep alert\nat {tA:.0f} s (~8 s of pattern)", color=RED, fontweight="bold", fontsize=12, va="top")
ax[0].axhline(50, color=RED, ls="--", lw=1.2)
ax[0].text(ax[0].get_xlim()[0] + 1, 52, "sustained-alert line (50)", color=RED, fontsize=11)
ax[0].set(ylim=(-3, 100), xlabel="Seconds", ylabel="Drowsiness score", title="Simulated pre-sleep transition (θ×2, α −85%)")
names_ = [n for n, _, _ in summary][::-1]
vals_ = [v for _, v, _ in summary][::-1]
cols_ = [ORANGE if "pre-sleep" in n else GREY for n in names_]
ax[1].barh(np.arange(len(names_)), vals_, color=cols_, height=0.6)
for i, (n, v) in enumerate(zip(names_, vals_)):
    a_t = dict((nm, at) for nm, _, at in summary)[n]
    ax[1].text(max(v, 0) + 2, i, f"max {v:.0f} · " + (f"alert at {a_t:.0f} s" if a_t else "no alert"), va="center", fontsize=12,
               color=RED if a_t else INK, fontweight="bold" if a_t else "normal")
ax[1].set_yticks(np.arange(len(names_)))
ax[1].set_yticklabels([n.replace(" (", "\n(") for n in names_], fontsize=12)
ax[1].set(xlim=(0, 140), xlabel="Highest score reached", title="Every awake case stays at 0")
fig.tight_layout()
save(fig, "detector_validation.png")

# ------------------------------------------------------------------ 5. driving decline
mins = A["mins"] + 0.5
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
rt = g["rt"]
ax[0].scatter(rt[:, 0] / 60, rt[:, 1], s=22, color=GREY, label="each drift")
ax[0].plot(mins, A["rt_min"], color=BLUE, lw=2.6, marker="o", label="per-minute median")
slope = np.polyfit(mins, A["rt_min"], 1)
ax[0].plot(mins, np.polyval(slope, mins), color=ORANGE, lw=2, ls="--", label=f"trend +{slope[0]:.2f} s per minute")
for lt in g["lapse"]:
    ax[0].axvline(lt / 60, color=RED, lw=1.6, ls=":")
    ax[0].text(lt / 60 - 0.3, 4.3, "lapse", color=RED, ha="right", fontsize=12)
r_ = stats.spearmanr(A["mins"], A["rt_min"])
ax[0].set(xlabel="Minutes into the drive", ylabel="Reaction time to drift (s)", ylim=(0, 4.8),
          title=f"Reaction time rose from 1.80 to 2.86 s (ρ = {r_.statistic:+.2f}, p < 0.001)")
ax[0].legend(loc="upper left", fontsize=12)
ax[1].bar(mins, A["sdlp"], color="#9DB0CF", width=0.8)
r2 = stats.spearmanr(A["mins"], A["sdlp"])
ax[1].plot(mins, np.polyval(np.polyfit(mins, A["sdlp"], 1), mins), color=ORANGE, lw=2, ls="--")
ax[1].set(xlabel="Minutes into the drive", ylabel="SDLP (m)", title=f"Weaving (SDLP) rose (ρ = {r2.statistic:+.2f}, p = {r2.pvalue:.3f})")
fig.tight_layout()
save(fig, "driving_decline.png")

# ------------------------------------------------------------------ 6. EEG detector timeline (final run)
rows = A["rows"]
T = np.array([r["t"] for r in rows]) / 60
ratio = np.array([r["θ/α (detector)"] for r in rows])
rep = sys.modules["make_drive_figure"].replay_eeg(M.RECORDING)
Rr = np.array([(rr[0] - t0, rr[1] if rr[1] is not None else np.nan, rr[2] if rr[2] is not None else np.nan,
                rr[3] if rr[3] is not None else np.nan) for rr in rep["rows"]])
fig, ax = plt.subplots(3, 1, figsize=(13, 7.4), sharex=True, gridspec_kw={"hspace": 0.35})
for a in ax:
    a.axvspan((st["calibration_eyes_open"] - t0) / 60, (st["calibration_eyes_closed"] - t0) / 60, color="#E3EAF5", lw=0)
    a.axvspan((st["calibration_eyes_closed"] - t0) / 60, (st["countdown"] - t0) / 60, color="#F1E3F2", lw=0)


def rmed(tt_, yy, w=60):
    o = np.full(len(yy), np.nan)
    for i, ti in enumerate(tt_):
        m_ = (tt_ > ti - w) & (tt_ <= ti) & np.isfinite(yy)
        if m_.sum() >= 3:
            o[i] = np.median(yy[m_])
    return o


ax[0].semilogy(Rr[:, 0] / 60, Rr[:, 1], color=GRID, lw=0.8)
ax[0].semilogy(Rr[:, 0] / 60, rmed(Rr[:, 0], Rr[:, 1]), color=BLUE, lw=2.4)
ax[0].axhline(1, color=RED, ls="--", lw=1.2)
ax[0].set(ylabel="θ/α", title="θ/α stayed near its eyes-open level (limit 1.0); eyes-closed baseline dips to 0.12")
ax[1].plot(Rr[:, 0] / 60, Rr[:, 2], color=GRID, lw=0.8)
ax[1].plot(Rr[:, 0] / 60, rmed(Rr[:, 0], Rr[:, 2]), color=ORANGE, lw=2.4)
ax[1].axhline(2, color=RED, ls="--", lw=1.2)
ax[1].set(ylabel="θ ÷ baseline", ylim=(0, 4), title="Theta never rose toward 2× baseline")
ax[2].fill_between(Rr[:, 0] / 60, np.nan_to_num(Rr[:, 3]), color=RED, alpha=0.25, lw=0)
ax[2].plot(Rr[:, 0] / 60, Rr[:, 3], color=RED, lw=1.6)
ax[2].axhline(50, color=RED, ls="--", lw=1.2)
ax[2].set(ylabel="Score", ylim=(0, 100), xlabel="Minutes into the drive (shaded: eyes-open / eyes-closed baseline)",
          title="Drowsiness score peaked at 8 / 100: no pre-sleep stage, no alert")
fig.align_ylabels(ax)
save(fig, "final_eeg_timeline.png")

# ------------------------------------------------------------------ 7. metric reanalysis (forest plot)
res = sorted(A["results"], key=lambda r: r["rho_time"])
names = [r["metric"] for r in res]
fig, ax = plt.subplots(1, 3, figsize=(14.5, 7), sharey=True)
bonf = 0.05 / len(res)
for k, (key, pkey, title) in enumerate([("rho_time", "p_time", "vs time on task"), ("rho_rt", "p_rt", "vs reaction time"),
                                        ("rho_rt_partial", "p_rt_partial", "vs reaction time,\ntime removed")]):
    vals = [r[key] for r in res]
    ps = [r[pkey] for r in res]
    cols = [ORANGE if p < bonf else (BLUE if p < 0.05 else GREY) for p in ps]
    ax[k].barh(np.arange(len(res)), vals, color=cols, height=0.65)
    ax[k].axvline(0, color=INK, lw=1)
    ax[k].set(xlim=(-0.9, 0.9), title=title, xlabel="Spearman ρ (per minute, n = 20)")
    for i, (v, p) in enumerate(zip(vals, ps)):
        if p < 0.05:  # label inside the bar so it never collides with the metric names
            ax[k].text(v / 2, i, f"p={p:.3f}", va="center", ha="center", fontsize=11, color="white", fontweight="bold")
ax[0].set_yticks(np.arange(len(res)))
ax[0].set_yticklabels(names)
fig.text(0.13, -0.02, f"Orange: survives Bonferroni for 15 metrics (p < {bonf:.4f}) · blue: p < 0.05 uncorrected · grey: n.s.  "
         "Only the drops in θ and β power over time survive; no link to reaction time does.", color=MUTED, fontsize=13)
fig.tight_layout()
save(fig, "metrics_forest.png")

# ------------------------------------------------------------------ 8. best usable trend: fatigue indices vs RT
pm = A["per_min"]


def norm(v):
    return 100 * (v / np.nanmean(v[:3]) - 1)


fig, ax = plt.subplots(figsize=(13, 5))
for k, col, lab in [("(θ+α)/β", ORANGE, "(θ+α)/β  (Jap et al. 2009)"), ("α/β", "#B4552B", "α/β"), ("log β central", GREY, "β power (central)")]:
    v = pm[k] if not k.startswith("log") else 10 ** pm[k]
    sm = np.convolve(np.pad(norm(v), 1, mode="edge"), np.ones(3) / 3, mode="valid")
    ax.plot(mins, sm, color=col, lw=2.6, label=lab, marker="o", ms=4)
ax.axhline(0, color=INK, lw=0.8)
ax.set(xlabel="Minutes into the drive", ylabel="Change vs first 3 min (%)",
       title="Classic fatigue indices rose as reaction time slowed (3-min smoothing)")
ax2 = ax.twinx()
ax2.plot(mins, A["rt_min"], color=BLUE, lw=2.2, ls="--", label="reaction time (right axis)")
ax2.set_ylabel("Reaction time (s)", color=BLUE)
ax2.tick_params(axis="y", colors=BLUE)
ax2.spines["right"].set_visible(True)
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=12, ncol=4)
fig.tight_layout()
save(fig, "fatigue_index_trend.png")

# ------------------------------------------------------------------ 9. blink duration vs reaction time
bd = pm["blink duration"] * 1000
ok = np.isfinite(bd)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(mins, bd, color=ORANGE, lw=2.4, marker="o")
ax[0].set(xlabel="Minutes into the drive", ylabel="Median blink duration (ms)", title="Blink duration from the frontal electrodes")
sc = ax[1].scatter(bd[ok], A["rt_min"][ok], c=mins[ok], cmap="Blues", s=70, edgecolor=INK, linewidth=0.6)
rr = stats.spearmanr(bd[ok], A["rt_min"][ok])
ax[1].set(xlabel="Median blink duration (ms)", ylabel="Median reaction time (s)",
          title=f"Longer blinks, slower reactions (ρ = {rr.statistic:+.2f}, p = {rr.pvalue:.3f}, uncorrected)")
cb = fig.colorbar(sc, ax=ax[1])
cb.set_label("minute")
fig.tight_layout()
save(fig, "blink_vs_rt.png")

# ------------------------------------------------------------------ summary numbers for the slides
rtv = g["rt"]
print("SUMMARY", dict(
    windows_valid=f"{A['n_windows_drive']}", base=A["base"], hum=hum_stats,
    rt_first5=float(np.median(rtv[rtv[:, 0] < 300, 1])), rt_last5=float(np.median(rtv[rtv[:, 0] >= 900, 1])),
    drifts=int(len(rtv) + len(g["lapse"])), departures=int(len(g["departure"])), lapses=int(len(g["lapse"])),
    max_score=float(np.nanmax(Rr[(Rr[:, 0] >= 0) & (Rr[:, 0] <= 1200), 3])),
))
