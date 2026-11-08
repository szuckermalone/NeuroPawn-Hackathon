"""Slide figure: eyes-open vs eyes-closed alpha from a NeuroPawn GUI recording.

Usage: PYTHONPATH=src python notebooks/make_alpha_figure.py <recording.csv> [switch_s=30] [closed_end_s=54]
Writes presentation/figures/eyes_open_closed_alpha.png
"""
import sys

import matplotlib
import numpy as np
from scipy import signal

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from neuropawn_gui.cleaning import CleaningConfig, design_sos  # noqa: E402
from neuropawn_gui.review import load_recording  # noqa: E402

path = sys.argv[1]
switch_s = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
closed_end_s = float(sys.argv[3]) if len(sys.argv) > 3 else 54.0  # blinks resumed at ~55 s in our session

r = load_recording(path)
fs = r.fs
eeg = r.eeg
n = eeg.shape[1]
y = signal.sosfiltfilt(design_sos(fs, CleaningConfig()), eeg - eeg.mean(1, keepdims=True), axis=1)
pad = int(2 * fs)
OPEN, CLOSED = slice(pad, int(switch_s * fs) - pad), slice(int(switch_s * fs) + pad, int(closed_end_s * fs))


def welch(idx, sl):
    return signal.welch(y[idx, sl], fs=fs, nperseg=int(4 * fs), axis=-1)


def alpha_power(P, ff):
    m = (ff >= 8) & (ff < 13)
    return np.trapezoid(P[..., m], ff[m], axis=-1)


f, Po = welch([4, 5], OPEN)
_, Pc = welch([4, 5], CLOSED)
ao, ac = np.sqrt(Po.mean(0)), np.sqrt(Pc.mean(0))
a = signal.sosfiltfilt(signal.butter(4, [8, 13], "bandpass", fs=fs, output="sos"), y, axis=1)
w = int(fs)
tt = np.arange(0, n - w + 1, w)
tc = (tt + w / 2) / fs


def env(idx):
    return np.array([np.sqrt((a[idx, i:i + w] ** 2).mean()) for i in tt])


BLUE, ORANGE, INK, MUTED = "#36579E", "#E0602D", "#1B1F23", "#6B7580"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 13, "axes.edgecolor": "#9AA3AD",
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(13.33, 7.5), dpi=200)
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.42, wspace=0.22,
                      left=0.06, right=0.98, top=0.86, bottom=0.08)
fig.suptitle("Closing the eyes switches on a ~11 Hz alpha rhythm", x=0.06, ha="left",
             fontsize=22, fontweight="bold", color=INK)
fig.text(0.06, 0.905, "NeuroPawn Knight, 8 ch @ 125 Hz · 30 s eyes open, then 30 s eyes closed · "
         "parietal electrodes P3/P4, 1–40 Hz + 60 Hz notch", fontsize=12, color=MUTED)

ax = fig.add_subplot(gs[0, 0])
ax.axvspan(8, 13, color="#F4E3B5", alpha=0.6, lw=0)
ax.text(10.5, max(ac) * 1.6, "alpha band", ha="center", color="#8A6D1F", fontsize=11)
ax.semilogy(f, ao, color=BLUE, lw=2.2, label="Eyes open")
ax.semilogy(f, ac, color=ORANGE, lw=2.6, label="Eyes closed")
pk = np.argmax(ac * ((f >= 8) & (f <= 13)))
ax.annotate(f"{f[pk]:.2f} Hz peak", (f[pk], ac[pk]), xytext=(f[pk] + 6, ac[pk] * 1.05), color=ORANGE,
            fontsize=12, fontweight="bold", arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.2))
band = (f > 1) & (f < 40)
ax.set_xlim(1, 40)
ax.set_ylim(min(ao[band].min(), ac[band].min()) * 0.7, max(ac) * 3)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude (µV/√Hz)")
ax.set_title("A  Spectrum (P3/P4 mean)", loc="left", fontsize=14, fontweight="bold", color=INK)
ax.legend(frameon=False, loc="upper right")

ax = fig.add_subplot(gs[0, 1])
regions = [("Frontal\nF3/F4", [0, 1]), ("Central\nC3/C4", [2, 3]), ("Parietal\nP3/P4", [4, 5]), ("Occipital\nO1", [6])]
ratios = []
for _, idx in regions:
    ff, po = welch(idx, OPEN)
    _, pc = welch(idx, CLOSED)
    ratios.append(alpha_power(pc, ff).mean() / alpha_power(po, ff).mean())
bars = ax.bar(range(4), ratios, color=["#B9C2CC", "#9DAFC6", ORANGE, ORANGE], width=0.62)
for b, v in zip(bars, ratios):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"×{v:.1f}", ha="center", fontsize=14, fontweight="bold", color=INK)
ax.axhline(1, color=MUTED, lw=1, ls="--")
ax.text(-0.42, 1.25, "no change", color=MUTED, fontsize=10, ha="left")
ax.set_xticks(range(4))
ax.set_xticklabels([r_[0] for r_ in regions])
ax.set_ylim(0, max(ratios) * 1.22)
ax.set_ylabel("Alpha power, closed ÷ open")
ax.set_title("B  Strongest at the back of the head", loc="left", fontsize=14, fontweight="bold", color=INK)

ax = fig.add_subplot(gs[1, :])
ax.axvspan(0, switch_s, color="#E8EEF6", lw=0)
ax.axvspan(switch_s, n / fs, color="#FBEADF", lw=0)
ax.text(switch_s / 2, 0.93, "EYES OPEN", transform=ax.get_xaxis_transform(), ha="center", color=BLUE, fontweight="bold")
ax.text((switch_s + n / fs) / 2, 0.93, "EYES CLOSED", transform=ax.get_xaxis_transform(), ha="center",
        color=ORANGE, fontweight="bold")
ax.plot(tc, env([4, 5]), color=ORANGE, lw=2.4, label="Parietal P3/P4")
ax.plot(tc, env([6]), color="#7C4B8D", lw=1.6, alpha=0.85, label="Occipital O1")
ax.plot(tc, env([0, 1]), color="#9AA3AD", lw=1.4, label="Frontal F3/F4")
ax.axvline(switch_s, color=INK, lw=1, ls=":")
ax.set_xlim(0, n / fs)
ax.set_ylim(0, None)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Alpha amplitude (µV rms)")
ax.set_title("C  Alpha appears within ~2 s of closing the eyes", loc="left", fontsize=14, fontweight="bold", color=INK)
ax.legend(frameon=False, loc="upper left", ncol=3)

out = "presentation/figures/eyes_open_closed_alpha.png"
fig.savefig(out, facecolor="white")
print("saved", out, "| alpha ratios", np.round(ratios, 1), "| peak", f[pk], "Hz")
