"""Validate the pre-sleep detector on real, expert-scored sleep onsets (PhysioNet Sleep-EDF).

The GUI's DrowsinessMonitor is run unchanged on each recording's first sleep onset of the
night: 30 min of scored wake before it, then 10 min after.

Sleep-EDF (sleep-cassette) has two EEG derivations at 100 Hz, so the detector's channel slots
are filled as follows:
    frontal (blink check)  <- Fpz-Cz
    central (theta, beta)  <- Fpz-Cz   (nearest to the central sites the detector expects)
    posterior (alpha)      <- Pz-Oz
Variant "no_blink" drops the frontal blink veto, because slow eye movements at sleep onset
are large on Fpz and would otherwise be read as "eyes open".

Sleep onset = first epoch scored N1/N2/N3/REM preceded by >= 10 min of continuous wake (AASM).
Calibration = the detector's normal 60 s at the start (30 min before onset, awake).
Alerts >= 5 min before scored onset count as false alarms (conservative: some may be
genuine pre-sleep drowsiness, which AASM still scores as wake).

Variants "posterior"/"posterior_no_blink" take theta, alpha and beta all from Pz-Oz: the ratio
threshold assumes theta and alpha come from comparable referential sites, which a frontal vs
posterior bipolar pair breaks (theta/alpha ~10 in every state). Pz-Oz alone is also closest to
the AASM N1 definition (occipital alpha replaced by theta). All variants are reported.

Results on 13 night-1 recordings (SC4001-SC4121), posterior mapping: 9/13 onsets caught within
-5..+10 min (6/13 within +5), median +1.5 min; 3 alerts in 5.2 h of in-bed wake; 3.0/h in
25.8 h of ambulatory daytime wake. Original frontal mapping: 6/13, 23 alerts in 5.2 h, 7.0/h.

Usage: python sleep_edf_validation.py [max_recordings] [repo_root]
Recordings are read from ~/mne_data/physionet-sleep-data (download with
mne.datasets.sleep_physionet.age.fetch_data(subjects=[...], recording=[1])).
Outputs: presentation/figures/final/sleep_edf_validation.png and data/sleep_edf_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from neuropawn_gui import drowsiness as dz  # noqa: E402

mne.set_log_level("ERROR")
EPOCH = 30.0
PRE_MIN, POST_MIN = 30, 10
FA_BEFORE_MIN = 5
SLEEP = {"Sleep stage 1": "N1", "Sleep stage 2": "N2", "Sleep stage 3": "N3",
         "Sleep stage 4": "N3", "Sleep stage R": "R"}
# name -> (signal feeding the central theta/beta slots, frontal blink veto on?)
VARIANT_MAP = {"faithful": ("fpz", True), "no_blink": ("fpz", False),
               "posterior": ("pz", True), "posterior_no_blink": ("pz", False)}
VARIANTS = tuple(VARIANT_MAP)


def stages(annot, n_epochs):
    st = np.full(n_epochs, "?", dtype=object)
    for on, dur, desc in zip(annot.onset, annot.duration, annot.description):
        lab = "W" if desc == "Sleep stage W" else SLEEP.get(desc, "?")
        a, b = int(on // EPOCH), int((on + dur) // EPOCH)
        st[a:min(b, n_epochs)] = lab
    return st


def first_onset(st, wake_epochs=20):
    for i in range(wake_epochs, len(st)):
        if st[i] in ("N1", "N2", "N3", "R") and all(s == "W" for s in st[i - wake_epochs:i]):
            return i
    return None


def run_detector(x8, fs, t_axis, usable):
    """x8: (8, n) uV. Feeds 4 s windows every 2 s, exactly as the GUI does."""
    mon = dz.DrowsinessMonitor(fs)
    n = int(dz.WINDOW_S * fs)
    rows = []
    for end in range(n, x8.shape[1] + 1, int(dz.STEP_S * fs)):
        t = t_axis[end - 1]
        st = mon.update(x8[:, end - n:end], t * 60.0, usable)   # detector clock is in seconds
        if st is None:
            continue
        w = mon.last_window
        rows.append(dict(t=float(t), phase=st.phase, score=st.score, ratio=w.ratio, label=w.label,
                         alert=st.alert, reason=st.alert_reason))
    return rows


def analyse(psg, hyp):
    raw = mne.io.read_raw_edf(psg, include=["EEG Fpz-Cz", "EEG Pz-Oz"], preload=True)
    fs = raw.info["sfreq"]
    fpz, pz = raw.get_data() * 1e6
    st = stages(mne.read_annotations(hyp), int(len(fpz) / fs // EPOCH))
    i_on = first_onset(st)
    if i_on is None:
        return None
    t_on = i_on * EPOCH
    a, b = int((t_on - PRE_MIN * 60) * fs), int((t_on + POST_MIN * 60) * fs)
    if a < 0 or b > len(fpz):
        return None
    out = {"psg": Path(psg).name, "onset_stage": str(st[i_on]),
           "stages": [str(s) for s in st[i_on - PRE_MIN * 2:i_on + POST_MIN * 2]]}
    sig = {"fpz": fpz, "pz": pz}
    da, db = int((t_on - 180 * 60) * fs), int((t_on - 60 * 60) * fs)
    ep = lambda t: int((t_on + t * 60) // EPOCH)
    clean = lambda t: all(s == "W" for s in st[max(0, ep(t) - 10):ep(t) + 11])
    for name, (central, blink) in VARIANT_MAP.items():
        usable = None if blink else np.array([0, 0, 1, 1, 1, 1, 1, 1], bool)
        mk = lambda lo, hi: np.vstack([fpz[lo:hi]] * 2 + [sig[central][lo:hi]] * 2 + [pz[lo:hi]] * 4)
        t_rel = (np.arange(a, b) / fs - t_on) / 60.0           # minutes from scored onset
        out[name] = run_detector(mk(a, b), fs, t_rel, usable)
        # Daytime wake check: 3 h -> 1 h before onset (ambulatory, awake). Alerts count as false
        # alarms when their epoch is scored W with no sleep epoch within 5 min.
        if da >= 0:
            rows = run_detector(mk(da, db), fs, (np.arange(da, db) / fs - t_on) / 60.0, usable)
            mon = [x for x in rows if x["phase"] == "monitoring"]
            out["day_" + name] = dict(
                alerts=[x["t"] for x in rows if x["alert"] and clean(x["t"])],
                wake_hours=sum(clean(x["t"]) for x in mon) * dz.STEP_S / 3600,
                valid_fraction=float(np.mean([x["label"] != "invalid" for x in mon])) if mon else 0.0)
    return out


def summarise(results, variant):
    hits, latencies, fa, wake_h, early = 0, [], 0, 0.0, 0
    for r in results:
        alerts = [x["t"] for x in r[variant] if x["alert"]]
        fa += sum(t < -FA_BEFORE_MIN for t in alerts)
        cal_end = next((x["t"] for x in r[variant] if x["phase"] == "monitoring"), -PRE_MIN)
        wake_h += max(0.0, (-FA_BEFORE_MIN - cal_end)) / 60.0
        window = [t for t in alerts if -FA_BEFORE_MIN <= t <= POST_MIN]
        if window:
            hits += 1
            latencies.append(window[0])
        early += any(t < -FA_BEFORE_MIN for t in alerts)
    n = len(results)
    day = [r["day_" + variant] for r in results if "day_" + variant in r]
    day_fa, day_h = sum(len(d["alerts"]) for d in day), sum(d["wake_hours"] for d in day)
    return dict(day_false_alarms=day_fa, day_wake_hours=day_h,
                day_fa_per_hour=day_fa / day_h if day_h else None,
                day_valid_fraction=float(np.mean([d["valid_fraction"] for d in day])) if day else None,
                n=n, detected=hits, missed=n - hits, sensitivity=hits / n if n else None,
                median_latency_min=float(np.median(latencies)) if latencies else None,
                latencies_min=latencies, false_alarms=fa, wake_hours=wake_h,
                fa_per_hour=fa / wake_h if wake_h else None, recordings_with_early_alert=early)


def figure(results, summ, out_png, variant="no_blink"):
    grid = np.arange(-PRE_MIN, POST_MIN, 2 / 60)
    ratios, scores = [], []
    for r in results:
        rows = [x for x in r[variant] if x["phase"] == "monitoring"]
        t = np.array([x["t"] for x in rows])
        rat = np.array([x["ratio"] if x["ratio"] is not None else np.nan for x in rows], float)
        sc = np.array([x["score"] if x["score"] is not None else np.nan for x in rows], float)
        ratios.append(np.interp(grid, t, rat, left=np.nan, right=np.nan))
        scores.append(np.interp(grid, t, sc, left=np.nan, right=np.nan))
    ratios, scores = np.array(ratios), np.array(scores)

    def smooth(v, k=15):
        v = np.where(np.isnan(v), np.nanmedian(v), v)
        return np.convolve(np.pad(v, k // 2, mode="edge"), np.ones(k) / k, mode="valid")

    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.1, 1.1, 1.2]))
    fig.patch.set_facecolor("#F6F5F1")
    for a in ax:
        a.set_facecolor("#F6F5F1")
        a.axvspan(-PRE_MIN, -FA_BEFORE_MIN, color="#2F5DA8", alpha=0.06, lw=0)
        a.axvline(0, color="#14213D", lw=1.2, ls="--")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    med, q1, q3 = (np.nanpercentile(ratios, q, axis=0) for q in (50, 25, 75))
    ax[0].fill_between(grid, smooth(q1), smooth(q3), color="#E0602D", alpha=0.18, lw=0)
    ax[0].plot(grid, smooth(med), color="#E0602D", lw=2.4)
    ax[0].axhline(1.0, color="#B8481C", lw=1, ls=":")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("theta / alpha\n(median, IQR)")
    ax[0].set_title("θ/α rises through real sleep onset", loc="left", fontsize=13, fontweight="bold")
    ax[1].plot(grid, smooth(np.nanmean(scores, axis=0)), color="#14213D", lw=2.4)
    ax[1].axhline(dz.ALERT_SCORE, color="#B8481C", lw=1, ls=":")
    ax[1].set_ylim(0, 100)
    ax[1].set_ylabel("detector score\n(mean)")
    ax[1].set_title("Mean detector score across recordings", loc="left", fontsize=13, fontweight="bold")
    for i, r in enumerate(results):
        al = [x["t"] for x in r[variant] if x["alert"]]
        ax[2].axhline(i, color="#DAD7CE", lw=0.6, zorder=1)
        ax[2].scatter(al, [i] * len(al), s=30, zorder=3,
                      color=["#2F5DA8" if t < -FA_BEFORE_MIN else "#E0602D" for t in al])
    ax[2].set_yticks([])
    ax[2].set_ylim(-1, len(results))
    ax[2].set_ylabel(f"{len(results)} recordings")
    ax[2].set_xlabel("minutes from expert-scored sleep onset (dashed line)")
    s = summ[variant]
    ax[2].set_title(f"Alerts: {s['detected']}/{s['n']} onsets caught within −5…+10 min · "
                    f"{s['false_alarms']} alerts >5 min before onset ({s['wake_hours']:.1f} h scored awake)",
                    loc="left", fontsize=13, fontweight="bold")
    chans = "Pz-Oz → theta & alpha" if VARIANT_MAP[variant][0] == "pz" else "Fpz-Cz → theta, Pz-Oz → alpha"
    fig.suptitle(f"Our unchanged detector on PhysioNet Sleep-EDF: {len(results)} real, expert-scored sleep "
                 f"onsets ({chans}, 100 Hz)", x=0.01, ha="left", fontsize=11, color="#4B5565")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=160)


def main():
    # Recordings already downloaded with mne.datasets.sleep_physionet.age.fetch_data (night 1).
    data_dir = Path.home() / "mne_data" / "physionet-sleep-data"
    n_sub = int(sys.argv[1]) if len(sys.argv) > 1 else 99
    pairs = []
    for psg in sorted(data_dir.glob("SC4??1E0-PSG.edf"))[:n_sub]:
        hyp = sorted(data_dir.glob(psg.name[:6] + "*-Hypnogram.edf"))
        if hyp:
            pairs.append((str(psg), str(hyp[0])))
    results = []
    for files in [pairs]:
        for psg, hyp in files:
            r = analyse(psg, hyp)
            if r is None:
                print("no usable onset", Path(psg).name, flush=True)
                continue
            results.append(r)
            for v in VARIANTS:
                al = [round(x["t"], 1) for x in r[v] if x["alert"]]
                print(f"{Path(psg).name} [{v:18s}] onset={r['onset_stage']} alerts(min)={al}", flush=True)
    summ = {v: summarise(results, v) for v in VARIANTS}
    for v, x in summ.items():
        print(f"{v:20s} caught {x['detected']}/{x['n']}  median latency {x['median_latency_min']} min  "
              f"pre-onset FA {x['false_alarms']} in {x['wake_hours']:.1f} h  "
              f"daytime FA {x['day_false_alarms']} in {x['day_wake_hours']:.1f} h "
              f"({x['day_fa_per_hour'] or 0:.1f}/h)", flush=True)
    (ROOT / "data" / "sleep_edf_results.json").write_text(json.dumps(
        {"summary": summ, "recordings": [{k: r[k] for k in ("psg", "onset_stage", "stages")} for r in results],
         "traces": {r["psg"]: {v: r[v] for v in VARIANTS} for r in results},
         "day": {r["psg"]: {v: r.get("day_" + v) for v in VARIANTS} for r in results}}))
    if results:
        figure(results, summ, ROOT / "presentation/figures/final/sleep_edf_validation.png", variant="posterior")


if __name__ == "__main__":
    main()
