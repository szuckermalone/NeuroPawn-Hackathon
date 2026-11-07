"""Reconstruct experimental EEG trend from the raw session recording."""
import csv
import json
from pathlib import Path
import numpy as np
from signals import extract
from vigilance import VigilanceMonitor


def make_report(session, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    session, out = Path(session), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((session / "session.json").read_text())
    fs = int(meta["fs"])
    data = np.loadtxt(session / "eeg.csv", delimiter=",", skiprows=1, ndmin=2)
    monitor = VigilanceMonitor(calibration_windows=max(3, min(8, meta.get("calibration", 8))))
    records = []
    if len(data) == 0:
        raise ValueError("No raw EEG samples")
    first, last = data[0, 0], data[-1, 0]
    for end in np.arange(first + 8.0, last, 2.0):
        f, q = extract(data[:, 0], data[:, 1:], end, fs)
        result = monitor.update(f, q)
        records.append(dict(elapsed_s=round(float(end-first), 2), **result))
    with (out / "vigilance.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["elapsed_s", "state", "score", "alert", "calibration", "signal_quality", "note"])
        writer.writeheader()
        writer.writerows(records)
    fig, ax = plt.subplots(figsize=(9, 4))
    times = [r["elapsed_s"] for r in records if r["score"] is not None]
    scores = [r["score"] for r in records if r["score"] is not None]
    if times:
        ax.plot(times, scores, label="EEG change index")
        ax.axhline(1.5, color="tab:red", linestyle=":", label="Illustrative threshold")
        for r in records:
            if r["alert"]:
                ax.axvspan(r["elapsed_s"] - 1, r["elapsed_s"] + 1, color="tab:red", alpha=.12)
        ax.legend()
    else:
        ax.text(.5, .5, "Insufficient accepted data to obtain a baseline", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Elapsed time (seconds)", ylabel="Baseline-relative EEG index",
           title="EXPERIMENTAL indicator: not validated driver drowsiness")
    fig.tight_layout()
    fig.savefig(out / "vigilance.png", dpi=150)
    plt.close(fig)
    summary = dict(session=str(session), windows=len(records), accepted=sum(r["signal_quality"] == "ok" for r in records),
                   alerts=sum(r["alert"] for r in records), simulation=bool(meta.get("demo_fatigue")),
                   explanation="Index is not a probability of drowsiness or a validated classifier")
    (out / "vigilance_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
