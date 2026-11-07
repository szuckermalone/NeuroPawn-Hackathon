"""Convert one original Cao EEGLAB session into the local replay format."""
import csv
import json
from pathlib import Path
import numpy as np
from signals import CHANNELS, extract
from analysis import Predictor


def convert(args):
    import mne
    raw = mne.io.read_raw_eeglab(args.set, preload=True)
    rows = {name.casefold(): name for name in raw.ch_names}
    raw.pick([rows[ch.casefold()] for ch in CHANNELS])
    markers, ids = mne.events_from_annotations(raw)
    by_id = {value: name.strip() for name, value in ids.items()}
    decoded = [(sample-raw.first_samp, by_id[code]) for sample, _, code in markers]
    if not {"253"}.issubset(by_id.values()) or not {"251","252"}.intersection(by_id.values()):
        raise ValueError(f"Expected Cao numeric annotations, found {ids}")
    fs = int(round(raw.info["sfreq"]))
    t = raw.times
    x = raw.get_data().T*1e6
    path = Path(args.out)
    path.mkdir(parents=True, exist_ok=False)
    (path/"session.json").write_text(json.dumps(dict(mode="public", fs=fs, channels=CHANNELS,
        calibration=args.calibration, source=Path(args.set).name, time_origin="recording seconds")))
    with (path/"eeg.csv").open("w") as f:
        f.write("recording_s,"+",".join(CHANNELS)+"\n")
        for i in range(0,len(t),20000):
            np.savetxt(f,np.column_stack((t[i:i+20000],x[i:i+20000])),delimiter=",",fmt="%.17g")
    predictor = Predictor(args.calibration)
    pending = None
    previous_end = -np.inf
    trial = 0
    with (path/"events.jsonl").open("w") as out:
        def finish(rt):
            pending["rt_ms"] = rt
            pending["outcome"] = "miss" if rt is None else "correct"
            out.write(json.dumps(pending)+"\n")
            predictor.observe(base, pending["features"], rt)
        for sample, code in decoded:
            when = sample/fs
            if code in {"251", "252"}:
                if pending:
                    finish(None)
                trial += 1
                features, quality = extract(t,x,when,fs)
                if when-previous_end < 8:
                    features, quality = None, "previous correction in window"
                base, pred = predictor.predict(when,features)
                pending = dict(**pred, trial=trial, elapsed_s=when, cutoff_unix=when,
                               features=features, quality=quality, direction=-1 if code=="251" else 1)
            elif code == "253" and pending:
                rt = 1000*(when-pending["elapsed_s"])
                finish(rt if 0 < rt <= 4000 else None)
                pending = None
                previous_end = when
            elif code == "254":
                previous_end = when
        if pending:
            finish(None)
