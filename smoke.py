"""Headless, reproducible check. Generated responses are software fixtures."""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from signals import CHANNELS, extract
from analysis import Predictor, summary


def check(root):
    fs = 125
    rng = np.random.default_rng(10)
    t = np.arange(200*fs)/fs
    eeg = rng.normal(0, 5, (len(t),8))
    predictor = Predictor(4)
    events = []
    # Drop a complete channel in one trial to exercise rejection and coverage.
    eeg[(t >= 102) & (t < 110),0] = 0
    for i, end in enumerate(range(10,191,10)):
        feat, quality = extract(t,eeg,end,fs)
        base,pred = predictor.predict(end,feat)
        rt = None if i == 8 else float(420+rng.normal(0,50))
        events.append(dict(**pred, trial=i+1, cutoff_unix=end, elapsed_s=end,
                           rt_ms=rt, quality=quality, features=feat,
                           outcome="miss" if rt is None else "correct"))
        predictor.observe(base,feat,rt)
    assert events[10]["features"] is None
    assert events[3]["phase"]=="calibration" and events[4]["phase"]=="test"
    # A movement after onset must never change the locked features.
    before = extract(t,eeg,100,fs)[0]
    altered = eeg.copy()
    altered[t>=100] = 1e6
    np.testing.assert_allclose(before,extract(t,altered,100,fs)[0])
    root.mkdir()
    (root/"session.json").write_text(json.dumps(dict(mode="synthetic",fs=fs,calibration=4)))
    with (root/"eeg.csv").open("w") as f:
        f.write("unix_s,"+",".join(CHANNELS)+"\n")
        np.savetxt(f,np.column_stack((t,eeg)),delimiter=",")
    (root/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
    def cli(*args):
        return subprocess.check_output([sys.executable,str(Path(__file__).with_name("main.py")),*args],text=True)
    replay = cli("replay",str(root))
    assert json.loads(replay[replay.index('{\n'):]) == summary(events)
    out = root.parent/"report"
    result = json.loads(cli("evaluate",str(root),"--out",str(out)))[0]
    assert result["paired"] == 13 and result["test_misses"] == 1
    assert (out/"comparison.png").stat().st_size > 1000
    print("PASS: calibration split, future-data isolation, quality rejection, missed response, raw replay, metrics and plot.")


if __name__=="__main__":
    with tempfile.TemporaryDirectory() as d:
        check(Path(d)/"fixture")
