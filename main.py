import argparse
import json
from pathlib import Path
import numpy as np
from analysis import Predictor, records, summary
from signals import extract


def replay(args):
    path = Path(args.session)
    meta = json.loads((path/"session.json").read_text())
    data = np.loadtxt(path/"eeg.csv", delimiter=",", skiprows=1, ndmin=2)
    predictor = Predictor(meta["calibration"])
    output = []
    for event in records(path/"events.jsonl"):
        if event.get("outcome") == "aborted":
            continue
        available = data[:event.get("samples_saved")]
        features, quality = extract(available[:,0], available[:,1:], event["cutoff_unix"], meta["fs"])
        if event["quality"] == "previous correction in window":
            features, quality = None, event["quality"]
        base, pred = predictor.predict(event["elapsed_s"], features)
        result = dict(event)
        result.update(pred, features=features, quality=quality)
        output.append(result)
        print(json.dumps(dict(trial=event["trial"], observed=event["rt_ms"], **pred)))
        predictor.observe(base, features, event["rt_ms"])
    print(json.dumps(summary(output), indent=2))


def evaluate(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    rows = []
    for session in args.sessions:
        path = Path(session)
        meta = json.loads((path/"session.json").read_text())
        events = [e for e in records(path/"events.jsonl") if e.get("outcome") != "aborted"]
        stats = summary(events)
        rows.append(dict(session=str(path), mode=meta["mode"], **stats))
        if stats["paired"]:
            ax.scatter(stats["behaviour_ms_mae"], stats["combined_ms_mae"],label=path.name)
    limit = max([100]+[max(r["behaviour_ms_mae"],r["combined_ms_mae"])*1.1 for r in rows if r["paired"]])
    ax.plot([0,limit],[0,limit], color="gray", linestyle=":")
    ax.set(xlabel="Behaviour-only MAE (ms)", ylabel="Behaviour + EEG MAE (ms)",
           title="Each point is a session; lower error is better")
    if any(r["paired"] for r in rows):
        ax.legend()
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=False)
    fig.savefig(dest/"comparison.png",bbox_inches="tight")
    (dest/"metrics.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


def main():
    p = argparse.ArgumentParser(description="Simulator-based steering response study")
    commands = p.add_subparsers(dest="command",required=True)
    drive = commands.add_parser("drive")
    drive.add_argument("--mode",choices=["synthetic","live"],required=True)
    drive.add_argument("--port")
    drive.add_argument("--imu",action="store_true")
    drive.add_argument("--joystick",action="store_true")
    drive.add_argument("--trials",type=int,default=32)
    drive.add_argument("--seed",type=int)
    public = commands.add_parser("import-cao")
    public.add_argument("--set",required=True)
    for cmd in [drive,public]:
        cmd.add_argument("--out",required=True)
        cmd.add_argument("--calibration",type=int,default=12)
    rep = commands.add_parser("replay")
    rep.add_argument("session")
    ev = commands.add_parser("evaluate")
    ev.add_argument("sessions",nargs="+")
    ev.add_argument("--out",required=True)
    args = p.parse_args()
    if args.command in ("drive","import-cao") and args.calibration < 3:
        p.error("Use at least 3 usable calibration trials.")
    if args.command == "drive":
        if args.mode=="live" and not args.port:
            p.error("Live mode needs --port.")
        if args.trials <= args.calibration:
            p.error("--trials must exceed --calibration.")
        from drive import run
        run(args)
    elif args.command=="import-cao":
        from research import convert
        convert(args)
    elif args.command=="replay":
        replay(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
