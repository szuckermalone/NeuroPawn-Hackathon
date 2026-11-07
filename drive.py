"""Desktop simulator. Each presented drift gets a prediction and an outcome."""
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
from acquisition import Source
from analysis import Predictor, summary
from signals import CHANNELS, extract
from vigilance import VigilanceMonitor


def run(args):
    import pygame as pg
    path = Path(args.out)
    path.mkdir(parents=True, exist_ok=False)
    pg.init()
    source = None
    try:
        screen = pg.display.set_mode((860, 520))
        pg.display.set_caption("Driver vigilance | " + args.mode)
        font = pg.font.Font(None, 28)
        clock = pg.time.Clock()
        rng = random.Random(args.seed)
        source = Source(args.mode, args.port, args.imu, args.seed, args.demo_fatigue)
        meta = dict(mode=args.mode, fs=source.fs, channels=CHANNELS, calibration=args.calibration,
                    seed=args.seed, demo_fatigue=args.demo_fatigue, response_timeout_s=4, board_variant="imu" if args.imu else "standard")
        (path/"session.json").write_text(json.dumps(meta, indent=2))
        history = np.empty((0, 9))
        predictor = Predictor(args.calibration)
        monitor = VigilanceMonitor(calibration_windows=max(3, min(8, args.calibration)))
        last_vigilance = {"state": "CALIBRATING", "score": None, "alert": False, "signal_quality": "pending"}
        last_vigilance_update = -999.0
        start = time.perf_counter()
        due = start + 12
        current = None
        completed = []
        held = set()
        axis_ready = True
        joy = pg.joystick.Joystick(0) if args.joystick else None
        if joy:
            joy.init()
        samples_saved = 0
        feedback = "Keep the car centred. Correct each drift with the opposite arrow."
        with (path/"events.jsonl").open("w", buffering=1) as events, (path/"eeg.csv").open("w") as raw:
            writer = csv.writer(raw)
            writer.writerow(["unix_s", *CHANNELS])
            running = True
            while running:
                chunk = source.read()
                if len(chunk):
                    writer.writerows(chunk)
                    samples_saved += len(chunk)
                    history = np.concatenate((history, chunk))[-12*source.fs:]
                now = time.perf_counter()
                if now - last_vigilance_update >= 2.0:
                    latest_features, latest_quality = extract(history[:, 0], history[:, 1:], time.time(), source.fs)
                    last_vigilance = monitor.update(latest_features, latest_quality)
                    last_vigilance_update = now
                response = None
                for ev in pg.event.get():
                    if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE):
                        running = False
                    elif ev.type == pg.KEYUP:
                        held.discard(ev.key)
                    elif ev.type == pg.KEYDOWN and ev.key in (pg.K_LEFT, pg.K_RIGHT):
                        if ev.key not in held and current and response is None and not joy:
                            direction = -1 if ev.key == pg.K_LEFT else 1
                            if direction == -current["direction"]:
                                response = time.perf_counter()
                            else:
                                current["wrong_inputs"] += 1
                        held.add(ev.key)
                if joy:
                    axis = joy.get_axis(0)
                    if abs(axis) < 0.2:
                        axis_ready = True
                    elif axis_ready and abs(axis) > 0.45:
                        axis_ready = False
                        if current and np.sign(axis) == -current["direction"]:
                            response = time.perf_counter()
                        elif current:
                            current["wrong_inputs"] += 1
                if current and (response is not None or now-onset >= 4):
                    rt = 1000*(response-onset) if response is not None and response-onset <= 4 else None
                    current["rt_ms"] = rt
                    current["outcome"] = "miss" if rt is None else "correct"
                    events.write(json.dumps(current)+"\n")
                    raw.flush()
                    predictor.observe(base, current["features"], rt)
                    completed.append(current)
                    feedback = "Missed correction" if rt is None else f"Correction: {rt:.0f} ms"
                    current = None
                    due = now + rng.uniform(12, 20)
                new = running and current is None and now >= due
                if new:
                    cutoff = time.time()
                    features, quality = extract(history[:, 0], history[:, 1:], cutoff, source.fs)
                    base, pred = predictor.predict(now-start, features)
                    current = dict(**pred, trial=len(completed)+1, cutoff_unix=cutoff, samples_saved=samples_saved, elapsed_s=now-start,
                                   features=features, quality=quality, vigilance=last_vigilance.copy(), direction=rng.choice([-1, 1]), wrong_inputs=0)
                offset = 0
                if current:
                    age = 0 if new else now-onset
                    offset = current["direction"]*min(165, 5+age*85)
                screen.fill((18, 35, 42))
                pg.draw.rect(screen, (54, 62, 70), (190, 65, 480, 420))
                for x in (275, 585):
                    pg.draw.line(screen, (230, 230, 220), (x, 65), (x, 485), 3)
                for y in range(-55, 440, 75):
                    py = 65+y+int((now-start)*130)%75
                    if 65 <= py <= 455:
                        pg.draw.rect(screen, (136, 144, 145), (427, py, 5, 28))
                pg.draw.rect(screen, (78, 195, 223), (int(405+offset), 345, 50, 86), border_radius=8)
                screen.blit(font.render(args.mode.upper()+"  |  "+feedback, True, "white"), (12, 18))
                label = "EEG trend: " + last_vigilance["state"]
                if last_vigilance["score"] is not None:
                    label += f"  |  Index: {last_vigilance['score']:+.2f}"
                color = (255, 120, 120) if last_vigilance["alert"] else (200, 225, 240)
                screen.blit(font.render(label, True, color), (12, 49))
                if args.demo_fatigue:
                    screen.blit(font.render("SIMULATED FATIGUE TRANSITION: NOT A REAL DETECTOR", True, (255, 204, 95)), (12, 492))
                if last_vigilance["alert"]:
                    screen.blit(font.render("EEG CHANGE ALERT (EXPERIMENTAL)", True, (255, 120, 120)), (258, 98))
                pg.display.flip()
                if new:
                    onset = time.perf_counter()
                    current["onset_unix"] = time.time()
                    source.mark(251 if current["direction"] == -1 else 252)
                if len(completed) >= args.trials:
                    running = False
                clock.tick(60)
            if current:
                events.write(json.dumps(dict(current, rt_ms=None, outcome="aborted"))+"\n")
        print(json.dumps(summary(completed), indent=2))
    finally:
        try:
            if source:
                source.close()
        finally:
            pg.quit()
