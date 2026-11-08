"""Night Drive: task logic, the GUI<->game link, and the game process end to end (headless)."""

import csv
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from neuropawn_gui.game.task import DriveTask, TaskConfig
from neuropawn_gui.link import MARKERS, LinkEndpoint

REPO = Path(__file__).resolve().parents[1]
DT = 1 / 60


def drive(task, seconds, steer=0.0):
    events = []
    for _ in range(int(round(seconds / DT))):
        events += task.step(DT, steer(task) if callable(steer) else steer)
    return events


def names(events):
    return [e["event"] for e in events]


# ------------------------------------------------------------------ task logic
def test_first_drift_and_reaction_time():
    task = DriveTask(TaskConfig(first_drift_s=2.0), seed=1)
    ev = drive(task, 1.9)
    assert ev == [] and task.x == 0.0
    ev = drive(task, 0.2)
    assert names(ev) == ["drift_onset"]
    d = ev[0]["dir"]
    x_before = task.x
    drive(task, 0.7)                           # drifting, no response yet
    assert abs(task.x) > abs(x_before) and task.x * d > 0
    ev = drive(task, 0.1, steer=-d)            # steer against the drift
    resp = [e for e in ev if e["event"] == "response"]
    assert len(resp) == 1 and resp[0]["rt"] == pytest.approx(0.8, abs=0.05)


def test_recovery_ends_trial_and_schedules_next():
    task = DriveTask(TaskConfig(first_drift_s=1.0, interval_s=(3.0, 3.0)), seed=2)
    drive(task, 1.6)
    d = task.trial.direction
    ev = drive(task, 3.0, steer=lambda tk: 0.0 if abs(tk.x) < 0.05 else (-1.0 if tk.x > 0 else 1.0) if tk.trial else 0.0)
    assert "response" in names(ev) and "recovered" in names(ev)
    assert task.trial is None and task.next_onset == pytest.approx(task.trials[-1].end + 3.0)
    assert task.trials[-1].direction == d and task.trials[-1].outcome == "recovered"


def test_lapse_and_lane_departure_when_asleep():
    cfg = TaskConfig(first_drift_s=1.0)
    task = DriveTask(cfg, seed=3)
    ev = drive(task, 1.0 + cfg.lapse_s + 0.2)    # never steer
    assert names(ev).count("lapse") == 1
    assert names(ev).count("lane_departure") == 1
    assert abs(task.x) > cfg.departure_m
    ev = drive(task, 20)                          # keeps sitting on the shoulder; events not repeated
    assert "lapse" not in names(ev) and abs(task.x) == pytest.approx(cfg.max_offset_m)


def test_wrong_direction_input_is_not_a_response():
    task = DriveTask(TaskConfig(first_drift_s=0.5), seed=4)
    drive(task, 0.6)
    d = task.trial.direction
    ev = drive(task, 0.2, steer=float(d))        # steering *with* the drift
    assert "response" not in names(ev) and task.trial.wrong_inputs == 1


def test_stats_and_sdlp():
    task = DriveTask(TaskConfig(first_drift_s=1.0, interval_s=(2.0, 2.0)), seed=5)
    def driver(tk):  # reacts 0.6 s after each drift starts, then steers back to the centre
        tr = tk.trial
        if tr is not None and tr.rt is None:
            return float(-tr.direction) if tk.t - tr.onset > 0.6 else 0.0
        return 0.0 if abs(tk.x) < 0.05 else (-1.0 if tk.x > 0 else 1.0)
    drive(task, 60, steer=driver)
    s = task.stats()
    assert s["drifts"] >= 10 and s["responses"] == s["drifts"] - (1 if task.trial and task.trial.rt is None else 0)
    assert 0.55 < s["median_rt_s"] < 0.7 and s["lapses"] == 0
    assert 0 < s["sdlp_m"] < 0.5


# ------------------------------------------------------------------ link
def test_link_roundtrip_and_garbage():
    a = LinkEndpoint(0, 0)
    b = LinkEndpoint(0, a.port)
    a.peer_port = b.port
    assert a.send("alert", score=61, test=False)
    b.sock.sendto(b"not json", ("127.0.0.1", b.port))  # garbage is ignored
    time.sleep(0.05)
    msgs = b.poll()
    assert [m["type"] for m in msgs] == ["alert"] and msgs[0]["score"] == 61 and msgs[0]["v"] == 1
    a.close()
    b.close()


def test_marker_codes_are_unique_and_avoid_manual_keys():
    vals = list(MARKERS.values())
    assert len(vals) == len(set(vals)) and all(v > 9 for v in vals)


# ------------------------------------------------------------------ the game process, headless
def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_game_process_end_to_end(tmp_path):
    """Fake GUI <-> real game: calibration, drifts, an alert that interrupts and is acknowledged."""
    pytest.importorskip("pygame")
    game_port = _free_port()
    gui = LinkEndpoint(0, game_port)
    cmd = [sys.executable, str(REPO / "run_game.py"), "--headless", "--mute", "--autopilot", "--autostart",
           "--minutes", "0.25", "--calibration-s", "2", "--first-drift", "1.0", "--drift-interval", "1.0", "2.0",
           "--seed", "7", "--out", str(tmp_path), "--game-port", str(game_port), "--gui-port", str(gui.port),
           "--max-runtime", "80"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    seen, events, alert_sent, ack = [], [], False, None
    deadline = time.monotonic() + 90
    last_status = 0.0
    try:
        while time.monotonic() < deadline:
            for m in gui.poll():
                seen.append(m["type"])
                if m["type"] == "event":
                    events.append(m["event"])
                    if m["event"] == "response" and not alert_sent:
                        gui.send("alert", score=72, test=False, reason="sustained")
                        alert_sent = True
                elif m["type"] == "alert_ack":
                    ack = m
            if time.monotonic() - last_status > 0.3:
                gui.send("status", streaming=True, recording=True, signal="8/8 channels good", phase="monitoring", score=4)
                last_status = time.monotonic()
            if "bye" in seen:
                break
            time.sleep(0.02)
        out = proc.communicate(timeout=20)[0].decode(errors="replace")
    finally:
        if proc.poll() is None:
            proc.kill()
        gui.close()
    assert proc.returncode == 0, out
    assert "hello" in seen and "calibrate" in seen and "bye" in seen
    for e in ("calibration_eyes_open", "calibration_eyes_closed", "calibration_end", "game_start",
              "drift_onset", "response", "game_end"):
        assert e in events, (e, events)
    assert ack is not None and 1.0 < ack["latency"] < 3.0 and ack["test"] is False

    sess = next(tmp_path.glob("night_drive_*"))
    meta = json.loads((sess / "session.json").read_text())
    assert meta["summary"]["drifts"] >= 3 and len(meta["summary"]["alerts"]) == 1
    rows = list(csv.DictReader(open(sess / "events.csv")))
    assert any(r["event"] == "alert_ack" for r in rows) and any(r["event"] == "response" for r in rows)
    tel = list(csv.DictReader(open(sess / "telemetry.csv")))
    assert len(tel) > 100  # 10 Hz while driving
