"""15-second demo video from the final run: Night Drive (reconstructed from its logs) + the live EEG.

Everything shown is real data from 7 Nov 2026, synced on the host clock: the EEG recording,
the game's state changes, events and 10 Hz telemetry. The game view is re-rendered with the
game's own Scene code from the logged lateral position and steering.

Run: PYTHONPATH=src python notebooks/final/make_video.py
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import signal

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame as pg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from neuropawn_gui.dsp import apparent_line_hz  # noqa: E402
from neuropawn_gui.game.app import Scene  # noqa: E402
from neuropawn_gui.review import load_recording  # noqa: E402

SESSION = REPO / "data/game_sessions/night_drive_2026-11-07_20-26-32"
RECORDING = REPO / "data/recordings/neuropawn_knight_2026-11-07_20-38-03.csv"
OUT = REPO / "presentation/video/night_drive_eeg_demo.mp4"
W, H, FPS = 1920, 1080, 30
START = dt.datetime(2026, 11, 7, 20, 39, 15, 600000).timestamp()
DURATION = 15.0
EEG_WINDOW = 6.0
CH = [(0, "F3"), (1, "F4"), (2, "C3"), (3, "C4"), (4, "P3"), (5, "P4"), (6, "O1")]
COLORS = [(129, 129, 129), (124, 75, 141), (54, 87, 158), (49, 113, 89), (221, 178, 13), (253, 94, 52), (224, 56, 45)]
BG, PANEL, INK, SOFT, ORANGE, BLUE = (14, 22, 40), (22, 33, 58), (240, 238, 232), (170, 182, 204), (240, 138, 93), (120, 160, 230)

# ------------------------------------------------------------------ data
rec = load_recording(RECORDING)
fs = rec.fs
ts = rec.col("timestamp_unix")
x = rec.eeg - rec.eeg.mean(1, keepdims=True)
f0 = apparent_line_hz(60, fs, rec.fs_true)
b, a = signal.iirnotch(f0, 30, fs=fs)
disp = signal.sosfiltfilt(np.vstack([signal.butter(4, [1, 40], "bandpass", fs=fs, output="sos"), signal.tf2sos(b, a)]), x, axis=1)
alpha_sig = signal.sosfiltfilt(signal.butter(4, [8, 13], "bandpass", fs=fs, output="sos"), x, axis=1)
theta_sig = signal.sosfiltfilt(signal.butter(4, [4, 8], "bandpass", fs=fs, output="sos"), x, axis=1)

ev = list(csv.DictReader(open(SESSION / "events.csv")))
states = [(float(r["unix_time"]), r["detail"]) for r in ev if r["event"] == "state"]
st = {name: t for t, name in states}
TONE, COUNT, DRIVE = st["calibration_eyes_closed"] + 30.0, st["countdown"], st["driving"]
tel = list(csv.DictReader(open(SESSION / "telemetry.csv")))
tel_t = np.array([float(r["unix_time"]) for r in tel])
tel_x = np.array([float(r["x_m"]) for r in tel])
tel_s = np.array([float(r["steer"]) for r in tel])
events = []
for r in ev:
    u = float(r["unix_time"])
    if START - 10 <= u <= START + DURATION + 1 and r["event"] in ("drift_onset", "response", "lane_departure"):
        if r["event"] == "response":
            lab = f"driver corrects (RT {float(r['rt_s']):.2f} s)"
        elif r["event"] == "drift_onset":
            lab = "drift starts " + ("right" if r["dir"] == "1" else "left")
        else:
            lab = "wheels cross the line"
        events.append((u, r["event"], lab))
marks = [(COUNT, "tone · eyes open"), (DRIVE, "drive starts")] + [(u, lab) for u, e, lab in events if e != "lane_departure"]


def band_ratio(t, win=2.0):
    m = (ts > t - win) & (ts <= t)
    if m.sum() < 50:
        return None, None
    al = np.sqrt((alpha_sig[[4, 5, 6]][:, m] ** 2).mean())
    th = np.sqrt((theta_sig[[2, 3]][:, m] ** 2).mean())
    return th ** 2 / al ** 2, al


# ------------------------------------------------------------------ drawing helpers
pg.init()
pg.display.set_mode((1, 1))
screen = pg.Surface((W, H))
GW, GH = 1120, 630
game = pg.Surface((GW, GH))
scene = Scene(pg, (GW, GH))
_fonts: dict = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = pg.font.SysFont("helveticaneue,helvetica,arial", size, bold=bold)
    return _fonts[key]


def text(surf, s, pos, size, color=INK, bold=False, anchor="topleft"):
    img = font(size, bold).render(s, True, color)
    r = img.get_rect(**{anchor: pos})
    surf.blit(img, r)
    return r


def game_view(t, steer_vis):
    if t < COUNT:  # eyes-closed half of the baseline
        game.fill((4, 4, 7))
        text(game, "Now close your eyes and relax.", (GW // 2, int(GH * 0.40)), 40, (235, 235, 235), anchor="center")
        text(game, "Open them when you hear the tone.", (GW // 2, int(GH * 0.50)), 30, (130, 138, 150), anchor="center")
        text(game, f"{max(0, COUNT - t):4.0f} s", (GW // 2, int(GH * 0.88)), 24, (70, 75, 85), anchor="center")
        return
    xm = float(np.interp(t, tel_t, tel_x)) if t >= DRIVE else 0.0
    dist = max(0.0, t - DRIVE) * 100 / 3.6
    scene.draw(game, xm, dist, steer_vis, t)
    if t < DRIVE:
        n = max(1, math.ceil(3 - (t - COUNT)))
        text(game, str(n), (GW // 2, int(GH * 0.30)), 150, (235, 235, 235), bold=True, anchor="center")


def meter(surf, x0, y0, w, label, value_txt, frac, thresh_frac=None, color=ORANGE, note=""):
    text(surf, label, (x0, y0), 24, SOFT, bold=True)
    text(surf, value_txt, (x0 + w, y0 - 6), 34, INK, bold=True, anchor="topright")
    pg.draw.rect(surf, (40, 52, 82), (x0, y0 + 40, w, 22), border_radius=11)
    pg.draw.rect(surf, color, (x0, y0 + 40, max(8, int(w * min(1, max(0, frac)))), 22), border_radius=11)
    if thresh_frac is not None:
        tx = x0 + int(w * thresh_frac)
        pg.draw.line(surf, (255, 90, 80), (tx, y0 + 34), (tx, y0 + 68), 3)
    if note:
        text(surf, note, (x0, y0 + 70), 20, SOFT)


def caption_for(t):
    if t < COUNT - 0.2:
        return "Eyes closed: 10 Hz alpha waves over the back of the head (P3, P4, O1)"
    if t < COUNT + 2.0:
        return "Tone, eyes open: the alpha disappears and theta/alpha jumps"
    if t < DRIVE:
        return "Vision blocks alpha: the signature of an awake brain"
    drift = next((u for u, e, _ in events if e == "drift_onset" and u <= t), None)
    resp = next((u for u, e, _ in events if e == "response" and u <= t), None)
    if resp:
        return "Driver steers back; every game event is written into the EEG recording"
    if drift:
        return "The car starts to drift: the driver has to notice and correct it"
    return "Monotonous night drive begins: constant speed, empty road"


# ------------------------------------------------------------------ render
OUT.parent.mkdir(parents=True, exist_ok=True)
ff = "/opt/homebrew/bin/ffmpeg" if Path("/opt/homebrew/bin/ffmpeg").exists() else "ffmpeg"
proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
                         "-movflags", "+faststart", str(OUT)], stdin=subprocess.PIPE)
steer_vis = 0.0
n_frames = int(DURATION * FPS)
for i in range(n_frames):
    t = START + i / FPS
    steer = float(np.interp(t, tel_t, tel_s)) if t >= DRIVE else 0.0
    steer_vis += (steer - steer_vis) * 0.25
    screen.fill(BG)
    # header
    text(screen, "NeuroPawn EEG × Night Drive", (40, 22), 34, INK, bold=True)
    text(screen, "Real data from our final run · 7 Nov 2026 · NeuroPawn Knight @ 125.6 Hz · EEG below: 1–40 Hz, last 6 s, ±40 µV per channel",
         (40, 64), 22, SOFT)
    text(screen, dt.datetime.fromtimestamp(t).strftime("%H:%M:%S.") + f"{int((t % 1) * 10)}", (W - 40, 26), 34, ORANGE, bold=True, anchor="topright")
    # game
    game_view(t, steer_vis)
    screen.blit(game, (40, 104))
    pg.draw.rect(screen, (60, 72, 100), (40, 104, GW, GH), 2)
    cap = caption_for(t)
    shade = pg.Surface((GW, 58), pg.SRCALPHA)
    shade.fill((0, 0, 0, 170))
    screen.blit(shade, (40, 104 + GH - 58))
    text(screen, cap, (40 + GW // 2, 104 + GH - 29), 26, (255, 255, 255), bold=True, anchor="center")
    # right panel
    px, py, pw = 1200, 104, 680
    pg.draw.rect(screen, PANEL, (px, py, pw, GH), border_radius=14)
    phase = "EYES CLOSED (baseline)" if t < COUNT else ("EYES OPEN (countdown)" if t < DRIVE else "DRIVING")
    text(screen, "STATE", (px + 32, py + 26), 22, SOFT, bold=True)
    text(screen, phase, (px + 32, py + 54), 40, ORANGE if t < COUNT else BLUE, bold=True)
    ratio, alpha_amp = band_ratio(t)
    if ratio is not None:
        lf = (math.log10(ratio) + 1.3) / 2.3  # log scale 0.05 .. 10
        meter(screen, px + 32, py + 140, pw - 64, "THETA / ALPHA  (live, 2 s)", f"{ratio:.2f}", lf, (math.log10(1.0) + 1.3) / 2.3,
              ORANGE if ratio < 1 else BLUE, "red line = 1.0: below it alpha dominates (eyes closed, relaxed)")
        meter(screen, px + 32, py + 260, pw - 64, "ALPHA AMPLITUDE  P3 / P4 / O1", f"{alpha_amp:.1f} µV", alpha_amp / 14.0,
              color=(253, 94, 52))
    det = "calibrating baseline (eyes open, then closed)" if t < COUNT else "baseline set · monitoring · no pre-sleep pattern"
    text(screen, "DROWSINESS DETECTOR", (px + 32, py + 380), 22, SOFT, bold=True)
    text(screen, det, (px + 32, py + 410), 24, INK)
    text(screen, "EVENTS", (px + 32, py + 470), 22, SOFT, bold=True)
    log = [(COUNT, "tone: eyes open"), (DRIVE, "drive starts")] + [(u, lab) for u, _, lab in events]
    y = py + 500
    for u, lab in [e for e in log if e[0] <= t][-4:]:
        text(screen, dt.datetime.fromtimestamp(u).strftime("%H:%M:%S") + f".{int((u % 1) * 10)}  " + lab, (px + 32, y), 22, INK)
        y += 30
    # EEG traces
    ex, ey, ew, eh = 120, 776, W - 160, 280
    pg.draw.rect(screen, (10, 16, 30), (40, ey - 12, W - 80, eh + 24), border_radius=10)
    m = (ts > t - EEG_WINDOW) & (ts <= t)
    tt = ts[m]
    row = eh / len(CH)
    for k, (ci, name) in enumerate(CH):
        yc = ey + row * (k + 0.5)
        text(screen, name, (60, yc), 22, COLORS[k], bold=True, anchor="midleft")
        if tt.size > 2:
            xs = ex + (tt - (t - EEG_WINDOW)) / EEG_WINDOW * ew
            ys = yc - np.clip(disp[ci, m], -40, 40) * (row * 0.5 / 40.0) * 1.6
            ys = np.clip(ys, yc - row * 0.55, yc + row * 0.55)
            pg.draw.lines(screen, COLORS[k], False, list(zip(xs, ys)), 2)
    for u, lab in marks:
        if t - EEG_WINDOW <= u <= t:
            mx = ex + (u - (t - EEG_WINDOW)) / EEG_WINDOW * ew
            pg.draw.line(screen, (235, 235, 235), (mx, ey - 6), (mx, ey + eh + 6), 2)
            right = mx > ex + ew - 260  # near the right edge: put the label on the left of the line
            text(screen, lab, (mx - 6 if right else mx + 6, ey - 34), 20, (235, 235, 235), bold=True,
                 anchor="topright" if right else "topleft")
    proc.stdin.write(pg.image.tostring(screen, "RGB"))
    if i in (45, 150, 360, 445):
        pg.image.save(screen, str(OUT.with_name(f"frame_{i:03d}.png")))
proc.stdin.close()
proc.wait()
print("saved", OUT, "| frames", n_frames)
