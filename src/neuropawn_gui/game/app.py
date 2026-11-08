"""Night Drive: a deliberately monotonous highway drive for drowsiness research.

Why it looks and sounds like this (the goal is to *let* people get drowsy):
* Night, an empty straight highway, constant 100 km/h, evenly spaced reflector posts,
  and a steady engine/road drone. Road monotony and low stimulation are what produce
  "highway hypnosis" in driving-fatigue studies.
* No score, no reaction-time feedback, no music: feedback and variety raise arousal.
* Every 8-14 s the car slowly drifts out of the lane and the driver steers it back
  (the Sustained-Attention Driving Task used in EEG drowsiness research). Reaction
  time, lapses, lane departures, and weaving (SDLP) are logged as behavioural ground
  truth for the EEG detector.

The EEG GUI runs in the background with the headset. When its detector sees the
pre-sleep pattern it sends an alert over the local link, and the game is interrupted
by a full-screen alarm until the driver acknowledges it.

Run:  python run_game.py [--minutes 30] [--fullscreen]
Keys: LEFT/RIGHT (or A/D) steer · SPACE start/continue · ESC pause · F fullscreen ·
      TAB experimenter overlay · K skip calibration (intro screen)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..link import GAME_PORT, GUI_PORT, LinkEndpoint
from .task import DriveTask, TaskConfig

REPO = Path(__file__).resolve().parents[3]
SESSIONS_DIR = REPO / "data" / "game_sessions"

INTRO, CAL_OPEN, CAL_CLOSED, COUNTDOWN, DRIVING, ALERT, PAUSED, END = (
    "intro", "calibration_eyes_open", "calibration_eyes_closed", "countdown", "driving", "alert", "paused", "ended")

WHITE = (235, 235, 235)
DIM = (130, 138, 150)
AMBER = (230, 160, 50)


# =============================================================================== audio
class Sounds:
    """All audio is generated at startup (no asset files)."""

    def __init__(self, pg, enabled: bool = True):
        self.ok = False
        self.drone = self.alarm = self.tone = self.chime = None
        self._drone_ch = self._alarm_ch = None
        if not enabled:
            return
        try:
            init = pg.mixer.get_init()
            if not init:
                return
            self.rate, _size, self.channels = init
            self.pg = pg
            self.drone = self._make(self._drone(6.0), 0.32)
            self.alarm = self._make(self._alarm(), 1.0)
            self.tone = self._make(self._bell([660, 990], 0.9), 0.6)
            self.chime = self._make(np.concatenate([self._bell([523], 0.25), self._bell([784], 0.5)]), 0.35)
            self.ok = True
        except Exception as e:  # audio is optional
            print(f"[night drive] audio disabled: {e}", file=sys.stderr)

    def _make(self, mono: np.ndarray, volume: float):
        a = np.clip(mono * 32767 * 0.9, -32767, 32767).astype(np.int16)
        arr = np.repeat(a[:, None], self.channels, axis=1) if self.channels > 1 else a
        snd = self.pg.sndarray.make_sound(np.ascontiguousarray(arr))
        snd.set_volume(volume)
        return snd

    def _drone(self, secs: float) -> np.ndarray:
        """Seamless loop: low engine hum + filtered road noise."""
        from scipy import signal
        n, fade = int(secs * self.rate), int(0.4 * self.rate)
        t = np.arange(n) / self.rate
        rng = np.random.default_rng(7)
        noise = signal.sosfilt(signal.butter(2, [30, 220], "bandpass", fs=self.rate, output="sos"),
                               rng.normal(0, 1, n + fade))
        loop = noise[:n].copy()  # crossfade the tail into the head so the loop has no click
        ramp = np.linspace(0, 1, fade)
        loop[:fade] = noise[:fade] * ramp + noise[n:n + fade] * (1 - ramp)
        loop /= np.abs(loop).max()
        wobble = 1 + 0.06 * np.sin(2 * np.pi * 0.5 * t)                 # 3 cycles per loop
        hum = (np.sin(2 * np.pi * 48 * t) + 0.5 * np.sin(2 * np.pi * 96 * t)
               + 0.2 * np.sin(2 * np.pi * 144 * t)) * wobble            # integer cycles per loop
        out = 0.55 * loop + 0.45 * hum / 1.7
        return out / np.abs(out).max()

    def _alarm(self) -> np.ndarray:
        beeps = []
        for k in range(4):
            f = 880 if k % 2 == 0 else 1175
            t = np.arange(int(0.22 * self.rate)) / self.rate
            env = np.minimum(1, t / 0.005) * np.minimum(1, (t[-1] - t) / 0.01 + 1e-9)
            tone = np.sign(np.sin(2 * np.pi * f * t)) * 0.5 + np.sin(2 * np.pi * 2 * f * t) * 0.3
            beeps += [tone * env, np.zeros(int(0.08 * self.rate))]
        beeps.append(np.zeros(int(0.5 * self.rate)))
        return np.concatenate(beeps)

    def _bell(self, freqs, secs) -> np.ndarray:
        t = np.arange(int(secs * self.rate)) / self.rate
        x = sum(np.sin(2 * np.pi * f * t) for f in freqs) * np.exp(-3.5 * t / secs)
        return x / max(1e-9, np.abs(x).max())

    def start_drone(self):
        if self.ok and self._drone_ch is None:
            self._drone_ch = self.drone.play(loops=-1, fade_ms=1500)

    def pause_drone(self, paused: bool):
        if self._drone_ch is not None:
            self._drone_ch.pause() if paused else self._drone_ch.unpause()

    def start_alarm(self):
        if self.ok:
            self._alarm_ch = self.alarm.play(loops=-1)

    def stop_alarm(self):
        if self.alarm is not None:
            self.alarm.stop()

    def play(self, which: str):
        snd = getattr(self, which, None)
        if self.ok and snd is not None:
            snd.play()


# =============================================================================== graphics
class Scene:
    """Pseudo-3D night highway seen from the driver's seat (car in the right lane)."""

    # (X0, X1, base RGB) world bands in metres, lane centre at X = 0
    BANDS = [
        (-40.0, -7.6, (9, 12, 10)),     # left verge
        (-7.6, -5.6, (21, 22, 24)),     # left shoulder
        (-5.6, 1.8, (34, 36, 40)),      # asphalt (two lanes)
        (1.8, 4.4, (23, 24, 26)),       # right shoulder
        (4.4, 40.0, (9, 12, 10)),       # right verge
    ]
    LINES = [(-5.6, -5.45, (150, 128, 60)), (1.72, 1.87, (205, 205, 200))]   # yellow left edge, white right edge
    DIVIDER_X, DASH_LEN, DASH_PERIOD = -1.8, 3.0, 12.0
    POST_SPACING = 40.0
    POSTS_X = (5.0, -8.0)

    def __init__(self, pg, size):
        self.pg = pg
        self.resize(size)

    def resize(self, size):
        self.W, self.H = size
        self.horizon = int(self.H * 0.44)
        self.dash_top = int(self.H * 0.80)
        self.f = self.H * 1.0
        self.cam_h = 1.2
        self.z_near = self.cam_h * self.f / (self.dash_top - self.horizon)
        self.z = np.geomspace(self.z_near, 320.0, 72)
        self.sky = self._make_sky()
        self.dash = self._make_dashboard()

    @staticmethod
    def light(z: float) -> float:
        """Headlights: bright near the car, fading into the dark."""
        return 0.10 + 0.90 * math.exp(-z / 30.0)

    def _make_sky(self):
        pg = self.pg
        s = pg.Surface((self.W, self.horizon + 2))
        top, bot = np.array([3, 5, 10]), np.array([15, 19, 32])
        for y in range(self.horizon + 2):
            c = top + (bot - top) * (y / max(1, self.horizon)) ** 1.6
            pg.draw.line(s, c.astype(int).tolist(), (0, y), (self.W, y))
        rng = random.Random(3)
        for _ in range(int(self.W * self.horizon / 9000)):
            b = rng.randint(40, 95)
            s.set_at((rng.randrange(self.W), rng.randrange(int(self.horizon * 0.8))), (b, b, b + 10))
        return s

    def _make_dashboard(self):
        pg = self.pg
        s = pg.Surface((self.W, self.H), pg.SRCALPHA)
        pts = [(0, self.H), (0, self.dash_top + 18)]
        for i in range(41):
            x = self.W * i / 40
            pts.append((x, self.dash_top + 18 - 22 * math.sin(math.pi * i / 40)))
        pts += [(self.W, self.dash_top + 18), (self.W, self.H)]
        pg.draw.polygon(s, (8, 8, 10, 255), pts)
        pg.draw.lines(s, (30, 32, 38, 255), False, pts[1:-1], 2)
        return s

    def sx(self, X, z, cam_x):
        return self.W / 2 + (X - cam_x) * self.f / z

    def sy(self, z, h=0.0, bob=0.0):
        return self.horizon + bob + (self.cam_h - h) * self.f / z

    def draw(self, screen, cam_x: float, distance: float, steer_vis: float, t: float, speed_kmh: int = 100):
        pg = self.pg
        bob = 0.8 * math.sin(2 * math.pi * 1.1 * t)
        screen.fill((5, 7, 7))  # the far road meets the sky in a thin strip that slices don't cover
        screen.blit(self.sky, (0, 0))
        z = self.z
        # ground slices, far to near
        for i in range(len(z) - 1, 0, -1):
            za, zb = z[i - 1], z[i]          # near, far
            ya, yb = self.sy(za, bob=bob), self.sy(zb, bob=bob)
            lt = self.light((za + zb) / 2)
            for X0, X1, base in self.BANDS + self.LINES:
                col = [int(c * lt) for c in base]
                pg.draw.polygon(screen, col, [(self.sx(X0, za, cam_x), ya), (self.sx(X1, za, cam_x), ya),
                                              (self.sx(X1, zb, cam_x), yb), (self.sx(X0, zb, cam_x), yb)])
        # dashed lane divider
        s = distance % self.DASH_PERIOD
        for k in range(1, 14):
            z0 = k * self.DASH_PERIOD - s
            z1 = z0 + self.DASH_LEN
            z0c, z1c = max(z0, self.z_near), min(z1, 160.0)
            if z1c <= z0c:
                continue
            lt = self.light((z0c + z1c) / 2)
            col = [int(205 * lt)] * 3
            xa, xb = self.DIVIDER_X - 0.075, self.DIVIDER_X + 0.075
            pg.draw.polygon(screen, col, [(self.sx(xa, z0c, cam_x), self.sy(z0c, bob=bob)),
                                          (self.sx(xb, z0c, cam_x), self.sy(z0c, bob=bob)),
                                          (self.sx(xb, z1c, cam_x), self.sy(z1c, bob=bob)),
                                          (self.sx(xa, z1c, cam_x), self.sy(z1c, bob=bob))])
        # reflector posts (far to near)
        s = distance % self.POST_SPACING
        for k in range(9, 0, -1):
            zp = k * self.POST_SPACING - s
            if zp < self.z_near * 0.9:
                continue
            for X in self.POSTS_X:
                x = self.sx(X, zp, cam_x)
                yg, yt = self.sy(zp, 0.0, bob), self.sy(zp, 1.05, bob)
                w = max(1, int(0.12 * self.f / zp))
                lt = self.light(zp)
                pg.draw.rect(screen, [int(c * lt) for c in (150, 150, 145)], (int(x - w / 2), int(yt), w, max(1, int(yg - yt))))
                glow = max(lt, 0.30)
                rh = max(1, int(0.14 * self.f / zp))
                pg.draw.rect(screen, [int(c * glow) for c in AMBER], (int(x - w / 2), int(yt + rh * 0.4), w, rh))
        # dashboard + steering wheel
        screen.blit(self.dash, (0, 0))
        self._wheel(screen, steer_vis)
        font = _font(pg, int(self.H * 0.035))
        txt = font.render(f"{speed_kmh} km/h", True, (70, 140, 150))
        screen.blit(txt, (int(self.W * 0.78), int(self.dash_top + self.H * 0.07)))

    def _wheel(self, screen, steer_vis: float):
        pg = self.pg
        cx, cy, r = self.W / 2, self.H * 1.12, self.H * 0.36
        ang = math.radians(32 * steer_vis)
        pg.draw.circle(screen, (20, 20, 24), (int(cx), int(cy)), int(r), max(6, int(self.H * 0.035)))
        for base in (-math.pi / 2 - 1.05, -math.pi / 2 + 1.05):  # spokes
            a = base + ang
            pg.draw.line(screen, (18, 18, 22), (cx, cy), (cx + r * math.cos(a), cy + r * math.sin(a)), max(4, int(self.H * 0.02)))
        a = -math.pi / 2 + ang  # top-centre mark makes the rotation visible
        pg.draw.line(screen, (60, 62, 70), (cx + (r - 10) * math.cos(a), cy + (r - 10) * math.sin(a)),
                     (cx + (r + 10) * math.cos(a), cy + (r + 10) * math.sin(a)), 5)


_FONTS: dict = {}


def _font(pg, size: int):
    if size not in _FONTS:
        _FONTS[size] = pg.font.Font(None, max(10, size))
    return _FONTS[size]


def _center_text(pg, screen, text, size, y, color=WHITE):
    surf = _font(pg, size).render(text, True, color)
    screen.blit(surf, (screen.get_width() / 2 - surf.get_width() / 2, y))


# =============================================================================== game
class NightDrive:
    def __init__(self, args):
        self.args = args
        if args.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame as pg
        self.pg = pg
        pg.mixer.pre_init(44100, -16, 2, 1024)
        pg.init()
        pg.display.set_caption("Night Drive · NeuroPawn EEG")
        self.fullscreen = bool(args.fullscreen)
        self._set_mode()
        self.scene = Scene(pg, self.screen.get_size())
        self.sounds = Sounds(pg, enabled=not args.mute)
        self.clock = pg.time.Clock()
        self.joy = None
        if pg.joystick.get_count() > 0:
            self.joy = pg.joystick.Joystick(0)
            self.joy.init()

        cfg = TaskConfig()
        if args.first_drift is not None:
            cfg.first_drift_s = args.first_drift
        if args.drift_interval:
            cfg.interval_s = tuple(args.drift_interval)
        self.task = DriveTask(cfg, seed=args.seed)
        self.rng = random.Random(args.seed)

        self.state = INTRO
        self.state_t0 = time.monotonic()
        self.prev_state = None
        self.distance = 0.0
        self.steer_vis = 0.0
        self.overlay = False
        self.alert_t0 = None
        self.alert_info: dict = {}
        self.alerts: list[dict] = []
        self.countdown_s = 3.0
        self.drive_s = args.minutes * 60.0
        self.calibration_s = args.calibration_s
        self.skip_calibration = False

        # link to the GUI
        self.link = None
        self.link_error = None
        try:
            self.link = LinkEndpoint(args.game_port, args.gui_port)
        except OSError as e:
            self.link_error = f"link unavailable ({e})"
        self.gui = {}
        self.gui_seen = -1e9
        self._last_hello = self._last_hb = -1e9

        # session logging
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_id = f"night_drive_{stamp}"
        self.dir = Path(args.out) / self.session_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self._events_f = open(self.dir / "events.csv", "w", newline="", encoding="utf-8")
        self.events_csv = csv.writer(self._events_f)
        self.events_csv.writerow(["unix_time", "drive_time_s", "event", "trial", "dir", "rt_s", "detail"])
        self._tel_f = open(self.dir / "telemetry.csv", "w", newline="", encoding="utf-8")
        self.tel_csv = csv.writer(self._tel_f)
        self.tel_csv.writerow(["unix_time", "drive_time_s", "x_m", "steer", "drifting"])
        self._tel_next = 0.0
        self.meta = {
            "session": self.session_id, "started": datetime.now().astimezone().isoformat(),
            "minutes": args.minutes, "task": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
            "autopilot": bool(args.autopilot), "seed": args.seed,
            "notes": "x_m = lateral offset from lane centre (+right). Times in events.csv are host Unix "
                     "time, the same clock as timestamp_unix in the EEG recording.",
        }
        self._write_meta()
        self._autopilot_rt = None
        self.send("hello", session=self.session_id, pid=os.getpid(), minutes=args.minutes)

    # ------------------------------------------------------------------ plumbing
    def _set_mode(self):
        pg = self.pg
        if self.fullscreen:
            self.screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
        else:
            self.screen = pg.display.set_mode(tuple(self.args.size))
        if hasattr(self, "scene"):
            self.scene.resize(self.screen.get_size())

    def _write_meta(self):
        (self.dir / "session.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def send(self, type_, **fields):
        if self.link is not None:
            self.link.send(type_, **fields)

    @property
    def gui_connected(self) -> bool:
        return time.monotonic() - self.gui_seen < 3.0

    @property
    def eeg_streaming(self) -> bool:
        return self.gui_connected and bool(self.gui.get("streaming"))

    def log(self, event: str, trial=None, direction=None, rt=None, detail=""):
        self.events_csv.writerow([f"{time.time():.6f}", round(self.task.t, 4), event, trial, direction, rt, detail])
        self._events_f.flush()

    def emit(self, ev: dict):
        """Task event -> CSV + GUI (which turns it into an EEG marker)."""
        name = ev["event"]
        self.log(name, ev.get("trial"), ev.get("dir"), ev.get("rt"), ev.get("side", ev.get("duration", "")))
        self.send("event", event=name, **{k: v for k, v in ev.items() if k != "event"})

    def set_state(self, state: str):
        self.prev_state, self.state = self.state, state
        self.state_t0 = time.monotonic()
        self.send("state", state=state)
        self.log("state", detail=state)
        if state == CAL_OPEN:
            self.send("calibrate")
            self.send("event", event="calibration_eyes_open")
        elif state == CAL_CLOSED:
            self.send("event", event="calibration_eyes_closed")
        elif state == DRIVING and self.prev_state == COUNTDOWN and not getattr(self, "_started", False):
            self._started = True
            self.send("event", event="game_start")
            self.sounds.start_drone()
        elif state == PAUSED:
            self.send("event", event="paused")
            self.sounds.pause_drone(True)
        elif state == DRIVING and self.prev_state == COUNTDOWN:
            self.send("event", event="resumed")
            self.sounds.pause_drone(False)

    def elapsed(self) -> float:
        return time.monotonic() - self.state_t0

    # ------------------------------------------------------------------ link
    def poll_link(self):
        if self.link is None:
            return
        now = time.monotonic()
        for msg in self.link.poll():
            kind = msg.get("type")
            if kind in ("status", "hello_ack"):
                self.gui.update({k: v for k, v in msg.items() if k not in ("v", "type", "t")})
                self.gui_seen = now
            elif kind == "alert":
                self.gui_seen = now
                self.on_alert(msg)
        if not self.gui_connected and now - self._last_hello > 2.0:
            self.send("hello", session=self.session_id, pid=os.getpid(), minutes=self.args.minutes)
            self._last_hello = now
        if now - self._last_hb > 1.0:
            self.send("heartbeat", state=self.state, stats=self.task.stats(), alerts=len(self.alerts))
            self._last_hb = now

    def on_alert(self, msg: dict):
        test = bool(msg.get("test"))
        score = msg.get("score")
        if self.state in (DRIVING, COUNTDOWN, PAUSED):
            self.alert_info = {"test": test, "score": score, "reason": msg.get("reason", "")}
            self.alert_t0 = time.monotonic()
            self.log("alert_received", detail=json.dumps(self.alert_info))
            self.set_state(ALERT)
            self.sounds.pause_drone(True)
            self.sounds.start_alarm()
        else:
            self.log("alert_ignored", detail=f"state={self.state}")

    def acknowledge_alert(self):
        latency = time.monotonic() - (self.alert_t0 or time.monotonic())
        self.sounds.stop_alarm()
        rec = {**self.alert_info, "ack_latency_s": round(latency, 3), "drive_time_s": round(self.task.t, 2)}
        self.alerts.append(rec)
        self.log("alert_ack", rt=round(latency, 4), detail=json.dumps(self.alert_info))
        self.send("alert_ack", latency=round(latency, 3), test=self.alert_info.get("test", False))
        self.set_state(COUNTDOWN)

    # ------------------------------------------------------------------ input
    def steering(self) -> float:
        pg = self.pg
        if self.args.autopilot:
            return self._autopilot()
        keys = pg.key.get_pressed()
        s = float(keys[pg.K_RIGHT] or keys[pg.K_d]) - float(keys[pg.K_LEFT] or keys[pg.K_a])
        if self.joy is not None and s == 0:
            ax = self.joy.get_axis(0)
            s = 0.0 if abs(ax) < 0.12 else ax
        return s

    def _autopilot(self) -> float:
        """Test driver: reacts after 0.5-1.1 s, steers back to the centre."""
        tr = self.task.trial
        if tr is not None and tr.rt is None:
            if self._autopilot_rt is None:
                self._autopilot_rt = self.rng.uniform(0.5, 1.1)
            return float(-tr.direction) if self.task.t - tr.onset >= self._autopilot_rt else 0.0
        self._autopilot_rt = None
        x = self.task.x
        return 0.0 if abs(x) < 0.08 else (-1.0 if x > 0 else 1.0)

    def handle_events(self) -> bool:
        pg = self.pg
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                return False
            if ev.type != pg.KEYDOWN:
                continue
            k = ev.key
            if k == pg.K_f:
                self.fullscreen = not self.fullscreen
                self._set_mode()
            elif k == pg.K_TAB:
                self.overlay = not self.overlay
            elif self.state == INTRO:
                if k == pg.K_ESCAPE:
                    return False
                if k == pg.K_k:
                    self.skip_calibration = not self.skip_calibration
                if k == pg.K_SPACE:
                    self.start()
            elif self.state in (DRIVING, COUNTDOWN) and k in (pg.K_ESCAPE, pg.K_p):
                self.set_state(PAUSED)
            elif self.state == PAUSED:
                if k == pg.K_SPACE:
                    self.set_state(COUNTDOWN)
                elif k == pg.K_q:
                    self.finish()
            elif self.state == ALERT and k in (pg.K_SPACE, pg.K_RETURN) and self.elapsed() > 0.8:
                self.acknowledge_alert()
            elif self.state in (CAL_OPEN, CAL_CLOSED) and k == pg.K_ESCAPE:
                self.set_state(INTRO)
            elif self.state == END and k in (pg.K_ESCAPE, pg.K_q, pg.K_SPACE, pg.K_RETURN):
                return False
        return True

    def start(self):
        self.sounds.play("chime")
        if self.eeg_streaming and not self.skip_calibration:
            self.set_state(CAL_OPEN)
        else:
            self.set_state(COUNTDOWN)

    def finish(self):
        self.sounds.stop_alarm()
        self.sounds.pause_drone(True)
        self.meta["ended"] = datetime.now().astimezone().isoformat()
        self.meta["summary"] = {**self.task.stats(), "alerts": self.alerts}
        self._write_meta()
        self.send("event", event="game_end")
        self.set_state(END)

    # ------------------------------------------------------------------ main loop
    def run(self) -> int:
        running = True
        try:
            while running:
                dt = min(self.clock.tick(60) / 1000.0, 0.1)
                self.poll_link()
                running = self.handle_events()
                self.update(dt)
                self.draw()
                self.pg.display.flip()
                if self.args.max_runtime and time.monotonic() - self._t_start() > self.args.max_runtime:
                    running = False
                if self.state == END and self.args.autopilot and self.elapsed() > 1.0:
                    running = False
        finally:
            self.close()
        return 0

    def _t_start(self):
        if not hasattr(self, "_t0"):
            self._t0 = time.monotonic()
        return self._t0

    def update(self, dt: float):
        st = self.state
        if st == INTRO and self.args.autostart and self.elapsed() > 0.5:
            self.start()
        elif st == CAL_OPEN and self.elapsed() >= self.calibration_s / 2:
            self.set_state(CAL_CLOSED)
        elif st == CAL_CLOSED and self.elapsed() >= self.calibration_s / 2:
            self.sounds.play("tone")
            self.send("event", event="calibration_end")
            self.set_state(COUNTDOWN)
        elif st == COUNTDOWN and self.elapsed() >= self.countdown_s:
            self.set_state(DRIVING)
        elif st == DRIVING:
            steer = self.steering()
            for ev in self.task.step(dt, steer):
                self.emit(ev)
            self.distance += dt * 100 / 3.6
            self.steer_vis += (steer - self.steer_vis) * min(1.0, dt * 8)
            if self.task.t >= self._tel_next:
                self.tel_csv.writerow([f"{time.time():.6f}", round(self.task.t, 3), round(self.task.x, 4),
                                       round(steer, 3), int(self.task.trial is not None and self.task.trial.rt is None)])
                self._tel_next = self.task.t + 0.1
            if self.task.t >= self.drive_s:
                self.finish()
        elif st == ALERT and self.args.autopilot and self.elapsed() > 1.2:
            self.acknowledge_alert()

    # ------------------------------------------------------------------ drawing
    def draw(self):
        pg, scr = self.pg, self.screen
        W, H = scr.get_size()
        t = time.monotonic()
        st = self.state
        if st in (CAL_OPEN, CAL_CLOSED):
            scr.fill((4, 4, 7))
            if st == CAL_OPEN:
                c = (W // 2, H // 2)
                pg.draw.line(scr, WHITE, (c[0] - 22, c[1]), (c[0] + 22, c[1]), 4)
                pg.draw.line(scr, WHITE, (c[0], c[1] - 22), (c[0], c[1] + 22), 4)
                _center_text(pg, scr, "Baseline: keep your eyes open and look at the cross", int(H * 0.045), H * 0.66, DIM)
            else:
                _center_text(pg, scr, "Now close your eyes and relax.", int(H * 0.06), H * 0.42, WHITE)
                _center_text(pg, scr, "Open them when you hear the tone.", int(H * 0.045), H * 0.52, DIM)
            left = max(0, self.calibration_s / 2 - self.elapsed())
            _center_text(pg, scr, f"{left:4.0f} s", int(H * 0.035), H * 0.9, (70, 75, 85))
            self._experimenter_overlay()
            return

        self.scene.draw(scr, self.task.x, self.distance, self.steer_vis, t)
        if st == INTRO:
            self._panel([
                ("NIGHT DRIVE", 0.085, WHITE),
                ("", 0.02, DIM),
                ("Keep the car in the middle of your lane.", 0.042, WHITE),
                ("Now and then it will start to drift. Steer it back with the LEFT / RIGHT arrow keys.", 0.034, DIM),
                ("Just drive calmly. There is no score.", 0.034, DIM),
                ("", 0.02, DIM),
                (self._link_line(), 0.03, (120, 200, 160) if self.eeg_streaming else AMBER),
                ("First a 1-minute baseline: 30 s eyes open, 30 s eyes closed." if self.eeg_streaming and not self.skip_calibration
                 else "Baseline skipped." if self.skip_calibration else "", 0.03, DIM),
                ("", 0.02, DIM),
                ("Press SPACE to start", 0.05, WHITE),
                ("ESC quit  ·  F fullscreen  ·  K skip baseline (experimenter)", 0.026, (90, 95, 105)),
            ])
        elif st == COUNTDOWN:
            n = max(1, math.ceil(self.countdown_s - self.elapsed()))
            _center_text(pg, scr, str(n), int(H * 0.2), H * 0.25, WHITE)
        elif st == PAUSED:
            self._panel([("PAUSED", 0.08, WHITE), ("SPACE continue   ·   Q end the drive", 0.04, DIM)])
        elif st == ALERT:
            self._alert_screen(t)
        elif st == END:
            s = self.task.stats()
            rt = "n/a" if s["median_rt_s"] is None else f"{s['median_rt_s']:.2f} s"
            self._panel([
                ("DRIVE COMPLETE", 0.075, WHITE),
                (f"{s['driving_s'] / 60:.1f} min  ·  {s['drifts']} drifts  ·  median reaction {rt}", 0.04, DIM),
                (f"{s['lapses']} lapses  ·  {s['lane_departures']} lane departures  ·  {len(self.alerts)} drowsiness alerts", 0.04, DIM),
                ("", 0.02, DIM),
                (f"Saved to data/game_sessions/{self.session_id}", 0.03, (90, 95, 105)),
                ("Press ESC to exit", 0.045, WHITE),
            ])
        self._experimenter_overlay()

    def _link_line(self) -> str:
        if self.link is None:
            return f"EEG: {self.link_error}"
        if not self.gui_connected:
            return "EEG: GUI not found. Practice mode, no drowsiness detection"
        if not self.gui.get("streaming"):
            return "EEG: GUI connected, but the headset isn't streaming. Start a session in the GUI"
        rec = "recording" if self.gui.get("recording") else "NOT recording"
        return f"EEG: connected · headset streaming · {rec} · signal {self.gui.get('signal', '?')}"

    def _panel(self, lines):
        pg, scr = self.pg, self.screen
        W, H = scr.get_size()
        shade = pg.Surface((W, H), pg.SRCALPHA)
        shade.fill((0, 0, 0, 175))
        scr.blit(shade, (0, 0))
        total = sum(int(H * s) * 1.25 for _, s, _ in lines)
        y = H * 0.42 - total / 2
        for text, size, col in lines:
            if text:
                _center_text(pg, scr, text, int(H * size), y, col)
            y += int(H * size) * 1.25

    def _alert_screen(self, t):
        pg, scr = self.pg, self.screen
        W, H = scr.get_size()
        on = int(t * 2.5) % 2 == 0
        scr.fill((175, 0, 22) if on else (95, 0, 12))
        test = self.alert_info.get("test")
        y = H * 0.18
        _center_text(pg, scr, "! DROWSINESS DETECTED !", int(H * 0.11), y, WHITE)
        _center_text(pg, scr, "Your brain activity shows the pattern of falling asleep.", int(H * 0.045), y + H * 0.17, WHITE)
        _center_text(pg, scr, "In a real car: pull over somewhere safe and take a break.", int(H * 0.045), y + H * 0.23, WHITE)
        if self.elapsed() > 0.8:
            _center_text(pg, scr, "Press SPACE to continue", int(H * 0.06), y + H * 0.42, WHITE)
        if test:
            _center_text(pg, scr, "(test alert from the experimenter)", int(H * 0.035), y + H * 0.55, (255, 210, 210))

    def _experimenter_overlay(self):
        if not self.overlay:
            return
        pg, scr = self.pg, self.screen
        s = self.task.stats()
        g = self.gui
        lines = [
            f"state {self.state}   drive {s['driving_s']:.0f}/{self.drive_s:.0f} s   x {self.task.x:+.2f} m",
            f"drifts {s['drifts']}  median RT {s['median_rt_s']}  lapses {s['lapses']}  departures {s['lane_departures']}  SDLP {s['sdlp_m']}",
            f"{self._link_line()}",
            f"drowsiness {g.get('score')}  phase {g.get('phase')}  alerts {len(self.alerts)}",
        ]
        f = _font(pg, 20)
        box = pg.Surface((scr.get_width(), 20 * len(lines) + 10), pg.SRCALPHA)
        box.fill((0, 0, 0, 160))
        scr.blit(box, (0, 0))
        for i, line in enumerate(lines):
            scr.blit(f.render(line, True, (160, 200, 170)), (8, 5 + 20 * i))

    def close(self):
        if self.state != END:
            self.meta["ended"] = datetime.now().astimezone().isoformat()
            self.meta["summary"] = {**self.task.stats(), "alerts": self.alerts, "aborted": True}
            self._write_meta()
            self.send("event", event="game_end")
        self.send("bye", session=self.session_id)
        for f in (self._events_f, self._tel_f):
            try:
                f.close()
            except OSError:
                pass
        if self.link is not None:
            self.link.close()
        self.pg.quit()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Night Drive: monotonous lane-keeping task for drowsiness research")
    ap.add_argument("--minutes", type=float, default=30.0, help="driving time (default 30)")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--out", type=Path, default=SESSIONS_DIR)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--calibration-s", type=float, default=60.0, help="baseline length (half eyes open, half closed)")
    # testing / demo aids
    ap.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--autopilot", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--autostart", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--first-drift", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--drift-interval", type=float, nargs=2, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--max-runtime", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--game-port", type=int, default=GAME_PORT, help=argparse.SUPPRESS)
    ap.add_argument("--gui-port", type=int, default=GUI_PORT, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    return NightDrive(args).run()
