"""Lane-keeping task logic (no graphics), modelled on the Sustained-Attention Driving Task.

In the SADT paradigm used in EEG drowsiness research, the car occasionally starts to
drift out of its lane and the driver steers it back. Slower responses, missed drifts
(lapses), lane departures, and wider weaving (SDLP) all track drowsiness.

Coordinates: ``x`` is the car's lateral offset from the lane centre in metres
(+ = right). Time only advances while driving (pauses and alerts don't count toward
reaction times).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class TaskConfig:
    lane_width_m: float = 3.6
    car_width_m: float = 1.8
    drift_mps: float = 0.55           # lateral drift speed during a drift event
    steer_mps: float = 1.2            # lateral speed at full steering input
    first_drift_s: float = 6.0
    interval_s: tuple[float, float] = (8.0, 14.0)   # quiet time between drifts
    lapse_s: float = 4.0              # no correction within this -> lapse
    recover_tol_m: float = 0.25       # back within this of the centre -> recovered
    unrecovered_s: float = 8.0        # give up waiting for recovery after the response
    max_offset_m: float = 3.2         # shoulder

    @property
    def departure_m(self) -> float:
        """Offset at which the wheels touch the lane line."""
        return (self.lane_width_m - self.car_width_m) / 2


@dataclass
class Trial:
    n: int
    direction: int
    onset: float
    rt: float | None = None
    lapse: bool = False
    departure: bool = False
    wrong_inputs: int = 0
    end: float | None = None
    outcome: str | None = None


@dataclass
class DriveTask:
    cfg: TaskConfig = field(default_factory=TaskConfig)
    seed: int | None = None

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.t = 0.0
        self.x = 0.0
        self.trial: Trial | None = None
        self.trials: list[Trial] = []
        self.next_onset = self.cfg.first_drift_s
        self._prev_dir = 0
        self._w = self._sx = self._sxx = 0.0   # time-weighted lateral position stats (SDLP)

    # ------------------------------------------------------------------ stepping
    def step(self, dt: float, steer: float) -> list[dict]:
        """Advance by ``dt`` seconds with steering input in [-1, 1]. Returns new events."""
        cfg = self.cfg
        events: list[dict] = []
        self.t += dt
        steer = max(-1.0, min(1.0, steer))
        sdir = 0 if abs(steer) < 0.5 else (1 if steer > 0 else -1)

        tr = self.trial
        if tr is None and self.t >= self.next_onset:
            tr = self.trial = Trial(len(self.trials) + 1, self.rng.choice((-1, 1)), self.t)
            events.append({"event": "drift_onset", "trial": tr.n, "dir": tr.direction})

        if tr is not None and tr.rt is None and sdir != 0 and sdir != self._prev_dir:
            if sdir == -tr.direction:
                tr.rt = self.t - tr.onset
                events.append({"event": "response", "trial": tr.n, "rt": round(tr.rt, 4)})
            else:
                tr.wrong_inputs += 1

        vx = steer * cfg.steer_mps
        if tr is not None and tr.rt is None:
            vx += tr.direction * cfg.drift_mps
        self.x = max(-cfg.max_offset_m, min(cfg.max_offset_m, self.x + vx * dt))

        if tr is not None:
            if tr.rt is None and not tr.lapse and self.t - tr.onset >= cfg.lapse_s:
                tr.lapse = True
                events.append({"event": "lapse", "trial": tr.n})
            if not tr.departure and abs(self.x) > cfg.departure_m:
                tr.departure = True
                events.append({"event": "lane_departure", "trial": tr.n, "side": "right" if self.x > 0 else "left"})
            if tr.rt is not None:
                if abs(self.x) <= cfg.recover_tol_m:
                    self._finish(tr, "recovered", events)
                elif self.t - (tr.onset + tr.rt) >= cfg.unrecovered_s:
                    self._finish(tr, "unrecovered", events)

        self._prev_dir = sdir
        self._w += dt
        self._sx += self.x * dt
        self._sxx += self.x * self.x * dt
        return events

    def _finish(self, tr: Trial, outcome: str, events: list[dict]) -> None:
        tr.end, tr.outcome = self.t, outcome
        events.append({"event": outcome, "trial": tr.n, "duration": round(tr.end - tr.onset, 4)})
        self.trials.append(tr)
        self.trial = None
        self.next_onset = self.t + self.rng.uniform(*self.cfg.interval_s)

    # ------------------------------------------------------------------ summary
    @property
    def sdlp(self) -> float | None:
        """Standard deviation of lateral position (m), the classic driving-drowsiness measure."""
        if self._w <= 0:
            return None
        mean = self._sx / self._w
        return math.sqrt(max(self._sxx / self._w - mean * mean, 0.0))

    def stats(self) -> dict:
        done = self.trials + ([self.trial] if self.trial else [])
        rts = sorted(t.rt for t in done if t.rt is not None)
        med = rts[len(rts) // 2] if len(rts) % 2 else (sum(rts[len(rts) // 2 - 1: len(rts) // 2 + 1]) / 2 if rts else None)
        return {
            "drifts": len(done),
            "responses": len(rts),
            "median_rt_s": None if med is None else round(med, 3),
            "mean_rt_s": round(sum(rts) / len(rts), 3) if rts else None,
            "lapses": sum(t.lapse for t in done),
            "lane_departures": sum(t.departure for t in done),
            "sdlp_m": None if self.sdlp is None else round(self.sdlp, 3),
            "driving_s": round(self.t, 1),
        }
