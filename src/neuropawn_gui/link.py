"""Local link between the EEG GUI and the driving game (JSON over UDP on 127.0.0.1).

The GUI owns the headset (only one program can hold the serial port), runs the
drowsiness detector, and stays in the background. The game is a separate process.

    game -> GUI   hello, state, event (drift/response/lapse/...), calibrate, alert_ack,
                  heartbeat (stats), bye
    GUI  -> game  status (streaming, drowsiness score, ...), alert (interrupts gameplay)

Game events become markers in the EEG recording (codes below), so behaviour and
EEG line up sample-for-sample. The game also logs every event with Unix time.
"""

from __future__ import annotations

import json
import socket
import time

GUI_PORT = 47361
GAME_PORT = 47362
HOST = "127.0.0.1"
VERSION = 1

# Marker values written into the recording's `marker` column (keys 1-9 are manual markers).
MARKERS = {
    "game_start": 30,
    "paused": 31,
    "resumed": 32,
    "game_end": 33,
    "calibration_eyes_open": 34,
    "calibration_eyes_closed": 35,
    "calibration_end": 36,
    "drift_left": 41,
    "drift_right": 42,
    "response": 43,
    "lapse": 44,
    "lane_departure": 45,
    "recovered": 46,
    "drowsiness_alert": 90,
    "alert_ack": 91,
}


class LinkEndpoint:
    """Non-blocking UDP endpoint. ``port=0`` picks a free port (tests)."""

    def __init__(self, port: int, peer_port: int, host: str = HOST):
        self.host = host
        self.peer_port = peer_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

    @property
    def port(self) -> int:
        return self.sock.getsockname()[1]

    def send(self, type_: str, **fields) -> bool:
        msg = {"v": VERSION, "type": type_, "t": time.time(), **fields}
        try:
            self.sock.sendto(json.dumps(msg).encode("utf-8"), (self.host, self.peer_port))
            return True
        except OSError:
            return False

    def poll(self, max_messages: int = 200) -> list[dict]:
        out = []
        for _ in range(max_messages):
            try:
                data, _addr = self.sock.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(msg, dict) and "type" in msg:
                out.append(msg)
        return out

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
