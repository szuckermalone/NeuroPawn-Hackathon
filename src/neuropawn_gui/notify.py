"""Desktop alerts: system notification, sound, optional speech. Never blocks or raises."""

from __future__ import annotations

import shutil
import subprocess
import sys

ALERT_SOUND_MAC = "/System/Library/Sounds/Sosumi.aiff"


def _spawn(cmd: list[str]) -> bool:
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, ValueError):
        return False


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def system_notification(title: str, message: str) -> bool:
    """macOS Notification Center / Linux notify-send. Returns False if unavailable."""
    if sys.platform == "darwin" and shutil.which("osascript"):
        script = f'display notification "{_escape(message)}" with title "{_escape(title)}" sound name "Sosumi"'
        return _spawn(["osascript", "-e", script])
    if shutil.which("notify-send"):
        return _spawn(["notify-send", "-u", "critical", title, message])
    return False


def play_sound() -> bool:
    if sys.platform == "darwin" and shutil.which("afplay"):
        return _spawn(["afplay", ALERT_SOUND_MAC])
    return False


def speak(message: str) -> bool:
    if sys.platform == "darwin" and shutil.which("say"):
        return _spawn(["say", message])
    if shutil.which("spd-say"):
        return _spawn(["spd-say", message])
    return False


def drowsiness_alert(score: float, speak_aloud: bool = False) -> dict:
    title = "NeuroPawn EEG: drowsiness alert"
    msg = f"Pre-sleep EEG pattern (score {score:.0f}/100). Theta has replaced alpha for most of the last 30 s."
    sent = {"notification": system_notification(title, msg), "sound": play_sound()}
    if speak_aloud:
        sent["speech"] = speak("Warning. Drowsiness detected. Please take a break.")
    return sent
