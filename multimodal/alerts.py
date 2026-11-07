"""Optional spoken and SMS alerts for stationary NeuroGuard demonstrations.

Messages are sent only on a confirmed transition to warning and are rate-limited.
Never infer alertness from unavailable sensors. SMS requires explicit opt-in.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

WARNING_TEXT = (
    "NeuroGuard research alert: possible reduced vigilance detected. "
    "Stop somewhere safe and rest. Do not rely on this prototype for driving safety."
)


def validate_e164(value: str) -> str:
    """Require international E.164 telephone format, e.g. +15551234567."""
    if not re.fullmatch(r"\+[1-9][0-9]{7,14}", value or ""):
        raise ValueError("Phone number must be in international E.164 format, e.g. +15551234567")
    return value


def twilio_send(message: str, recipient: str, *, timeout: float = 8.0) -> str:
    """Send one SMS via Twilio REST API; credentials only from environment.

    An API acceptance response is not proof of delivery to a phone.
    """
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    origin = os.environ.get("TWILIO_FROM_NUMBER", "")
    if not (sid.startswith("AC") and token and origin):
        raise RuntimeError("Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER")
    recipient = validate_e164(recipient)
    origin = validate_e164(origin)
    if not message.strip():
        raise ValueError("Message cannot be empty")
    payload = urllib.parse.urlencode({"From": origin, "To": recipient, "Body": message}).encode()
    request = urllib.request.Request(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
        data=payload,
        method="POST",
        headers={
            "Authorization": "Basic " + base64.b64encode(f"{sid}:{token}".encode()).decode(),
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "NeuroGuard-research-prototype/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        # Avoid printing credentials and raw phone numbers from the response.
        raise RuntimeError(f"Twilio rejected SMS request (HTTP {exc.code})") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Cannot reach SMS service; check internet connection") from exc
    return str(result.get("sid", "unknown"))


def mac_say(message: str) -> None:
    """Speak via macOS' selected audio output, which may be Bluetooth."""
    if sys.platform != "darwin":
        raise RuntimeError("Local spoken alerts currently require macOS")
    subprocess.Popen(["say", message], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@dataclass
class AlertConfig:
    sms_enabled: bool = False
    speech_enabled: bool = False
    recipient: str = ""
    cooldown_seconds: float = 300.0
    # A sender may be injected in tests. Production SMS always uses twilio_send.


class AlertDispatcher:
    """Send only on warning transitions; never block camera capture on network I/O.

    The upstream detector must require validated, fresh sensor inputs and a
    sustained warning before calling update('EXPERIMENTAL WARNING', ...).
    """

    def __init__(self, config: AlertConfig, *, sms_sender: Callable = twilio_send,
                 speaker: Callable = mac_say, clock: Callable = time.monotonic):
        if config.cooldown_seconds < 0:
            raise ValueError("Cooldown must be >= 0")
        if config.sms_enabled:
            validate_e164(config.recipient)
        self.config = config
        self.sms_sender = sms_sender
        self.speaker = speaker
        self.clock = clock
        self.last_sent = -float("inf")
        self.prev_warning = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neuroguard-alert")
        self.pending = []

    def update(self, state: str, *, genuine_hardware: bool, validated_inputs: bool) -> bool:
        warning = state == "EXPERIMENTAL WARNING" and genuine_hardware and validated_inputs
        rising_edge = warning and not self.prev_warning
        self.prev_warning = warning
        if not rising_edge or self.clock() - self.last_sent < self.config.cooldown_seconds:
            return False
        if not (self.config.sms_enabled or self.config.speech_enabled):
            return False
        self.last_sent = self.clock()
        if self.config.speech_enabled:
            self.pending.append(self.executor.submit(self.speaker, WARNING_TEXT))
        if self.config.sms_enabled:
            self.pending.append(self.executor.submit(self.sms_sender, WARNING_TEXT, self.config.recipient))
        return True

    def check_errors(self) -> None:
        remaining = []
        for future in self.pending:
            if future.done():
                try:
                    result = future.result()
                    if result:
                        print("SMS accepted by provider; delivery not confirmed (id: " + str(result) + ")")
                except Exception as exc:
                    print("Alert delivery failed: " + str(exc), file=sys.stderr)
            else:
                remaining.append(future)
        self.pending = remaining

    def close(self) -> None:
        # Wait only for already-submitted work at shutdown (bounded by Twilio timeout).
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.check_errors()


def main():
    parser = argparse.ArgumentParser(description="NeuroGuard alert output self-test (stationary only)")
    parser.add_argument("--test-speech", action="store_true", help="Speak a test alert via Mac audio output")
    parser.add_argument("--send-test-sms", action="store_true", help="Explicitly send a real test SMS (may incur charges)")
    parser.add_argument("--phone", default=os.environ.get("NEUROGUARD_ALERT_PHONE", ""))
    args = parser.parse_args()
    if not (args.test_speech or args.send_test_sms):
        parser.print_help(); return
    if args.test_speech:
        mac_say("NeuroGuard test. Voice warning output is working.")
        print("Speech requested on the Mac's selected output device")
    if args.send_test_sms:
        sid = twilio_send("NeuroGuard TEST only. This is not a genuine drowsiness warning.", args.phone)
        print("Test SMS accepted by provider. Message id: " + sid)


if __name__ == "__main__":
    main()
