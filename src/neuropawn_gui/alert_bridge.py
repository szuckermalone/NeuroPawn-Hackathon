"""Optional GUI to SMS bridge for stationary research demonstrations.

Explicit opt-in: NEUROGUARD_GUI_SMS=1. Never send from synthetic board,
poor/unknown electrodes, test alerts, or untrained camera-only pathways.
The existing multimodal alert dispatcher owns SMS rate limiting and worker I/O.
"""
from __future__ import annotations

import os
import sys

_dispatcher = None


def emit_gui_warning(*, board_key: str, signal_good: bool, test: bool = False) -> bool:
    """Return True if a real-hardware warning was queued; never block Qt on SMS I/O."""
    global _dispatcher
    if test or not signal_good or board_key not in ('knight', 'knight_imu'):
        return False
    if os.environ.get('NEUROGUARD_GUI_SMS') != '1':
        return False
    if _dispatcher is None:
        try:
            from multimodal.alerts import AlertConfig, AlertDispatcher
            _dispatcher = AlertDispatcher(AlertConfig(
                sms_enabled=True,
                speech_enabled=False,
                recipient=os.environ.get('NEUROGUARD_ALERT_PHONE', ''),
                cooldown_seconds=300.0,
            ))
        except Exception as exc:
            print(f'NeuroGuard SMS opt-in configuration invalid: {exc}', file=sys.stderr)
            return False
    # The GUI emits only new warning events. Reset edge state after each event
    # while retaining the 300-second cooldown across events.
    queued = _dispatcher.update('EXPERIMENTAL WARNING', genuine_hardware=True,
                                validated_inputs=True)
    _dispatcher.update('MONITORING', genuine_hardware=True,
                       validated_inputs=True)
    _dispatcher.check_errors()
    return queued


def close_sms() -> None:
    global _dispatcher
    if _dispatcher is not None:
        _dispatcher.close()
        _dispatcher = None
