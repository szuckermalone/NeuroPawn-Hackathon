"""Safety contract: SMS stays disabled except for explicit real-device opt-in."""
from neuropawn_gui import alert_bridge


def test_gui_sms_is_default_off(monkeypatch):
    monkeypatch.delenv('NEUROGUARD_GUI_SMS', raising=False)
    assert not alert_bridge.emit_gui_warning(board_key='knight', signal_good=True)


def test_gui_sms_never_uses_synthetic_or_bad_quality(monkeypatch):
    monkeypatch.setenv('NEUROGUARD_GUI_SMS', '1')
    assert not alert_bridge.emit_gui_warning(board_key='synthetic', signal_good=True)
    assert not alert_bridge.emit_gui_warning(board_key='knight', signal_good=False)
    assert not alert_bridge.emit_gui_warning(board_key='knight', signal_good=True, test=True)
