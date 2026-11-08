"""Safely patch the current GUI SMS notification hook (idempotent).

Run from repository root after applying the patch files. No secrets are written.
Refuses to modify unexpected source so teammates' changes are not clobbered.
"""
from pathlib import Path

TARGET = Path('src/neuropawn_gui/main_window.py')
HOOK = '''        # Optional research-only SMS: use the sole GUI board owner and good signal.
        # An explicit NEUROGUARD_GUI_SMS=1 opt-in is also required. Tests never send.
        if enabled:
            from .alert_bridge import emit_gui_warning
            q = self._quality
            good = bool(q is not None and len(q.channels) == 8
                        and q.n_good == 8 and not q.warning)
            if emit_gui_warning(board_key=self._current_spec().key, signal_good=good):
                self.statusBar().showMessage("Experimental SMS warning queued; delivery unconfirmed", 6000)
'''
CLOSE_HOOK = '''        # Flush any already queued opt-in SMS before closing the application.
        from .alert_bridge import close_sms
        close_sms()
'''


def patch_numpy_compatibility():
    """Presentation code should work with both NumPy 1.26 and 2.x."""
    for name in ('notebooks/final/metrics.py', 'notebooks/make_alpha_figure.py'):
        target = Path(name)
        if not target.is_file():
            continue
        data = target.read_text(encoding='utf-8')
        if 'np.trapezoid(' not in data:
            continue
        if name.endswith('metrics.py'):
            expected = 'from scipy import signal, stats'
            replacement = 'from scipy import signal, stats, integrate'
        else:
            expected = 'from scipy import signal'
            replacement = 'from scipy import signal, integrate'
        if data.count(expected) != 1:
            raise SystemExit(f'Cannot safely patch scipy import in {target}')
        changed = data.replace(expected, replacement, 1).replace('np.trapezoid(', 'integrate.trapezoid(')
        compile(changed, str(target), 'exec')
        target.write_text(changed, encoding='utf-8')
        print('Made NumPy 1.26-compatible:', target)


def main():
    if not TARGET.is_file():
        raise SystemExit(f'Missing GUI source file: {TARGET}. Run from repo root.')
    text = TARGET.read_text(encoding='utf-8')
    if HOOK not in text:
        anchor = '        self.tabs.setCurrentWidget(self._drowsy_tab)\n\n    # ------------------------------------------------------------------ driving game link'
        if text.count(anchor) != 1:
            raise SystemExit('Cannot find unique warning hook anchor. No changes made.')
        text = text.replace(anchor, HOOK + anchor, 1)
    if CLOSE_HOOK not in text:
        anchor = '        event.accept()\n'
        if text.count(anchor) != 1:
            raise SystemExit('Cannot find unique close hook anchor. No changes made.')
        text = text.replace(anchor, CLOSE_HOOK + anchor, 1)
    compile(text, str(TARGET), 'exec')
    TARGET.write_text(text, encoding='utf-8')
    print('GUI SMS hook present, source syntax valid:', TARGET)
    patch_numpy_compatibility()


if __name__ == '__main__':
    main()
