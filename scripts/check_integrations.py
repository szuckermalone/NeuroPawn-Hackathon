"""Preflight: show which NeuroGuard modules and presentation inputs are available.

Does not open EEG hardware, camera or internet connections; safe to run in CI.
Run: python scripts/check_integrations.py [--require-presentation-data]
"""
import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEEDED = {
    'Legacy EEG + reaction-time': ['acquisition.py', 'analysis.py', 'main.py', 'signals.py'],
    'Offline dashboard': ['dashboard.py', 'vigilance.py'],
    'GUI and Night Drive': ['run_gui.py', 'run_game.py', 'src/neuropawn_gui/main_window.py',
                            'src/neuropawn_gui/game/app.py', 'src/neuropawn_gui/alert_bridge.py'],
    'Camera and Random Forest': ['multimodal/live.py', 'multimodal/train.py'],
    'SMS': ['multimodal/alerts.py', 'src/neuropawn_gui/notify.py'],
    'Reproducible bridge': ['scripts/gui_to_dashboard.py'],
    'Presentation scripts': ['notebooks/final/metrics.py', 'notebooks/final/figures.py',
                              'notebooks/final/make_video.py'],
}
PRESENTATION_SOURCE = [
    'data/recordings/neuropawn_knight_2026-11-07_20-38-03.csv',
    'data/recordings/neuropawn_knight_2026-11-07_17-29-49.csv',
    'data/game_sessions/night_drive_2026-11-07_20-26-32',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-presentation-data', action='store_true')
    args = parser.parse_args()
    missing = []
    for group, paths in NEEDED.items():
        absent = [p for p in paths if not (ROOT / p).exists()]
        status = 'OK' if not absent else 'MISSING'
        print(f'{status:7s} {group}' + (': ' + ', '.join(absent) if absent else ''))
        missing.extend(absent)
    print('\nLocal presentation input data (intentionally not tracked by Git):')
    data_missing = [p for p in PRESENTATION_SOURCE if not (ROOT / p).exists()]
    for p in PRESENTATION_SOURCE:
        print(('MISSING' if p in data_missing else 'PRESENT'), p)
    print('\nOptional runtime dependencies:')
    for pkg in ('brainflow', 'PyQt6', 'pyqtgraph', 'pygame', 'plotly', 'cv2', 'mediapipe'):
        print(f'  {pkg}: {"installed" if importlib.util.find_spec(pkg) else "not installed"}')
    if missing or (args.require_presentation_data and data_missing):
        raise SystemExit(1)
    if data_missing:
        print('Preflight passes code layout; original presentation analysis cannot be rerun without local source data.')


if __name__ == '__main__':
    main()
