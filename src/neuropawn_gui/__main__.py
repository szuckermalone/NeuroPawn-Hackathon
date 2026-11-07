"""Entry point: ``python run_gui.py`` or ``python -m neuropawn_gui`` (with src/ on the path)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_RECORDINGS = Path(__file__).resolve().parents[2] / "data" / "recordings"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="NeuroPawn Knight EEG acquisition GUI")
    ap.add_argument("--board", choices=["knight", "knight_imu", "synthetic"], help="board to preselect")
    ap.add_argument("--port", help="serial port to preselect, e.g. /dev/cu.usbserial-XXXX")
    ap.add_argument("--out", type=Path, default=DEFAULT_RECORDINGS, help="recordings folder")
    ap.add_argument("--review", type=Path, help="open a saved CSV recording in the viewer")
    ap.add_argument("--autostart", action="store_true", help="start the session immediately")
    args = ap.parse_args(argv)

    from PyQt6.QtWidgets import QApplication

    from .theme import STYLESHEET

    app = QApplication(sys.argv[:1])
    app.setApplicationName("NeuroPawn EEG")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    if args.review:
        from .review import ReviewWindow
        w = ReviewWindow(args.review)
    else:
        from .main_window import MainWindow
        w = MainWindow(args.out, board_key=args.board, port=args.port)
        if args.autostart:
            w._toggle_session()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
