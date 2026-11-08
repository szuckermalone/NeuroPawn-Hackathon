#!/usr/bin/env python3
"""Launch Night Drive (start the EEG GUI first so drowsiness alerts can interrupt the drive).

Usage: python run_game.py [--minutes 30] [--fullscreen]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from neuropawn_gui.game.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
