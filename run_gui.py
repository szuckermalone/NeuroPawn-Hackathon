#!/usr/bin/env python3
"""Launch the NeuroPawn EEG GUI.  Usage: python run_gui.py [--board knight] [--port /dev/cu.usbserial-XXXX]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from neuropawn_gui.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
