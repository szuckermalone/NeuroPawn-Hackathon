# NeuroPawn EEG GUI

An OpenBCI-style desktop app for streaming, viewing, and recording the NeuroPawn Knight board (8 channels, 125 Hz) through BrainFlow.

## Setup

```bash
python3 -m venv .venv.nosync && ln -s .venv.nosync .venv
source .venv/bin/activate
pip install -r requirements-gui.txt
python run_gui.py
```

> **Why `.nosync`?** If the repo is inside `~/Documents` or `~/Desktop` with iCloud Drive turned on, iCloud evicts venv files to the cloud. Qt then hangs while loading, or fails with *"Could not find the Qt platform plugin 'cocoa'"*. iCloud ignores anything ending in `.nosync`. If your repo isn't in an iCloud folder, a plain `python3 -m venv .venv` is fine.

Options: `--board knight|knight_imu|synthetic`, `--port /dev/cu.usbserial-XXXX`, `--autostart`, `--out <folder>`, and `--review <file.csv>` to open a saved recording.

## Using it

1. **Close the NeuroPawn EXG Visualizer.** Only one program can hold the serial port.
2. Plug in the headset. Pick **Board**: "NeuroPawn Knight" (or "Knight IMU" if your box says IMU), choose the **Port** (on a Mac it's `/dev/cu.usbserial-…` or `/dev/cu.usbmodem…`; press ⟳ to rescan), then press **Start Session**.
3. The app powers on each channel at gain 12 and adds it to the RLD (bias) loop, using the same `chon_{ch}_12` / `rldadd_{ch}` commands as NeuroPawn's example code. This takes **about 16 seconds**. You'll see channels come alive one at a time.
4. When the status says **Streaming**, press **● Start Recording** (or Ctrl+R). Press it again to stop. Files go to `data/recordings/`, which is gitignored so recordings aren't pushed to the public repo.
5. **Markers:** press **Marker**, or the keys `M` / `1`–`9`, to stamp an event into the data (it shows as a dashed line and lands in the CSV's `marker` column).
6. **Open Recording…** opens any saved CSV in a full-length viewer with pan and zoom.

Use **Synthetic demo** to try everything without the headset.

## Screen layout

| Area | What it shows |
|---|---|
| **Time Series** | 8 stacked channels in OpenBCI colors. Click a channel's number to hide it (display only). The right column shows µVrms over the last second, plus a railed warning when the raw signal is at ≥90% / ≥99% of the ADC's ±4190 µV range (usually means bad electrode contact). |
| **FFT Plot** | Amplitude spectrum (Welch, 0.5 Hz bins, last 4 s) per visible channel. |
| **Band Power** | Relative delta/theta/alpha/beta/gamma power, averaged over visible channels. |
| **Live Values** | Latest, mean, min, max, and standard deviation per channel over the last second, unfiltered (the same numbers the CSV gets), plus the raw lead-off bits. |
| **IMU** | Accelerometer and gyroscope (Knight IMU board only). |
| **Status bar** | Measured sample rate, total samples, packet continuity, recording progress. |

**Filters (Bandpass, Notch) only change what's drawn on screen.** The CSV is always the unfiltered signal.

## Recording format

Each recording produces two files:

- `neuropawn_<board>_<date>_<time>.csv`: one row per sample
- `neuropawn_<board>_<date>_<time>.json`: details about the recording (board, port, sample rate, gain, µV-per-count scale, the exact config commands sent, software versions, and an integrity summary written when the recording stops)

| Column | Meaning |
|---|---|
| `sample_index` | 0, 1, 2, … within the file |
| `timestamp_unix` | Host time when BrainFlow parsed the packet. USB delivers packets in bursts, so for evenly spaced timing use `sample_index / 125`. |
| `package_num` | Board's 8-bit packet counter (0–255, wraps) |
| `gap_before` | Packets missing right before this row (0 = none; -1 = counter repeated) |
| `ch1_uV` … `ch8_uV` | EEG in microvolts, exactly as BrainFlow returned it |
| `loff_statp`, `loff_statn` | Raw lead-off status bytes from the board |
| `accel_*`, `gyro_*`, `mag_*` | IMU (Knight IMU board only) |
| `marker` | Event marker (0 = none) |

Load in Python:

```python
import pandas as pd
df = pd.read_csv("data/recordings/neuropawn_knight_2026-11-07_16-00-00.csv")
eeg = df[[f"ch{i}_uV" for i in range(1, 9)]].to_numpy()
t = df["sample_index"] / 125  # seconds
```

### How the readings are kept accurate

- **Each sample is read once.** A background thread calls BrainFlow's `get_board_data()`, which removes what it returns from the buffer. Nothing is duplicated or skipped. BrainFlow's buffer holds about an hour of data, so even if the window freezes, no samples are lost.
- **Values are written unmodified.** Numbers go into the CSV before any display processing, in Python's exact round-trip float format, so `float(cell)` gives back the identical value. You can recover the board's original integer ADC counts with `round(uV / 0.12786…)` (the exact scale is in the JSON).
- **Data is protected on disk.** The file is flushed after every chunk and fsynced every second. On stop, the board is drained one last time so the final samples make it into the file. An existing file is never overwritten.
- **Dropped samples are detected.** The packet counter is checked on every sample. Any gap is flagged in `gap_before`, counted in the status bar, and summarized in the JSON.
- **The pipeline is tested.** `pytest tests/` runs a real BrainFlow stream (synthetic board) through the app and asserts that the CSV matches what BrainFlow delivered, value for value.

## µV scale correction (BrainFlow ≤ 5.23.0)

The current pip release of BrainFlow (5.23.0) reports Knight EEG **79.57× too large**. BrainFlow fixed this upstream in PR #867 on Oct 21, 2026, after that release came out. The app corrects it automatically and exactly:
- It recovers each value's integer ADC count and multiplies by the corrected 0.12785 µV/count. The result is bit-identical to what a fixed BrainFlow would output.
- It checks every chunk instead of relying on the version number, so upgrading BrainFlow just turns the correction off.
- The status bar shows "µV scale: corrected" (or "OK" on a fixed BrainFlow). The JSON records the per-chunk verdicts.

**Recordings made before this fix** are corrected automatically when opened in the viewer or the cleaner. To write a corrected copy of the file (the original is never touched):

```bash
PYTHONPATH=src python -m neuropawn_gui.scale data/recordings/<file>.csv   # -> <file>_scalefixed.csv
```

## Noise cleaning

`src/neuropawn_gui/cleaning.py` is tuned to the noise found in our first real recording: 60 Hz hum on poorly contacting channels, drift below 1 Hz, blinks on ch1/ch2, and noise shared across channels.

**In the GUI**, the default display filter is **"Clean 1–40 Hz + artifact flags"**:
- It applies a 1–40 Hz bandpass plus the 60 Hz notch.
- Blinks (ch1/ch2 above 100 µV) and artifacts (other channels above 150 µV) are shaded red.
- The **CAR** button subtracts the average of channels 3–8 from every channel. Ch1/2 are left out of that average so blinks don't get copied into every channel.

All of this affects the display only.

**Offline**, to clean a saved recording into a new file:

```bash
PYTHONPATH=src python -m neuropawn_gui.cleaning data/recordings/<file>.csv [--car] [--no-regress] [--frontal 1,2]
# -> <file>_clean.csv: cleaned ch1..8 + artifact / epoch_bad / epoch_blink flags, and <file>_clean.json report
```

This runs the filters without time shift (zero-phase), detects blinks, and removes blink leakage from the other channels using regression fitted on blink moments only. It also flags 1 s epochs whose non-frontal channels exceed 100 µV peak-to-peak.

**In your own live / BCI code** (causal, works chunk by chunk):

```python
from neuropawn_gui.cleaning import RealtimeCleaner, CleaningConfig
rt = RealtimeCleaner(fs=125, cfg=CleaningConfig(car=True))
clean, artifact = rt.process(chunk)   # chunk: (8, n) µV; artifact: (n,) bool
```

Live filtering delays the signal slightly. A blink gets flagged about 80 ms after its true peak.

## Right-leg drive (RLD)

During session start, `rldadd_N` adds each channel to the board's **right-leg drive**. The RLD circuit:
1. Measures the voltage that all the electrodes have in common (mostly mains hum picked up by your whole body).
2. Inverts and amplifies it.
3. Feeds it back into your body through the ear-clip electrode on the RLD pin, which cancels most of it.

Signs it's working in our data: DC offsets are small (20–150 µV), and the 60 Hz hum shows up only on some channels rather than all of them. The hum that's left comes from **poor contact on specific electrodes**, which RLD can't fix. So:
- Keep the ear clip snug.
- Fix contact on the noisy channels.
- Don't leave a channel switched on and in the RLD loop if its electrode isn't on the head. A dangling electrode feeds noise into the loop and makes it worse for every channel.

## Notes and limitations

- The Knight board's µV conversion assumes **gain 12** (hard-coded in BrainFlow). The app always configures gain 12. Don't change the gain elsewhere or the µV values will be wrong.
- The board's clock runs at about 125.6 Hz by the host clock, slightly faster than the nominal 125 Hz. The notch is wide enough to cover this.
- `timestamp_unix` is host time, not a hardware clock. Use the sample index for timing within a recording.
- A 1-byte counter can't detect a loss of exactly 256 (or a multiple of 256) consecutive packets.
- Lead-off bits are shown as reported. They're only meaningful if the board firmware enables lead-off detection.
