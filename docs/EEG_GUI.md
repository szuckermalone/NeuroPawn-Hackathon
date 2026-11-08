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

## Connection check (do this before every recording)

As soon as a session starts, each channel gets a badge next to its number, **✓ 92** (good, ≥80), **! 61** (fair, 50–79) or **✗ 0** (poor, <50), and the top bar shows **"✓ Signal 8/8 good"**. The **Signal Check** tab (opened automatically) lists each channel's problem and how to fix it. Aim for **8/8 ✓ and no red banner** before you press Record.

The Knight doesn't report electrode impedance, so the score comes from the last 4 s of signal. The worst of these sets the score:

| Check | Good | What a bad value usually means |
|---|---|---|
| 60 Hz hum (peak above nearby noise) | < 2 µV/√Hz | Poor contact. Press it in, part the hair, add paste. Check the ear clip |
| Noise (1–40 Hz) | 2–40 µV | > 100: loose electrode, movement, jaw tension. < 0.5: flat (off or unplugged) |
| Drift (< 1 Hz) | < 600 µV p-p | Electrode still settling, or the cable is being tugged |
| Railed | < 90% of ±4190 µV | No contact at all |

**Red banner: "ChX… are near-copies of each other."** Several channels carrying the same signal (r ≥ 0.95) means one floating electrode in the RLD loop is spreading its noise to the others. This is what happened in our eyes-open/closed session, where a loose Ch8 contaminated Ch1–6. Fix the channel the banner names first.

Every recording's JSON stores the check as it was when you pressed Record (`board.signal_check_at_start`), so you can always tell how good the contact was.

## Screen layout

| Area | What it shows |
|---|---|
| **Time Series** | 8 stacked channels in OpenBCI colors, each with a connection badge (✓/!/✗ + score). Click a channel's number to hide it (display only). The right column shows µVrms over the last second, plus a railed warning when the raw signal is at ≥90% / ≥99% of the ADC's ±4190 µV range (usually means bad electrode contact). |
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
from neuropawn_gui.dsp import apparent_line_hz
notch = apparent_line_hz(60, 125, board_clock_hz)   # ~59.68 on the Knight (see "60 Hz hum")
rt = RealtimeCleaner(fs=125, cfg=CleaningConfig(car=True, notch_hz=notch))
clean, artifact = rt.process(chunk)   # chunk: (8, n) µV; artifact: (n,) bool
```

Live filtering delays the signal slightly. A blink gets flagged about 80 ms after its true peak.

## Drowsiness score and pre-sleep alert

The **Drowsiness** tab shows a 0–100 score and fires an alert (a macOS notification and sound, optionally spoken) when the EEG shows the **pre-sleep pattern**: posterior alpha replaced by central theta. The top bar shows the score at all times. Each alert also writes **marker 90** into the recording.

**How to use it:**
1. Start a session and get the Signal Check to 8/8 ✓.
2. Stay awake for the first **60 s** while it calibrates your personal baseline. About 30 s eyes open, then 30 s eyes closed, works well. Use **Recalibrate** for a new person or after re-seating electrodes.
3. Use **Test alert** to check that notifications reach you. On macOS you may need to allow notifications for the terminal or Python.

**What counts as pre-sleep.** Every 2 s, a 4 s window is labeled pre-sleep only if all of these hold:

| Criterion | Threshold | Why |
|---|---|---|
| θ (4–8 Hz, C3/C4) ÷ α (8–13 Hz, P3/P4/O1/O2) | ≥ 1.0 | Alpha no longer dominates. AASM N1 rule: alpha *replaced by* 4–7 Hz activity |
| θ vs. personal awake baseline | ≥ 2.0× | Theta has actually risen (Strijkstra et al. 2003; Hori sleep-onset stages) |
| β (13–30 Hz) vs. baseline | ≤ 1.5× | Not attention-related theta, which comes with maintained beta. Drowsiness lowers beta (Lal & Craig 2002; Jap et al. 2009) |
| No blink on F3/F4 (> 100 µV) | — | Blinking means eyes open and awake |
| Clean signal | channels not ✗, < 200 µV | Artifacts and bad electrodes are excluded |

**Score** = % of valid windows in the last 30 s that meet all five. An **alert fires at ≥ 50**, the same ">50% of a 30 s epoch" rule used to score N1. It then needs the score to drop below 25 and a 60 s cooldown before it can fire again.

**Why not just θ/α?** On our own recordings θ/α is **1.3–2.1 in alert, eyes-open wakefulness**, because vision suppresses alpha, and 0.08–0.23 eyes closed. A bare ratio threshold would fire every time someone opens their eyes. Requiring theta to *rise* above baseline is what separates sleep onset from normal alpha attenuation.

**Validation so far:**

| Test | Score | Alert |
|---|---|---|
| Real sessions (eyes open, eyes closed, both full sessions) | 0 | none |
| Normal alpha blocking (alpha −90%, theta unchanged) | 0 | none |
| Focused attention (alpha −90%, theta ×2.5, beta ×2) | 0 | none |
| Mild drowsiness (alpha −50%, theta ×1.5) | 20 | none |
| Simulated pre-sleep (alpha −85%, theta ×2–3, beta −30%) | 87–100 | once, ~20 s after the pattern is established |

**Limitations:**
- **We have no real sleep-onset recordings yet.** The awake side is validated on real data; the pre-sleep side only on simulated transitions. Record a real drowsy session (lying down, late in the day, eyes closed) to confirm the 2× theta threshold, or test on the SADT driving dataset.
- Detection lags by design: about 15–30 s, because the pattern must hold for most of a 30 s epoch.
- It targets *eyes-closed* sleep onset. Drowsiness with the eyes open (e.g. a driver) shows up differently (alpha intrusions, slow blinks) and would need the camera features.
- This is a research prototype, not a validated sleepiness detector or a safety device.

## Night Drive: the driving task

A deliberately boring night-time highway drive that lets participants get drowsy while the GUI watches their EEG in the background. When the detector fires, the drive is **interrupted by a full-screen alarm** until the driver presses SPACE.

**Running a session (experimenter):**
1. In the GUI: Start Session, then get the Signal Check to 8/8 ✓.
2. On the **Drowsiness** tab, choose the length (default 30 min), tick **Fullscreen** for a real participant, and press **Launch driving task**. Recording starts automatically if it isn't already on.
3. The game guides a **1-minute baseline** (30 s eyes open on a fixation cross, 30 s eyes closed until a tone), which recalibrates the detector. Then the drive starts.
4. Leave the GUI running behind the game. The Drowsiness tab shows live driving stats (reaction time, lapses, lane departures, SDLP), and lapses and departures appear as ▲ on the score plot.
5. When the drive ends the game shows a summary and closes. Stop the recording in the GUI.

You can also start it on its own with `python run_game.py [--minutes 30] [--fullscreen]`. Start the GUI first, or there's no drowsiness detection.

**Controls:** LEFT/RIGHT (or A/D, or a USB wheel/gamepad) steer · SPACE start/continue · ESC pause · F fullscreen · TAB experimenter overlay · K skip the baseline (intro screen).

**Why it's monotonous on purpose:**
- Night, an empty straight highway, constant 100 km/h, evenly spaced reflector posts, and a steady engine/road drone. This is the kind of monotony that produces "highway hypnosis".
- No score, no reaction-time feedback, no music, and a minimal dashboard, because feedback and variety keep people alert.

**The task (based on the Sustained-Attention Driving Task):**
- Every 8–14 s the car slowly drifts out of the lane (0.55 m/s), and the driver steers it back.
- Logged per drift: reaction time; a **lapse** if there's no correction within 4 s; a **lane departure** if the wheels cross the line; recovery time.
- Logged continuously: lateral position at 10 Hz, and **SDLP** (how much the car weaves), the standard driving-drowsiness measure.

These give behavioral ground truth to check the EEG detector against. Files go to `data/game_sessions/night_drive_<date>/` (`events.csv`, `telemetry.csv`, `session.json`), timestamped with the same Unix clock as the EEG recording.

**Markers written into the EEG recording:**

| Code | Event | Code | Event |
|---|---|---|---|
| 30 | drive started | 41 / 42 | drift starts left / right |
| 31 / 32 | paused / resumed | 43 | driver responded (steered back) |
| 33 | drive ended | 44 | lapse (no response within 4 s) |
| 34 / 35 / 36 | baseline: eyes open / eyes closed / end | 45 | lane departure |
| 90 | drowsiness alert | 46 | back in lane centre |
| 91 | alert acknowledged by the driver | | |

**Driver mode (microsleep alert).** For a driver, waiting until half of a 30 s epoch shows the pre-sleep pattern is too slow. The detector also alerts after **3 consecutive pre-sleep windows** (about 8 s of continuous pattern), in line with microsleep durations of 3–15 s (Boyle et al. 2008; Poudel et al. 2014). On our real awake recordings this added no false alarms. On simulated transitions it fires about 12 s sooner than the 30 s rule.

**Keeping the GUI awake:** while the game covers the screen, macOS would normally throttle the hidden GUI (App Nap) and could let the system sleep during a long session. The GUI registers a user-initiated activity at startup to prevent both. You can see it in `pmset -g assertions`.

## 60 Hz hum: what we found and what to do

Measured on our three real sessions:

- **The hum is electrode-specific, not common to all channels.** Each channel's hum has a different phase, and the bad channels are 10–300× worse than the good ones. It comes from poor contact (an impedance mismatch turns body-wide mains pickup into a channel-specific signal), which RLD can't cancel. **Ch8 got worse every session** (2.7 → 65 → 484 µV rms) and Ch7 is consistently second-worst, so the back-of-head electrodes are the problem.
- **It's steady** (it varies only 1.1–1.2× within a session), so it comes from the surroundings plus contact, not from movement.
- **It doesn't sit at 60.00 Hz in the data.** The Knight's clock runs at ~125.66 Hz, not the 125 Hz BrainFlow assumes, so hum shows up at **~59.68 Hz**. A fixed 60.00 Hz notch only removed 21 dB (saved files) / **10 dB (live)**. The app now measures the board clock during the session (status bar: "notch tuned to 59.68 Hz") and centers the notch there: **42–51 dB / 28–33 dB**, with the EEG bands untouched. The cleaner and the recording viewer do the same, using the file's timestamps.
- Mains harmonics fold back into the EEG band (240 Hz lands at 11.27 Hz, inside alpha), but none were present at measurable levels. Our 11.25 Hz eyes-closed alpha peak is 0.75 Hz wide, so it's real alpha, not a hum line.

**Fixing it at the source (best payoff first):**
1. **Swap test for Ch8:** swap its electrode with a good channel's (e.g. Ch3). If the hum follows the electrode, the electrode or lead is bad. If it stays on Ch8, check the board side: pin on the **positive rail**, switch flicked **away** from the header.
2. **Back-of-head contact (O1/O2):** part the hair under the electrode, add conductive paste, and adjust the comfort pads so the electrode presses on the scalp.
3. **Ear clip (RLD/COMM):** snug, with a little paste. It's the reference and the RLD return for every channel.
4. **Environment A/B test:** watch the Signal Check hum column for 20 s with the laptop charger plugged in, then 20 s unplugged. Also move away from power strips, monitors, and lamps, and keep the USB cable away from the electrode leads.
5. **Don't leave a floating electrode in the RLD loop.** It injects its noise into every channel (our second session).

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
