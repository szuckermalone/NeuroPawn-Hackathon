# NeuroGuard | EEG and computer-vision driver-vigilance research prototype

**Project:** [NeuroPawn Hackathon](https://github.com/szuckermalone/NeuroPawn-Hackathon)  
**Status:** Software prototype integrated; physical EEG, multimodal timing, trained drowsiness detection, and end-to-end alert delivery remain to be validated.  
**Intended use:** Consenting participants in seated, stationary driving-simulator experiments **only**.

> **Safety notice:** NeuroGuard is **not** a validated sleepiness detector or an automotive safety device. It must not be relied upon while driving on public roads. A warning does not make continuing to drive while drowsy safe. No electrical brain stimulation is performed. Follow the NeuroPawn manufacturer's instructions for safe human EEG acquisition.

## Overview

NeuroGuard explores whether scalp EEG and camera-derived eye behaviour can provide useful information about reduced vigilance while a person performs a simulated driving task. The project combines:

- Eight-channel EEG from a **NeuroPawn Knight** or explicitly labelled synthetic signals
- Pygame lane-drift simulator that measures steering response time
- Spectral EEG features and a personalised, rule-based EEG-change indicator
- Webcam or compatible video-source features, including eyelid closure, blinking and PERCLOS
- An **optional** Random Forest training/inference path for independently labelled, paired EEG and camera observations
- A standalone, interactive **offline EEG replay dashboard** (Plotly)
- Optional spoken Mac alerts and SMS delivery via Twilio, with safety gates and rate limiting
- An **OpenBCI-style live EEG GUI** (`run_gui.py`) for signal checking, value-for-value exact CSV recording, and blink/noise cleaning, see [docs/EEG_GUI.md](docs/EEG_GUI.md)

**Important:** The real-time camera overlay (`multimodal.live`) is **not** the same as the offline Plotly dashboard (`dashboard.py`). This release does not include a unified streaming browser dashboard. It does **not** use PiEEG-server.

## Implementation status

| Component | Implementation | What is verified / missing |
| --- | --- | --- |
| EEG source | `acquisition.py` | Synthetic mode available; Knight requires compatible vendor BrainFlow package and real hardware test |
| Steering simulator | `drive.py`, `main.py` | Existing simulator and offline replay/evaluation; real-hardware timing remains to be measured |
| EEG features | `signals.py`, `multimodal/features.py` | Eight-channel spectral feature extraction; physical montage and reference must be verified |
| EEG-change indicator | `vigilance.py` | Rule-based baseline score and warning, **not a trained sleepiness classifier** |
| Reaction-time ML | `analysis.py` | Personalised Ridge regression predicting steering response time, **not sleepiness** |
| Camera features | `multimodal/camera_features.py` | Blink/closure/mouth measures; camera and lighting must be tested on target Mac |
| Multimodal live UI | `multimodal/live.py` | OpenCV overlay and CSV logs; accurate EEG/video synchronisation is not established |
| Multimodal ML | `multimodal/train.py` | Random Forest training code; no independently validated, pretrained model bundled |
| Offline dashboard | `dashboard.py` | Interactive HTML replay, not continuous real-time streaming |
| Spoken/SMS alerts | `multimodal/alerts.py` | Local test utilities exist; delivery and in-car playback not verified |
| Live EEG GUI + recorder | `run_gui.py`, `src/neuropawn_gui/` | CSV exactness tested end-to-end against a live BrainFlow stream; run on a real Knight (60 s, 0 dropped packets). Corrects the BrainFlow 5.23.0 µV scale bug |
| EEG cleaning | `src/neuropawn_gui/cleaning.py` | Real-time (causal) and offline filters, blink detection/regression, CAR; validated on one real recording |
| CI | None in repository at last audit | Run test suite locally; GitHub Actions recommended next |

## Architecture

```text
NeuroPawn Knight (or synthetic EEG)         Webcam / supported video input
                 |                                      |
     BrainFlow acquisition                      MediaPipe Face Mesh
                 |                                      |
  EEG quality + band powers                EAR, blink, PERCLOS, mouth
                 |                                      |
                 +-------- multimodal features ---------+
                                      |
                         Optional trained Random Forest
                                      |
                       Experimental state / quality gate
                                      |
                    OpenCV display + CSV + optional alerts

Separate experiment path:
EEG + Pygame steering events -> personalised Ridge regression -> session recordings
                                                  |
                                     Offline Plotly dashboard/replay
```

The driving simulator, offline dashboard, and multimodal webcam monitor are **separate entry points**. Do not try to open the same Knight serial connection from two applications at once.

## Repository layout

```text
.
├── acquisition.py              # Synthetic or Knight EEG acquisition
├── signals.py                  # Legacy EEG feature extraction and quality check
├── analysis.py                 # Steering reaction-time Ridge regression
├── drive.py                    # Pygame driving experiment
├── main.py                     # Drive, replay, evaluate, import and report CLI
├── research.py                 # Public sustained-attention driving data import
├── vigilance.py                # Baseline-relative EEG-change indicator
├── vigilance_report.py         # Offline EEG-change report
├── dashboard.py                # Interactive standalone HTML session replay
├── multimodal/
│   ├── features.py             # Shared eight-channel feature schema and resampling
│   ├── camera_features.py      # Eye and mouth feature calculations
│   ├── live.py                 # Camera + optional EEG real-time OpenCV interface
│   ├── train.py                # Group-held-out Random Forest training
│   ├── import_sadt.py          # Eight-site public EEG feature export
│   └── alerts.py               # Mac speech and Twilio SMS
├── run_gui.py                  # Launches the live EEG GUI
├── src/neuropawn_gui/
│   ├── acquisition.py          # Board session thread; each sample read exactly once
│   ├── recorder.py             # Lossless CSV + JSON sidecar
│   ├── scale.py                # Exact fix for the BrainFlow <= 5.23.0 Knight uV scale
│   ├── cleaning.py             # Real-time / offline noise and blink cleaning
│   ├── main_window.py, widgets.py, review.py   # GUI and recording viewer
│   └── boards.py, integrity.py, dsp.py, ...
├── tests/
│   ├── test_multimodal.py
│   ├── test_alerts.py
│   └── test_recorder.py, test_scale.py, test_cleaning.py, ...  # GUI/recorder (pytest)
├── smoke.py
├── test_vigilance.py
├── requirements.txt
├── requirements-gui.txt
├── requirements-dashboard.txt
├── requirements-multimodal.txt
├── MULTIMODAL_INTEGRATION.md
├── SMS_BLUETOOTH_ALERTS.md
├── NEUROGUARD_24H_GUIDE.md
├── docs/
└── data/                      # Raw data should remain untracked/private
```

## Requirements and installation

**Recommended:** macOS with **Python 3.11**, a webcam, and optionally a supported NeuroPawn Knight. The project also has desktop simulator code for Windows/Linux, but the alert speech path currently uses macOS `say`.

Python 3.14 can cause the pinned Pygame build to compile from source and fail if SDL headers are missing. Prefer Python 3.11 for the hackathon. On macOS with Homebrew:

```bash
brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
python --version            # Should report Python 3.11.x
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-dashboard.txt
python -m pip install -r requirements-multimodal.txt
python -m pip install -r requirements-gui.txt   # live EEG GUI (PyQt6 + pyqtgraph)
```

> **iCloud note:** if the repo lives in `~/Documents` or `~/Desktop` with iCloud Drive turned on, iCloud evicts files from `.venv`. Imports then hang, or Qt fails with *"Could not find the Qt platform plugin 'cocoa'"*. Name the environment `.venv.nosync` instead (`python3.11 -m venv .venv.nosync && ln -s .venv.nosync .venv`); iCloud skips `*.nosync`.

`requirements-multimodal.txt` includes the earlier requirements and pins `mediapipe==0.10.21`; use the same environment for the live camera path. BrainFlow must expose the **NeuroPawn Knight board ID** in the installed/vendor distribution. Never substitute an unrelated board ID merely to make the code run.

Check core imports:

```bash
python -c "import pygame, sklearn, scipy, numpy, plotly; print('Core imports OK')"
```

For VS Code, select the interpreter at `.venv/bin/python` using **Python: Select Interpreter**. Confirm `python --version` in the integrated terminal.

## Quick-start: no EEG hardware required

Run the automated local checks:

```bash
python smoke.py
python test_vigilance.py
python -m unittest discover -s tests -v
```

Build and open the **synthetic** interactive dashboard:

```bash
python dashboard.py --demo --out neuroguard_demo.html
open neuroguard_demo.html           # macOS
```

Run the Pygame driving simulator with a *deliberately simulated* EEG transition:

```bash
python main.py drive \
  --mode synthetic --demo-fatigue \
  --trials 8 --calibration 3 --seed 22 \
  --out runs/synthetic_demo_01

python dashboard.py --session runs/synthetic_demo_01 \
  --out runs/synthetic_demo_01/dashboard.html

python main.py vigilance-report runs/synthetic_demo_01 \
  --out reports/synthetic_demo_01
```

The simulator uses arrow keys to counteract steering deviations. Each output directory used by the simulator/reporter must be new. **Synthetic EEG is not physiological evidence or proof of model accuracy.**

Camera-only monitoring (requires webcam permission and compatible MediaPipe):

```bash
python -m multimodal.live \
  --eeg none --camera 0 \
  --record runs/multimodal/camera_test.csv
```

Camera with synthetic EEG, for integration testing only:

```bash
python -m multimodal.live \
  --eeg synthetic --camera 0 \
  --record runs/multimodal/synthetic_camera_test.csv
```

Press **Q** to close the camera overlay. Camera-only/synthetic mode is for measuring features; without an appropriately trained model, the UI reports an **untrained/monitoring** state rather than a verified sleepy/alert classification.

## Hardware bring-up: NeuroPawn Knight

**Do this before any AI or alert demonstration using real EEG.** Use a seated, consenting, awake participant and follow the [NeuroPawn documentation](https://docs.neuropawn.tech/software/brainflow/board-setup/) for reference, RLD, electrode contact, gain and electrical safety.

### 1. Verify the physical montage

The current eight-channel *software assumption* is:

| Channel | Site | Region |
| --- | --- | --- |
| 1 | F3 | Left frontal |
| 2 | F4 | Right frontal |
| 3 | C3 | Left central |
| 4 | C4 | Right central |
| 5 | P3 | Left parietal |
| 6 | P4 | Right parietal |
| 7 | O1 | Left occipital |
| 8 | O2 | Right occipital |

The **10–20 system defines electrode positions, not a fixed channel count**. Verify every lead and the actual acquisition reference before interpreting features or rendering a scalp map. The Knight board is assumed to use a vendor-supported ear/COMM and RLD setup; consult its exact documentation. Do not connect an ESP32 directly to human electrodes.

### 2. Confirm port and vendor BrainFlow board ID

```bash
ls /dev/cu.*
python -m serial.tools.list_ports
python -c "from brainflow.board_shim import BoardIds; print([b.name for b in BoardIds if 'NEUROPAWN' in b.name])"
```

The code expects `NEUROPAWN_KNIGHT_BOARD`, or `NEUROPAWN_KNIGHT_BOARD_IMU` with `--imu`. If neither appears, install the **matching vendor SDK**. Plugging in a USB device alone does not establish BrainFlow compatibility.

Check channels and signal contact with the manufacturer's EXG Visualizer, or with `python run_gui.py` (live µV, FFT, railed and blink flags, packet-continuity check). Then **close that program** before launching another one, so the serial port is free.

> **⚠ BrainFlow 5.23.0 µV scale bug:** the current pip release reports Knight EEG **79.57× too large**. It's missing the `/79.57` term that upstream added in [PR #867](https://github.com/brainflow-dev/brainflow) on Oct 21, 2026. On our first real recording, values read as thousands of "µV" when the true EEG was about ±20 µV. **`acquisition.py` (NeuroGuard) reads BrainFlow directly and is affected.** Any absolute µV threshold or quality gate there is off by that factor until it's corrected. Relative features (band-power ratios, z-scores against a baseline) are unaffected. The GUI corrects the scale exactly and per chunk with `neuropawn_gui.scale.KnightScaleCorrector`, which NeuroGuard can reuse. Saved files can be fixed with `python -m neuropawn_gui.scale <file.csv>`.

### 3. Record EEG in the driving simulator

```bash
python main.py drive \
  --mode live --port /dev/cu.YOUR_KNIGHT_PORT \
  --trials 12 --calibration 4 \
  --out runs/knight_001

python dashboard.py --session runs/knight_001 \
  --out runs/knight_001/dashboard.html
```

Add `--imu` **only** if the physical board is the Knight IMU variant. In `acquisition.py`, the live source starts streaming, waits briefly, and sequentially enables eight EEG channels and their RLD settings. The code uses gain `12`; ensure it matches the board configuration. The sampling rate is read from BrainFlow rather than guessed.

### 4. Run the combined EEG + camera monitor

```bash
python -m multimodal.live \
  --eeg knight --port /dev/cu.YOUR_KNIGHT_PORT \
  --camera 0 \
  --record runs/multimodal/knight_camera_001.csv
```

Start the simulator and multimodal monitor in **separate sessions**, not simultaneously on the same serial port. Camera features become available only after enough valid face observations. In this version, EEG and video alignment is approximate; it is **not hardware-verified synchronisation**.

### 5. First physical validation protocol

1. Confirm all eight channels and the verified reference configuration
2. Save a continuous awake eyes-open recording, then a brief **awake** eyes-closed interval
3. Record a few deliberate blinks and gentle jaw/head movements as artifact controls
4. Check sample frequency, timestamp gaps, microvolt scaling, saturation and quality flags
5. Confirm that missing/flat EEG and missing face are shown as **UNAVAILABLE**, never automatically **ALERT**
6. Save participant consent, anonymised session metadata and electrode map outside the public repository

Eyes-closed posterior alpha enhancement is a possible **signal sanity check**, not a drowsiness label. Never attempt deliberate sleep-deprivation driving experiments on public roads.

## EEG features, model types and interpretation

### EEG

`multimodal/features.py` computes features from approximately eight-second EEG windows:

- `frontal_theta`: log power, 4–8 Hz, channels F3/F4
- `posterior_alpha`: log power, 8–13 Hz, channels P3/P4/O1/O2
- `central_beta`: log power, 13–30 Hz, channels C3/C4
- `theta_alpha`: ratio of **linear** theta to alpha band power
- `theta_beta`: ratio of **linear** theta to beta band power

Power is estimated using Welch's method and basic quality gating. These features can correlate with reduced vigilance, but their direction and interpretation vary by participant, reference, artifact and task; **no single EEG frequency band proves sleepiness**.

### Camera

`multimodal/camera_features.py` uses face landmarks and time-weighted observations to generate:

- `eye_closed`, `perclos` (fraction of observed time with estimated eyes closed)
- `blink_rate`, `closure_seconds`, `mouth_open`

The current fixed eye-aspect-ratio threshold is `0.20` by default. It needs participant and camera validation. The feature `mouth_open` is **not** a validated yawn detector. Occlusion or loss of tracking is unknown data, not an alert state.

### Three *different* inference paths

1. **Ridge regression**, `analysis.py`: predicts **steering reaction time**, not sleepy/awake state
2. **Rule-based EEG-change index**, `vigilance.py`: baseline-normalised hand-weighted theta/alpha/beta indicator; has no trained classifier and its threshold is provisional
3. **Random Forest**, `multimodal/train.py`: supervised binary classifier **training infrastructure**, only usable when a team-trained model with correctly paired, independently labelled EEG and camera feature rows is supplied

The Random Forest uses `n_estimators=150`, `min_samples_leaf=5`, class balancing and group-based train/test splitting. The live overlay will not show a model prediction unless a model file is supplied and both EEG and camera features are valid. Its current `0.65` score threshold and three fresh EEG-update persistence rule are **unvalidated design choices**. No validated pretrained sleepiness model ships with this repository.

### Training a *real* multimodal classifier

The input CSV must have independent `label` values (`0` for alert, `1` for drowsy under a documented research protocol), `participant_id` or `session_id`, and **all 10 feature columns** listed above. Camera and EEG observations must come from the **same time-aligned, labelled sessions**. The live recorder currently writes `participant_id=UNSET` and does **not** create ground-truth labels.

```bash
python -m multimodal.train \
  data/derived/independently_labeled.csv \
  --group participant_id \
  --output models/neuroguard_multimodal_rf.joblib
```

Training requires at least 100 complete labelled windows, three distinct groups and both classes in the held-out split. These are **minimum software guards**, not sufficient evidence of clinical or generalisable performance. Evaluate by held-out participant, sensitivity/specificity, false-alert frequency, calibration and alert latency. Do not train and evaluate overlapping EEG windows from the same session on both sides of a split. Only load trusted team-created `.joblib` models because Python deserialisation can execute code.

## Public datasets: what is and is not transferable

- **Sustained-attention driving task (SADT), [NEMAR nm000275](https://www.nemar.org/dataset/nm000275)**: 32-channel dataset with F3/F4/C3/C4/P3/P4/O1/O2 available and steering-response events. Useful for **reaction-time research**, but slow reactions are **not independently verified sleepy labels**.
- **[SEED-VIG](https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html)**: vigilance-related EEG features and PERCLOS labels. Its released channels, band definitions and feature representation do **not** match our Knight model exactly. Requires a **dataset-specific** training/transfer experiment.
- **[DD-Database](https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp9c)**: smaller montage with drowsiness annotations; useful for alternative research, not an eight-channel replacement.

To extract the five EEG features from a local SADT EEGLAB `.set` file:

```bash
python -m multimodal.import_sadt /path/to/recording.set \
  --out data/derived/sadt_features.csv
```

This importer selects channels **by exact electrode name**, converts units to microvolts and anti-alias resamples to 125 Hz. It exports features **without sleepiness labels** and does not parse steering events. For the older reaction-time import pipeline, use `python main.py import-cao --set ... --out runs/public_01 --calibration 12` and inspect event alignment.

**Never pad four channels into eight, duplicate electrodes, rename different positions as matching sites, or join unrelated people's EEG and camera data.** Document reference electrode, gain, ADC, sensor type, filters, sample rates, task, and montage because public data and Knight data can differ significantly even at the same sites. Raw recordings may be large and should not be committed to Git.

## Recorded sessions, reports and dashboard

Simulator session directories generally contain:

- `session.json`: recording metadata and acquisition configuration
- `eeg.csv`: timestamped EEG channels
- `events.jsonl`: steering stimuli, responses, feature quality and outcomes
- Optional reports: `vigilance.csv`, `vigilance.png`, `metrics.json`, `comparison.png`, generated dashboard HTML

Examples:

```bash
python main.py replay runs/knight_001
python main.py evaluate runs/knight_001 --out reports/knight_001_eval
python main.py vigilance-report runs/knight_001 --out reports/knight_001_vigilance
python dashboard.py --session runs/knight_001 --out runs/knight_001/dashboard.html
```

The **offline** dashboard supports eight EEG waveforms, PSD and frequency-band charts, spectrograms, approximate scalp activity mapping, an experimental EEG-change timeline and steering-event replay. A generated HTML file can be opened without Python or an internet connection. It is not a continuous live stream.

The multimodal live camera CSV logs feature estimates approximately every second, with availability/quality fields and experimental model output if loaded. It is **not** identical to a raw EEG recording. Preserve raw Knight sessions separately when needed.

## Optional spoken and SMS alerts

`multimodal/alerts.py` provides two outputs:

1. **Mac speech** via macOS `say` and the computer's selected audio output (including supported Bluetooth speakers)
2. **SMS** through Twilio to a recipient who has consented to testing

Run local speech test:

```bash
python -m multimodal.alerts --test-speech
```

For a single deliberate SMS test, supply credentials **only via local environment variables or a secure secret manager**:

```bash
export TWILIO_ACCOUNT_SID='AC_YOUR_ACCOUNT_SID'
export TWILIO_AUTH_TOKEN='YOUR_SECRET_TOKEN'
export TWILIO_FROM_NUMBER='+1YOUR_TWILIO_NUMBER'
export NEUROGUARD_ALERT_PHONE='+1YOUR_PHONE_NUMBER'
python -m multimodal.alerts --send-test-sms
```

Sending may incur charges. A Twilio accepted-message ID is **not proof of delivery**. Trial accounts, sender registration and regional restrictions may apply. Do not publish credentials or real phone numbers.

**Automatic alert output requires real Knight EEG, a supplied trained compatible classifier, valid camera + EEG features, and a sustained `EXPERIMENTAL WARNING`.** The SMS path is explicitly disallowed for synthetic and camera-only EEG modes.

After appropriate model training and offline validation, the experimental live command is:

```bash
python -m multimodal.live \
  --eeg knight --port /dev/cu.YOUR_KNIGHT_PORT \
  --camera 0 \
  --model models/neuroguard_multimodal_rf.joblib \
  --speak-alerts --sms-alerts \
  --alert-phone "$NEUROGUARD_ALERT_PHONE" \
  --alert-cooldown 300 \
  --record runs/multimodal/knight_alert_test.csv
```

SMS dispatch occurs on warning transitions with a default 300-second cooldown. SMS is **not** a guaranteed emergency service. Mac speech through a car speaker requires the **Mac** to be the connected audio source. If an **iPhone** receives SMS via CarPlay or an **Android** phone via Android Auto, spoken message announcements depend on the phone/car settings; basic Bluetooth pairing does not guarantee automatic SMS readout. Test while parked.

See [`SMS_BLUETOOTH_ALERTS.md`](SMS_BLUETOOTH_ALERTS.md) for detailed instructions.

## Testing, current readiness and acceptance criteria

Run locally after each change:

```bash
python smoke.py
python test_vigilance.py
python -m unittest discover -s tests -v
python -m pytest                 # everything, incl. GUI/recorder tests (what CI runs)
python dashboard.py --demo --out neuroguard_demo.html
```

The source code has unit tests for parts of the algorithms, but **GitHub Actions CI has not been configured** and an end-to-end Knight hardware test has **not** been recorded as passing in this repository. Test results on synthetic inputs do not establish actual driver sleepiness detection.

**Suggested criteria for a pitch-ready engineering demo (not safety approval):**

- [ ] Verified eight-channel Knight recording with documented physical electrode map, reference, gain, sampling and timestamp continuity
- [ ] Five-minute seated recording; aim for at least 90% usable windows as an **internal prototype target**
- [ ] EEG eyes-open/closed and blink artifact checks, with expected physiological interpretation qualified
- [ ] Webcam detects face, blinks and sustained closure with missing-face recovery
- [ ] EEG and camera CSV sessions saved with participant codes and provenance
- [ ] Offline dashboard opens, navigates and shows actual recorded data
- [ ] Signal failures display `UNAVAILABLE`, not an assumed safe/alert classification
- [ ] Local speech and deliberate SMS tests verified, including real phone delivery if used
- [ ] Synthetic demonstration clearly distinguished from human measurement
- [ ] Any classifier claim includes independently labelled data, held-out-group evaluation and documented false-alert rates
- [ ] Offline demonstration backup available without the physical board

**Next engineering priorities:** hardware bring-up; instrumented synchronisation and latency testing; independently labelled paired EEG/video collection; improved camera calibration/artifact handling; live WebSocket dashboard; CI; robust failure/recovery tests; external validation.

## Research ethics, privacy and licensing

Use informed consent for EEG, camera and phone notification tests. Avoid storing identifiable face video unless genuinely needed. Keep participant data, recordings, local model artifacts and messaging credentials outside the public GitHub repository. Review each external dataset's licence and attribution requirements before redistribution. Do not interpret an elevated EEG score as a diagnosis, and never run fatigue-induction experiments while someone operates a moving vehicle.

**Suggested `.gitignore` additions** beyond the current repository rules: `runs/`, `reports/`, `data/raw/`, `data/derived/`, `models/*.joblib`, `.env.*` (with an exception for `.env.example`), `neuroguard_demo.html`, and `.pytest_cache/`. Check `git status` before every commit.

## References and further documentation

- [Live EEG GUI, recording format, cleaning and RLD notes](docs/EEG_GUI.md)
- [Hackathon schedule, rooms and **submission checklist** (due Sun Nov 8, 11:45 AM)](docs/HACKATHON_INFO.md)

- [NeuroPawn BrainFlow setup](https://docs.neuropawn.tech/software/brainflow/board-setup/)
- [NeuroPawn BrainFlow examples](https://docs.neuropawn.tech/software/brainflow/examples/)
- [Project multimodal integration guide](MULTIMODAL_INTEGRATION.md)
- [Project SMS/Bluetooth setup](SMS_BLUETOOTH_ALERTS.md)
- [Project 24-hour dashboard guide](NEUROGUARD_24H_GUIDE.md)
- [SADT public driving EEG dataset](https://www.nemar.org/dataset/nm000275)
- [SEED-VIG](https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html)
- [Bitbrain overview of EEG and fatigue](https://www.bitbrain.com/blog/eeg-driver-fatigue)
- [PiEEG-server, architectural reference only](https://github.com/pieeg-club/PiEEG-server)

---

**Project maturity statement:** NeuroGuard currently provides working software components for simulated experiments, recording analysis, EEG/camera feature monitoring and optional alert delivery. The team must verify the physical Knight and camera, collect trustworthy labels and independently assess performance before describing the system as a real driver drowsiness detector.
