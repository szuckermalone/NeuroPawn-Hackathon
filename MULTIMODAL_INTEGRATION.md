# NeuroGuard multimodal integration: Knight EEG + camera + datasets

**Status:** Tested feature extraction and offline/synthetic code. Physical Knight, GoPro, human drowsiness classification and real-time web streaming have **not** been verified. This is for stationary simulator research, never live-road use.

## What changed

Existing `main.py`, Pygame `drive.py`, `analysis.py`, `vigilance.py`, and offline `dashboard.py` are preserved. The opt-in `multimodal/` package adds:

- `features.py`: shared eight-site montage, 8-second microvolt Welch band-power features and safe resampling; refuses absent channels
- `camera_features.py`: MediaPipe-style facial landmark calculations, time-weighted closure fraction and blink events
- `live.py`: webcam + Knight, camera-only or synthetic EEG interface; CSV output; optional trusted RF model and warning display
- `train.py`: Random Forest with held-out participant/session groups and **independent** labels
- `import_sadt.py`: import full raw EEGLAB `.set` recordings from the driving task to the matching eight sites, without inventing sleepiness labels
- `tests/test_multimodal.py`: channel-mapping, quality and resampling tests

## macOS setup (Python 3.11)

```bash
brew install python@3.11      # if needed
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-multimodal.txt
python smoke.py
python test_vigilance.py
python -m unittest discover -s tests -v
```

The `.venv` must report Python 3.11. Grant macOS camera permission to Terminal or VS Code. Try built-in webcam **before** GoPro.

```bash
python -m multimodal.live --eeg none --camera 0 --record runs/camera01.csv
python -m multimodal.live --eeg synthetic --camera 0 --record runs/synthetic01.csv
ls /dev/cu.*
python -m multimodal.live --eeg knight --port /dev/cu.usbmodemYOURPORT --camera 0 --record runs/knight01.csv
# Knight with IMU:
python -m multimodal.live --eeg knight --imu --port /dev/cu.YOURPORT --camera 0
```

The Knight adapter **reuses `acquisition.Source`**: configures BrainFlow Knight board ID, 125 SPS by default, waits for start-up, enables all 8 channels (`chon_1_12` through `chon_8_12`) and configures RLD. Follow the official vendor electrode/reference safety requirements. Confirm the device's gain, input switches, COM/USB port, packet format and correct electrode positions. Keep the EXG Visualizer closed when BrainFlow owns the serial port. `--eeg synthetic` is software-only.

Camera data are logged once per second; eye closure is uncertain when the face is missing. No real pupil tracking or formal yawn identification is claimed. Video and EEG timestamps here are approximately host-aligned; **hardware synchronisation and precise latency are not yet established**. Built-in beep is optional terminal bell and is not a safety alert.

## Model status, precise language

- Existing `analysis.py`: Ridge regression predicts steering reaction time
- Existing `vigilance.py`: rule-based, baseline-relative theta/alpha/beta experimental score
- New `multimodal.train`: Random Forest classifier **architecture** but **no pretrained, validated drowsiness model is bundled**
- New `multimodal.live`: runs RF only if a team-trained model is supplied, a valid EEG window exists and face coverage is sufficient. Otherwise displays FEATURE MONITORING, EEG UNAVAILABLE or CAMERA UNAVAILABLE; **not falsely classified as alert**

Train only when you have independently annotated CSV windows that include `participant_id`, `session_id`, `label` (0 alert, 1 drowsy) and **all 10 columns**: `frontal_theta, posterior_alpha, central_beta, theta_alpha, theta_beta, eye_closed, perclos, blink_rate, closure_seconds, mouth_open`. The live recorder uses `participant_id=UNSET`; replace with coded IDs and collect independent labels after review. Ensure different people/sessions are held out: overlapping adjacent windows from one session in both train/test is leakage.

```bash
python -m multimodal.train data/derived/independently_labeled.csv --group participant_id --output models/neuroguard_multimodal_rf.joblib
python -m multimodal.live --eeg knight --port /dev/cu.YOURPORT --camera 0 --model models/neuroguard_multimodal_rf.joblib
```

RF probabilities and `.65` warning threshold are **unvalidated placeholders**. Never promote trained artifacts to a real-world driving safety device without rigorous independent testing.

## Dataset strategy: never fabricate electrode signals

**First choice for 8-site transfer:** [Sustained-attention driving EEG, NEMAR nm000275](https://www.nemar.org/dataset/nm000275), originally Cao et al. 2019. 27 participants, 32 channels including A1/A2 references, 500 Hz, steering deviation / response event timestamps; CC BY 4.0. All eight of our proposed scalp sites are present: `F3 F4 C3 C4 P3 P4 O1 O2`. It is excellent for the **reaction-time regression task**, but contains **no independent standard sleepiness labels**: slow steering reaction does not prove drowsiness. Raw files may be large; download a single `.set` + `.fdt` pair first, then:

```bash
python -m multimodal.import_sadt /path/to/sub-XX_task-driving_eeg.set --out data/derived/sadt_features.csv
```

The importer checks exact channel labels, selects only the matching eight channels, converts from V to µV through MNE, resamples **500→125 Hz with anti-alias filtering**, uses the same 8-second power-feature algorithm and exports rows WITHOUT drowsiness labels. It does not yet parse or assign stimulus reaction times. For reaction-time supervised learning, add event parsing and alignment separately.

**First choice for true vigilance labels:** [SEED-VIG](https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html). 23 participants, 17 EEG feature channels and eye-tracker PERCLOS labels. It includes 8-second feature windows, but **its provided spectral band definitions differ** (alpha 8–14, beta 14–31), it does not contain our full frontal/central montage, and the public release mainly supplies extracted EEG features rather than the same raw 8-channel measurements. It must be treated as a **separate dataset-specific model**, or a rigorously documented feature-space transfer experiment. Do **not** claim that a SEED-VIG model has been validated on the Knight.

**Additional externally labelled option:** [DD-Database](https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp9c). Four EEG + two EOG + ECG signals, 128 Hz, 10 volunteers, simulator studies and drowsiness-event annotations. Good for testing reduced-channel drowsiness research, but its channels and labels require a dataset-specific importer. Never zero-pad or duplicate four leads into eight.

### Different channel count and hardware

The 10–20 or 10–10 system specifies *electrode locations*, not an eight-channel protocol. Knight can use eight chosen positions, but other EEG machines may use 4, 17, 32 or more sites. Record actual Knight electrode positions and common reference **before** comparing. Differences in reference electrode (Knight ear COMM vs mastoids / CPz), amplifier filters, ADC bit depth/gain, sample clock, dry vs wet contact, noise profile and recording task produce **domain shift** even with the same electrode names.

1. Select electrodes by **exact name and position**; never choose `first 8 channels` by index without checking.
2. Use an explicit montage mapping for `F3 F4 C3 C4 P3 P4 O1 O2`. If a dataset lacks a needed site, train a *different common-channel feature model* or collect your own data. Do not create missing data by interpolation then claim it is measured EEG.
3. Match references only with documented, mathematically appropriate rereferencing using available recorded signals. A mastoid-referenced recording cannot always be made exactly identical to ear-clip recording.
4. Match bandwidth and anti-alias resampling, use microvolts, correct gain calibration, reject artifacts and control for signal quality.
5. Use per-session/participant baseline and evaluate train-on-public/test-on-Knight separately; never equate public-dataset accuracy to hardware accuracy.
6. For camera + EEG RF, ensure webcam/video features were recorded in the SAME labelled sessions as EEG: do not join unrelated subjects or fabricate eye metrics.

## Recommended 24-hour validation ladder

1. **Hour 0–3:** enumerate `/dev/cu.*`, check board ID and Knight channel activation; check scalp sites and reference; eyes-open/closed alpha sanity experiment and eye blinks (stationary)
2. **Hour 3–7:** camera-only demonstration; quality-gated EEG with a real recording; verify timestamps, units, gaps, eye-camera validity and artifacts
3. **Hour 7–12:** record simultaneous simulator/EEG/camera sessions. Log coded participant IDs, keyboard/lane events, external sleepiness ratings (when safely paused), opt-in video annotations and calibration metadata
4. **Hour 12–17:** produce dashboard/replay and compare theta/beta and eye metrics against **independent** labels. Import one SADT recording to check full eight-site mapping
5. **Hour 17–21:** if valid labels for multiple groups exist, train RF with subject/session holdout; otherwise keep rules-only and demonstrate untrained state explicitly
6. **Hour 21–24:** safety/quality failure tests, reproducible offline demo, technical pitch, record known limitations and freeze release

**Pitch-ready prototype exit criteria:** verified 5+ min eight-channel recording with channel provenance; usable camera video and face-detection logs; deliberate signal-loss produces UNAVAILABLE not ALERT; an offline rerun produces same features; clear synthetic-vs-human provenance; held-out group metrics if a supervised model is claimed; signed-off limitations and no claims of collision prevention. Live-road use is expressly excluded.
