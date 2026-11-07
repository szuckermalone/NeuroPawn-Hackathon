# NeuroGuard LAB: 24-hour delivery guide

## What is in this package

- `dashboard.py`: standalone interactive HTML session dashboard, using the *existing* `signals.extract` and `VigilanceMonitor` components
- `requirements-dashboard.txt`: dashboard-only installation dependencies
- `neuroguard_dashboard_demo.html`: a complete offline preview built with SYNTHETIC EEG
- Existing `drive.py`, `analysis.py`, `signals.py`, `acquisition.py`, `vigilance.py`, `smoke.py`, and `test_vigilance.py` remain available

The HTML contains Plotly JS and data. No CDN, server, PiEEG board, internet or running Python process is needed **after** generation. Works as an interactive **session replay**, not a real-time stream. Rebuild after recording a new session. It is deliberately labelled as a research prototype, not a validated driving-safety device.

## Fast start (10 minutes)

With Python 3.10+ from this folder:

```bash
python -m pip install -r requirements-dashboard.txt
python dashboard.py --demo --out neuroguard_demo.html
```

Double-click `neuroguard_demo.html`. Use **Play/Pause**, scrub the session, select channel, replay speed, EEG waveform window and scalp-map metric. The controls update the visualisations.

Full original simulator (Pygame plus other requirements):

```bash
python -m pip install -r requirements.txt
python smoke.py
python test_vigilance.py
python main.py drive --mode synthetic --demo-fatigue --trials 8 --calibration 3 --seed 22 --out runs/demo01
python dashboard.py --session runs/demo01 --out runs/demo01/dashboard.html
```

The simulator takes wall-clock time to collect sample data and must finish before dashboard generation; do not open the same hardware acquisition device in two processes. The board requires a BrainFlow installation that includes `NEUROPAWN_KNIGHT_BOARD` and the correct port configuration. `brainflow` is intentionally not a required dependency for synthetic replay.

## Current interactive elements

1. Eight EEG waveform traces, zoomable with Plotly, plus adjustable replay time window
2. EEG spectral power density (mean over channels, 1 to 45 Hz)
3. Delta, theta, alpha, beta, gamma band-power bars for selected channel or channel average
4. Time-frequency spectrogram (mean over channels)
5. Approximate 8-electrode scalp map with selectable band; only meaningful when electrode labels truly match the recorded montage
6. Baseline-normalised experimental EEG-change trend, calibration progress, quality status
7. Steering response events, reaction-time summary, replay seek, speed, play/pause

The EEG trend is *not* a probability of drowsiness. It uses hand-selected research hypotheses, a short per-session baseline, smoothing and persistence. Some signals may be rejected by the quality gate.

## Hardware compatibility

**Our acquisition device is the NeuroPawn Knight** as encoded in `acquisition.py`. **PiEEG-server is NOT included, installed or used.** PiEEG-server currently documents support for PiEEG, JNEEG and IronBCI and has its own WebSocket stream and UI. A real NeuroPawn-to-PiEEG bridge requires checking its transport schema, channel ordering, sample timing, montage, filters, resampling, and handling of packet gaps; do not connect by assumption. For the 24-hour demo, keep the existing BrainFlow-backed acquisition and use the independent dashboard.

## Recommended 24-hour work allocation (parallel if possible)

| Window | Assignment | Acceptance criterion |
| --- | --- | --- |
| 0-2 h | Run offline dashboard and tests | Every chart updates on timeline scrub; screenshot and screen capture |
| 2-5 h | EEG hardware hookup and montage verification | 30 s stationary EEG with 8 channels, plausible µV scale, timestamps without gaps |
| 5-8 h | Add session export after one stationary recording | `eeg.csv` and `session.json` load into `dashboard.py` |
| 8-11 h | Quality assessment and artifacts | Deliberately disconnected electrode produces a clear unavailable/quality state |
| 11-15 h | Simulator use and annotation | 3 replayable sessions, stimulus-to-response markers checked |
| 15-19 h | UI polish and demonstration narrative | Controls work at presentation resolution, explicit simulated vs recorded badge |
| 19-22 h | Evaluation and repeatability | Run smoke and vigilance tests; preserve reproducible console output |
| 22-24 h | Freeze/demo and failure recovery | Offline HTML and backup videos opened without dependencies |

## Team split

- Hardware: electrode montage, skin contact, acquisition timing, electrode safety and device compatibility
- Signal processing: quality flags, Welch bands, baseline calibration, event alignment
- UI: responsive browser demo, replay, charts and annotations
- QA/presentation: independent verification of tests, reproducibility and nonclinical claims

## Immediate next technical milestone: live dashboard

A genuine live dashboard needs a single **acquisition process** owning the board and a nonblocking output stream. Recommended messages, at 10-30 Hz visual frame update rather than 125 Hz re-rendering, are:

```json
{"type":"eeg","ts":1731000000.04,"fs":125,"channels":["F3","F4","C3","C4","P3","P4","O1","O2"],"samples_uv":[[0.2,0.5,1.3,0.8,1.1,0.7,0.9,0.4]]}
```

A second 0.5 Hz status message could send `state`, `score`, `quality`, and `calibration`. Prefer a local WebSocket or Lab Streaming Layer feed with ring buffers, bounded history, reconnection and drop detection. This release does **not** implement that live bridge. Never pretend a replay is streaming. Never connect raw EEG to remote web services without consent.

## Validation requirements before any public safety claims

- Test only in a stationary vehicle or driving simulator; never test by making a sleepy person drive on public roads
- Separate synthetic data, labelled data and independent held-out human data
- Report sensitivity, false-positive alerts per hour, detection latency and missing-data rate
- Collect informed consent and document storage, retention and privacy protections
- Disconnected/flat channels and timestamp gaps must lead to UNAVAILABLE, not ALERT
- Safe instruction on apparent drowsiness: stop in a safe location and rest; no claim that an alert makes continued driving safe

## Sources

- https://github.com/pieeg-club/PiEEG-server
- https://www.bitbrain.com/blog/eeg-driver-fatigue
