# Driver vigilance

Predict time to corrective steering in a desktop simulator from the eight seconds preceding each drift. A separate model sees only elapsed time and recent steering history. The outcome is reaction time in milliseconds. Slow responses have multiple causes. This research prototype is intended for seated simulator trials only.

## First run on macOS, Windows or Linux

Use desktop Python 3.11 or 3.12. Open a terminal in this extracted directory. On macOS or Linux:

~~~sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke.py
python main.py drive --mode synthetic --trials 8 --calibration 3 --out runs/practice
~~~

On Windows create the environment with `py -3.11 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1` in PowerShell. Each output directory must be new. Synthetic mode generates mock EEG and measures actual keyboard responses. Its results have no physiological meaning. The headless smoke check takes seconds. An interactive eight-event practice takes roughly two to three minutes. At the first drift the car moves slightly off centre, then continues sideways. Press and release the opposite arrow. Escape exits. An unfinished trial is marked aborted.

## Record with the Knight board

1. Follow the manufacturer's guide for reference, RLD, contact and wearer setup. Use channels 1 through 8 in the order F3, F4, C3, C4, P3, P4, O1, O2. The feature groups assume this order. Check that all eight sites and channels are available before collection. Spare-channel EOG is a future extension that would require a different channel map and gate.
2. Use EXG Visualizer to check contact and inspect blink and jaw artefacts. Disconnect it to release the serial port. Record the port and whether the box says IMU.
3. Install and check the board support in this same virtual environment:

~~~sh
python -m pip install brainflow
python -c "from brainflow.board_shim import BoardIds; print([x.name for x in BoardIds if 'NEUROPAWN' in x.name])"
python -m pip install pyserial
python -m serial.tools.list_ports
~~~

If the board ID is missing, obtain the event's matching SDK from the NeuroPawn mentor or the vendor links below. Do not substitute a numeric ID from another board. Supply the actual serial device path or Windows COM port:

~~~sh
python main.py drive --mode live --port /dev/cu.YOUR_PORT --trials 40 --calibration 12 --out runs/p01_day1
~~~

Append `--imu` only for the IMU board. Append `--joystick` for a connected USB gamepad with horizontal axis 0 (left negative, right positive). Verify orientation during practice. Use a fresh directory for the scored run. The headset streams at the sampling rate reported by BrainFlow, with matched gain 12 and paced channel activation. Initialisation takes about twenty seconds. Eight stable channels are required by this version.

## Replay and score

~~~sh
python main.py replay runs/p01_day1
python main.py evaluate runs/p01_day1 --out reports/p01
~~~

Replay reconstructs the predictor in trial order from raw EEG and recorded responses. Local replay respects the number of samples available when each prediction was made. It writes trial predictions to the terminal. Evaluate scores the originally logged predictions and writes metrics.json and comparison.png. Pass several session directories before --out for a per-session comparison. Keep synthetic, public and live sessions separately labelled. The chart compares prediction errors, not driving safety.

## Public recordings

Install `python -m pip install mne`. Download one original EEGLAB .set session and its associated .fdt if supplied. See [original files](https://doi.org/10.6084/m9.figshare.6427334.v5) and [event documentation](https://github.com/nemarDatasets/nm000275). Use numeric annotations 251/252 (left/right drift), 253 (response onset), 254 (correction ends). Keep the files together.

~~~sh
python main.py import-cao --set data/session.set --out runs/public_01 --calibration 12
python main.py replay runs/public_01
python main.py evaluate runs/public_01 --out reports/public_01
~~~

Importer selects the eight named channels in the live order and converts MNE volts to microvolts. For public data the CSV time and event cutoff are seconds from recording start, including the legacy field named cutoff_unix. Windows overlapping a preceding correction are excluded; closely spaced drifts may consequently yield few usable events. Inspect the first ten source marker pairs and retained counts before interpreting results. The importer loads a single session into RAM; raw CSV export and replay can require substantial disk and memory for long recordings.

## How prediction works

- Record eight seconds of past data. Reject missing/stale data, a large gap, flat channels or a peak-to-peak excursion above 500 microvolts. These are initial engineering settings. A rejected window does not prove poor attention.
- Apply Welch spectral estimation to that window only. Calculate frontal 4–8 Hz, posterior 8–13 Hz and central 13–30 Hz log band power. Three features keep the pilot model small. There is no universal fatigue threshold.
- Train two standardised ridge regressions on log reaction time using exactly the same calibration trials. Both receive time elapsed and the median of up to five prior valid responses. Only the combined model receives the three EEG features. The first trial uses zero for missing history. A frozen personal median is also scored.
- Freeze models after the requested number of usable, completed calibration trials. History may update with past outcomes. Model weights do not update in the test block. A noisy session can finish without a fitted model. Inspect quality reasons before collecting more data.
- A trial times out at four seconds. Misses are counted separately. Wrong-direction inputs do not finish a trial. Display onset and keyboard polling are software estimates, with approximately one frame of polling uncertainty and additional device/display latency. Column onset_unix is measured immediately after flipping the display. The marker is sent afterwards. The CSV and JSON timestamps are the replay alignment source.
- Compare mean absolute error on identical paired trials. Report counts of calibration/test, paired trials, misses, and EEG coverage. A smaller combined-model error in one person's short run is a pilot observation. Longitudinal and held-out-driver validation are separate work.
- Hardware noise, eye and muscle artefacts, practice and keyboard/gamepad differences can affect results. Use the same input device and montage within comparisons. Intentional sleep loss is unnecessary.

## Session files

| File | Contents |
| --- | --- |
| session.json | Acquisition mode, sampling rate, channel order and calibration configuration |
| eeg.csv | Continuous samples with timestamps and one microvolt column per channel |
| events.jsonl | Direction, locked prediction, quality, feature values, response time and outcome per event |
| metrics.json | Paired mean absolute errors and coverage, written by evaluate |
| comparison.png | Session-level comparison plot, written by evaluate |

## Five-person workflow

1. Acquisition owner: verify board ID, montage, microvolt scaling and a one-minute saved recording.
2. Simulator owner: run practice, verify early/incorrect/held presses and joystick orientation.
3. Public-data owner: import one session, verify marker pairs and measure usable coverage.
4. Analysis owner: run smoke.py, inspect calibration/test counts and compare all three baselines.
5. Integration owner: rehearse a fresh seated live run, save the report and prepare a labelled replay. Give judges results from complete runs, including sessions where EEG added no benefit.

## References

- [NeuroPawn board setup and channel commands](https://docs.neuropawn.tech/software/brainflow/board-setup/)
- [NeuroPawn acquisition examples](https://docs.neuropawn.tech/software/brainflow/examples/)
- [NeuroPawn BrainFlow repository](https://github.com/NeuroPawn/brainflow)
- [Cao et al., Multi-channel EEG recordings during a sustained-attention driving task](https://doi.org/10.1038/s41597-019-0027-4)
- [Bitbrain overview of driver fatigue and EEG](https://www.bitbrain.com/blog/eeg-driver-fatigue)
