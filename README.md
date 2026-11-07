# NeuroPawn Hackathon: Team 14

Our project for the **Neurotech @ NC State BCI Hackathon** (Nov 7–8, 2026), **NeuroPawn track**.

> **Project:** _TBD: add a one-line description of what we're building._

## NeuroPawn EEG GUI

An OpenBCI-style app for the 8-channel NeuroPawn Knight headset. It shows a live time series, FFT, band power, numeric readouts and IMU data, and records **every sample, unfiltered and value-for-value exact**, to CSV.

```bash
python3 -m venv .venv.nosync && ln -s .venv.nosync .venv   # .nosync keeps iCloud away from it
source .venv/bin/activate
pip install -r requirements.txt
python run_gui.py               # pick board + port, Start Session, Start Recording
```

Close the NeuroPawn EXG Visualizer first, because only one app can use the serial port. Choose **Synthetic demo** to try it without hardware. Full guide and CSV format: [docs/EEG_GUI.md](docs/EEG_GUI.md).

## Repo layout

```
run_gui.py      launches the GUI
src/            neuropawn_gui package (acquisition, recorder, plots)
tests/          pytest suite (includes an end-to-end CSV-exactness test)
notebooks/      exploration and analysis
data/           EEG recordings (gitignored; keep local)
presentation/   final .pptx for submission
docs/           GUI guide, hackathon logistics, submission checklist
```

## Key info

- **Submissions due Sun Nov 8, 11:45 AM.** This repo must be **public** by then, and we need a .pptx. See the [submission checklist](docs/HACKATHON_INFO.md#submission-due-sun-nov-8-1145-am-no-exceptions).
- Headset setup steps: [docs/HACKATHON_INFO.md#neuropawn-headset-setup](docs/HACKATHON_INFO.md#neuropawn-headset-setup)
- Full schedule and rooms: [docs/HACKATHON_INFO.md](docs/HACKATHON_INFO.md)

## Team

Yamin Alshayeb · Zaid Abdeen · Emily Propst · Colin Liu · Sebastian Zucker-Malone
