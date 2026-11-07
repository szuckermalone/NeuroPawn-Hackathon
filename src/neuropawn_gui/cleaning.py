"""Noise removal for Knight EEG, tuned to what the first real recording showed:

* 60 Hz mains hum, concentrated on channels with poor contact -> notch
* slow drift / motion below 1 Hz                                 -> high-pass
* muscle noise and hum above ~40 Hz                              -> low-pass
* eye blinks on the frontal channels (ch1/ch2), 150-260 uV       -> detect, flag, regress out
* shared noise across channels (r = 0.6-0.8)                     -> optional common average reference

Two entry points:

``RealtimeCleaner``  causal, chunk-by-chunk, for live use / BCI code. Processing a stream in
                     chunks gives the same output as processing it in one go. Adds the small
                     delay that any causal filter has.
``clean_offline``    zero-phase (no delay) for saved recordings, plus blink regression and
                     epoch rejection.

Cleaning always produces new arrays. Recordings on disk are never modified. The CLI writes
``<name>_clean.csv`` next to the original.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
from scipy import signal

FRONTAL_DEFAULT = (0, 1)  # 0-based channel indices that pick up blinks (ch1, ch2)


@dataclass(frozen=True)
class CleaningConfig:
    highpass_hz: float | None = 1.0
    lowpass_hz: float | None = 40.0
    notch_hz: float | None = 60.0
    notch_q: float = 30.0                 # ~2 Hz wide at 60 Hz; covers the board's 125.6 vs 125 Hz clock offset
    car: bool = False                     # common average reference
    frontal_channels: tuple[int, ...] = FRONTAL_DEFAULT
    blink_threshold_uv: float = 100.0     # |filtered frontal| above this = blink
    artifact_threshold_uv: float = 150.0  # |filtered| on any non-frontal channel above this = artifact
    artifact_pad_s: float = 0.25          # widen flagged spans by this much
    epoch_s: float = 1.0
    epoch_reject_p2p_uv: float = 100.0


# ----------------------------------------------------------------------------- filters
@lru_cache(maxsize=32)
def _design(fs: float, hp: float | None, lp: float | None, notch: float | None, q: float) -> np.ndarray | None:
    sections = []
    nyq = fs / 2
    if hp:
        sections.append(signal.butter(4, hp, btype="highpass", fs=fs, output="sos"))
    if notch and notch < nyq:
        b, a = signal.iirnotch(notch, q, fs=fs)
        sections.append(signal.tf2sos(b, a))
    if lp and lp < nyq:
        sections.append(signal.butter(4, lp, btype="lowpass", fs=fs, output="sos"))
    return np.vstack(sections) if sections else None


def design_sos(fs: float, cfg: CleaningConfig) -> np.ndarray | None:
    return _design(float(fs), cfg.highpass_hz, cfg.lowpass_hz, cfg.notch_hz, cfg.notch_q)


def common_average(x: np.ndarray, exclude: tuple[int, ...] = FRONTAL_DEFAULT) -> np.ndarray:
    """Subtract the mean of the non-excluded channels from every channel.

    The frontal (blink) channels are left out of the average. Otherwise every blink
    would be copied into all the other channels.
    """
    keep = [i for i in range(x.shape[0]) if i not in exclude]
    if not keep:
        keep = list(range(x.shape[0]))
    return x - x[keep].mean(axis=0, keepdims=True)


def _dilate(mask: np.ndarray, pad: int) -> np.ndarray:
    """Widen each True by ``pad`` samples on both sides. Output length always equals input length."""
    if pad <= 0 or not mask.any():
        return mask
    n = mask.size
    idx = np.flatnonzero(mask)
    edges = np.zeros(n + 1, dtype=np.int64)
    np.add.at(edges, np.clip(idx - pad, 0, n), 1)
    np.add.at(edges, np.clip(idx + pad + 1, 0, n), -1)
    return np.cumsum(edges[:-1]) > 0


def artifact_mask(filtered: np.ndarray, fs: float, cfg: CleaningConfig = CleaningConfig()) -> np.ndarray:
    """Boolean per-sample mask: blinks on frontal channels or large excursions elsewhere."""
    fr = [i for i in cfg.frontal_channels if i < filtered.shape[0]]
    other = [i for i in range(filtered.shape[0]) if i not in fr]
    m = np.zeros(filtered.shape[1], dtype=bool)
    if fr:
        m |= np.abs(filtered[fr]).max(axis=0) > cfg.blink_threshold_uv
    if other:
        m |= np.abs(filtered[other]).max(axis=0) > cfg.artifact_threshold_uv
    return _dilate(m, int(cfg.artifact_pad_s * fs))


def detect_blinks(filtered: np.ndarray, fs: float, cfg: CleaningConfig = CleaningConfig()) -> np.ndarray:
    """Sample indices of blink peaks on the frontal channels (>= 0.4 s apart)."""
    fr = [i for i in cfg.frontal_channels if i < filtered.shape[0]]
    if not fr:
        return np.array([], dtype=int)
    eog = filtered[fr].mean(axis=0)
    pk, _ = signal.find_peaks(np.abs(eog), height=cfg.blink_threshold_uv, distance=max(1, int(0.4 * fs)))
    return pk


# ----------------------------------------------------------------------------- real-time
class RealtimeCleaner:
    """Causal cleaning for streaming data.

    >>> rt = RealtimeCleaner(fs=125)
    >>> clean, artifact = rt.process(chunk)   # chunk: (8, n) uV, any n
    """

    def __init__(self, fs: float, n_channels: int = 8, cfg: CleaningConfig = CleaningConfig()):
        self.fs = float(fs)
        self.cfg = cfg
        self.n_channels = n_channels
        self.sos = design_sos(self.fs, cfg)
        self._zi: np.ndarray | None = None
        self._hold = 0
        self._pad = int(cfg.artifact_pad_s * self.fs)

    def reset(self) -> None:
        self._zi = None
        self._hold = 0

    def process(self, chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(chunk, dtype=np.float64)
        if x.shape[1] == 0:
            return x.copy(), np.zeros(0, dtype=bool)
        if self.sos is None:
            y = x.copy()
        else:
            if self._zi is None:
                # Start the filter in steady state for the first sample's DC offset, so it
                # doesn't ring for seconds at startup.
                self._zi = signal.sosfilt_zi(self.sos)[:, None, :] * x[None, :, :1]
            y, self._zi = signal.sosfilt(self.sos, x, axis=-1, zi=self._zi)
        if self.cfg.car:
            y = common_average(y, self.cfg.frontal_channels)

        # Causal artifact flags: a detection can only extend forward in time.
        fr = [i for i in self.cfg.frontal_channels if i < y.shape[0]]
        other = [i for i in range(y.shape[0]) if i not in fr]
        hit = np.zeros(y.shape[1], dtype=bool)
        if fr:
            hit |= np.abs(y[fr]).max(axis=0) > self.cfg.blink_threshold_uv
        if other:
            hit |= np.abs(y[other]).max(axis=0) > self.cfg.artifact_threshold_uv
        mask = np.zeros_like(hit)
        hold = self._hold
        for j in range(hit.size):
            if hit[j]:
                hold = self._pad + 1
            if hold > 0:
                mask[j] = True
                hold -= 1
        self._hold = hold
        return y, mask


# ----------------------------------------------------------------------------- offline
@dataclass
class CleanResult:
    data: np.ndarray                 # (channels, samples) cleaned uV
    artifact: np.ndarray             # (samples,) bool
    blink_samples: np.ndarray        # indices of blink peaks
    epoch_bad: np.ndarray            # (n_epochs,) bool: non-frontal p2p > threshold
    epoch_has_blink: np.ndarray      # (n_epochs,) bool
    regression_coefs: np.ndarray | None  # blink propagation factor per channel
    config: CleaningConfig
    report: dict = field(default_factory=dict)

    def sample_epoch_flags(self, fs: float) -> tuple[np.ndarray, np.ndarray]:
        w = int(self.config.epoch_s * fs)
        idx = np.minimum(np.arange(self.data.shape[1]) // max(w, 1), self.epoch_bad.size - 1)
        if self.epoch_bad.size == 0:
            z = np.zeros(self.data.shape[1], dtype=bool)
            return z, z
        return self.epoch_bad[idx], self.epoch_has_blink[idx]


def _line_asd(x: np.ndarray, fs: float, f0: float = 60.0) -> np.ndarray:
    if x.shape[1] < fs * 2 or f0 >= fs / 2:
        return np.full(x.shape[0], np.nan)
    f, p = signal.welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    m = (f > f0 - 1.5) & (f < f0 + 1.5)
    return np.sqrt(p[:, m].max(axis=1))


def clean_offline(eeg: np.ndarray, fs: float, cfg: CleaningConfig = CleaningConfig(),
                  regress_blinks: bool = True) -> CleanResult:
    """Zero-phase cleaning of a whole recording. ``eeg``: (channels, samples) in uV."""
    x = np.asarray(eeg, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    sos = design_sos(fs, cfg)
    y = signal.sosfiltfilt(sos, x, axis=-1) if sos is not None and x.shape[1] > 3 * (2 * len(sos) + 1) else x.copy()

    fr = [i for i in cfg.frontal_channels if i < y.shape[0]]
    blinks = detect_blinks(y, fs, cfg)
    blink_mask = np.zeros(y.shape[1], dtype=bool)
    blink_mask[blinks] = True
    blink_mask = _dilate(blink_mask, int(0.3 * fs))

    coefs = None
    if regress_blinks and fr and blink_mask.sum() > fs * 0.2:
        # Estimate how much of the frontal blink waveform leaks into each other channel,
        # using only blink samples (so ongoing brain activity doesn't bias the fit), then
        # subtract the slow (< 6 Hz) blink component everywhere.
        eog = y[fr].mean(axis=0)
        eog_lf = signal.sosfiltfilt(signal.butter(4, 6.0, "lowpass", fs=fs, output="sos"), eog)
        e = eog_lf[blink_mask]
        coefs = np.zeros(y.shape[0])
        for ch in range(y.shape[0]):
            if ch in fr:
                continue
            coefs[ch] = float(np.dot(e, y[ch, blink_mask]) / np.dot(e, e))
            y[ch] -= coefs[ch] * eog_lf

    if cfg.car:
        y = common_average(y, cfg.frontal_channels)

    art = artifact_mask(y, fs, cfg)

    w = int(cfg.epoch_s * fs)
    n_ep = y.shape[1] // w if w else 0
    other = [i for i in range(y.shape[0]) if i not in fr]
    if n_ep:
        ep = y[other][:, : n_ep * w].reshape(len(other), n_ep, w)
        epoch_bad = np.ptp(ep, axis=2).max(axis=0) > cfg.epoch_reject_p2p_uv
        epoch_blink = np.zeros(n_ep, dtype=bool)
        epoch_blink[np.minimum(blinks // w, n_ep - 1)] = True
    else:
        epoch_bad = epoch_blink = np.zeros(0, dtype=bool)

    before, after = _line_asd(x, fs), _line_asd(y, fs)
    report = {
        "samples": int(y.shape[1]),
        "fs": fs,
        "blinks": int(blinks.size),
        "blink_times_s": [round(b / fs, 2) for b in blinks.tolist()],
        "artifact_fraction": float(art.mean()) if art.size else 0.0,
        "epochs": int(n_ep),
        "epochs_bad": int(epoch_bad.sum()),
        "epochs_with_blink": int(epoch_blink.sum()),
        "std_uv_before": np.round(x.std(axis=1), 2).tolist(),
        "std_uv_after": np.round(y.std(axis=1), 2).tolist(),
        "line_60hz_asd_before": np.round(before, 3).tolist(),  # uV/sqrt(Hz) peak near 60 Hz
        "line_60hz_asd_after": np.round(after, 4).tolist(),
        "blink_regression_coefs": None if coefs is None else np.round(coefs, 3).tolist(),
    }
    return CleanResult(y, art, blinks, epoch_bad, epoch_blink, coefs, cfg, report)


# ----------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    import argparse
    import csv
    import json
    from pathlib import Path

    from .review import load_recording

    ap = argparse.ArgumentParser(description="Clean a NeuroPawn GUI recording -> <name>_clean.csv (original untouched)")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--hp", type=float, default=1.0, help="high-pass Hz (0 = off)")
    ap.add_argument("--lp", type=float, default=40.0, help="low-pass Hz (0 = off)")
    ap.add_argument("--notch", type=float, default=60.0, help="notch Hz (0 = off; use 50 outside the Americas)")
    ap.add_argument("--car", action="store_true", help="common average reference (frontal channels excluded from the mean)")
    ap.add_argument("--no-regress", action="store_true", help="don't regress blinks out of non-frontal channels")
    ap.add_argument("--frontal", default="1,2", help="1-based blink channels, comma separated ('' = none)")
    args = ap.parse_args(argv)

    frontal = tuple(int(c) - 1 for c in args.frontal.split(",") if c.strip())
    cfg = CleaningConfig(highpass_hz=args.hp or None, lowpass_hz=args.lp or None, notch_hz=args.notch or None,
                         car=args.car, frontal_channels=frontal)
    rec = load_recording(args.csv)
    res = clean_offline(rec.eeg, rec.fs, cfg, regress_blinks=not args.no_regress)

    out = args.csv.with_name(args.csv.stem + "_clean.csv")
    if out.exists():
        print(f"{out} already exists; delete it or rename it first.")
        return 1
    ep_bad, ep_blink = res.sample_epoch_flags(rec.fs)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["sample_index", "time_s", *[f"ch{i}_uV" for i in range(1, 9)], "artifact", "epoch_bad", "epoch_blink"])
        t = rec.time_s
        si = rec.col("sample_index").astype(int)
        for j in range(res.data.shape[1]):
            w.writerow([si[j], t[j], *res.data[:, j].tolist(), int(res.artifact[j]), int(ep_bad[j]), int(ep_blink[j])])
    meta = {"source": args.csv.name, "scale_note": rec.scale_note, "config": asdict(cfg),
            "regress_blinks": not args.no_regress, "report": res.report}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    r = res.report
    print(f"Wrote {out.name}")
    if rec.scale_note:
        print(f"  note: {rec.scale_note}")
    print(f"  blinks: {r['blinks']} at {r['blink_times_s']} s")
    print(f"  epochs flagged (non-frontal p2p > {cfg.epoch_reject_p2p_uv:g} uV): {r['epochs_bad']}/{r['epochs']}, "
          f"with blinks: {r['epochs_with_blink']}")
    print(f"  60 Hz hum (uV/rtHz) before: {r['line_60hz_asd_before']}")
    print(f"  60 Hz hum (uV/rtHz) after:  {r['line_60hz_asd_after']}")
    print(f"  std uV before: {r['std_uv_before']}")
    print(f"  std uV after:  {r['std_uv_after']}")
    if r["blink_regression_coefs"] is not None:
        print(f"  blink leakage removed (fraction of frontal blink per channel): {r['blink_regression_coefs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
