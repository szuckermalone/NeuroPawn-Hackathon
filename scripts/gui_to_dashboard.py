"""Convert a NeuroPawn GUI CSV recording to the legacy offline dashboard format.

Does not invent steering events, EEG labels, or drowsiness ground truth.
Use only after confirming the physical electrode montage and microvolt scale.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPECTED = ("F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2")


def convert(source: Path, destination: Path, channels: tuple[str, ...], fs: float | None = None) -> dict:
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Output directory exists: {destination}")
    if channels != EXPECTED:
        raise ValueError("This legacy dashboard's EEG analysis requires an explicitly confirmed F3,F4,C3,C4,P3,P4,O1,O2 montage")
    sidecar = source.with_suffix('.json')
    if not sidecar.exists():
        raise ValueError("A GUI JSON sidecar is required to establish device and scale provenance")
    meta = json.loads(sidecar.read_text(encoding='utf-8'))
    fs = float(fs or meta.get('board', {}).get('sampling_rate_hz') or 0)
    if not (0 < fs <= 10000):
        raise ValueError("Invalid or missing sampling rate; supply --fs or check sidecar")
    scale = meta.get('board', {}).get('scale_correction')
    if not scale and meta.get('board', {}).get('name') in ('Knight', 'KnightIMU'):
        raise ValueError('Cannot verify Knight microvolt scale. Fix/verify old recording first with GUI scale tools.')
    destination.mkdir(parents=True)
    count = 0
    first_time = None
    last_idx = -1
    last_sample_t = None
    accumulated_gaps = 0
    with source.open(newline='', encoding='utf-8') as inp, (destination / 'eeg.csv').open('w', newline='', encoding='utf-8') as out:
        reader = csv.DictReader(inp)
        required = {'sample_index', 'timestamp_unix', 'gap_before', *(f'ch{i}_uV' for i in range(1,9))}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('Not a NeuroPawn GUI CSV recording; required columns are missing')
        writer = csv.writer(out)
        writer.writerow(['unix_s', *channels])
        for row in reader:
            idx = int(float(row['sample_index']))
            gap = int(float(row['gap_before']))
            if idx != last_idx + 1 or gap < 0:
                raise ValueError('Nonconsecutive/repeated sample indices or counter-backstep detected; inspect source integrity')
            if first_time is None:
                first_time = float(row['timestamp_unix'])
            accumulated_gaps += gap
            ts = first_time + (idx + accumulated_gaps) / fs
            if last_sample_t is not None and ts <= last_sample_t:
                raise ValueError('Non-increasing reconstructed timestamp')
            data = [float(row[f'ch{i}_uV']) for i in range(1,9)]
            writer.writerow([format(ts, '.9f'), *[repr(x) for x in data]])
            last_idx, last_sample_t = idx, ts
            count += 1
    if count < 20:
        raise ValueError('Recording must contain at least 20 rows')
    # The original game uses a different marker/event scheme and must be aligned separately.
    session = {
        'mode': 'gui_recording', 'fs': int(fs) if fs.is_integer() else fs,
        'channels': list(channels), 'calibration': 8,
        'source_gui_csv': str(source), 'source_sidecar': str(sidecar),
        'data_provenance': 'REAL RECORDED EEG, not labelled for drowsiness',
        'gap_samples_reported': accumulated_gaps,
        'event_import': 'none: no steering events fabricated',
        'timestamp_policy': 'first host timestamp + (sample_index + reported packet gaps)/fs',
    }
    (destination / 'session.json').write_text(json.dumps(session, indent=2), encoding='utf-8')
    (destination / 'events.jsonl').write_text('', encoding='utf-8')
    return {'rows': count, 'reported_missing_packets': accumulated_gaps, 'destination': str(destination)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fs', type=float, help='Override sample rate if metadata is absent')
    p.add_argument('--channels', required=True, help='Confirmed physical CH1..CH8 montage as comma-separated labels')
    args = p.parse_args()
    try:
        report = convert(args.csv, args.out, tuple(x.strip().upper() for x in args.channels.split(',')), args.fs)
    except Exception:
        # Do not silently claim successful conversion with a partial directory.
        raise
    print(json.dumps(report, indent=2))
    print('Now run: python dashboard.py --session', args.out, '--out', args.out / 'dashboard.html')


if __name__ == '__main__':
    main()
