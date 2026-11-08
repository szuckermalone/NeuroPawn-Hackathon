"""Format bridge: no invented events; preserve channel order, scale and gaps."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from gui_to_dashboard import convert  # noqa: E402


def test_gui_csv_to_dashboard_with_verified_scale(tmp_path):
    src = tmp_path / 'record.csv'
    names = ['sample_index', 'timestamp_unix', 'gap_before'] + [f'ch{i}_uV' for i in range(1, 9)]
    with src.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for i in range(100):
            writer.writerow([i, 1000 + i / 125, int(i == 40)] + [0.1 * i + j for j in range(8)])
    src.with_suffix('.json').write_text(json.dumps({'board': {'sampling_rate_hz': 125,
        'name': 'Knight', 'scale_correction': {'mode': 'auto'}}}))
    out = tmp_path / 'converted'
    summary = convert(src, out, ('F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'))
    assert summary['rows'] == 100
    assert summary['reported_missing_packets'] == 1
    assert (out / 'events.jsonl').read_text() == ''
    assert 'no steering events fabricated' in (out / 'session.json').read_text()
