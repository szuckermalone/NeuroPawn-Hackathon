"""Select genuinely matching scalp channels from sustained-attention driving EEG.

This is a reaction-time dataset: do NOT relabel reaction-time data as sleepiness.
Produces real EEG feature rows and event info for exploratory analysis.
"""
import argparse
import csv
from pathlib import Path
from .features import MONTAGE,eeg_features,channel_selection,resample_eeg

def main():
    p=argparse.ArgumentParser(description='Import one EEGLAB .set from NEMAR nm000275')
    p.add_argument('set_file')
    p.add_argument('--out',default='data/derived/sadt_features.csv')
    p.add_argument('--target-fs',type=int,default=125)
    a=p.parse_args()
    import mne
    raw=mne.io.read_raw_eeglab(a.set_file,preload=False,verbose='ERROR')
    # Match actual sensor locations, never channel numbers.
    indices=channel_selection(raw.ch_names)
    print('Selected:',[(site,raw.ch_names[i]) for site,i in zip(MONTAGE,indices)])
    print('Reference MUST be documented: source mastoid setup vs Knight ear COMM.')
    raw.load_data(); eeg=raw.get_data(picks=indices,units='uV')
    transformed=resample_eeg(eeg,raw.info['sfreq'],a.target_fs)
    rows=[]; window=8*a.target_fs
    for start in range(0,transformed.shape[1]-window+1,window):
        vec,reason=eeg_features(transformed[:,start:start+window],a.target_fs,MONTAGE)
        if vec:
            rows.append({'window_start_sec':start/a.target_fs,'signal_quality':reason,**vec})
    target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
    if not rows:raise RuntimeError('No valid 8-second EEG windows')
    with target.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Wrote',len(rows),'EEG feature windows to',target)
    print('NO drowsiness labels added. Event latencies require separate event alignment.')

if __name__=='__main__':main()
