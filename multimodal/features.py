"""EEG-feature schema shared by Knight recordings and external datasets.

No channel fabrication: every required site must be explicitly present.
This is an experimental research feature extractor, not a medical classifier.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import welch, resample_poly
from math import gcd

MONTAGE = ('F3','F4','C3','C4','P3','P4','O1','O2')
EEG_FEATURES = ('frontal_theta','posterior_alpha','central_beta','theta_alpha','theta_beta')
CAMERA_FEATURES = ('eye_closed','perclos','blink_rate','closure_seconds','mouth_open')
ALL_FEATURES = EEG_FEATURES + CAMERA_FEATURES


def channel_selection(ch_names, required=MONTAGE):
    """Return exact channel indices, refusing missing or duplicate electrode names."""
    names = [str(n).upper().strip().replace('EEG ', '') for n in ch_names]
    absent = [n for n in required if names.count(n) != 1]
    if absent:
        raise ValueError('Missing or ambiguous electrodes: '+', '.join(absent)+
                         '. Do not copy/interpolate channels or relabel different sites.')
    return [names.index(n) for n in required]


def resample_eeg(data, input_fs, target_fs=125):
    """Anti-alias resample channels x samples; use same operation for all recordings."""
    if input_fs <= 0 or target_fs <= 0:
        raise ValueError('Sample rates must be positive')
    a = np.asarray(data, dtype=float)
    if a.ndim != 2:
        raise ValueError('Expected channels x samples')
    # Accept integer sample rates for reproducible polyphase resampling.
    inp, out = int(round(input_fs)), int(round(target_fs))
    if abs(input_fs-inp)>1e-6 or abs(target_fs-out)>1e-6:
        raise ValueError('Resampling currently requires integer rates')
    factor = gcd(inp, out)
    return resample_poly(a, out//factor, inp//factor, axis=1)


def eeg_features(eeg_uv, fs, ch_names=MONTAGE):
    """Compute log-power features from 8 seconds of real microvolt EEG.

    Returns (features, reason). A failed quality gate never becomes 'alert'.
    """
    x=np.asarray(eeg_uv,dtype=float)
    if x.ndim!=2 or x.shape[0]!=len(ch_names):
        return None, 'channel count mismatch'
    if x.shape[1]<int(7.6*fs) or fs < 65:
        return None, 'insufficient samples or rate'
    try:
        x=x[channel_selection(ch_names)]
    except ValueError as ex:
        return None, str(ex)
    if not np.isfinite(x).all() or np.min(np.std(x,axis=1)) < .1:
        return None,'flat/invalid EEG'
    if np.max(np.ptp(x,axis=1)) > 500:
        return None,'high-amplitude artifact'
    # Matched feature extraction across source devices. No arbitrary re-reference.
    f,p=welch(x,fs=fs,nperseg=min(int(2*fs),x.shape[1]),axis=1,detrend='constant')
    def band(lo,hi,indices):
        mask=(f>=lo)&(f<hi)
        return float(np.mean(np.sum(p[indices][:,mask],axis=1)*(f[1]-f[0])))
    theta=band(4,8,[0,1]); alpha=band(8,13,[4,5,6,7]); beta=band(13,30,[2,3])
    eps=1e-12
    v=dict(frontal_theta=float(np.log(max(theta,eps))),
           posterior_alpha=float(np.log(max(alpha,eps))),
           central_beta=float(np.log(max(beta,eps))),
           theta_alpha=theta/max(alpha,eps), theta_beta=theta/max(beta,eps))
    return v,'ok'
