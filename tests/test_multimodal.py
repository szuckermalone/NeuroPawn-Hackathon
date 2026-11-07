import unittest
import numpy as np
from multimodal.features import MONTAGE, ALL_FEATURES, channel_selection, resample_eeg, eeg_features
from multimodal.camera_features import EyeMonitor

class TestMultimodal(unittest.TestCase):
    def test_channel_mapping(self):
        names=['CZ','O2','F4','P4','C3','F3','O1','P3','C4']
        self.assertEqual([names[i] for i in channel_selection(names)],list(MONTAGE))
        with self.assertRaises(ValueError):channel_selection(['F3','F4','C3'])
    def test_features(self):
        fs=125;t=np.arange(fs*8)/fs
        x=np.array([8*np.sin(2*np.pi*6*t)+6*np.sin(2*np.pi*10*t)+2*np.sin(2*np.pi*18*t)+i*.01*np.sin(2*np.pi*12*t) for i in range(8)])
        values,reason=eeg_features(x,fs)
        self.assertEqual(reason,'ok');self.assertEqual(set(values),set(ALL_FEATURES[:5]))
        invalid,reason=eeg_features(np.zeros_like(x),fs)
        self.assertIsNone(invalid)
    def test_resample(self):
        x=np.ones((8,4000));y=resample_eeg(x,500,125)
        self.assertEqual(y.shape,(8,1000))
    def test_camera_invalid(self):
        self.assertIsNone(EyeMonitor().update(1.,None,640,480))

if __name__=='__main__':unittest.main()
