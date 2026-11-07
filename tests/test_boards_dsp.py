import numpy as np

from neuropawn_gui.boards import (
    KNIGHT_ID, KNIGHT_IMU_ID, KNIGHT_UV_PER_COUNT, ChannelLayout, channel_setup_commands,
)
from neuropawn_gui.dsp import (
    DISPLAY_FILTERS, amplitude_spectrum, apply_display_filter, relative_band_powers,
)


def test_knight_layout_matches_brainflow():
    L = ChannelLayout.for_board(KNIGHT_ID)
    assert L.sampling_rate == 125
    assert L.eeg_rows == [1, 2, 3, 4, 5, 6, 7, 8]
    assert L.package_row == 0 and L.timestamp_row == 11 and L.marker_row == 12
    assert L.extra_names == ["loff_statp", "loff_statn"]
    assert len(L.csv_columns()) == 4 + 8 + 2 + 1


def test_knight_imu_layout():
    L = ChannelLayout.for_board(KNIGHT_IMU_ID)
    assert L.has_imu and L.sampling_rate == 125
    assert len(L.extra_rows) == 11


def test_scale_matches_brainflow_header():
    # knight_base.h: 4.0 / (32768.0 - 1.0) / 12.0 / 79.57 * 1000000.0
    assert KNIGHT_UV_PER_COUNT == 4.0 / (32768.0 - 1.0) / 12.0 / 79.57 * 1000000.0


def test_setup_commands_match_neuropawn_examples():
    cmds = channel_setup_commands()
    assert cmds[0] == ("chon_1_12", "rldadd_1")
    assert cmds[-1] == ("chon_8_12", "rldadd_8")


def _sine(freq, fs=125, secs=4, amp=10.0):
    t = np.arange(int(fs * secs)) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def test_filter_never_mutates_input():
    x = np.vstack([_sine(10) + 500.0] * 8)
    before = x.copy()
    for flt in DISPLAY_FILTERS:
        apply_display_filter(x, 125, flt, 60.0)
    assert np.array_equal(x, before)


def test_notch_and_bandpass_behave():
    fs = 125
    band = DISPLAY_FILTERS[0]  # 1-50 Hz
    x = np.vstack([_sine(10, fs) + _sine(60, fs) + 300.0])
    y = apply_display_filter(x, fs, band, 60.0)[0, fs:-fs]  # ignore edges
    f, asd = amplitude_spectrum(y[None, :], fs)
    a10 = asd[0, np.argmin(abs(f - 10))]
    a60 = asd[0, np.argmin(abs(f - 60))]
    assert a10 > 50 * a60
    assert abs(y.mean()) < 1.0  # DC removed


def test_alpha_dominates_for_10hz():
    f, asd = amplitude_spectrum(_sine(10)[None, :], 125)
    rel = relative_band_powers(f, asd)[0]
    assert rel.argmax() == 2 and rel[2] > 90
