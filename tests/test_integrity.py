import numpy as np

from neuropawn_gui.integrity import COUNTER_REPEAT, SequenceTracker


def test_contiguous_with_wraparound():
    t = SequenceTracker()
    pkg = np.arange(250, 250 + 20) % 256
    gaps = t.process(pkg)
    assert (gaps == 0).all()
    assert t.gap_events == 0 and t.missing_packets == 0


def test_gap_inside_chunk_and_across_wrap():
    t = SequenceTracker()
    gaps = t.process(np.array([253, 254, 2, 3]))  # 255, 0, 1 missing
    assert gaps.tolist() == [0, 0, 3, 0]
    assert t.gap_events == 1 and t.missing_packets == 3


def test_gap_across_chunk_boundary():
    t = SequenceTracker()
    t.process(np.array([10, 11, 12]))
    gaps = t.process(np.array([15, 16]))
    assert gaps.tolist() == [2, 0]
    assert t.missing_packets == 2


def test_repeat_is_flagged_not_counted_as_loss():
    t = SequenceTracker()
    gaps = t.process(np.array([5, 5, 6]))
    assert gaps.tolist() == [0, COUNTER_REPEAT, 0]
    assert t.repeat_events == 1 and t.missing_packets == 0


def test_empty_chunk():
    t = SequenceTracker()
    assert t.process(np.array([])).size == 0
