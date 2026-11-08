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


def test_stale_packet_flip_is_backsteps_not_millions_missing():
    """Real Knight fault: counter flipping 235, 236, 235, 236, ... (re-sent stale packets)."""
    from neuropawn_gui.integrity import COUNTER_BACKSTEP
    t = SequenceTracker()
    gaps = t.process(np.array([235, 236] * 500))
    assert t.missing_packets == 0 and t.gap_events == 0
    assert t.backstep_events == 499 and (gaps == COUNTER_BACKSTEP).sum() == 499


def test_stream_health_flags_faults_and_clears():
    from neuropawn_gui.acquisition import StreamHealth
    h = StreamHealth(125.0)
    for k in range(20):                                   # healthy: 5 samples per 40 ms
        assert h.add(k * 0.04, 5, 0, 0, 0) is None
    msg = h.add(0.84, 5, 3, 0, 0)                          # out-of-order burst
    assert msg is None or "out of order" in msg
    for k in range(10):
        msg = h.add(0.88 + k * 0.04, 5, 2, 0, 0)
    assert "out of order" in msg
    h2 = StreamHealth(125.0)                               # stuck board: ~3000 packets/s of zeros
    for k in range(60):
        msg = h2.add(k * 0.04, 120, 0, 120, 0)
    assert "all-zero" in msg
    h3 = StreamHealth(125.0)                               # too fast but non-zero
    for k in range(60):
        msg = h3.add(k * 0.04, 120, 0, 0, 0)
    assert "arriving at" in msg
    for k in range(120):                                   # recovers once the window is clean again
        msg = h3.add(2.4 + k * 0.04, 5, 0, 0, 0)
    assert msg is None
