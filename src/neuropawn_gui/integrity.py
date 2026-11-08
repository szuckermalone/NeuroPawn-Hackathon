"""Sample-continuity checking using the board's 8-bit package counter."""

from __future__ import annotations

import numpy as np

COUNTER_MODULO = 256
# gap_before codes for samples where the counter did not move forward by >= 1:
COUNTER_REPEAT = -1    # counter did not advance (duplicate packet)
COUNTER_BACKSTEP = -2  # counter went backwards (stale / re-sent packet, a board fault)


class SequenceTracker:
    """Track the package counter across chunks and classify every step.

    ``process`` returns, for every sample, how many packets were missing immediately
    before it (0 = contiguous, >0 = that many lost), or a negative code for a repeat
    or a backwards step. The state carries across calls, so a gap that straddles two
    chunks is still detected.

    Steps are read as signed values in [-128, 127]. A board fault seen on our Knight,
    where it re-sends stale packets with the counter flipping n, n+1, n, n+1 ..., is
    therefore reported as backsteps, not as 254 "missing" packets per flip. Limitation:
    with a 1-byte counter, losing 128 or more consecutive packets (about 1 s) can't be
    told apart from a backwards step.
    """

    def __init__(self, modulo: int = COUNTER_MODULO):
        self.modulo = modulo
        self.prev: int | None = None
        self.gap_events = 0
        self.missing_packets = 0
        self.repeat_events = 0
        self.backstep_events = 0

    def process(self, package_nums: np.ndarray) -> np.ndarray:
        pkg = np.rint(np.asarray(package_nums, dtype=np.float64)).astype(np.int64) % self.modulo
        n = pkg.size
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        prev = np.empty(n, dtype=np.int64)
        prev[1:] = pkg[:-1]
        prev[0] = (pkg[0] - 1) % self.modulo if self.prev is None else self.prev
        half = self.modulo // 2
        step = (pkg - prev + half) % self.modulo - half      # signed step in [-128, 127]
        gaps = step - 1
        gaps[step == 0] = COUNTER_REPEAT
        gaps[step < 0] = COUNTER_BACKSTEP

        real_gaps = gaps > 0
        self.gap_events += int(real_gaps.sum())
        self.missing_packets += int(gaps[real_gaps].sum())
        self.repeat_events += int((gaps == COUNTER_REPEAT).sum())
        self.backstep_events += int((gaps == COUNTER_BACKSTEP).sum())
        self.prev = int(pkg[-1])
        return gaps
