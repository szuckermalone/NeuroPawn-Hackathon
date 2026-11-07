"""Sample-continuity checking using the board's 8-bit package counter."""

from __future__ import annotations

import numpy as np

COUNTER_MODULO = 256
# gap_before value used when the counter did not advance (duplicate packet, or
# exactly a multiple of 256 packets lost, which a 1-byte counter cannot tell apart).
COUNTER_REPEAT = -1


class SequenceTracker:
    """Track the package counter across chunks and report missing packets.

    ``process`` returns, for every sample, how many packets were missing
    immediately before it (0 = contiguous). The state carries across calls, so a
    gap that straddles two chunks is still detected.

    Limitation: with a 1-byte counter, losing >= 256 consecutive packets is
    indistinguishable from losing (n mod 256).
    """

    def __init__(self, modulo: int = COUNTER_MODULO):
        self.modulo = modulo
        self.prev: int | None = None
        self.gap_events = 0
        self.missing_packets = 0
        self.repeat_events = 0

    def process(self, package_nums: np.ndarray) -> np.ndarray:
        pkg = np.rint(np.asarray(package_nums, dtype=np.float64)).astype(np.int64) % self.modulo
        n = pkg.size
        gaps = np.zeros(n, dtype=np.int64)
        if n == 0:
            return gaps
        prev = np.empty(n, dtype=np.int64)
        prev[1:] = pkg[:-1]
        if self.prev is None:
            prev[0] = (pkg[0] - 1) % self.modulo  # first sample ever: no gap by definition
        else:
            prev[0] = self.prev
        step = (pkg - prev) % self.modulo
        gaps = step - 1
        repeats = step == 0
        gaps[repeats] = COUNTER_REPEAT

        real_gaps = gaps > 0
        self.gap_events += int(real_gaps.sum())
        self.missing_packets += int(gaps[real_gaps].sum())
        self.repeat_events += int(repeats.sum())
        self.prev = int(pkg[-1])
        return gaps
