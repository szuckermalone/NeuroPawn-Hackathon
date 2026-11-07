"""Thread-safe fixed-size ring buffer feeding the display (not used for recording)."""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, n_rows: int, capacity: int):
        self._buf = np.zeros((n_rows, capacity), dtype=np.float64)
        self._cap = capacity
        self._write = 0
        self._count = 0
        self._total = 0
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def append(self, block: np.ndarray) -> None:
        n_in = int(block.shape[1])
        if n_in == 0:
            return
        with self._lock:
            n = n_in
            if n >= self._cap:
                block = block[:, -self._cap:]
                n = self._cap
            end = self._write + n
            if end <= self._cap:
                self._buf[:, self._write:end] = block
            else:
                k = self._cap - self._write
                self._buf[:, self._write:] = block[:, :k]
                self._buf[:, : n - k] = block[:, k:]
            self._write = end % self._cap
            self._count = min(self._cap, self._count + n)
            self._total += n_in

    def latest(self, n: int) -> tuple[np.ndarray, int]:
        """Copy of the newest ``n`` samples (oldest first) and the running sample total."""
        with self._lock:
            n = max(0, min(n, self._count))
            start = (self._write - n) % self._cap
            if start + n <= self._cap:
                out = self._buf[:, start:start + n].copy()
            else:
                out = np.concatenate((self._buf[:, start:], self._buf[:, : (start + n) - self._cap]), axis=1)
            return out, self._total
