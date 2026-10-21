"""Minimal in-place progress bar for training epochs (no extra dependency).

Writes to stderr with carriage returns and redraws at most every 0.2 s:

    epoch  3/25 [##########----------]  214/429  loss 1.5432  acc 0.912  41.2 it/s  ETA 5s

Enabled by NGRunConfig.progress (run_snn.py --progress). Disabled, it does
nothing, so logs written to files stay free of carriage returns. If the
environment variable NG_PROGRESS_FD names an open file descriptor, the bar is
drawn there instead of stderr.
"""
from __future__ import annotations

import os
import sys
import time

_STREAM = None


def _default_stream():
    """stderr, or the file descriptor named by NG_PROGRESS_FD (used by
    run_comparison.sh to send the bar to the terminal but not to train.log)."""
    global _STREAM
    if _STREAM is None:
        fd = os.environ.get("NG_PROGRESS_FD")
        try:
            _STREAM = os.fdopen(int(fd), "w", buffering=1, closefd=False) if fd else sys.stderr
        except (OSError, ValueError):
            _STREAM = sys.stderr
    return _STREAM


class EpochProgress:
    def __init__(self, enabled: bool, epoch: int, epochs: int, total: int,
                 width: int = 24, min_interval: float = 0.2, stream=None):
        self.enabled = enabled and total > 0
        self.epoch, self.epochs, self.total = epoch, epochs, total
        self.width, self.min_interval = width, min_interval
        self.stream = stream or _default_stream()
        self.t0 = self._last = time.perf_counter()
        self.n = 0
        self.loss_sum = self.acc_sum = 0.0

    def update(self, loss: float, acc: float) -> None:
        if not self.enabled:
            return
        self.n += 1
        self.loss_sum += loss
        self.acc_sum += acc
        now = time.perf_counter()
        if now - self._last >= self.min_interval or self.n == self.total:
            self._last = now
            self._draw(now)

    def _draw(self, now: float) -> None:
        elapsed = max(now - self.t0, 1e-9)
        rate = self.n / elapsed
        eta = (self.total - self.n) / rate if rate > 0 else 0.0
        filled = int(self.width * self.n / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        w = len(str(self.epochs))
        line = (f"\repoch {self.epoch:>{w}}/{self.epochs} [{bar}] {self.n:>{len(str(self.total))}}"
                f"/{self.total}  loss {self.loss_sum / self.n:.4f}  acc {self.acc_sum / self.n:.3f}"
                f"  {rate:5.1f} it/s  ETA {eta:4.0f}s ")
        self.stream.write(line)
        self.stream.flush()

    def close(self) -> None:
        if self.enabled:
            self.stream.write("\r" + " " * 110 + "\r")   # clear the bar; the epoch summary line follows
            self.stream.flush()
