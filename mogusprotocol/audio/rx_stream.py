"""Sounddevice input stream wrapper for RX."""

import threading
from collections import deque

import numpy as np
import sounddevice as sd

from ..protocol.constants import SAMPLE_RATE


class RxStream:
    """Audio input stream that captures into a ring buffer.

    Usage:
        rx = RxStream()
        rx.start()
        # ... wait ...
        rx.stop()
        audio = rx.get_audio()
    """

    def __init__(self, device=None, blocksize: int = 2048):
        self._buffer: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._blocksize = blocksize
        self._device = device
        self._stream: sd.InputStream | None = None
        self._peak_level: float = 0.0

    @property
    def peak_level(self) -> float:
        """Peak amplitude of the most recent audio chunk (0.0–1.0)."""
        with self._lock:
            return self._peak_level

    def _callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0].copy()
        with self._lock:
            self._buffer.append(chunk)
            self._peak_level = float(np.max(np.abs(chunk)))

    def start(self):
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=self._blocksize,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_audio(self) -> np.ndarray:
        """Return all captured audio as a single array."""
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self._buffer))
