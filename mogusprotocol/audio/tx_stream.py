"""Sounddevice output stream wrapper for TX."""

import threading
from collections import deque

import numpy as np
import sounddevice as sd

from ..protocol.constants import SAMPLE_RATE


class TxStream:
    """Audio output stream that plays from a ring buffer.

    Usage:
        tx = TxStream()
        tx.write(audio_samples)
        tx.start()
        tx.wait_done()
        tx.stop()
    """

    def __init__(self, device=None, blocksize: int = 2048):
        self._buffer: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._done_event = threading.Event()
        self._playing = False
        self._blocksize = blocksize
        self._device = device
        self._stream: sd.OutputStream | None = None

    def write(self, samples: np.ndarray):
        """Queue audio samples for playback."""
        # Split into blocks
        for i in range(0, len(samples), self._blocksize):
            chunk = samples[i:i + self._blocksize].astype(np.float32)
            with self._lock:
                self._buffer.append(chunk)

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            if self._buffer:
                chunk = self._buffer.popleft()
                if len(chunk) < frames:
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0.0
                else:
                    outdata[:, 0] = chunk[:frames]
            else:
                outdata[:, 0] = 0.0
                if self._playing:
                    self._done_event.set()

    def start(self):
        self._playing = True
        self._done_event.clear()
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=self._blocksize,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def wait_done(self, timeout: float | None = None):
        self._done_event.wait(timeout=timeout)

    def stop(self):
        self._playing = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
