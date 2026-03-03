"""Streaming BPSK demodulator for live audio decoding.

Incrementally processes audio chunks, yielding decoded characters as they
arrive.  Two phases:

1. SEARCHING – accumulate audio and periodically cross-correlate with the
   reference preamble to find frame start.
2. DEMODULATING – extract one symbol at a time, feed bits to RxFrameParser,
   yield characters immediately.

Supports multi-frame messages: after completing one frame, resets to
searching mode to find the next frame's preamble.
"""

import numpy as np

from ..protocol.constants import SAMPLE_RATE, SAMPLES_PER_SYMBOL, PREAMBLE_SYMBOLS
from ..protocol.melody import HopScheduler
from ..protocol.frame import RxFrameParser


def _build_reference_preamble() -> np.ndarray:
    from .modulator import PSKModulator
    preamble_bits = [1 if i % 2 == 0 else 0 for i in range(PREAMBLE_SYMBOLS)]
    mod = PSKModulator()
    return mod.modulate(preamble_bits)


class StreamingDemodulator:
    """Incremental demodulator that yields decoded characters as audio arrives.

    Tracks decoded frames by index and reassembles text across frames.
    """

    def __init__(self, energy_threshold: float = 0.1):
        self.energy_threshold = energy_threshold
        self._ref_preamble = _build_reference_preamble()
        self._ref_len = len(self._ref_preamble)
        self._ref_energy = float(np.sum(self._ref_preamble ** 2))
        self.reset()

    def reset(self):
        """Reset all state for a new reception."""
        self._audio_buf = np.array([], dtype=np.float64)
        self._searching = True
        self._frame_offset = 0
        self._demod_pos = 0  # sample position within frame-aligned signal
        self._scheduler = HopScheduler()
        self._parser = RxFrameParser()
        self._lo_theta = 0.0
        self._prev_phase = 0.0
        self._done = False
        self._last_search_len = 0  # avoid re-searching same audio
        self._frames: dict[int, str] = {}
        self._total_frames: int | None = None
        self._frames_received = 0

    @property
    def done(self) -> bool:
        return self._done

    @property
    def synced(self) -> bool:
        return not self._searching

    @property
    def decoded_text(self) -> str:
        return self._parser.decoded_text

    @property
    def crc_ok(self) -> bool | None:
        return self._parser.crc_ok

    @property
    def frames_received(self) -> int:
        return self._frames_received

    @property
    def total_frames(self) -> int | None:
        return self._total_frames

    def feed(self, audio_chunk: np.ndarray) -> list[str]:
        """Feed an audio chunk and return any newly decoded characters.

        Args:
            audio_chunk: Float64 audio samples at 48 kHz.

        Returns:
            List of decoded characters (may be empty).
        """
        if self._done:
            return []

        self._audio_buf = np.concatenate([self._audio_buf, audio_chunk.astype(np.float64)])
        chars: list[str] = []

        if self._searching:
            found = self._try_sync()
            if not found:
                # Cap buffer at ~30 seconds to avoid unbounded growth,
                # but keep enough overlap so preamble at boundary isn't lost.
                max_buf = SAMPLE_RATE * 30
                if len(self._audio_buf) > max_buf:
                    trim_to = max_buf // 2
                    self._audio_buf = self._audio_buf[-trim_to:]
                    self._last_search_len = 0
                return []

        # Demodulate as many symbols as we have audio for
        chars.extend(self._demod_available())
        return chars

    def _try_sync(self) -> bool:
        """Attempt cross-correlation sync.  Returns True if frame found."""
        audio = self._audio_buf
        ref = self._ref_preamble
        ref_len = self._ref_len

        if len(audio) < ref_len:
            return False

        # Only re-search if we have meaningful new audio since last attempt
        min_new = ref_len // 4
        if len(audio) - self._last_search_len < min_new:
            return False

        search_len = len(audio)
        self._last_search_len = search_len

        n_fft = 1
        while n_fft < search_len + ref_len:
            n_fft *= 2

        audio_pad = np.zeros(n_fft)
        audio_pad[:search_len] = audio[:search_len]
        ref_pad = np.zeros(n_fft)
        ref_pad[:ref_len] = ref[::-1]

        corr = np.fft.irfft(np.fft.rfft(audio_pad) * np.fft.rfft(ref_pad))
        corr = np.abs(corr[ref_len - 1:search_len])

        peak = np.argmax(corr)
        if corr[peak] < self.energy_threshold:
            return False

        # Ensure we have enough audio past the peak to contain the full
        # preamble — otherwise we matched a partial preamble and the
        # offset may be unreliable.  Wait for more audio.
        if search_len - peak < ref_len:
            return False

        # Normalized cross-correlation: divide by geometric mean of reference
        # and signal energies so the check is amplitude-independent (0–1).
        sig_end = min(peak + ref_len, search_len)
        sig_energy = float(np.sum(audio[peak:sig_end] ** 2))
        if sig_energy <= 0:
            return False
        normalized = corr[peak] / np.sqrt(self._ref_energy * sig_energy)
        if normalized < 0.5:
            return False

        # Found frame start
        self._frame_offset = int(peak)
        # Trim buffer to start at frame
        self._audio_buf = self._audio_buf[self._frame_offset:]
        self._frame_offset = 0
        self._demod_pos = 0
        self._searching = False
        self._scheduler.reset(0)
        self._lo_theta = 0.0
        self._prev_phase = 0.0
        return True

    def _demod_available(self) -> list[str]:
        """Demodulate all available complete symbols from the buffer."""
        chars: list[str] = []
        sps = SAMPLES_PER_SYMBOL

        while self._demod_pos + sps <= len(self._audio_buf):
            chunk = self._audio_buf[self._demod_pos:self._demod_pos + sps]
            freq = self._scheduler.current_freq

            phase_inc = 2.0 * np.pi * freq / SAMPLE_RATE
            phases = self._lo_theta + phase_inc * np.arange(1, sps + 1)
            self._lo_theta = phases[-1]

            i_val = np.mean(chunk * np.cos(phases))
            q_val = np.mean(chunk * (-np.sin(phases)))
            phase = np.arctan2(q_val, i_val)

            delta = phase - self._prev_phase
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
            bit = 1 if abs(delta) > np.pi / 2 else 0

            self._prev_phase = phase
            self._scheduler.advance_symbol()
            self._demod_pos += sps

            ch = self._parser.feed_bit(bit)
            if ch is not None:
                chars.append(ch)

            if self._parser.is_idle():
                # Commit any pending frame data (e.g. varicode)
                self._parser.finalize_frame()

                # Frame complete — update tracking
                if self._parser.frame_idx is not None:
                    self._frames_received = len(self._parser.frames)
                if self._parser.total_frames is not None:
                    self._total_frames = self._parser.total_frames

                if self._parser.all_frames_received:
                    self._done = True
                else:
                    # Reset for next frame: go back to searching
                    self._parser.reset_for_next_frame()
                    self._searching = True
                    self._last_search_len = 0
                    # Trim consumed audio
                    self._audio_buf = self._audio_buf[self._demod_pos:]
                    self._demod_pos = 0
                break

        # Trim consumed audio to keep memory bounded
        if self._demod_pos > 0 and not self._searching:
            self._audio_buf = self._audio_buf[self._demod_pos:]
            self._demod_pos = 0

        return chars
