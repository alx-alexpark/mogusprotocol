"""BPSK demodulator with carrier hopping sync and tracking."""

import numpy as np

from ..protocol.constants import (
    SAMPLE_RATE,
    SAMPLES_PER_SYMBOL,
    PREAMBLE_SYMBOLS,
)
from ..protocol.melody import HopScheduler, HOP_SEQUENCE_HZ
from ..protocol.frame import RxFrameParser
from .agc import AGC
from .timing import GardnerTimingRecovery


def _build_reference_preamble() -> np.ndarray:
    """Build a reference preamble waveform using the same method as the modulator."""
    from .modulator import PSKModulator
    # Preamble is alternating 1,0,1,0...
    preamble_bits = [1 if i % 2 == 0 else 0 for i in range(PREAMBLE_SYMBOLS)]
    mod = PSKModulator()
    return mod.modulate(preamble_bits)


class PSKDemodulator:
    """BPSK demodulator with Among Us Drip hop tracking.

    Strategy: cross-correlate with a reference preamble to find exact frame
    start, then demodulate with a continuous-phase LO locked to the hop
    schedule.  Loops to find multiple frames in batch audio.
    """

    def __init__(self, energy_threshold: float = 0.1):
        self.energy_threshold = energy_threshold
        self._ref_preamble = _build_reference_preamble()

    def demodulate(self, audio: np.ndarray) -> str:
        """Demodulate a complete audio buffer to text.

        Searches for multiple frames and reassembles text from all.
        """
        parser = RxFrameParser()
        remaining = audio
        min_frame_samples = SAMPLES_PER_SYMBOL * (PREAMBLE_SYMBOLS + 32)

        while len(remaining) >= min_frame_samples:
            offset = self._find_frame_start(remaining)
            if offset < 0:
                break

            sig = remaining[offset:]
            if len(sig) < SAMPLES_PER_SYMBOL * 2:
                break

            # Demodulate one frame
            scheduler = HopScheduler()
            scheduler.reset(0)
            n_symbols = len(sig) // SAMPLES_PER_SYMBOL

            lo_theta = 0.0
            prev_phase = 0.0
            symbols_consumed = 0

            for sym in range(n_symbols):
                freq = scheduler.current_freq
                s0 = sym * SAMPLES_PER_SYMBOL
                s1 = s0 + SAMPLES_PER_SYMBOL
                chunk = sig[s0:s1]

                phase_inc = 2.0 * np.pi * freq / SAMPLE_RATE
                phases = lo_theta + phase_inc * np.arange(1, SAMPLES_PER_SYMBOL + 1)
                lo_theta = phases[-1]

                i_val = np.mean(chunk * np.cos(phases))
                q_val = np.mean(chunk * (-np.sin(phases)))

                phase = np.arctan2(q_val, i_val)

                delta = phase - prev_phase
                delta = (delta + np.pi) % (2 * np.pi) - np.pi
                bit = 1 if abs(delta) > np.pi / 2 else 0

                prev_phase = phase
                scheduler.advance_symbol()
                symbols_consumed = sym + 1

                parser.feed_bit(bit)
                if parser.is_idle():
                    break

            # Commit any pending frame data (e.g. varicode)
            parser.finalize_frame()

            # Advance past the frame we just decoded
            samples_used = offset + symbols_consumed * SAMPLES_PER_SYMBOL
            remaining = remaining[samples_used:]

            if parser.all_frames_received:
                break

            # Reset parser for next frame search
            parser.reset_for_next_frame()

        return parser.decoded_text

    def _find_frame_start(self, audio: np.ndarray) -> int:
        """Find the exact sample offset where the next TX frame begins.

        Uses cross-correlation with a reference preamble, returning the
        first significant peak (not necessarily the global maximum) so
        that earlier frames are found before later ones.
        """
        ref = self._ref_preamble
        ref_len = len(ref)

        if len(audio) < ref_len:
            return -1

        search_len = len(audio)

        # Use FFT-based correlation for speed on long signals
        n_fft = 1
        while n_fft < search_len + ref_len:
            n_fft *= 2

        audio_pad = np.zeros(n_fft)
        audio_pad[:search_len] = audio[:search_len]
        ref_pad = np.zeros(n_fft)
        ref_pad[:ref_len] = ref[::-1]  # time-reversed for correlation

        corr = np.fft.irfft(np.fft.rfft(audio_pad) * np.fft.rfft(ref_pad))
        # Valid correlation region
        corr = np.abs(corr[ref_len - 1:search_len])

        max_val = np.max(corr)
        if max_val < self.energy_threshold:
            return -1

        # Find the first significant peak: scan for the first point where
        # correlation exceeds 40% of max, then find the local maximum
        # within a preamble-length window from that point.
        peak_threshold = max_val * 0.4
        above = np.where(corr >= peak_threshold)[0]
        if len(above) == 0:
            return -1

        first_above = int(above[0])
        window_end = min(first_above + ref_len, len(corr))
        local_peak = first_above + np.argmax(corr[first_above:window_end])
        return int(local_peak)
