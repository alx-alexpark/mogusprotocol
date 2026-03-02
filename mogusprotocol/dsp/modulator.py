"""BPSK modulator with Among Us Drip carrier hopping."""

import numpy as np

from ..protocol.constants import (
    SAMPLE_RATE,
    SAMPLES_PER_SYMBOL,
    CROSSFADE_SAMPLES,
)
from ..protocol.melody import HopScheduler
from .filters import rrc_pulse


class PSKModulator:
    """BPSK modulator with frequency-hopping carrier.

    Uses differential encoding: a 1 bit causes a pi phase flip relative
    to the previous symbol. Carrier frequency follows the hop scheduler.
    Phase accumulator is continuous across hop boundaries.
    """

    def __init__(self):
        self.scheduler = HopScheduler()
        self.theta = 0.0  # continuous phase accumulator
        self._prev_symbol = 1  # differential encoding state
        self._pulse = rrc_pulse()

    def modulate(self, bits: list[int]) -> np.ndarray:
        """Modulate a complete bit sequence into audio samples.

        Args:
            bits: List of 0/1 values.

        Returns:
            Audio samples as float64 array, normalized to [-1, 1].
        """
        self.scheduler.reset()
        self.theta = 0.0
        self._prev_symbol = 1

        if not bits:
            return np.array([], dtype=np.float64)

        # Differential encoding: flip phase on 1
        symbols = []
        for bit in bits:
            if bit == 1:
                self._prev_symbol *= -1
            symbols.append(self._prev_symbol)

        num_samples = len(symbols) * SAMPLES_PER_SYMBOL

        # Generate baseband NRZ symbols upsampled to sample rate
        baseband = np.zeros(num_samples)
        for i, sym in enumerate(symbols):
            start = i * SAMPLES_PER_SYMBOL
            baseband[start:start + SAMPLES_PER_SYMBOL] = sym

        # Apply pulse shaping (keep output same length as input)
        shaped = np.convolve(baseband, self._pulse, mode="same")

        # Generate carrier with hopping
        output = np.zeros(num_samples)
        sample_idx = 0
        symbol_count = 0
        total_symbols = len(symbols)

        while symbol_count < total_symbols:
            freq = self.scheduler.current_freq
            sym_samples = SAMPLES_PER_SYMBOL
            end_idx = min(sample_idx + sym_samples, len(shaped))

            for j in range(sample_idx, end_idx):
                self.theta += 2.0 * np.pi * freq / SAMPLE_RATE
                output[j] = shaped[j] * np.cos(self.theta)

            sample_idx = end_idx
            symbol_count += 1
            hopped = self.scheduler.advance_symbol()

            # Apply crossfade at hop boundaries
            if hopped and sample_idx > CROSSFADE_SAMPLES and sample_idx < len(output):
                fade_len = min(CROSSFADE_SAMPLES, len(output) - sample_idx, sample_idx)
                fade_out = np.linspace(1.0, 0.0, fade_len)
                fade_in = np.linspace(0.0, 1.0, fade_len)
                # Smooth the transition region
                region = output[sample_idx - fade_len:sample_idx]
                output[sample_idx - fade_len:sample_idx] = region * (fade_out * 0.5 + 0.5)

        # Normalize
        peak = np.max(np.abs(output))
        if peak > 0:
            output /= peak

        return output

    def modulate_streaming(self, bit_queue) -> np.ndarray:
        """Modulate bits from a queue, yielding audio chunks.

        Args:
            bit_queue: Queue-like object with .get() returning int bits,
                       or None to signal end.

        Yields:
            Audio sample arrays (one symbol at a time).
        """
        chunks = []
        while True:
            bit = bit_queue.get()
            if bit is None:
                break

            # Differential encode
            if bit == 1:
                self._prev_symbol *= -1
            symbol_val = self._prev_symbol

            freq = self.scheduler.current_freq
            samples = np.zeros(SAMPLES_PER_SYMBOL)
            for j in range(SAMPLES_PER_SYMBOL):
                self.theta += 2.0 * np.pi * freq / SAMPLE_RATE
                samples[j] = symbol_val * np.cos(self.theta)

            self.scheduler.advance_symbol()
            chunks.append(samples)

        return np.concatenate(chunks) if chunks else np.array([])
