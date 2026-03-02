"""Tests for PSK modulator."""

import numpy as np

from mogusprotocol.dsp.modulator import PSKModulator
from mogusprotocol.protocol.constants import SAMPLE_RATE, SAMPLES_PER_SYMBOL
from mogusprotocol.protocol.melody import HOP_SEQUENCE_HZ


def test_modulate_produces_audio():
    mod = PSKModulator()
    bits = [1, 0, 1, 0, 1, 0, 1, 0]
    audio = mod.modulate(bits)
    assert len(audio) == len(bits) * SAMPLES_PER_SYMBOL
    assert audio.dtype == np.float64


def test_modulate_normalized():
    mod = PSKModulator()
    bits = [1, 0, 1, 1, 0, 0, 1, 0] * 10
    audio = mod.modulate(bits)
    assert np.max(np.abs(audio)) <= 1.0 + 1e-6


def test_modulate_empty():
    mod = PSKModulator()
    audio = mod.modulate([])
    assert len(audio) == 0


def test_carrier_frequency_present():
    """Check that the first carrier frequency has energy in the output spectrum."""
    mod = PSKModulator()
    bits = [1, 0] * 50
    audio = mod.modulate(bits)

    # FFT
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / SAMPLE_RATE)

    # Find energy near first carrier
    first_freq = float(HOP_SEQUENCE_HZ[0])
    mask = np.abs(freqs - first_freq) < 50  # within 50 Hz
    energy_near_carrier = np.sum(spectrum[mask] ** 2)
    total_energy = np.sum(spectrum ** 2)

    # At least some energy should be near the first carrier
    assert energy_near_carrier > 0.01 * total_energy
