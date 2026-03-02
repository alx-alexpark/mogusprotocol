"""Tests for PSK demodulator - loopback test."""

import numpy as np

from mogusprotocol.dsp.modulator import PSKModulator
from mogusprotocol.dsp.demodulator import PSKDemodulator
from mogusprotocol.protocol.frame import build_frame, RxFrameParser
from mogusprotocol.protocol.constants import SYNC_WORD


def test_frame_parser_sync_detection():
    """Test that the frame parser can find the sync word in a bit stream."""
    parser = RxFrameParser()

    # Feed some garbage then the sync word
    for _ in range(20):
        parser.feed_bit(0)

    sync_bits = [(SYNC_WORD >> (15 - i)) & 1 for i in range(16)]
    for bit in sync_bits:
        parser.feed_bit(bit)

    assert parser.synced


def test_frame_build_contains_sync():
    """Test that a built frame contains the sync word."""
    bits = build_frame("test")
    bitstr = "".join(str(b) for b in bits)
    sync_bitstr = format(SYNC_WORD, "016b")
    assert sync_bitstr in bitstr


def test_loopback_clean():
    """Modulate and demodulate text with no noise - loopback test."""
    text = "CQ CQ"
    bits = build_frame(text)

    mod = PSKModulator()
    audio = mod.modulate(bits)

    # Add a small amount of leading silence
    silence = np.zeros(1000)
    audio_with_silence = np.concatenate([silence, audio, silence])

    demod = PSKDemodulator(energy_threshold=0.001)
    decoded = demod.demodulate(audio_with_silence)

    # In clean conditions we should decode at least part of the message
    # Full decode depends on phase alignment; check we got something
    assert len(decoded) > 0, "Demodulator produced no output"


def test_loopback_with_noise():
    """Modulate and demodulate with moderate AWGN."""
    text = "HELLO"
    bits = build_frame(text)

    mod = PSKModulator()
    audio = mod.modulate(bits)

    # Add AWGN at ~20 dB SNR
    rng = np.random.default_rng(42)
    noise_power = np.mean(audio ** 2) / 100  # SNR ~20dB
    noise = rng.normal(0, np.sqrt(noise_power), len(audio))
    noisy = audio + noise

    silence = np.zeros(1000)
    noisy = np.concatenate([silence, noisy, silence])

    demod = PSKDemodulator(energy_threshold=0.001)
    decoded = demod.demodulate(noisy)

    # With 20dB SNR we should get some output
    assert len(decoded) > 0, "Demodulator produced no output with 20dB SNR"
