"""Minimum SNR benchmark — find the lowest SNR where decoding still works.

Sweeps SNR from high to low and records the minimum at which we get
perfect decode (FEC + CRC OK).  Runs multiple trials per SNR level to
account for random noise realisation.  The final assertion guards against
regressions: if a code change raises the minimum SNR floor, this test
fails.
"""

import numpy as np
import pytest

from mogusprotocol.protocol.frame import encode_audio, build_frames
from mogusprotocol.protocol.constants import (
    MODE_FEC_BPSK,
    MODE_FEC_COMPRESSED,
)
from mogusprotocol.dsp.demodulator import PSKDemodulator
from mogusprotocol.dsp.streaming_demod import StreamingDemodulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_awgn(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add AWGN to audio at the specified SNR (dB)."""
    sig_power = np.mean(audio ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise


def _try_decode_batch(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> str:
    """Add noise at snr_db and attempt batch decode."""
    silence = np.zeros(1000)
    noisy = np.concatenate([silence, _add_awgn(audio, snr_db, rng), silence])
    demod = PSKDemodulator(energy_threshold=0.001)
    return demod.demodulate(noisy)


def _try_decode_streaming(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> str:
    """Add noise at snr_db and attempt streaming decode."""
    silence = np.zeros(2000)
    noisy = np.concatenate([silence, _add_awgn(audio, snr_db, rng), silence])
    demod = StreamingDemodulator(energy_threshold=0.001)
    chunk_size = 4096
    for i in range(0, len(noisy), chunk_size):
        demod.feed(noisy[i:i + chunk_size])
        if demod.done:
            break
    return demod.decoded_text


def _find_min_snr(
    text: str,
    mode: int,
    decode_fn,
    snr_range: np.ndarray,
    trials: int = 5,
    required_successes: int = 3,
) -> float | None:
    """Sweep from high SNR to low, return lowest where decode succeeds.

    Returns the lowest SNR (dB) at which at least `required_successes`
    out of `trials` attempts produce the correct text, or None if even
    the highest SNR fails.
    """
    audio = encode_audio(text, mode=mode)
    min_passing_snr = None

    for snr_db in snr_range:
        successes = 0
        for trial in range(trials):
            seed = 1000 * int((snr_db + 100) * 10) + trial
            rng = np.random.default_rng(seed=seed)
            decoded = decode_fn(audio, float(snr_db), rng)
            if decoded == text:
                successes += 1
        if successes >= required_successes:
            min_passing_snr = float(snr_db)

    return min_passing_snr


# ---------------------------------------------------------------------------
# Benchmarks — regression gates
# ---------------------------------------------------------------------------

# Focused sweep: 6 dB down to -2 dB in 2 dB steps (fast, covers the
# interesting region around the decode threshold).
SNR_RANGE = np.arange(6, -4, -2, dtype=float)  # [6, 4, 2, 0, -2]


class TestSNRBenchmark:
    """Minimum SNR floor benchmarks.  Fail if sensitivity regresses."""

    def test_fec_bpsk_min_snr_batch(self):
        """FEC-BPSK batch decode must work at 0 dB SNR or better."""
        min_snr = _find_min_snr(
            "CQ CQ DE MOGUS", MODE_FEC_BPSK, _try_decode_batch, SNR_RANGE,
        )
        assert min_snr is not None, "Could not decode at any SNR"
        assert min_snr <= 0, f"Regression: min SNR = {min_snr} dB (limit 0 dB)"

    def test_fec_bpsk_min_snr_streaming(self):
        """FEC-BPSK streaming decode must work at 0 dB SNR or better."""
        min_snr = _find_min_snr(
            "CQ CQ DE MOGUS", MODE_FEC_BPSK, _try_decode_streaming, SNR_RANGE,
        )
        assert min_snr is not None, "Could not decode at any SNR"
        assert min_snr <= 0, f"Regression: min SNR = {min_snr} dB (limit 0 dB)"

    def test_fec_compressed_min_snr_batch(self):
        """FEC-compressed batch decode must work at 0 dB SNR or better."""
        min_snr = _find_min_snr(
            "HELLO WORLD 73", MODE_FEC_COMPRESSED, _try_decode_batch, SNR_RANGE,
        )
        assert min_snr is not None, "Could not decode at any SNR"
        assert min_snr <= 0, f"Regression: min SNR = {min_snr} dB (limit 0 dB)"

    def test_multiframe_min_snr_batch(self):
        """Multi-frame batch decode must work at 2 dB SNR or better."""
        text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 1234567890"
        assert len(build_frames(text, mode=MODE_FEC_BPSK)) >= 2
        min_snr = _find_min_snr(
            text, MODE_FEC_BPSK, _try_decode_batch, SNR_RANGE,
        )
        assert min_snr is not None, "Could not decode at any SNR"
        assert min_snr <= 2, f"Regression: min SNR = {min_snr} dB (limit 2 dB)"

    def test_multiframe_min_snr_streaming(self):
        """Multi-frame streaming decode must work at 2 dB SNR or better."""
        text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 1234567890"
        assert len(build_frames(text, mode=MODE_FEC_BPSK)) >= 2
        min_snr = _find_min_snr(
            text, MODE_FEC_BPSK, _try_decode_streaming, SNR_RANGE,
        )
        assert min_snr is not None, "Could not decode at any SNR"
        assert min_snr <= 2, f"Regression: min SNR = {min_snr} dB (limit 2 dB)"


# ---------------------------------------------------------------------------
# Report — prints actual floors (run with pytest -s)
# ---------------------------------------------------------------------------

def test_snr_report(capsys):
    """Print actual minimum SNR floors (informational, always passes)."""
    snr_range = np.arange(10, -4, -2, dtype=float)

    configs = [
        ("FEC-BPSK  batch    short", "CQ CQ DE MOGUS", MODE_FEC_BPSK, _try_decode_batch),
        ("FEC-BPSK  stream   short", "CQ CQ DE MOGUS", MODE_FEC_BPSK, _try_decode_streaming),
        ("FEC-COMP  batch    short", "HELLO WORLD 73", MODE_FEC_COMPRESSED, _try_decode_batch),
        ("FEC-BPSK  batch    multi", "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 1234567890", MODE_FEC_BPSK, _try_decode_batch),
        ("FEC-BPSK  stream   multi", "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 1234567890", MODE_FEC_BPSK, _try_decode_streaming),
    ]

    lines = ["", "SNR Benchmark Results", "=" * 50]
    for label, text, mode, fn in configs:
        min_snr = _find_min_snr(text, mode, fn, snr_range)
        snr_str = f"{min_snr:+.0f} dB" if min_snr is not None else "FAIL"
        lines.append(f"  {label}:  {snr_str}")
    lines.append("=" * 50)

    with capsys.disabled():
        print("\n".join(lines))
