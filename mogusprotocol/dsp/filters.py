"""FIR filter factories: root-raised-cosine pulse shaper, lowpass, bandpass."""

import numpy as np
from scipy.signal import firwin


from ..protocol.constants import SAMPLE_RATE, SAMPLES_PER_SYMBOL


def rrc_pulse(beta: float = 0.5, span: int = 6) -> np.ndarray:
    """Root-raised-cosine pulse for PSK symbol shaping.

    Args:
        beta: Roll-off factor (0-1).
        span: Number of symbols the pulse spans on each side.
    """
    N = span * SAMPLES_PER_SYMBOL
    t = np.arange(-N, N + 1) / SAMPLES_PER_SYMBOL  # in symbol periods

    h = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-10:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den

    h /= np.sqrt(np.sum(h ** 2))
    return h


def lowpass_fir(cutoff_hz: float, num_taps: int = 127) -> np.ndarray:
    """Design a lowpass FIR filter."""
    return firwin(num_taps, cutoff_hz, fs=SAMPLE_RATE)


def bandpass_fir(center_hz: float, bandwidth_hz: float = 100.0, num_taps: int = 255) -> np.ndarray:
    """Design a bandpass FIR filter centered on a carrier frequency."""
    low = center_hz - bandwidth_hz / 2
    high = center_hz + bandwidth_hz / 2
    low = max(low, 1.0)
    high = min(high, SAMPLE_RATE / 2 - 1)
    return firwin(num_taps, [low, high], pass_zero=False, fs=SAMPLE_RATE)
