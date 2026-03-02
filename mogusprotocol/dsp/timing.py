"""Gardner symbol timing recovery for PSK demodulation."""

import numpy as np

from ..protocol.constants import SAMPLES_PER_SYMBOL


class GardnerTimingRecovery:
    """Gardner timing error detector with proportional-integral loop filter.

    Operates on baseband (post-downconversion) I or Q samples.
    Outputs one sample per symbol at the optimal sampling instant.
    """

    def __init__(self, kp: float = 0.01, ki: float = 0.001):
        self.kp = kp
        self.ki = ki
        self._mu = 0.0  # fractional sample offset
        self._integrator = 0.0
        self._prev_sample = 0.0
        self._prev_mid = 0.0

    def process(self, samples: np.ndarray) -> list[float]:
        """Process a block of baseband samples, return symbol-rate samples.

        Args:
            samples: Baseband samples at SAMPLE_RATE.

        Returns:
            List of symbol-spaced sample values.
        """
        symbols_out: list[float] = []
        idx = 0
        sps = SAMPLES_PER_SYMBOL

        while idx + sps < len(samples):
            # Current optimal sample point
            sample_point = int(idx + self._mu)
            if sample_point + sps >= len(samples):
                break

            # Midpoint between previous and current symbol
            mid_point = sample_point + sps // 2
            current = samples[min(sample_point + sps, len(samples) - 1)]
            mid = samples[min(mid_point, len(samples) - 1)]

            # Gardner TED: e = mid * (prev - current)
            error = mid * (self._prev_sample - current)

            # Loop filter (PI)
            self._integrator += self.ki * error
            adjust = self.kp * error + self._integrator

            # Clamp adjustment
            adjust = np.clip(adjust, -sps / 4, sps / 4)
            self._mu += adjust

            self._prev_sample = current
            self._prev_mid = mid

            symbols_out.append(current)
            idx += sps

        return symbols_out

    def reset(self):
        self._mu = 0.0
        self._integrator = 0.0
        self._prev_sample = 0.0
        self._prev_mid = 0.0
