"""Automatic Gain Control for RX signal normalization."""

import numpy as np


class AGC:
    """Power-estimating AGC with separate attack and decay time constants."""

    def __init__(
        self,
        target_power: float = 1.0,
        attack: float = 0.01,
        decay: float = 0.001,
        max_gain: float = 1000.0,
    ):
        self.target_power = target_power
        self.attack = attack
        self.decay = decay
        self.max_gain = max_gain
        self._power_est = target_power
        self._gain = 1.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply AGC to a block of samples."""
        output = np.empty_like(samples)
        for i, s in enumerate(samples):
            instant_power = s * s
            if instant_power > self._power_est:
                alpha = self.attack
            else:
                alpha = self.decay
            self._power_est += alpha * (instant_power - self._power_est)
            self._power_est = max(self._power_est, 1e-12)

            desired_gain = np.sqrt(self.target_power / self._power_est)
            self._gain = min(desired_gain, self.max_gain)
            output[i] = s * self._gain
        return output

    def reset(self):
        self._power_est = self.target_power
        self._gain = 1.0
