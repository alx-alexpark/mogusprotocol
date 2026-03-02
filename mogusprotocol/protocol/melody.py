import numpy as np

from .constants import (
    HOP_DURATION_SAMPLES,
    SAMPLE_RATE,
    SYMBOLS_PER_HOP,
    SAMPLES_PER_SYMBOL,
)

# Among Us Drip melody - frequencies taken directly from the reference
# buzzer implementation. These are the actual carrier frequencies in Hz,
# already in the SSB passband (~525-1510 Hz).
#
# Rests in the original melody are mapped to the preceding tone's
# frequency to maintain carrier continuity (data still flows during rests).
#
# Part 1:  C6  Eb6  F6  F#6  F6  Eb6  C6  [rest]  Bb5  D6  C6
# Bridge:  [rest]  G5  C5  [rest]
# Part 2:  C6  Eb6  F6  F#6  F6  Eb6  F6  [rest]
# Fast:    [rest] F#6  F6  Eb6  F#6  F6  Eb6  ~Eb6  C6  [rest]

HOP_SEQUENCE_HZ = np.array([
    # Part 1 (7 tones + rest + 3 tones)
    1046, 1244, 1400, 1510, 1400, 1244, 1046,
    1046,  # rest -> hold carrier
    932, 1174, 1046,
    # Bridge (rest + 2 tones + rest)
    1046,  # rest -> hold
    780, 525,
    525,   # rest -> hold
    # Part 2 (7 tones + rest)
    1046, 1244, 1400, 1510, 1400, 1244, 1400,
    1400,  # rest -> hold
    # Fast part (rest + 8 tones + rest)
    1400,  # rest -> hold
    1510, 1400, 1244, 1510, 1400, 1244, 1200, 1050,
    1050,  # rest -> hold
], dtype=np.float64)

NUM_HOPS = len(HOP_SEQUENCE_HZ)

# Human-readable note labels (for debugging/display)
HOP_SEQUENCE_NOTES = [
    "C6", "Eb6", "F6", "F#6", "F6", "Eb6", "C6",
    "C6",
    "Bb5", "D6", "C6",
    "C6",
    "G5", "C5",
    "C5",
    "C6", "Eb6", "F6", "F#6", "F6", "Eb6", "F6",
    "F6",
    "F6",
    "F#6", "F6", "Eb6", "F#6", "F6", "Eb6", "~Eb6", "C6",
    "C6",
]


class HopScheduler:
    """Tracks position in the hop sequence for TX or RX."""

    def __init__(self, start_index: int = 0):
        self.hop_index = start_index
        self.samples_remaining = HOP_DURATION_SAMPLES
        self.symbols_remaining = SYMBOLS_PER_HOP

    @property
    def current_freq(self) -> float:
        return float(HOP_SEQUENCE_HZ[self.hop_index % NUM_HOPS])

    def advance_symbol(self) -> bool:
        """Advance by one symbol. Returns True if a hop boundary was crossed."""
        self.symbols_remaining -= 1
        self.samples_remaining -= SAMPLES_PER_SYMBOL
        if self.symbols_remaining <= 0:
            self.hop_index = (self.hop_index + 1) % NUM_HOPS
            self.samples_remaining = HOP_DURATION_SAMPLES
            self.symbols_remaining = SYMBOLS_PER_HOP
            return True
        return False

    def reset(self, index: int = 0):
        self.hop_index = index
        self.samples_remaining = HOP_DURATION_SAMPLES
        self.symbols_remaining = SYMBOLS_PER_HOP
