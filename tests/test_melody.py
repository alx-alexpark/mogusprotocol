"""Tests for melody and hop scheduler."""

from mogusprotocol.protocol.melody import (
    HopScheduler,
    HOP_SEQUENCE_HZ,
    HOP_SEQUENCE_NOTES,
    NUM_HOPS,
)
from mogusprotocol.protocol.constants import SYMBOLS_PER_HOP


def test_hop_count():
    assert NUM_HOPS == 33


def test_first_note_is_c6():
    assert HOP_SEQUENCE_NOTES[0] == "C6"
    assert HOP_SEQUENCE_HZ[0] == 1046


def test_freq_range_in_ssb_passband():
    for freq in HOP_SEQUENCE_HZ:
        assert 300 <= freq <= 3000, f"Frequency {freq} Hz out of SSB passband (300-3000)"


def test_scheduler_advances_after_symbols_per_hop():
    sched = HopScheduler()
    initial_freq = sched.current_freq
    for i in range(SYMBOLS_PER_HOP - 1):
        hopped = sched.advance_symbol()
        assert not hopped
    hopped = sched.advance_symbol()
    assert hopped


def test_scheduler_wraps_around():
    sched = HopScheduler()
    for _ in range(NUM_HOPS * SYMBOLS_PER_HOP):
        sched.advance_symbol()
    # Should wrap back to index 0
    assert sched.hop_index == 0
    assert sched.current_freq == float(HOP_SEQUENCE_HZ[0])


def test_scheduler_reset():
    sched = HopScheduler()
    for _ in range(50):
        sched.advance_symbol()
    sched.reset(5)
    assert sched.hop_index == 5
    assert sched.current_freq == float(HOP_SEQUENCE_HZ[5])
    assert sched.symbols_remaining == SYMBOLS_PER_HOP


def test_lowest_freq():
    assert min(HOP_SEQUENCE_HZ) == 525
