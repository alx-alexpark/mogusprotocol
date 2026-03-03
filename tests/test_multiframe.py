"""Tests for multi-frame per-frame sync support."""

import numpy as np

from mogusprotocol.protocol.frame import (
    build_frame,
    build_frames,
    build_multiframe,
    encode_audio,
    RxFrameParser,
)
from mogusprotocol.protocol.constants import (
    SYNC_WORD,
    SYNC_BITS,
    PREAMBLE_SYMBOLS,
    MAX_FRAME_PAYLOAD_BYTES,
    MODE_FEC_BPSK,
    MODE_FEC_COMPRESSED,
)
from mogusprotocol.dsp.modulator import PSKModulator
from mogusprotocol.dsp.demodulator import PSKDemodulator
from mogusprotocol.dsp.streaming_demod import StreamingDemodulator


def test_build_frames_short_message():
    """Short message fits in one frame."""
    frames = build_frames("Hi", mode=MODE_FEC_BPSK)
    assert len(frames) == 1


def test_build_frames_long_message():
    """Long message splits into multiple frames."""
    text = "A" * (MAX_FRAME_PAYLOAD_BYTES * 3)
    frames = build_frames(text, mode=MODE_FEC_BPSK)
    assert len(frames) == 3


def test_each_frame_has_preamble_and_sync():
    """Each frame has its own preamble and sync word."""
    text = "Hello world! This is a long test message for multi-frame."
    frames = build_frames(text, mode=MODE_FEC_BPSK)
    assert len(frames) > 1

    sync_bitstr = format(SYNC_WORD, "016b")
    for i, frame_bits in enumerate(frames):
        bitstr = "".join(str(b) for b in frame_bits)
        assert sync_bitstr in bitstr, f"Frame {i} missing sync word"
        # Check preamble (first 64 bits should be alternating)
        for j in range(PREAMBLE_SYMBOLS):
            expected = 1 if j % 2 == 0 else 0
            assert frame_bits[j] == expected, f"Frame {i} preamble bit {j} wrong"


def test_frame_idx_and_total_frames():
    """Each frame has correct frame_idx and total_frames in header."""
    text = "A" * (MAX_FRAME_PAYLOAD_BYTES * 3)
    frames = build_frames(text, mode=MODE_FEC_BPSK)
    assert len(frames) == 3

    for idx, frame_bits in enumerate(frames):
        parser = RxFrameParser()
        for bit in frame_bits:
            parser.feed_bit(bit)
        assert parser.frame_idx == idx, f"Frame {idx}: got frame_idx={parser.frame_idx}"
        assert parser._frame_total == 3, f"Frame {idx}: got total_frames={parser._frame_total}"


def test_single_frame_roundtrip():
    """Single-frame message round-trips through parser."""
    text = "CQ CQ"
    bits = build_frame(text, mode=MODE_FEC_BPSK)

    parser = RxFrameParser()
    for bit in bits:
        parser.feed_bit(bit)

    assert parser.crc_ok is True
    assert parser.decoded_text == text
    assert parser.frame_idx == 0
    assert parser.total_frames == 1


def test_multiframe_roundtrip():
    """Multi-frame message round-trips through parser (bit-level)."""
    text = "Hello world! This is a longer message that should be split across multiple frames for testing."
    bits = build_multiframe(text, mode=MODE_FEC_BPSK)

    parser = RxFrameParser()
    for bit in bits:
        ch = parser.feed_bit(bit)
        if parser.is_idle() and not parser.all_frames_received:
            parser.finalize_frame()
            parser.reset_for_next_frame()

    parser.finalize_frame()
    assert parser.decoded_text == text
    assert parser.all_frames_received


def test_multiframe_compressed_roundtrip():
    """Multi-frame compressed message round-trips (bit-level)."""
    text = "Hello world! " * 20  # Long enough to need multiple frames even compressed
    bits = build_multiframe(text, mode=MODE_FEC_COMPRESSED)

    parser = RxFrameParser()
    for bit in bits:
        parser.feed_bit(bit)
        if parser.is_idle() and not parser.all_frames_received:
            parser.finalize_frame()
            parser.reset_for_next_frame()

    parser.finalize_frame()
    assert parser.crc_ok is True
    assert parser.decoded_text == text


def test_mid_stream_join():
    """Dropping the first frame(s) should still decode remaining frames."""
    text = "AAAA" * MAX_FRAME_PAYLOAD_BYTES  # ~4 frames
    frames = build_frames(text, mode=MODE_FEC_BPSK)
    assert len(frames) >= 2

    # Drop first frame, concatenate remaining
    remaining_bits: list[int] = []
    for frame in frames[1:]:
        remaining_bits.extend(frame)

    parser = RxFrameParser()
    for bit in remaining_bits:
        parser.feed_bit(bit)
        if parser.is_idle() and not parser.all_frames_received:
            parser.finalize_frame()
            parser.reset_for_next_frame()

    parser.finalize_frame()

    # Should have decoded frames 1..N but not frame 0
    assert 0 not in parser.frames
    assert len(parser.frames) == len(frames) - 1
    # Each frame's text should be present
    for idx in range(1, len(frames)):
        assert idx in parser.frames


def test_loopback_single_frame_audio():
    """Encode single frame -> modulate -> demodulate -> verify."""
    text = "CQ CQ DE MOGUS"
    audio = encode_audio(text)

    silence = np.zeros(1000)
    audio_with_silence = np.concatenate([silence, audio, silence])

    demod = PSKDemodulator(energy_threshold=0.001)
    decoded = demod.demodulate(audio_with_silence)

    assert len(decoded) > 0, "Single-frame loopback produced no output"


def test_loopback_multiframe_audio():
    """Encode multi-frame -> modulate each frame independently -> demodulate."""
    text = "HELLO WORLD THIS IS A MULTI FRAME TEST MESSAGE 73"
    frames = build_frames(text)
    assert len(frames) >= 2, "Test needs a multi-frame message"

    audio = encode_audio(text)
    silence = np.zeros(1000)
    audio_with_silence = np.concatenate([silence, audio, silence])

    demod = PSKDemodulator(energy_threshold=0.001)
    decoded = demod.demodulate(audio_with_silence)

    assert decoded == text, f"Expected {text!r}, got {decoded!r}"


def test_streaming_demod_single_frame():
    """Streaming demod decodes a single-frame message."""
    text = "CQ CQ"
    audio = encode_audio(text)
    silence = np.zeros(2000)
    audio_full = np.concatenate([silence, audio, silence])

    demod = StreamingDemodulator(energy_threshold=0.001)
    chunk_size = 2048
    for i in range(0, len(audio_full), chunk_size):
        demod.feed(audio_full[i:i + chunk_size])
        if demod.done:
            break

    assert demod.done
    assert demod.decoded_text == text
    assert demod.crc_ok is True


def test_streaming_demod_multiframe():
    """Streaming demod decodes a multi-frame message."""
    text = "HELLO WORLD THIS IS A MULTI FRAME TEST MESSAGE 73"
    audio = encode_audio(text)
    silence = np.zeros(2000)
    audio_full = np.concatenate([silence, audio, silence])

    demod = StreamingDemodulator(energy_threshold=0.001)
    chunk_size = 2048
    for i in range(0, len(audio_full), chunk_size):
        demod.feed(audio_full[i:i + chunk_size])
        if demod.done:
            break

    assert demod.done, f"Streaming demod never finished (received {demod.frames_received}/{demod.total_frames})"
    assert demod.decoded_text == text
    assert demod.crc_ok is True


def test_build_frame_backward_compat():
    """build_frame() still returns a flat bit list."""
    bits = build_frame("test")
    assert isinstance(bits, list)
    assert all(b in (0, 1) for b in bits)
