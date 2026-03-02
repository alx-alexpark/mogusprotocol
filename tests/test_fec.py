"""Tests for the convolutional FEC encoder/decoder and CRC-16."""

import random

from mogusprotocol.dsp.fec import ConvEncoder, ViterbiDecoder, crc16_ccitt, crc16_bits


def test_conv_roundtrip_clean():
    """Encode then decode with no errors — should recover original bits."""
    data = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1]
    encoder = ConvEncoder()
    encoded = encoder.encode(data)
    # Rate 1/2: output is 2 * (len(data) + 6 flush bits)
    assert len(encoded) == 2 * (len(data) + 6)

    decoder = ViterbiDecoder()
    decoded = decoder.decode(encoded)
    assert decoded == data


def test_conv_roundtrip_random():
    """Roundtrip random data of various lengths."""
    rng = random.Random(42)
    for length in [8, 32, 64, 128, 256]:
        data = [rng.randint(0, 1) for _ in range(length)]
        encoder = ConvEncoder()
        encoded = encoder.encode(data)
        decoder = ViterbiDecoder()
        decoded = decoder.decode(encoded)
        assert decoded == data, f"Failed for length {length}"


def test_conv_corrects_single_errors():
    """Viterbi should correct isolated single-bit errors."""
    data = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1]
    encoder = ConvEncoder()
    encoded = encoder.encode(data)

    # Flip one bit at various positions
    for pos in range(0, len(encoded), 7):
        corrupted = list(encoded)
        corrupted[pos] ^= 1
        decoder = ViterbiDecoder()
        decoded = decoder.decode(corrupted)
        assert decoded == data, f"Failed to correct error at position {pos}"


def test_conv_corrects_burst_errors():
    """Viterbi should handle small burst errors."""
    rng = random.Random(123)
    data = [rng.randint(0, 1) for _ in range(64)]
    encoder = ConvEncoder()
    encoded = encoder.encode(data)

    # Flip 3 consecutive bits (small burst)
    corrupted = list(encoded)
    start = len(corrupted) // 3
    for i in range(3):
        corrupted[start + i] ^= 1

    decoder = ViterbiDecoder()
    decoded = decoder.decode(corrupted)
    assert decoded == data


def test_conv_empty():
    """Empty input encodes only flush bits."""
    encoder = ConvEncoder()
    encoded = encoder.encode([])
    # Only flush bits: 2 * 6 = 12
    assert len(encoded) == 12

    decoder = ViterbiDecoder()
    decoded = decoder.decode(encoded)
    # Decoder returns empty since all 6 decoded bits are stripped as flush
    assert decoded == []


def test_crc16_known_value():
    """CRC-16-CCITT of 'A' should match known value."""
    crc = crc16_ccitt(b"A")
    assert isinstance(crc, int)
    assert 0 <= crc <= 0xFFFF


def test_crc16_detects_errors():
    """CRC should differ if data is modified."""
    original = b"Hello, world!"
    crc_orig = crc16_ccitt(original)
    corrupted = bytearray(original)
    corrupted[3] ^= 0x01
    crc_bad = crc16_ccitt(bytes(corrupted))
    assert crc_orig != crc_bad


def test_crc16_bits_roundtrip():
    """crc16_bits should produce 16 bits consistent with crc16_ccitt."""
    data_bits = [1, 0, 1, 0, 0, 0, 0, 1]  # 0xA1
    crc_bits_result = crc16_bits(data_bits)
    assert len(crc_bits_result) == 16
    assert all(b in (0, 1) for b in crc_bits_result)


def test_frame_fec_roundtrip():
    """Full frame build + parse roundtrip with FEC modes."""
    from mogusprotocol.protocol.frame import build_frame, RxFrameParser
    from mogusprotocol.protocol.constants import MODE_FEC_BPSK, MODE_FEC_COMPRESSED

    for mode in [MODE_FEC_BPSK, MODE_FEC_COMPRESSED]:
        text = "Hello FEC!"
        bits = build_frame(text, mode=mode)

        parser = RxFrameParser()
        result = None
        for bit in bits:
            ch = parser.feed_bit(bit)
            if ch is not None:
                result = ch

        assert parser.synced
        assert parser.crc_ok is True
        assert result == text, f"Mode {mode}: got {result!r}"


def test_frame_fec_detects_corruption():
    """FEC frame with too many errors should fail CRC check."""
    from mogusprotocol.protocol.frame import build_frame, RxFrameParser
    from mogusprotocol.protocol.constants import MODE_FEC_BPSK

    text = "Test"
    bits = build_frame(text, mode=MODE_FEC_BPSK)

    # Corrupt a large burst in the payload area (past header)
    corrupted = list(bits)
    payload_start = 64 + 16 + 8 + 8 + 16  # preamble + sync + ver + mode + length
    for i in range(20):
        if payload_start + i < len(corrupted):
            corrupted[payload_start + i] ^= 1

    parser = RxFrameParser()
    for bit in corrupted:
        parser.feed_bit(bit)

    # With 20 flipped bits, CRC should fail
    assert parser.crc_ok is False
