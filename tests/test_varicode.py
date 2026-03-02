"""Tests for varicode encoding/decoding."""

from mogusprotocol.protocol.varicode import encode, decode, VARICODE_TABLE


def test_encode_decode_roundtrip():
    text = "Hello, World!"
    bits = encode(text)
    result = decode(bits)
    assert result == text


def test_encode_decode_lowercase():
    text = "the quick brown fox"
    bits = encode(text)
    result = decode(bits)
    assert result == text


def test_encode_decode_numbers():
    text = "12345"
    bits = encode(text)
    result = decode(bits)
    assert result == text


def test_encode_decode_all_printable():
    text = "".join(chr(i) for i in range(32, 127))
    bits = encode(text)
    result = decode(bits)
    assert result == text


def test_space_is_single_bit():
    # Space (0x20) should be encoded as "1"
    assert VARICODE_TABLE[32] == "1"


def test_e_is_short():
    # 'e' should be one of the shortest codes
    assert VARICODE_TABLE[ord("e")] == "11"


def test_separator_between_chars():
    bits = encode("ab")
    # Should contain 00 separator between a and b
    bitstr = "".join(str(b) for b in bits)
    # 'a' = 1011, sep = 00, 'b' = 1011111, sep = 00
    assert "00" in bitstr


def test_empty_string():
    assert encode("") == []
    assert decode([]) == ""


def test_no_consecutive_zeros_in_codes():
    for i, code in enumerate(VARICODE_TABLE):
        assert "00" not in code, f"Character {i} ({chr(i)!r}) has consecutive zeros in code: {code}"
