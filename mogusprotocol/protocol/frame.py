"""Frame assembly and parsing for the mogus protocol.

Frame structure (MODE_BPSK / varicode):
  [preamble 1010... (64 symbols)]
  [SYNC_WORD 16 bits]
  [VERSION 8 bits]
  [MODE 8 bits = 0]
  [payload varicode bits]
  [tail zeros (16 symbols)]

Frame structure (MODE_COMPRESSED / zlib):
  [preamble 1010... (64 symbols)]
  [SYNC_WORD 16 bits]
  [VERSION 8 bits]
  [MODE 8 bits = 1]
  [LENGTH 16 bits - compressed byte count]
  [payload: zlib compressed bytes, 8 bits each]
  [tail zeros (16 symbols)]

Frame structure (MODE_FEC_BPSK / MODE_FEC_COMPRESSED):
  Same as above but payload is wrapped:
  [LENGTH 16 bits - encoded bit count]
  [FEC-encoded payload: conv(payload_bits + CRC-16)]
  FEC uses rate 1/2 K=7 convolutional code (NASA/CCSDS).
"""

import zlib
from enum import Enum, auto

from . import varicode
from .constants import (
    PREAMBLE_SYMBOLS,
    SYNC_WORD,
    SYNC_BITS,
    VERSION_BITS,
    MODE_BITS,
    LENGTH_BITS,
    CRC_BITS,
    TAIL_SYMBOLS,
    PROTOCOL_VERSION,
    MODE_BPSK,
    MODE_COMPRESSED,
    MODE_FEC_BPSK,
    MODE_FEC_COMPRESSED,
)
from ..dsp.fec import ConvEncoder, ViterbiDecoder, crc16_bits


def _int_to_bits(value: int, num_bits: int) -> list[int]:
    """Convert integer to MSB-first bit list."""
    return [(value >> (num_bits - 1 - i)) & 1 for i in range(num_bits)]


def _bits_to_int(bits: list[int]) -> int:
    """Convert MSB-first bit list to integer."""
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val


def _bytes_to_bits(data: bytes) -> list[int]:
    """Convert bytes to MSB-first bit list."""
    bits: list[int] = []
    for byte in data:
        bits.extend(_int_to_bits(byte, 8))
    return bits


def _bits_to_bytes(bits: list[int]) -> bytes:
    """Convert bit list to bytes (groups of 8, MSB-first)."""
    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        out.append(_bits_to_int(bits[i:i + 8]))
    return bytes(out)


def _is_fec_mode(mode: int) -> bool:
    return mode in (MODE_FEC_BPSK, MODE_FEC_COMPRESSED)


def _base_mode(mode: int) -> int:
    """Map FEC mode to its underlying encoding (varicode or compressed)."""
    if mode == MODE_FEC_BPSK:
        return MODE_BPSK
    if mode == MODE_FEC_COMPRESSED:
        return MODE_COMPRESSED
    return mode


def build_frame(text: str, mode: int | None = None) -> list[int]:
    """Assemble a complete TX frame from text payload.

    Args:
        text: The text to encode.
        mode: Encoding mode. None = auto-select best mode (prefers FEC).
    """
    # Auto-select: pick FEC variant; use compressed if it saves bits
    if mode is None:
        vc_bits = varicode.encode(text)
        compressed = zlib.compress(text.encode("utf-8"), 9)
        compressed_bits = LENGTH_BITS + len(compressed) * 8
        if compressed_bits < len(vc_bits):
            mode = MODE_FEC_COMPRESSED
        else:
            mode = MODE_FEC_BPSK

    bits: list[int] = []

    # Preamble: alternating 10 pattern
    for _ in range(PREAMBLE_SYMBOLS):
        bits.append(1 if len(bits) % 2 == 0 else 0)

    # Sync word
    bits.extend(_int_to_bits(SYNC_WORD, SYNC_BITS))

    # Header
    bits.extend(_int_to_bits(PROTOCOL_VERSION, VERSION_BITS))
    bits.extend(_int_to_bits(mode, MODE_BITS))

    # Payload
    if _is_fec_mode(mode):
        payload_bits = _encode_payload(text, _base_mode(mode))
        # Append CRC-16 over raw payload
        crc = crc16_bits(payload_bits)
        payload_with_crc = payload_bits + crc
        # Convolutional encode
        encoder = ConvEncoder()
        encoded = encoder.encode(payload_with_crc)
        # Write length (encoded bit count) then encoded bits
        bits.extend(_int_to_bits(len(encoded), LENGTH_BITS))
        bits.extend(encoded)
    elif mode == MODE_COMPRESSED:
        compressed = zlib.compress(text.encode("utf-8"), 9)
        bits.extend(_int_to_bits(len(compressed), LENGTH_BITS))
        bits.extend(_bytes_to_bits(compressed))
    else:
        bits.extend(varicode.encode(text))

    # Tail
    bits.extend([0] * TAIL_SYMBOLS)

    return bits


def _encode_payload(text: str, base_mode: int) -> list[int]:
    """Produce raw payload bits for the given base mode (before FEC)."""
    if base_mode == MODE_COMPRESSED:
        compressed = zlib.compress(text.encode("utf-8"), 9)
        return _int_to_bits(len(compressed), LENGTH_BITS) + _bytes_to_bits(compressed)
    else:
        return varicode.encode(text)


class RxState(Enum):
    HUNTING_SYNC = auto()
    READING_HEADER = auto()
    READING_LENGTH = auto()
    READING_PAYLOAD = auto()


class RxFrameParser:
    """Stateful bit-by-bit frame parser for RX."""

    def __init__(self):
        self.state = RxState.HUNTING_SYNC
        self._shift_reg = 0
        self._bits_seen = 0
        self._header_bits: list[int] = []
        self._length_bits: list[int] = []
        self._payload_bits: list[int] = []
        self._payload_byte_count = 0
        self._encoded_bit_count = 0
        self._zero_count = 0
        self._idle_zeros = 0
        self.version: int | None = None
        self.mode: int | None = None
        self.decoded_text = ""
        self.crc_ok: bool | None = None

    @property
    def synced(self) -> bool:
        return self.state != RxState.HUNTING_SYNC

    def feed_bit(self, bit: int) -> str | None:
        """Feed one bit. Returns decoded character when available, or None."""
        if self.state == RxState.HUNTING_SYNC:
            self._shift_reg = ((self._shift_reg << 1) | bit) & 0xFFFF
            self._bits_seen += 1
            if self._bits_seen >= SYNC_BITS and self._shift_reg == SYNC_WORD:
                self.state = RxState.READING_HEADER
                self._header_bits = []
            return None

        elif self.state == RxState.READING_HEADER:
            self._header_bits.append(bit)
            if len(self._header_bits) == VERSION_BITS + MODE_BITS:
                self.version = _bits_to_int(self._header_bits[:VERSION_BITS])
                self.mode = _bits_to_int(self._header_bits[VERSION_BITS:])
                if self.mode in (MODE_COMPRESSED, MODE_FEC_BPSK,
                                 MODE_FEC_COMPRESSED):
                    self.state = RxState.READING_LENGTH
                    self._length_bits = []
                else:
                    self.state = RxState.READING_PAYLOAD
                    self._payload_bits = []
                    self._zero_count = 0
                    self._idle_zeros = 0
            return None

        elif self.state == RxState.READING_LENGTH:
            self._length_bits.append(bit)
            if len(self._length_bits) == LENGTH_BITS:
                count = _bits_to_int(self._length_bits)
                if _is_fec_mode(self.mode):
                    self._encoded_bit_count = count
                else:
                    self._payload_byte_count = count
                self.state = RxState.READING_PAYLOAD
                self._payload_bits = []
                self._idle_zeros = 0
            return None

        elif self.state == RxState.READING_PAYLOAD:
            if _is_fec_mode(self.mode):
                return self._feed_fec_bit(bit)
            elif self.mode == MODE_COMPRESSED:
                return self._feed_compressed_bit(bit)
            else:
                return self._feed_varicode_bit(bit)

        return None

    def _feed_fec_bit(self, bit: int) -> str | None:
        """Accumulate FEC-encoded bits, decode when complete."""
        self._payload_bits.append(bit)

        if len(self._payload_bits) >= self._encoded_bit_count:
            encoded = self._payload_bits[:self._encoded_bit_count]
            # Viterbi decode
            decoder = ViterbiDecoder()
            decoded_bits = decoder.decode(encoded)
            # Split off CRC-16 (last 16 bits)
            if len(decoded_bits) < CRC_BITS:
                self.crc_ok = False
                self._idle_zeros = 999
                return None
            payload_bits = decoded_bits[:-CRC_BITS]
            rx_crc = decoded_bits[-CRC_BITS:]
            expected_crc = crc16_bits(payload_bits)
            self.crc_ok = (rx_crc == expected_crc)
            if not self.crc_ok:
                self._idle_zeros = 999
                return None
            # Decode inner payload
            base = _base_mode(self.mode)
            if base == MODE_COMPRESSED:
                self.decoded_text = _decode_compressed_bits(payload_bits)
            else:
                self.decoded_text = _decode_varicode_bits(payload_bits)
            self._idle_zeros = 999
            return self.decoded_text if self.decoded_text else None

        return None

    def _feed_compressed_bit(self, bit: int) -> str | None:
        """Accumulate bits for compressed mode. Decodes when all bytes received."""
        self._payload_bits.append(bit)
        total_bits_needed = self._payload_byte_count * 8

        if len(self._payload_bits) >= total_bits_needed:
            raw = _bits_to_bytes(self._payload_bits[:total_bits_needed])
            try:
                self.decoded_text = zlib.decompress(raw).decode("utf-8")
            except (zlib.error, UnicodeDecodeError):
                self.decoded_text = ""
            self._idle_zeros = 999
            return self.decoded_text if self.decoded_text else None

        return None

    def _feed_varicode_bit(self, bit: int) -> str | None:
        """Varicode decoding for MODE_BPSK."""
        if bit == 0:
            self._idle_zeros += 1
        else:
            self._idle_zeros = 0

        if bit == 0:
            self._zero_count += 1
            if self._zero_count >= 2 and self._payload_bits:
                if self._payload_bits and self._payload_bits[-1] == 0:
                    self._payload_bits.pop()
                bitstr = "".join(str(b) for b in self._payload_bits)
                ch = varicode._DECODE_MAP.get(bitstr)
                self._payload_bits = []
                self._zero_count = 0
                if ch is not None:
                    self.decoded_text += ch
                    return ch
            else:
                self._payload_bits.append(0)
        else:
            self._zero_count = 0
            self._payload_bits.append(1)

        return None

    def is_idle(self, threshold: int = 32) -> bool:
        """True if we've seen enough consecutive zeros to consider the frame done."""
        return self._idle_zeros >= threshold


def _decode_compressed_bits(bits: list[int]) -> str:
    """Decode compressed payload bits (length field + zlib bytes)."""
    if len(bits) < LENGTH_BITS:
        return ""
    byte_count = _bits_to_int(bits[:LENGTH_BITS])
    data_bits = bits[LENGTH_BITS: LENGTH_BITS + byte_count * 8]
    raw = _bits_to_bytes(data_bits)
    try:
        return zlib.decompress(raw).decode("utf-8")
    except (zlib.error, UnicodeDecodeError):
        return ""


def _decode_varicode_bits(bits: list[int]) -> str:
    """Decode varicode payload bits to text."""
    text = ""
    char_bits: list[int] = []
    zero_count = 0
    for bit in bits:
        if bit == 0:
            zero_count += 1
            if zero_count >= 2 and char_bits:
                if char_bits and char_bits[-1] == 0:
                    char_bits.pop()
                bitstr = "".join(str(b) for b in char_bits)
                ch = varicode._DECODE_MAP.get(bitstr)
                if ch is not None:
                    text += ch
                char_bits = []
                zero_count = 0
            else:
                char_bits.append(0)
        else:
            zero_count = 0
            char_bits.append(1)
    return text
