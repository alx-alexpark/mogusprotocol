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
    TAIL_SYMBOLS,
    PROTOCOL_VERSION,
    MODE_BPSK,
    MODE_COMPRESSED,
)


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


def build_frame(text: str, mode: int | None = None) -> list[int]:
    """Assemble a complete TX frame from text payload.

    Args:
        text: The text to encode.
        mode: Encoding mode. None = auto-select best mode.
    """
    # Auto-select: use compressed if it saves bits
    if mode is None:
        vc_bits = varicode.encode(text)
        compressed = zlib.compress(text.encode("utf-8"), 9)
        compressed_bits = LENGTH_BITS + len(compressed) * 8
        mode = MODE_COMPRESSED if compressed_bits < len(vc_bits) else MODE_BPSK

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
    if mode == MODE_COMPRESSED:
        compressed = zlib.compress(text.encode("utf-8"), 9)
        bits.extend(_int_to_bits(len(compressed), LENGTH_BITS))
        bits.extend(_bytes_to_bits(compressed))
    else:
        bits.extend(varicode.encode(text))

    # Tail
    bits.extend([0] * TAIL_SYMBOLS)

    return bits


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
        self._zero_count = 0
        self._idle_zeros = 0
        self.version: int | None = None
        self.mode: int | None = None
        self.decoded_text = ""

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
                if self.mode == MODE_COMPRESSED:
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
                self._payload_byte_count = _bits_to_int(self._length_bits)
                self.state = RxState.READING_PAYLOAD
                self._payload_bits = []
                self._idle_zeros = 0
            return None

        elif self.state == RxState.READING_PAYLOAD:
            if self.mode == MODE_COMPRESSED:
                return self._feed_compressed_bit(bit)
            else:
                return self._feed_varicode_bit(bit)

        return None

    def _feed_compressed_bit(self, bit: int) -> str | None:
        """Accumulate bits for compressed mode. Decodes when all bytes received."""
        self._payload_bits.append(bit)
        total_bits_needed = self._payload_byte_count * 8

        if len(self._payload_bits) >= total_bits_needed:
            # All payload bytes received - decompress
            raw = _bits_to_bytes(self._payload_bits[:total_bits_needed])
            try:
                self.decoded_text = zlib.decompress(raw).decode("utf-8")
            except (zlib.error, UnicodeDecodeError):
                self.decoded_text = ""
            # Signal idle so demod stops
            self._idle_zeros = 999
            return self.decoded_text if self.decoded_text else None

        return None

    def _feed_varicode_bit(self, bit: int) -> str | None:
        """Varicode decoding for MODE_BPSK."""
        # Track consecutive zeros for end-of-frame detection
        if bit == 0:
            self._idle_zeros += 1
        else:
            self._idle_zeros = 0

        if bit == 0:
            self._zero_count += 1
            if self._zero_count >= 2 and self._payload_bits:
                # Two zeros = end of character. Strip speculative trailing 0.
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
