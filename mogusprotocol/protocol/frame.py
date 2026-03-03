"""Frame assembly and parsing for the mogus protocol.

Frame structure (v2, per-frame sync):
  [preamble 1010... (64 symbols)]
  [SYNC_WORD 16 bits]
  [VERSION 8 bits]
  [MODE 8 bits]
  [FRAME_IDX 8 bits]
  [TOTAL_FRAMES 8 bits]
  [LENGTH 16 bits]  (for modes that need it)
  [payload]
  [tail zeros (16 symbols)]

Each frame is independently decodable with its own preamble, sync,
header, FEC + CRC.  Long messages are split across multiple frames.
"""

import zlib
import math
from enum import Enum, auto

from . import varicode
from .constants import (
    PREAMBLE_SYMBOLS,
    SYNC_WORD,
    SYNC_BITS,
    VERSION_BITS,
    MODE_BITS,
    FRAME_IDX_BITS,
    TOTAL_FRAMES_BITS,
    LENGTH_BITS,
    CRC_BITS,
    TAIL_SYMBOLS,
    PROTOCOL_VERSION,
    MODE_BPSK,
    MODE_COMPRESSED,
    MODE_FEC_BPSK,
    MODE_FEC_COMPRESSED,
    MAX_FRAME_PAYLOAD_BYTES,
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


def _auto_select_mode(text: str) -> int:
    """Auto-select best encoding mode for the given text."""
    vc_bits = varicode.encode(text)
    compressed = zlib.compress(text.encode("utf-8"), 9)
    compressed_bits = LENGTH_BITS + len(compressed) * 8
    if compressed_bits < len(vc_bits):
        return MODE_FEC_COMPRESSED
    return MODE_FEC_BPSK


def _is_compressed_base(mode: int) -> bool:
    """True if the mode uses compression (raw or FEC-wrapped)."""
    return mode in (MODE_COMPRESSED, MODE_FEC_COMPRESSED)


def _build_single_frame(chunk_payload: str | bytes, mode: int,
                        frame_idx: int, total_frames: int) -> list[int]:
    """Build one frame with preamble, sync, header, payload, tail.

    Args:
        chunk_payload: For varicode modes, a str chunk. For compressed modes,
                       pre-compressed bytes chunk.
        mode: Encoding mode.
        frame_idx: 0-indexed frame number.
        total_frames: Total number of frames.
    """
    bits: list[int] = []

    # Preamble: alternating 10 pattern
    for _ in range(PREAMBLE_SYMBOLS):
        bits.append(1 if len(bits) % 2 == 0 else 0)

    # Sync word
    bits.extend(_int_to_bits(SYNC_WORD, SYNC_BITS))

    # Header
    bits.extend(_int_to_bits(PROTOCOL_VERSION, VERSION_BITS))
    bits.extend(_int_to_bits(mode, MODE_BITS))
    bits.extend(_int_to_bits(frame_idx, FRAME_IDX_BITS))
    bits.extend(_int_to_bits(total_frames, TOTAL_FRAMES_BITS))

    # Payload
    if _is_fec_mode(mode):
        base = _base_mode(mode)
        if base == MODE_COMPRESSED:
            # chunk_payload is already compressed bytes
            payload_bits = _int_to_bits(len(chunk_payload), LENGTH_BITS) + _bytes_to_bits(chunk_payload)
        else:
            # varicode encode the text chunk
            payload_bits = varicode.encode(chunk_payload)
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
        # chunk_payload is already compressed bytes
        bits.extend(_int_to_bits(len(chunk_payload), LENGTH_BITS))
        bits.extend(_bytes_to_bits(chunk_payload))
    else:
        # varicode
        bits.extend(varicode.encode(chunk_payload))

    # Tail
    bits.extend([0] * TAIL_SYMBOLS)

    return bits


def build_frame(text: str, mode: int | None = None) -> list[int]:
    """Assemble a single TX frame from text payload.

    Builds one frame with frame_idx=0 and total_frames=1. For messages
    that fit in a single frame this is the simplest API.  For longer
    messages use build_frames() + encode_audio().

    Args:
        text: The text to encode.
        mode: Encoding mode. None = auto-select best mode (prefers FEC).
    """
    if mode is None:
        mode = _auto_select_mode(text)

    if _is_compressed_base(mode):
        chunk = zlib.compress(text.encode("utf-8"), 9)
    else:
        chunk = text

    return _build_single_frame(chunk, mode, frame_idx=0, total_frames=1)


def build_frames(text: str, mode: int | None = None) -> list[list[int]]:
    """Split text into chunks and build independent frames.

    Args:
        text: The text to encode.
        mode: Encoding mode. None = auto-select best mode.

    Returns:
        List of per-frame bit lists, each independently decodable.
    """
    if mode is None:
        mode = _auto_select_mode(text)

    if _is_compressed_base(mode):
        # Compress full text first, then split compressed bytes into chunks
        compressed = zlib.compress(text.encode("utf-8"), 9)
        chunks = []
        for i in range(0, len(compressed), MAX_FRAME_PAYLOAD_BYTES):
            chunks.append(compressed[i:i + MAX_FRAME_PAYLOAD_BYTES])
    else:
        # Split by character count for varicode modes
        chunks = []
        for i in range(0, len(text), MAX_FRAME_PAYLOAD_BYTES):
            chunks.append(text[i:i + MAX_FRAME_PAYLOAD_BYTES])

    if not chunks:
        chunks = [b"" if _is_compressed_base(mode) else ""]

    total_frames = len(chunks)
    frames = []
    for idx, chunk in enumerate(chunks):
        frames.append(_build_single_frame(chunk, mode, idx, total_frames))

    return frames


def build_multiframe(text: str, mode: int | None = None) -> list[int]:
    """Build multi-frame bit stream (concatenated frames).

    NOTE: This concatenates bits, which is only correct for direct
    bit-level parser testing.  For audio transmission use encode_audio()
    instead, which modulates each frame independently so each preamble
    starts at the correct carrier frequency.
    """
    frames = build_frames(text, mode)
    bits: list[int] = []
    for frame in frames:
        bits.extend(frame)
    return bits


def encode_audio(text: str, mode: int | None = None):
    """Encode text into multi-frame audio, ready for playback.

    Each frame is modulated independently (carrier/hops reset per frame)
    so the receiver can sync to each preamble separately.

    Returns:
        numpy float64 audio array at SAMPLE_RATE.
    """
    import numpy as np
    from ..dsp.modulator import PSKModulator

    frames = build_frames(text, mode)
    mod = PSKModulator()
    parts = []
    for frame_bits in frames:
        parts.append(mod.modulate(frame_bits))
    return np.concatenate(parts) if parts else np.array([], dtype=np.float64)


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
    """Stateful bit-by-bit frame parser for RX.

    After completing a frame (idle detected), automatically resets to
    HUNTING_SYNC to catch the next frame's preamble.
    """

    def __init__(self):
        self._reset_frame_state()
        self.decoded_text = ""
        self.frames: dict[int, str] = {}  # frame_idx -> decoded chunk
        self.total_frames: int | None = None
        self.crc_ok: bool | None = None

    def _reset_frame_state(self):
        """Reset per-frame parsing state (not accumulated text)."""
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
        self.frame_idx: int | None = None
        self._frame_total: int | None = None
        self._frame_decoded_text = ""
        self._frame_crc_ok: bool | None = None

    @property
    def synced(self) -> bool:
        return self.state != RxState.HUNTING_SYNC

    @property
    def all_frames_received(self) -> bool:
        if self.total_frames is None or self.total_frames == 0:
            return False
        return len(self.frames) >= self.total_frames

    def _reassemble(self) -> str:
        """Reassemble decoded text from all received frames in order."""
        if not self.frames:
            return ""
        return "".join(self.frames[i] for i in sorted(self.frames.keys()))

    def _commit_frame(self, text: str):
        """Store a completed frame's text and update reassembled output."""
        if self.frame_idx is not None:
            self.frames[self.frame_idx] = text
        if self._frame_total is not None and self._frame_total > 0:
            self.total_frames = self._frame_total
        self.crc_ok = self._frame_crc_ok
        self.decoded_text = self._reassemble()

    def reset_for_next_frame(self):
        """Reset parser to hunt for the next frame's sync word."""
        self._reset_frame_state()

    def feed_bit(self, bit: int) -> str | None:
        """Feed one bit. Returns decoded text when a frame completes, or None."""
        if self.state == RxState.HUNTING_SYNC:
            self._shift_reg = ((self._shift_reg << 1) | bit) & 0xFFFF
            self._bits_seen += 1
            if self._bits_seen >= SYNC_BITS and self._shift_reg == SYNC_WORD:
                self.state = RxState.READING_HEADER
                self._header_bits = []
            return None

        elif self.state == RxState.READING_HEADER:
            self._header_bits.append(bit)
            header_len = VERSION_BITS + MODE_BITS + FRAME_IDX_BITS + TOTAL_FRAMES_BITS
            if len(self._header_bits) == header_len:
                offset = 0
                self.version = _bits_to_int(self._header_bits[offset:offset + VERSION_BITS])
                offset += VERSION_BITS
                self.mode = _bits_to_int(self._header_bits[offset:offset + MODE_BITS])
                offset += MODE_BITS
                self.frame_idx = _bits_to_int(self._header_bits[offset:offset + FRAME_IDX_BITS])
                offset += FRAME_IDX_BITS
                self._frame_total = _bits_to_int(self._header_bits[offset:offset + TOTAL_FRAMES_BITS])

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
                self._frame_crc_ok = False
                self._idle_zeros = 999
                return None
            payload_bits = decoded_bits[:-CRC_BITS]
            rx_crc = decoded_bits[-CRC_BITS:]
            expected_crc = crc16_bits(payload_bits)
            self._frame_crc_ok = (rx_crc == expected_crc)
            if not self._frame_crc_ok:
                self._idle_zeros = 999
                self.crc_ok = False
                return None
            # Decode inner payload
            base = _base_mode(self.mode)
            if base == MODE_COMPRESSED:
                self._frame_decoded_text = _decode_compressed_bits(payload_bits)
            else:
                self._frame_decoded_text = _decode_varicode_bits(payload_bits)
            self._idle_zeros = 999
            self._commit_frame(self._frame_decoded_text)
            return self._frame_decoded_text if self._frame_decoded_text else None

        return None

    def _feed_compressed_bit(self, bit: int) -> str | None:
        """Accumulate bits for compressed mode. Decodes when all bytes received."""
        self._payload_bits.append(bit)
        total_bits_needed = self._payload_byte_count * 8

        if len(self._payload_bits) >= total_bits_needed:
            raw = _bits_to_bytes(self._payload_bits[:total_bits_needed])
            try:
                self._frame_decoded_text = zlib.decompress(raw).decode("utf-8")
            except (zlib.error, UnicodeDecodeError):
                self._frame_decoded_text = ""
            self._idle_zeros = 999
            self._commit_frame(self._frame_decoded_text)
            return self._frame_decoded_text if self._frame_decoded_text else None

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
                    self._frame_decoded_text += ch
                    self.decoded_text += ch
                    return ch
            else:
                self._payload_bits.append(0)
        else:
            self._zero_count = 0
            self._payload_bits.append(1)

        return None

    def finalize_frame(self):
        """Commit any pending frame data (call when is_idle() returns True).

        For FEC/compressed modes the frame is already committed when the
        payload is fully received.  For varicode mode, text accumulates
        character by character and needs an explicit commit at idle.
        """
        if self.frame_idx is not None and self.frame_idx not in self.frames:
            if self._frame_decoded_text:
                self._commit_frame(self._frame_decoded_text)
            elif self._frame_total is not None:
                # Even if empty, record total_frames from header
                if self._frame_total > 0:
                    self.total_frames = self._frame_total

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
