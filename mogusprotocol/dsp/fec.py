"""Forward Error Correction for mogusprotocol.

Implements a rate 1/2, constraint length K=7 convolutional code with
Viterbi decoding — the standard NASA/CCSDS scheme used across ham radio
digital modes (PSK63F, AX.25 FEC, AMSAT satellites).

Generator polynomials (octal): G1=171, G2=133
CRC-16-CCITT for error detection (poly 0x1021, init 0xFFFF).
"""

from __future__ import annotations

# K=7 convolutional code, rate 1/2
# Generator polynomials in octal: 171 and 133
K = 7
NUM_STATES = 1 << (K - 1)  # 64
G1 = 0o171  # 0x79
G2 = 0o133  # 0x5B
TRACEBACK_DEPTH = 5 * K  # 35


def _parity(x: int) -> int:
    """Return parity (0 or 1) of integer x."""
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


# ---------- Convolutional Encoder ----------

class ConvEncoder:
    """Rate 1/2, K=7 convolutional encoder."""

    def __init__(self) -> None:
        self.state = 0

    def encode(self, bits: list[int]) -> list[int]:
        """Encode bits. Appends K-1 flush bits to terminate trellis."""
        out: list[int] = []
        flushed = list(bits) + [0] * (K - 1)
        for bit in flushed:
            reg = (self.state << 1) | bit
            out.append(_parity(reg & G1))
            out.append(_parity(reg & G2))
            self.state = reg & (NUM_STATES - 1)
        return out


# ---------- Viterbi Decoder ----------

class ViterbiDecoder:
    """Hard-decision Viterbi decoder for the K=7, rate 1/2 code."""

    def __init__(self) -> None:
        # Precompute branch outputs for each state and input bit
        self._branch_table: list[list[tuple[int, int]]] = []
        for state in range(NUM_STATES):
            entries = []
            for input_bit in (0, 1):
                reg = (state << 1) | input_bit
                o1 = _parity(reg & G1)
                o2 = _parity(reg & G2)
                entries.append((o1, o2))
            self._branch_table.append(entries)

    def decode(self, bits: list[int]) -> list[int]:
        """Decode received bits (pairs). Returns decoded data bits (excluding flush)."""
        n_pairs = len(bits) // 2
        if n_pairs == 0:
            return []

        INF = 0x7FFFFFFF

        # Path metrics: distance for each state
        prev_metrics = [INF] * NUM_STATES
        prev_metrics[0] = 0

        # Survivor paths stored as list of predecessor states per step
        survivors: list[list[int]] = []
        decisions: list[list[int]] = []  # input bit chosen

        for i in range(n_pairs):
            r0 = bits[2 * i]
            r1 = bits[2 * i + 1]

            curr_metrics = [INF] * NUM_STATES
            surv = [0] * NUM_STATES
            dec = [0] * NUM_STATES

            for prev_state in range(NUM_STATES):
                if prev_metrics[prev_state] == INF:
                    continue
                for input_bit in (0, 1):
                    next_state = (prev_state << 1 | input_bit) & (NUM_STATES - 1)
                    o1, o2 = self._branch_table[prev_state][input_bit]
                    branch_metric = (o1 ^ r0) + (o2 ^ r1)
                    candidate = prev_metrics[prev_state] + branch_metric

                    if candidate < curr_metrics[next_state]:
                        curr_metrics[next_state] = candidate
                        surv[next_state] = prev_state
                        dec[next_state] = input_bit

            prev_metrics = curr_metrics
            survivors.append(surv)
            decisions.append(dec)

        # Traceback from state 0 (encoder was flushed)
        state = 0
        decoded: list[int] = []
        for i in range(n_pairs - 1, -1, -1):
            decoded.append(decisions[i][state])
            state = survivors[i][state]

        decoded.reverse()

        # Strip K-1 flush bits
        if len(decoded) >= K - 1:
            decoded = decoded[: len(decoded) - (K - 1)]

        return decoded


# ---------- CRC-16-CCITT ----------

_CRC_POLY = 0x1021
_CRC_INIT = 0xFFFF


def crc16_ccitt(data: bytes) -> int:
    """Compute CRC-16-CCITT over bytes."""
    crc = _CRC_INIT
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ _CRC_POLY
            else:
                crc = crc << 1
            crc &= 0xFFFF
    return crc


def crc16_bits(bits: list[int]) -> list[int]:
    """Compute CRC-16-CCITT over a bit list, return 16 CRC bits (MSB-first)."""
    # Pack bits into bytes (pad last byte with zeros if needed)
    data = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i: i + 8]
        byte = 0
        for b in chunk:
            byte = (byte << 1) | b
        # Left-justify if partial byte
        byte <<= (8 - len(chunk))
        data.append(byte)

    crc = crc16_ccitt(bytes(data))
    return [(crc >> (15 - i)) & 1 for i in range(16)]
