SAMPLE_RATE = 48000
BAUD_RATE = 62.5  # PSK63 - 2x throughput over PSK31
SAMPLES_PER_SYMBOL = int(SAMPLE_RATE / BAUD_RATE)  # 768

# Hop timing: ~130 BPM quarter notes
HOP_DURATION_S = 60.0 / 130.0  # ~0.4615 s
HOP_DURATION_SAMPLES = int(HOP_DURATION_S * SAMPLE_RATE)  # ~22153
SYMBOLS_PER_HOP = int(HOP_DURATION_S * BAUD_RATE)  # 14

# Crossfade at hop boundaries (samples)
CROSSFADE_SAMPLES = 64

# Frame structure
PREAMBLE_SYMBOLS = 64  # alternating 1010... for sync
SYNC_WORD = 0b0111_0001_0100_1110  # 16-bit magic: 0x714E
SYNC_BITS = 16
VERSION_BITS = 8
MODE_BITS = 8
HEADER_BITS = SYNC_BITS + VERSION_BITS + MODE_BITS
TAIL_SYMBOLS = 16  # trailing zeros to flush

# Protocol version
PROTOCOL_VERSION = 1
MODE_BPSK = 0            # varicode encoding
MODE_COMPRESSED = 1      # zlib + raw 8-bit bytes
MODE_FEC_BPSK = 2        # varicode + convolutional FEC + CRC-16
MODE_FEC_COMPRESSED = 3  # zlib + convolutional FEC + CRC-16

# Length field for compressed mode (16-bit payload byte count)
LENGTH_BITS = 16

# CRC
CRC_BITS = 16

# Varicode end-of-character
VARICODE_SEPARATOR = [0, 0]
