"""CLI interface for mogusprotocol: mogus tx / mogus rx."""

import math
import sys

import click
import numpy as np

from ..protocol.frame import encode_audio
from ..protocol.constants import SAMPLE_RATE
from ..dsp.demodulator import PSKDemodulator


def _render_vu(peak: float, width: int = 20) -> str:
    """Render a colored VU meter bar from peak amplitude."""
    if peak > 1e-10:
        db = max(-60.0, 20.0 * math.log10(peak))
    else:
        db = -60.0

    fill = int((db + 60) / 60 * width)
    fill = max(0, min(width, fill))

    green_end = int(width * 0.6)
    yellow_end = int(width * 0.85)

    bar = []
    for i in range(width):
        if i < fill:
            if i < green_end:
                bar.append("\033[32m█")
            elif i < yellow_end:
                bar.append("\033[33m█")
            else:
                bar.append("\033[31m█")
        else:
            bar.append("\033[90m░")

    return f'▕{"".join(bar)}\033[0m▏ {db:+5.0f} dB'


def _get_device_info(device: int | None) -> tuple[str, int]:
    """Return (device_name, device_index) for the given input device."""
    import sounddevice as sd

    if device is not None:
        info = sd.query_devices(device, "input")
        return info["name"], device
    info = sd.query_devices(kind="input")
    return info["name"], info["index"]


@click.group()
def cli():
    """mogusprotocol - PSK31 digital mode with Among Us Drip carrier hopping."""
    pass


@cli.command()
def devices():
    """List available audio devices."""
    import sounddevice as sd

    devs = sd.query_devices()
    default_in, default_out = sd.default.device

    click.echo("Input devices:")
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0:
            marker = " \033[32m← default\033[0m" if i == default_in else ""
            rate = int(d["default_samplerate"])
            click.echo(f"  [{i}] {d['name']}  ({d['max_input_channels']}ch, {rate} Hz){marker}")

    click.echo("\nOutput devices:")
    for i, d in enumerate(devs):
        if d["max_output_channels"] > 0:
            marker = " \033[32m← default\033[0m" if i == default_out else ""
            rate = int(d["default_samplerate"])
            click.echo(f"  [{i}] {d['name']}  ({d['max_output_channels']}ch, {rate} Hz){marker}")

    click.echo(f"\nUsage: mogus rx -d <index>")


@cli.command()
@click.argument("text")
@click.option("--output", "-o", default=None, help="Output WAV file path (omit for audio playback)")
@click.option("--device", "-d", default=None, type=int, help="Audio device index")
def tx(text: str, output: str | None, device: int | None):
    """Transmit a text message."""
    click.echo(f"Encoding: {text!r}")

    audio = encode_audio(text)
    click.echo(f"Audio: {len(audio)} samples ({len(audio)/SAMPLE_RATE:.2f}s)")

    if output:
        import soundfile as sf
        sf.write(output, audio, SAMPLE_RATE)
        click.echo(f"Written to {output}")
    else:
        from ..audio.tx_stream import TxStream
        tx_stream = TxStream(device=device)
        tx_stream.write(audio)
        tx_stream.start()
        click.echo("Transmitting...")
        tx_stream.wait_done(timeout=len(audio) / SAMPLE_RATE + 2.0)
        tx_stream.stop()
        click.echo("Done.")


@cli.command()
@click.option("--input", "-i", "input_file", default=None, help="Input WAV file (omit for live audio)")
@click.option("--device", "-d", default=None, type=int, help="Audio device index")
@click.option("--duration", "-t", default=None, type=float, help="Listen duration in seconds (default: until Ctrl-C)")
@click.option("--live", "-l", is_flag=True, help="Live decode: print characters as they arrive")
def rx(input_file: str | None, device: int | None, duration: float | None, live: bool):
    """Receive and decode a transmission."""
    if live:
        _rx_live(device, duration)
        return

    if input_file:
        import soundfile as sf
        audio, sr = sf.read(input_file, dtype="float64")
        if sr != SAMPLE_RATE:
            click.echo(f"Warning: WAV sample rate {sr} != {SAMPLE_RATE}, resampling not implemented")
            return
        if audio.ndim > 1:
            audio = audio[:, 0]
        click.echo(f"Read {len(audio)} samples from {input_file}")
    else:
        import time
        from ..audio.rx_stream import RxStream

        dev_name, dev_idx = _get_device_info(device)
        click.echo(f"🎤 {dev_name} (#{dev_idx}) │ {SAMPLE_RATE} Hz")

        rx_stream = RxStream(device=device)
        rx_stream.start()

        # Recording progress with VU meter
        start_time = time.monotonic()
        dur_str = f"{duration:.0f}s" if duration else "∞"
        sys.stdout.write("\n")  # placeholder for status line
        silence_warned = False
        try:
            while duration is None or time.monotonic() - start_time < duration:
                elapsed = time.monotonic() - start_time
                peak = rx_stream.peak_level
                vu = _render_vu(peak)
                sys.stdout.write(
                    f"\r  Recording {vu}  {elapsed:5.1f}s / {dur_str}\033[K"
                )
                sys.stdout.flush()
                if elapsed > 2.0 and peak < 1e-10 and not silence_warned:
                    silence_warned = True
                    sys.stdout.write(
                        "\n  \033[33m⚠  Mic is silent — check permissions "
                        "(System Settings → Privacy → Microphone)\033[0m"
                    )
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        sys.stdout.write("\n")

        rx_stream.stop()
        audio = rx_stream.get_audio().astype(np.float64)
        click.echo(f"Captured {len(audio)} samples ({len(audio)/SAMPLE_RATE:.1f}s)")

    demod = PSKDemodulator()
    text = demod.demodulate(audio)

    if text:
        click.echo(f"Decoded: {text}")
    else:
        click.echo("No signal decoded.")


def _rx_live(device: int | None, duration: float | None):
    """Live decoding mode with TUI: VU meter, status, and streaming output."""
    import time
    import threading
    from ..audio.rx_stream import RxStream
    from ..dsp.streaming_demod import StreamingDemodulator

    dev_name, dev_idx = _get_device_info(device)
    dur_str = f"{duration:.0f}s" if duration else "∞"

    demod = StreamingDemodulator()
    rx_stream = RxStream(device=device, blocksize=2048)
    stop_event = threading.Event()

    # Print static header
    click.echo("mogus rx ─ live decode")
    click.echo(f"🎤 {dev_name} (#{dev_idx}) │ {SAMPLE_RATE} Hz")
    click.echo("─" * 46)

    # Two placeholder lines: status + decoded text
    sys.stdout.write("\n\n")
    sys.stdout.flush()

    def feeder():
        """Pull audio from RxStream buffer and feed to demodulator."""
        read_pos = 0
        while not stop_event.is_set():
            with rx_stream._lock:
                buf_list = list(rx_stream._buffer)
            if not buf_list:
                time.sleep(0.02)
                continue
            all_audio = np.concatenate(buf_list)
            if read_pos >= len(all_audio):
                time.sleep(0.02)
                continue

            new_audio = all_audio[read_pos:]
            read_pos += len(new_audio)
            demod.feed(new_audio)

            if demod.done:
                stop_event.set()
                break

            time.sleep(0.02)

    rx_stream.start()
    feed_thread = threading.Thread(target=feeder, daemon=True)
    feed_thread.start()

    start_time = time.monotonic()
    silence_warned = False
    try:
        while not stop_event.is_set() and (duration is None or (time.monotonic() - start_time) < duration):
            elapsed = time.monotonic() - start_time
            peak = rx_stream.peak_level
            vu = _render_vu(peak)

            if elapsed > 2.0 and peak < 1e-10 and not silence_warned:
                silence_warned = True

            if demod.done:
                status = "\033[32m✓ Done\033[0m"
            elif demod.synced:
                status = "\033[32m◉ Locked\033[0m"
            elif silence_warned:
                status = "\033[31m⚠ Silent\033[0m"
            else:
                status = "\033[33m● Searching\033[0m"

            timer = f"{elapsed:5.1f}s / {dur_str}"

            decoded = demod.decoded_text
            if decoded:
                crc_str = ""
                if demod.crc_ok is not None:
                    crc_str = (
                        " \033[32m[CRC OK]\033[0m"
                        if demod.crc_ok
                        else " \033[31m[CRC FAIL]\033[0m"
                    )
                decoded_line = f"  \033[1m>\033[0m {decoded}{crc_str}"
            else:
                decoded_line = ""

            # Redraw: move up 2 lines, overwrite, clear trailing chars
            sys.stdout.write("\033[2F")
            sys.stdout.write(f"  {vu}  {status}  {timer}\033[K\n")
            sys.stdout.write(f"{decoded_line}\033[K\n")
            sys.stdout.flush()

            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        feed_thread.join(timeout=1.0)
        rx_stream.stop()

    # Final redraw
    elapsed = time.monotonic() - start_time
    decoded = demod.decoded_text

    sys.stdout.write("\033[2F")
    if decoded:
        crc_str = ""
        if demod.crc_ok is not None:
            crc_str = (
                " \033[32m[CRC OK]\033[0m"
                if demod.crc_ok
                else " \033[31m[CRC FAIL]\033[0m"
            )
        sys.stdout.write(f"  \033[32m✓\033[0m Decoded in {elapsed:.1f}s\033[K\n")
        sys.stdout.write(f"  \033[1m>\033[0m {decoded}{crc_str}\033[K\n")
    else:
        sys.stdout.write(f"  \033[31m✗\033[0m No signal detected ({elapsed:.1f}s)\033[K\n")
        if silence_warned:
            sys.stdout.write(
                "  \033[33m⚠  Mic is silent — check permissions "
                "(System Settings → Privacy → Microphone)\033[K\033[0m\n"
            )
        else:
            sys.stdout.write("\033[K\n")
    sys.stdout.flush()


if __name__ == "__main__":
    cli()
