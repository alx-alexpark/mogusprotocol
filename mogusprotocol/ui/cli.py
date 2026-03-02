"""CLI interface for mogusprotocol: mogus tx / mogus rx."""

import sys

import click
import numpy as np

from ..protocol.frame import build_frame
from ..protocol.constants import SAMPLE_RATE
from ..dsp.modulator import PSKModulator
from ..dsp.demodulator import PSKDemodulator


@click.group()
def cli():
    """mogusprotocol - PSK31 digital mode with Among Us Drip carrier hopping."""
    pass


@cli.command()
@click.argument("text")
@click.option("--output", "-o", default=None, help="Output WAV file path (omit for audio playback)")
@click.option("--device", "-d", default=None, type=int, help="Audio device index")
def tx(text: str, output: str | None, device: int | None):
    """Transmit a text message."""
    click.echo(f"Encoding: {text!r}")

    bits = build_frame(text)
    click.echo(f"Frame: {len(bits)} bits")

    mod = PSKModulator()
    audio = mod.modulate(bits)
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
@click.option("--duration", "-t", default=30.0, help="Listen duration in seconds (live mode)")
def rx(input_file: str | None, device: int | None, duration: float):
    """Receive and decode a transmission."""
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
        rx_stream = RxStream(device=device)
        rx_stream.start()
        click.echo(f"Listening for {duration}s...")
        time.sleep(duration)
        rx_stream.stop()
        audio = rx_stream.get_audio().astype(np.float64)
        click.echo(f"Captured {len(audio)} samples")

    demod = PSKDemodulator()
    text = demod.demodulate(audio)

    if text:
        click.echo(f"Decoded: {text}")
    else:
        click.echo("No signal decoded.")


if __name__ == "__main__":
    cli()
