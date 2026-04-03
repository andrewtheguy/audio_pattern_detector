#!/usr/bin/env python3
"""Compare loudness normalization between pyloudnorm and native_helper.

Reads a WAV file, normalizes with both approaches, and writes the results
so they can be compared (e.g. in Audition or any audio editor).

Usage:
    uv run python scripts/compare_loudness.py <input.wav> [--target-lufs -16]
"""

import argparse
import struct
import wave
from pathlib import Path

import numpy as np
import pyloudnorm as pyln

import native_helper


def load_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        n_samples = len(raw) // 3
        arr = np.empty(n_samples, dtype=np.int32)
        for i in range(n_samples):
            b = raw[3 * i : 3 * i + 3]
            arr[i] = struct.unpack_from("<i", b + (b"\xff" if b[2] & 0x80 else b"\x00"))[0]
        data = arr.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1).astype(np.float32)

    return data, sr


def write_wav(path: str, data: np.ndarray, sr: int) -> None:
    int16 = np.clip(data * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("--target-lufs", type=float, default=-16.0, help="Target LUFS (default: -16)")
    parser.add_argument("--output-dir", default=".", help="Output directory (default: cwd)")
    args = parser.parse_args()

    input_path = args.input
    target = args.target_lufs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    print(f"Loading {input_path}...")
    audio, sr = load_wav(input_path)
    duration = len(audio) / sr
    print(f"  {duration:.2f}s, {sr} Hz, {len(audio)} samples")

    # ── pyloudnorm ──
    block = duration if duration < 0.5 else 0.4
    meter = pyln.Meter(sr, block_size=block)
    pyln_lufs = meter.integrated_loudness(audio)
    pyln_out = pyln.normalize.loudness(audio, pyln_lufs, target)
    print("\npyloudnorm:")
    print(f"  measured LUFS: {pyln_lufs:.2f}")
    print(f"  output range:  [{pyln_out.min():.4f}, {pyln_out.max():.4f}]")
    print(f"  clipped:       {np.any(np.abs(pyln_out) > 1.0)}")

    # ── native_helper ──
    rust_lufs = native_helper.integrated_loudness(audio, sr, block_size=block)
    rust_out = native_helper.loudness_normalize(audio, rust_lufs, target)
    print("\nnative_helper:")
    print(f"  measured LUFS: {rust_lufs:.2f}")
    print(f"  output range:  [{rust_out.min():.4f}, {rust_out.max():.4f}]")
    print("  clipped:       hard-clipped to [-1, 1]")

    # ── comparison ──
    # Clip pyloudnorm output to match for fair comparison
    pyln_clipped = np.clip(pyln_out, -1.0, 1.0).astype(np.float32)
    diff = rust_out - pyln_clipped
    print("\ndifference (rust - pyloudnorm_clipped):")
    print(f"  max abs diff:  {np.max(np.abs(diff)):.6f}")
    print(f"  mean abs diff: {np.mean(np.abs(diff)):.6f}")
    print(f"  LUFS diff:     {rust_lufs - pyln_lufs:.4f} dB")

    # ── write outputs ──
    pyln_path = out_dir / f"{stem}_pyloudnorm.wav"
    rust_path = out_dir / f"{stem}_native_helper.wav"
    write_wav(str(pyln_path), pyln_clipped, sr)
    write_wav(str(rust_path), rust_out, sr)
    print("\nWritten:")
    print(f"  {pyln_path}")
    print(f"  {rust_path}")


if __name__ == "__main__":
    main()
