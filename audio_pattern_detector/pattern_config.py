"""Loader for `.apd.toml` pattern config files.

`.apd.toml` is a TOML document with two sections that mirror the detection
pipeline:

* `[clip]` — Step 1 audio source used by FFT cross-correlation. The clip can
  be synthesised from a formula (`source = "sine"`) or carried inline as a
  base64-encoded WAV (`source = "wav_base64"`).
* `[verification]` — Step 2 verification logic. Declares the strategy
  (currently only `marker_tone`) and per-strategy thresholds.

TOML parses via stdlib `tomllib` and supports `#` comments natively, so the
format stays human-editable without any custom parsing code.
"""

from __future__ import annotations

import base64
import binascii
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from audio_pattern_detector.audio_utils import (
    load_wav_from_bytes,
    resample_audio,
)


APD_EXTENSION = ".apd.toml"

# Strategies understood by the detector. Each strategy reads a different
# subset of strategy_params and triggers a different verification path.
VALID_STRATEGIES: frozenset[str] = frozenset({"marker_tone"})

# Clip-source kinds understood by the loader.
VALID_CLIP_SOURCES: frozenset[str] = frozenset({"sine", "wav_base64"})

# Per-source allowed fields (excluding `source` itself).
_SINE_FIELDS: frozenset[str] = frozenset({"frequency_hz", "duration_seconds", "amplitude"})
_WAV_BASE64_FIELDS: frozenset[str] = frozenset({"data"})

VALID_VERIFICATION_THRESHOLDS: frozenset[str] = frozenset({
    "minimum_band_purity",
    "minimum_active_frame_ratio",
    "minimum_longest_active_run",
    "minimum_active_frame_mean_purity",
    "maximum_min_flank_purity",
    "maximum_max_flank_purity",
})

# All keys allowed inside the [verification] table.
_VERIFICATION_FIELDS: frozenset[str] = (
    VALID_VERIFICATION_THRESHOLDS | frozenset({"strategy", "dominant_frequency_hz"})
)

# Keys allowed at the top level of the document.
_TOP_LEVEL_FIELDS: frozenset[str] = frozenset({"description", "clip", "verification"})


@dataclass(frozen=True)
class PatternConfig:
    """Parsed .apd.toml file."""
    strategy: str
    strategy_params: dict[str, Any]
    audio: NDArray[np.float32]


def _get_required(obj: dict[str, Any], key: str, kind: type | tuple[type, ...], path: str) -> Any:
    if key not in obj:
        raise ValueError(f"{path}: missing required field '{key}'")
    value = obj[key]
    if not isinstance(value, kind):
        kind_name = kind.__name__ if isinstance(kind, type) else "/".join(k.__name__ for k in kind)
        raise ValueError(
            f"{path}: field '{key}' must be {kind_name}, got {type(value).__name__}"
        )
    return value


def _clip_from_sine(
    params: dict[str, Any], sample_rate: int, source_path: str
) -> NDArray[np.float32]:
    unknown = sorted(set(params) - _SINE_FIELDS - {"source"})
    if unknown:
        raise ValueError(
            f"{source_path}: unknown [clip] field(s) for source='sine': {unknown}. "
            f"Valid fields: {sorted(_SINE_FIELDS)}"
        )
    frequency_hz = float(_get_required(params, "frequency_hz", (int, float), source_path))
    duration_seconds = float(_get_required(params, "duration_seconds", (int, float), source_path))
    amplitude = float(params.get("amplitude", 0.9))
    if frequency_hz <= 0:
        raise ValueError(f"{source_path}: frequency_hz must be positive, got {frequency_hz}")
    if duration_seconds <= 0:
        raise ValueError(f"{source_path}: duration_seconds must be positive, got {duration_seconds}")
    if not (frequency_hz * 2 < sample_rate):
        raise ValueError(
            f"{source_path}: frequency_hz {frequency_hz} exceeds Nyquist "
            f"({sample_rate / 2}) for sample_rate {sample_rate}"
        )
    n_samples = int(round(duration_seconds * sample_rate))
    t = np.arange(n_samples, dtype=np.float32) / np.float32(sample_rate)
    return (amplitude * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)


def _clip_from_wav_base64(
    params: dict[str, Any], sample_rate: int, source_path: str
) -> NDArray[np.float32]:
    unknown = sorted(set(params) - _WAV_BASE64_FIELDS - {"source"})
    if unknown:
        raise ValueError(
            f"{source_path}: unknown [clip] field(s) for source='wav_base64': {unknown}. "
            f"Valid fields: {sorted(_WAV_BASE64_FIELDS)}"
        )
    data_str = _get_required(params, "data", str, source_path)
    # Strip whitespace so callers can use TOML triple-quoted strings
    # (`data = """..."""`) and break the base64 across multiple lines.
    cleaned = "".join(data_str.split())
    try:
        wav_bytes = base64.b64decode(cleaned, validate=True)
    except binascii.Error as e:
        raise ValueError(f"{source_path}: invalid base64 in [clip].data: {e}") from e

    audio, source_sr = load_wav_from_bytes(wav_bytes, name=source_path)
    if source_sr != sample_rate:
        audio = resample_audio(audio, source_sr, sample_rate)
    return audio


def load_apd_file(path: str | Path, sample_rate: int) -> PatternConfig:
    """Parse an `.apd.toml` file and return the clip audio + strategy metadata.

    Args:
        path: Path to the .apd.toml file.
        sample_rate: Target sample rate for the clip audio.
    """
    source_path = str(path)
    with open(path, "rb") as f:
        try:
            obj = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{source_path}: invalid TOML: {e}") from e

    unknown_top = sorted(set(obj) - _TOP_LEVEL_FIELDS)
    if unknown_top:
        raise ValueError(
            f"{source_path}: unknown top-level field(s): {unknown_top}. "
            f"Valid fields: {sorted(_TOP_LEVEL_FIELDS)} "
            f"(note: 'strategy' moved into [verification] in the v2 schema)"
        )

    clip_raw = _get_required(obj, "clip", dict, source_path)
    clip_section = cast(dict[str, Any], clip_raw)
    source_kind = _get_required(clip_section, "source", str, source_path)
    if source_kind not in VALID_CLIP_SOURCES:
        raise ValueError(
            f"{source_path}: unknown [clip].source '{source_kind}'. "
            f"Valid sources: {sorted(VALID_CLIP_SOURCES)}"
        )

    if source_kind == "sine":
        audio = _clip_from_sine(clip_section, sample_rate, source_path)
    elif source_kind == "wav_base64":
        audio = _clip_from_wav_base64(clip_section, sample_rate, source_path)
    else:
        # VALID_CLIP_SOURCES gate above guarantees this is unreachable.
        raise AssertionError(f"unhandled clip source {source_kind}")

    verification_raw = _get_required(obj, "verification", dict, source_path)
    verification = cast(dict[str, Any], verification_raw)
    unknown_v = sorted(set(verification) - _VERIFICATION_FIELDS)
    if unknown_v:
        raise ValueError(
            f"{source_path}: unknown [verification] field(s): {unknown_v}. "
            f"Valid fields: {sorted(_VERIFICATION_FIELDS)}"
        )

    strategy = _get_required(verification, "strategy", str, source_path)
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"{source_path}: unknown strategy '{strategy}'. "
            f"Valid strategies: {sorted(VALID_STRATEGIES)}"
        )

    strategy_params: dict[str, Any] = {}

    if "dominant_frequency_hz" in verification:
        strategy_params["dominant_frequency_hz"] = float(
            _get_required(verification, "dominant_frequency_hz", (int, float), source_path)
        )
    elif source_kind == "sine":
        # For sine clips the declared generator frequency is authoritative;
        # store it so the detector doesn't need to re-derive it from the
        # synthesised samples.
        strategy_params["dominant_frequency_hz"] = float(clip_section["frequency_hz"])
    # else: leave unset; AudioPatternDetector falls back to
    # get_pure_tone_frequency on the loaded audio for marker_tone clips.

    threshold_keys = sorted(set(verification) & VALID_VERIFICATION_THRESHOLDS)
    if threshold_keys:
        parsed_thresholds: dict[str, float | int] = {}
        for key in threshold_keys:
            if key == "minimum_longest_active_run":
                parsed_thresholds[key] = int(_get_required(verification, key, int, source_path))
            else:
                parsed_thresholds[key] = float(
                    _get_required(verification, key, (int, float), source_path)
                )
        strategy_params["verification"] = parsed_thresholds

    return PatternConfig(
        strategy=strategy,
        strategy_params=strategy_params,
        audio=audio,
    )
