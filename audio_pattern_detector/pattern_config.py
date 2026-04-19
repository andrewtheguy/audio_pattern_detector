"""Loader for `.apd.toml` pattern config files.

`.apd.toml` is a TOML document that declares a detection strategy and,
optionally, parameters for synthesising the pattern clip. TOML parses via
`tomllib` in the stdlib and supports `#` comments natively, so the format
stays human-editable without any custom parsing code.

The extension is the extensibility point for special detection paths
(currently `pure_tone` and `marker_tone`); ordinary patterns continue to use `.wav`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


APD_EXTENSION = ".apd.toml"

# Strategies understood by the detector. Each strategy reads a different
# subset of strategy_params and triggers a different verification path.
VALID_STRATEGIES: frozenset[str] = frozenset({"pure_tone", "marker_tone"})

# Generator types understood by the loader.
VALID_GENERATOR_TYPES: frozenset[str] = frozenset({"sine"})
VALID_VERIFICATION_FIELDS: frozenset[str] = frozenset({
    "minimum_band_purity",
    "minimum_active_frame_ratio",
    "minimum_longest_active_run",
    "minimum_active_frame_mean_purity",
    "maximum_min_flank_purity",
    "maximum_max_flank_purity",
})


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


def _generate_sine(params: dict[str, Any], sample_rate: int, source_path: str) -> NDArray[np.float32]:
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


def load_apd_file(path: str | Path, sample_rate: int) -> PatternConfig:
    """Parse an `.apd.toml` file and return the generated clip + strategy metadata.

    Args:
        path: Path to the .apd.toml file.
        sample_rate: Target sample rate for the generated clip.
    """
    with open(path, "rb") as f:
        try:
            obj = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{path}: invalid TOML: {e}") from e

    strategy = _get_required(obj, "strategy", str, str(path))
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"{path}: unknown strategy '{strategy}'. "
            f"Valid strategies: {sorted(VALID_STRATEGIES)}"
        )

    generator_raw = _get_required(obj, "generator", dict, str(path))
    generator = cast(dict[str, Any], generator_raw)
    gen_type = _get_required(generator, "type", str, str(path))
    if gen_type not in VALID_GENERATOR_TYPES:
        raise ValueError(
            f"{path}: unknown generator.type '{gen_type}'. "
            f"Valid types: {sorted(VALID_GENERATOR_TYPES)}"
        )

    if gen_type == "sine":
        audio = _generate_sine(generator, sample_rate, str(path))
    else:
        # VALID_GENERATOR_TYPES gate above guarantees this is unreachable.
        raise AssertionError(f"unhandled generator type {gen_type}")

    strategy_params: dict[str, Any] = {}
    if strategy in {"pure_tone", "marker_tone"}:
        # Tone-based strategies use the declared generator frequency directly.
        strategy_params["dominant_frequency_hz"] = float(generator["frequency_hz"])

    verification_raw = obj.get("verification")
    if verification_raw is not None:
        if not isinstance(verification_raw, dict):
            raise ValueError(f"{path}: field 'verification' must be table/object")
        verification = cast(dict[str, Any], verification_raw)
        unknown_fields = sorted(set(verification) - VALID_VERIFICATION_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"{path}: unknown verification field(s): {unknown_fields}. "
                f"Valid fields: {sorted(VALID_VERIFICATION_FIELDS)}"
            )

        parsed_verification: dict[str, float | int] = {}
        for key in sorted(verification):
            if key == "minimum_longest_active_run":
                parsed_verification[key] = int(_get_required(verification, key, int, str(path)))
            else:
                parsed_verification[key] = float(
                    _get_required(verification, key, (int, float), str(path))
                )
        strategy_params["verification"] = parsed_verification

    return PatternConfig(
        strategy=strategy,
        strategy_params=strategy_params,
        audio=audio,
    )
