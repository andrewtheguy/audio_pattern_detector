"""Loader for `.apd` pattern config files.

`.apd` is a JSON-with-comments ("JSONC") format that declares a detection
strategy and, optionally, parameters for synthesising the pattern clip.
The extension is the extensibility point for special detection paths
(currently `pure_tone`); ordinary patterns continue to use `.wav`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


APD_EXTENSION = ".apd"

# Strategies understood by the detector. Each strategy reads a different
# subset of strategy_params and triggers a different verification path.
VALID_STRATEGIES: frozenset[str] = frozenset({"pure_tone"})

# Generator types understood by the loader.
VALID_GENERATOR_TYPES: frozenset[str] = frozenset({"sine"})


@dataclass(frozen=True)
class PatternConfig:
    """Parsed .apd file."""
    strategy: str
    strategy_params: dict[str, Any]
    audio: NDArray[np.float32]


def _strip_jsonc_comments(text: str) -> str:
    """Strip `// line` and `/* block */` comments outside of string literals.

    Handles escaped quotes inside strings. Keeps newlines so that `json`
    module error messages point at the right line.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    string_quote = ""
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == string_quote:
                in_string = False
            i += 1
            continue
        if c in ('"', "'"):
            in_string = True
            string_quote = c
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                while i < n and text[i] != "\n":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                    if text[i] == "\n":
                        out.append("\n")
                    i += 1
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


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
    """Parse an `.apd` file and return the generated clip + strategy metadata.

    Args:
        path: Path to the .apd file.
        sample_rate: Target sample rate for the generated clip.
    """
    text = Path(path).read_text(encoding="utf-8")
    stripped = _strip_jsonc_comments(text)
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON/JSONC: {e.msg} (line {e.lineno})") from e
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level value must be an object")
    obj = cast(dict[str, Any], raw)

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
    if strategy == "pure_tone":
        # For pure_tone, the generator's frequency IS the dominant frequency.
        strategy_params["dominant_frequency_hz"] = float(generator["frequency_hz"])

    return PatternConfig(
        strategy=strategy,
        strategy_params=strategy_params,
        audio=audio,
    )
