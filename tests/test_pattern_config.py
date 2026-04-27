import base64
import io
import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from audio_pattern_detector.audio_utils import DEFAULT_TARGET_SAMPLE_RATE
from audio_pattern_detector.pattern_config import load_apd_file


def _write_toml(tmp_path: Path, body: str, name: str = "clip.apd.toml") -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def _sine_wav_bytes(frequency_hz: float, duration_seconds: float, sample_rate: int) -> bytes:
    n = int(round(duration_seconds * sample_rate))
    samples = [
        int(max(-1.0, min(1.0, math.sin(2 * math.pi * frequency_hz * i / sample_rate))) * 32767)
        for i in range(n)
    ]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *samples))
    return buf.getvalue()


def test_sine_source_round_trip(tmp_path: Path) -> None:
    body = """\
[clip]
source = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1
amplitude = 1.0

[verification]
strategy = "marker_tone"
"""
    path = _write_toml(tmp_path, body)
    sr = DEFAULT_TARGET_SAMPLE_RATE
    config = load_apd_file(path, sample_rate=sr)

    assert config.strategy == "marker_tone"
    assert config.audio.dtype == np.float32
    assert config.audio.shape == (round(0.1 * sr),)
    assert pytest.approx(float(np.max(np.abs(config.audio))), rel=1e-3) == 1.0
    # Sine source auto-populates dominant_frequency_hz from the declared frequency.
    assert config.strategy_params["dominant_frequency_hz"] == 1040.0
    assert "verification" not in config.strategy_params  # no thresholds provided


def test_sine_source_with_thresholds_and_explicit_dominant_frequency(tmp_path: Path) -> None:
    body = """\
[clip]
source = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1

[verification]
strategy = "marker_tone"
dominant_frequency_hz = 1041.5
minimum_band_purity = 0.72
minimum_active_frame_ratio = 0.70
minimum_longest_active_run = 7
minimum_active_frame_mean_purity = 0.77
maximum_min_flank_purity = 0.02
maximum_max_flank_purity = 0.14
"""
    path = _write_toml(tmp_path, body)
    config = load_apd_file(path, sample_rate=DEFAULT_TARGET_SAMPLE_RATE)

    assert config.strategy_params["dominant_frequency_hz"] == 1041.5
    assert config.strategy_params["verification"] == {
        "minimum_band_purity": 0.72,
        "minimum_active_frame_ratio": 0.70,
        "minimum_longest_active_run": 7,
        "minimum_active_frame_mean_purity": 0.77,
        "maximum_min_flank_purity": 0.02,
        "maximum_max_flank_purity": 0.14,
    }


def test_wav_base64_round_trip(tmp_path: Path) -> None:
    sr = DEFAULT_TARGET_SAMPLE_RATE
    freq = 1040.0
    dur = 0.1
    wav_bytes = _sine_wav_bytes(freq, dur, sr)
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    body = f"""\
[clip]
source = "wav_base64"
data = "{b64}"

[verification]
strategy = "marker_tone"
dominant_frequency_hz = {freq}
"""
    path = _write_toml(tmp_path, body)
    config = load_apd_file(path, sample_rate=sr)

    n = round(dur * sr)
    expected = np.array(
        [math.sin(2 * math.pi * freq * i / sr) for i in range(n)],
        dtype=np.float32,
    )
    assert config.audio.shape == (n,)
    assert config.audio.dtype == np.float32
    # int16 round-trip introduces ~1.5e-4 quantisation error, well below 1e-3.
    assert float(np.max(np.abs(config.audio - expected))) < 1e-3
    assert config.strategy_params["dominant_frequency_hz"] == freq


def test_wav_base64_accepts_multiline_string(tmp_path: Path) -> None:
    """[clip].data may span multiple lines via TOML triple-quoted strings."""
    sr = DEFAULT_TARGET_SAMPLE_RATE
    wav_bytes = _sine_wav_bytes(1040.0, 0.05, sr)
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    wrapped = "\n".join(b64[i : i + 76] for i in range(0, len(b64), 76))
    body = f"""\
[clip]
source = "wav_base64"
data = \"\"\"
{wrapped}
\"\"\"

[verification]
strategy = "marker_tone"
dominant_frequency_hz = 1040.0
"""
    path = _write_toml(tmp_path, body)
    inline_body = f"""\
[clip]
source = "wav_base64"
data = "{b64}"

[verification]
strategy = "marker_tone"
dominant_frequency_hz = 1040.0
"""
    inline_path = _write_toml(tmp_path, inline_body, name="inline.apd.toml")

    multiline = load_apd_file(path, sample_rate=sr)
    inline = load_apd_file(inline_path, sample_rate=sr)
    np.testing.assert_array_equal(multiline.audio, inline.audio)


def test_wav_base64_resamples_to_target(tmp_path: Path) -> None:
    # WAV recorded at 16 kHz, loader must resample to the requested 8 kHz target.
    source_sr = 16000
    target_sr = 8000
    wav_bytes = _sine_wav_bytes(1000.0, 0.1, source_sr)
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    body = f"""\
[clip]
source = "wav_base64"
data = "{b64}"

[verification]
strategy = "marker_tone"
dominant_frequency_hz = 1000.0
"""
    path = _write_toml(tmp_path, body)
    config = load_apd_file(path, sample_rate=target_sr)

    assert config.audio.shape == (round(0.1 * target_sr),)


def test_top_level_strategy_is_rejected(tmp_path: Path) -> None:
    body = """\
strategy = "marker_tone"

[clip]
source = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1

[verification]
strategy = "marker_tone"
"""
    path = _write_toml(tmp_path, body)
    with pytest.raises(ValueError, match="unknown top-level field"):
        load_apd_file(path, sample_rate=DEFAULT_TARGET_SAMPLE_RATE)


def test_legacy_generator_section_is_rejected(tmp_path: Path) -> None:
    body = """\
strategy = "marker_tone"

[generator]
type = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1
"""
    path = _write_toml(tmp_path, body)
    with pytest.raises(ValueError, match="unknown top-level field"):
        load_apd_file(path, sample_rate=DEFAULT_TARGET_SAMPLE_RATE)


def test_unknown_clip_source_is_rejected(tmp_path: Path) -> None:
    body = """\
[clip]
source = "square"
frequency_hz = 1040.0

[verification]
strategy = "marker_tone"
"""
    path = _write_toml(tmp_path, body)
    with pytest.raises(ValueError, match="unknown \\[clip\\].source 'square'"):
        load_apd_file(path, sample_rate=DEFAULT_TARGET_SAMPLE_RATE)


def test_unknown_strategy_is_rejected(tmp_path: Path) -> None:
    body = """\
[clip]
source = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1

[verification]
strategy = "pure_tone"
"""
    path = _write_toml(tmp_path, body)
    with pytest.raises(ValueError, match="unknown strategy 'pure_tone'"):
        load_apd_file(path, sample_rate=DEFAULT_TARGET_SAMPLE_RATE)


def test_unknown_clip_field_for_sine_is_rejected(tmp_path: Path) -> None:
    body = """\
[clip]
source = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1
data = "abc"

[verification]
strategy = "marker_tone"
"""
    path = _write_toml(tmp_path, body)
    with pytest.raises(ValueError, match="unknown \\[clip\\] field"):
        load_apd_file(path, sample_rate=DEFAULT_TARGET_SAMPLE_RATE)


def test_unknown_clip_field_for_wav_base64_is_rejected(tmp_path: Path) -> None:
    body = """\
[clip]
source = "wav_base64"
data = "AAAA"
frequency_hz = 1040.0

[verification]
strategy = "marker_tone"
"""
    path = _write_toml(tmp_path, body)
    with pytest.raises(ValueError, match="unknown \\[clip\\] field"):
        load_apd_file(path, sample_rate=DEFAULT_TARGET_SAMPLE_RATE)


def test_unknown_verification_field_is_rejected(tmp_path: Path) -> None:
    body = """\
[clip]
source = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1

[verification]
strategy = "marker_tone"
not_a_real_threshold = 0.5
"""
    path = _write_toml(tmp_path, body)
    with pytest.raises(ValueError, match="unknown \\[verification\\] field"):
        load_apd_file(path, sample_rate=DEFAULT_TARGET_SAMPLE_RATE)


def test_invalid_base64_is_rejected(tmp_path: Path) -> None:
    body = """\
[clip]
source = "wav_base64"
data = "not!valid!base64!"

[verification]
strategy = "marker_tone"
"""
    path = _write_toml(tmp_path, body)
    with pytest.raises(ValueError, match="invalid base64"):
        load_apd_file(path, sample_rate=DEFAULT_TARGET_SAMPLE_RATE)


def test_missing_clip_section_is_rejected(tmp_path: Path) -> None:
    body = """\
[verification]
strategy = "marker_tone"
"""
    path = _write_toml(tmp_path, body)
    with pytest.raises(ValueError, match="missing required field 'clip'"):
        load_apd_file(path, sample_rate=DEFAULT_TARGET_SAMPLE_RATE)


def test_missing_verification_section_is_rejected(tmp_path: Path) -> None:
    body = """\
[clip]
source = "sine"
frequency_hz = 1040.0
duration_seconds = 0.1
"""
    path = _write_toml(tmp_path, body)
    with pytest.raises(ValueError, match="missing required field 'verification'"):
        load_apd_file(path, sample_rate=DEFAULT_TARGET_SAMPLE_RATE)
