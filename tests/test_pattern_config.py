from pathlib import Path

import pytest

from audio_pattern_detector.audio_utils import DEFAULT_TARGET_SAMPLE_RATE
from audio_pattern_detector.pattern_config import load_apd_file


LEGACY_PURE_TONE_PATTERN = """# rthk_beep: RTHK hourly station beep.
# Uses the pure_tone verification strategy (short-time spectral analysis)
# instead of the normal MSE + Pearson envelope check, because a clean
# sine does not cross-correlate distinctively against real airtime audio.
#
# The clip is synthesised from frequency + duration at the target sample
# rate, so a single .apd.toml works at 8 kHz, 16 kHz, or any other rate.

strategy = "pure_tone"
description = "RTHK hourly beep — ~1040 Hz pure tone, ~0.23s"

[generator]
type = "sine"
# Measured from the original rthk_beep.wav capture (parabolic-interp FFT peak).
frequency_hz = 1040.19
duration_seconds = 0.228375
amplitude = 1.0
"""


def test_legacy_pure_tone_strategy_is_rejected(tmp_path: Path) -> None:
    pattern_path = tmp_path / "legacy_rthk_beep.apd.toml"
    pattern_path.write_text(LEGACY_PURE_TONE_PATTERN)

    with pytest.raises(ValueError) as exc_info:
        load_apd_file(pattern_path, sample_rate=DEFAULT_TARGET_SAMPLE_RATE)

    assert str(exc_info.value) == (
        f"{pattern_path}: unknown strategy 'pure_tone'. "
        "Valid strategies: ['marker_tone']"
    )
