from pathlib import Path

import pytest

from audio_pattern_detector.match import match_pattern


RTHK_BEEP_PATTERN = "sample_audios/clips/rthk_beep.wav"

RTHK_BEEP_STRAY_CLIPS_V2_DIR = "sample_audios/regressions/rthk_beep_stray_clips_v2"

RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_1 = (
    f"{RTHK_BEEP_STRAY_CLIPS_V2_DIR}/tp_09-10_beep1.wav"
)
RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_2 = (
    f"{RTHK_BEEP_STRAY_CLIPS_V2_DIR}/tp_09-10_beep2.wav"
)
RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_3 = (
    f"{RTHK_BEEP_STRAY_CLIPS_V2_DIR}/tp_09-10_beep3.wav"
)

RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_1 = (
    f"{RTHK_BEEP_STRAY_CLIPS_V2_DIR}/v2_10-11_20m21s.wav"
)
RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_2 = (
    f"{RTHK_BEEP_STRAY_CLIPS_V2_DIR}/v2_10-11_50m40s.wav"
)
RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_3 = (
    f"{RTHK_BEEP_STRAY_CLIPS_V2_DIR}/v2_20-21_35m13s.wav"
)
RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_4 = (
    f"{RTHK_BEEP_STRAY_CLIPS_V2_DIR}/v2_22-23_19m48s.wav"
)

RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_CASES = [
    (RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_1, [2.00525, 3.004875]),
    (RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_2, [1.01525, 2.014875, 3.015]),
    (RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_3, [0.01525, 1.014875, 2.015, 3.01225]),
]

RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_CASES = [
    (RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_1, []),
    (RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_2, []),
    (RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_3, []),
    (RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_4, []),
]


def _assert_expected_timestamps(
    actual_timestamps: list[float],
    expected_timestamps: list[float],
) -> None:
    assert len(actual_timestamps) == len(expected_timestamps), (
        f"Expected {len(expected_timestamps)} matches, found "
        f"{len(actual_timestamps)}: {actual_timestamps}"
    )
    for actual, expected in zip(sorted(actual_timestamps), expected_timestamps):
        assert abs(actual - expected) < 0.01, (
            f"Expected timestamp ~{expected}s, got {actual}s"
        )


@pytest.mark.parametrize(
    ("audio_file", "expected_timestamps"),
    RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_CASES,
    ids=[
        Path(audio_file).stem
        for audio_file, _expected_timestamps in RTHK_BEEP_STRAY_CLIPS_V2_TRUE_POSITIVE_CASES
    ],
)
def test_rthk_beep_stray_clips_v2_true_positives(
    audio_file: str,
    expected_timestamps: list[float],
) -> None:
    assert Path(RTHK_BEEP_PATTERN).exists(), f"Pattern file {RTHK_BEEP_PATTERN} not found"
    assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

    peak_times, _ = match_pattern(audio_file, [RTHK_BEEP_PATTERN], debug_mode=False)

    assert peak_times is not None
    assert "rthk_beep" in peak_times
    _assert_expected_timestamps(peak_times["rthk_beep"], expected_timestamps)


@pytest.mark.parametrize(
    ("audio_file", "expected_timestamps"),
    RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_CASES,
    ids=[
        Path(audio_file).stem
        for audio_file, _expected_timestamps in RTHK_BEEP_STRAY_CLIPS_V2_FALSE_POSITIVE_CASES
    ],
)
def test_rthk_beep_stray_clips_v2_false_positives(
    audio_file: str,
    expected_timestamps: list[float],
) -> None:
    assert Path(RTHK_BEEP_PATTERN).exists(), f"Pattern file {RTHK_BEEP_PATTERN} not found"
    assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

    peak_times, _ = match_pattern(audio_file, [RTHK_BEEP_PATTERN], debug_mode=False)

    assert peak_times is not None
    assert "rthk_beep" in peak_times
    assert peak_times["rthk_beep"] == expected_timestamps
