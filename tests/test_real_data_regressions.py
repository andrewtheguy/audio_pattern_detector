from pathlib import Path

import pytest

from audio_pattern_detector.match import match_pattern


RTHK_BEEP_PATTERN = "sample_audios/clips/rthk_beep.apd.toml"

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

RTHK_BEEP_HOURLY_LEADINS_DIR = "sample_audios/regressions/rthk_beep_hourly_leadins"
RTHK_BEEP_HOURLY_OPENINGS_DIR = "sample_audios/regressions/rthk_beep_hourly_openings"

RTHK_BEEP_HOURLY_LEADIN_12_TO_13 = (
    f"{RTHK_BEEP_HOURLY_LEADINS_DIR}/radio1_2026-04-06_12_to_13_28m51_leadin.wav"
)
RTHK_BEEP_HOURLY_LEADIN_17_TO_18 = (
    f"{RTHK_BEEP_HOURLY_LEADINS_DIR}/radio1_2026-04-06_17_to_18_59m01_leadin.wav"
)

RTHK_BEEP_HOURLY_LEADIN_CASES = [
    (RTHK_BEEP_HOURLY_LEADIN_12_TO_13, [1.0085, 2.0, 3.013125, 3.987875, 5.025125]),
    (RTHK_BEEP_HOURLY_LEADIN_17_TO_18, [0.014125, 1.02625, 2.01, 3.015375, 4.017875]),
]

RTHK_BEEP_HOURLY_OPENING_12_TO_13 = (
    f"{RTHK_BEEP_HOURLY_OPENINGS_DIR}/radio1_2026-04-06_12_to_13_28m49_opening.wav"
)
RTHK_BEEP_HOURLY_OPENING_17_TO_18 = (
    f"{RTHK_BEEP_HOURLY_OPENINGS_DIR}/radio1_2026-04-06_17_to_18_58m58_opening.wav"
)

RTHK_BEEP_HOURLY_OPENING_CASES = [
    (
        RTHK_BEEP_HOURLY_OPENING_12_TO_13,
        [1.02325, 2.0335, 3.025, 4.038125, 5.012875, 6.050125],
    ),
    (
        RTHK_BEEP_HOURLY_OPENING_17_TO_18,
        [1.06975, 2.068875, 3.090625, 4.074375, 5.07975, 6.08225],
    ),
]


def _assert_expected_timestamps(
    actual_timestamps: list[float],
    expected_timestamps: list[float],
) -> None:
    # Tolerance is 0.02s: the .apd.toml pattern is a synthesised pure sine, so the
    # cross-correlation peak can land at a phase-aligned offset up to ~1 cycle
    # away from the true beep start (~1ms at 1 kHz, but accumulates across the
    # clip). 20 ms keeps regression sensitivity without over-fitting to the
    # specific phase of whichever WAV happened to generate the golden values.
    assert len(actual_timestamps) == len(expected_timestamps), (
        f"Expected {len(expected_timestamps)} matches, found "
        f"{len(actual_timestamps)}: {actual_timestamps}"
    )
    for actual, expected in zip(sorted(actual_timestamps), sorted(expected_timestamps)):
        assert abs(actual - expected) < 0.02, (
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


@pytest.mark.parametrize(
    ("audio_file", "expected_timestamps"),
    RTHK_BEEP_HOURLY_LEADIN_CASES,
    ids=[
        Path(audio_file).stem
        for audio_file, _expected_timestamps in RTHK_BEEP_HOURLY_LEADIN_CASES
    ],
)
def test_rthk_beep_hourly_leadins_recover_opening_beeps(
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
    RTHK_BEEP_HOURLY_OPENING_CASES,
    ids=[
        Path(audio_file).stem
        for audio_file, _expected_timestamps in RTHK_BEEP_HOURLY_OPENING_CASES
    ],
)
def test_rthk_beep_hourly_openings_recover_first_cluster_beeps(
    audio_file: str,
    expected_timestamps: list[float],
) -> None:
    assert Path(RTHK_BEEP_PATTERN).exists(), f"Pattern file {RTHK_BEEP_PATTERN} not found"
    assert Path(audio_file).exists(), f"Audio file {audio_file} not found"

    peak_times, _ = match_pattern(audio_file, [RTHK_BEEP_PATTERN], debug_mode=False)

    assert peak_times is not None
    assert "rthk_beep" in peak_times
    _assert_expected_timestamps(peak_times["rthk_beep"], expected_timestamps)
