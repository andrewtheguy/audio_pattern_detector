from datetime import UTC, datetime
from pathlib import Path

import match_directory


def test_build_detection_report_matches_expected_shape(tmp_path, monkeypatch):
    pattern_file = tmp_path / "903_beep.apd.toml"
    pattern_file.write_text("strategy = 'marker_tone'\n", encoding="utf-8")

    source_dir = tmp_path / "captures"
    source_dir.mkdir()
    first_file = source_dir / "segments_903_278532_278831_20260417090000.m4a"
    second_file = source_dir / "segments_903_278832_279131_20260417100000.m4a"
    first_file.write_bytes(b"")
    second_file.write_bytes(b"")

    results = {
        first_file.name: ({"903_beep": [12.163125, 1812.848875]}, 3600.022),
        second_file.name: ({"903_beep": [12.757125]}, 3599.883),
    }

    def fake_match_pattern(
        audio_source: str | None,
        pattern_files: list[str],
        debug_mode: bool = False,
        on_pattern_detected=None,
        accumulate_results: bool = True,
        seconds_per_chunk: int | None = 60,
        from_stdin: bool = False,
        target_sample_rate: int | None = None,
        debug_dir: str = "./tmp",
        height_min: float | None = None,
    ):
        assert audio_source is not None
        assert pattern_files == [str(pattern_file)]
        return results[Path(audio_source).name]

    monkeypatch.setattr(match_directory, "match_pattern", fake_match_pattern)

    report = match_directory.build_detection_report(
        pattern_file=pattern_file,
        source_dir=source_dir,
        audio_files=[first_file, second_file],
        target_sample_rate=None,
        seconds_per_chunk=60,
        debug_mode=False,
        debug_dir="./tmp",
        height_min=None,
    )

    assert report["pattern_file"] == str(pattern_file.resolve())
    assert report["source_dir"] == str(source_dir.resolve())
    assert report["file_count"] == 2
    assert report["per_file_results"] == [
        {
            "file": first_file.name,
            "file_start_utc": "2026-04-17T09:00:00Z",
            "duration_seconds": 3600.022,
            "detections_seconds": [12.163, 1812.849],
            "detections_formatted": ["00:00:12.163", "00:30:12.849"],
        },
        {
            "file": second_file.name,
            "file_start_utc": "2026-04-17T10:00:00Z",
            "duration_seconds": 3599.883,
            "detections_seconds": [12.757],
            "detections_formatted": ["00:00:12.757"],
        },
    ]
    assert report["all_detections"] == [
        {
            "source_file": first_file.name,
            "timestamp_utc": "2026-04-17T09:00:12.163125Z",
            "seconds_from_file_start": 12.163,
            "delta_from_previous_seconds": None,
        },
        {
            "source_file": first_file.name,
            "timestamp_utc": "2026-04-17T09:30:12.848875Z",
            "seconds_from_file_start": 1812.849,
            "delta_from_previous_seconds": 1800.686,
        },
        {
            "source_file": second_file.name,
            "timestamp_utc": "2026-04-17T10:00:12.757125Z",
            "seconds_from_file_start": 12.757,
            "delta_from_previous_seconds": 1799.908,
        },
    ]
    generated_at = datetime.fromisoformat(report["generated_at_utc"].replace("Z", "+00:00"))
    assert generated_at.tzinfo == UTC


def test_default_output_path_uses_pattern_name_and_first_file_date(tmp_path):
    pattern_file = tmp_path / "903_beep.apd.toml"
    audio_file = tmp_path / "segments_903_278532_278831_20260417090000.m4a"

    output_path = match_directory._default_output_path(pattern_file, [audio_file])

    assert output_path == Path("./tmp/903_beep_2026-04-17_detection_results.json")
