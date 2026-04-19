#!/usr/bin/env python3
"""Run a single pattern across every audio file in a directory and emit one JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
import re
from typing import Any

from audio_pattern_detector.audio_utils import seconds_to_time
from audio_pattern_detector.match import match_pattern
from audio_pattern_detector.pattern_config import APD_EXTENSION


TIMESTAMP_SUFFIX_RE = re.compile(r"_(\d{14})$")


@dataclass(frozen=True)
class DetectionEvent:
    source_file: str
    timestamp_utc: datetime
    seconds_from_file_start: float


def _pattern_name_from_path(pattern_file: Path) -> str:
    filename = pattern_file.name
    if filename.lower().endswith(APD_EXTENSION):
        return filename[: -len(APD_EXTENSION)]
    return pattern_file.stem


def _parse_file_start_utc(audio_file: Path) -> datetime:
    match = TIMESTAMP_SUFFIX_RE.search(audio_file.stem)
    if match is None:
        raise ValueError(
            f"Audio filename must end with _YYYYMMDDHHMMSS before the extension: {audio_file.name}"
        )
    timestamp_text = match.group(1)
    try:
        parsed_dt = datetime.strptime(timestamp_text, "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise ValueError(
            f"Audio filename {audio_file.name} has invalid timestamp "
            f"{timestamp_text!r}: {exc}"
        ) from exc
    if not (2000 <= parsed_dt.year <= 2100):
        raise ValueError(
            f"Audio filename {audio_file.name} has out-of-range timestamp year "
            f"{parsed_dt.year}; expected a year between 2000 and 2100"
        )
    return parsed_dt.replace(tzinfo=UTC)


def _format_utc(dt: datetime, *, include_microseconds: bool) -> str:
    if include_microseconds:
        return dt.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round_millis(value: float) -> float:
    return round(value, 3)


def _default_output_path(pattern_file: Path, audio_files: list[Path]) -> Path:
    if not audio_files:
        raise ValueError("Cannot derive output path with no audio files")
    first_audio_file = audio_files[0]
    try:
        first_file_start = _parse_file_start_utc(first_audio_file)
    except ValueError as exc:
        raise ValueError(
            "Cannot derive default output path from "
            f"{first_audio_file.name} while deriving the output filename: {exc}"
        ) from exc
    date_str = first_file_start.strftime("%Y-%m-%d")
    return Path("./tmp") / f"{_pattern_name_from_path(pattern_file)}_{date_str}_detection_results.json"


def _collect_audio_files(source_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    matcher = source_dir.rglob if recursive else source_dir.glob
    return sorted(path for path in matcher(pattern) if path.is_file())


def build_detection_report(
    *,
    pattern_file: Path,
    source_dir: Path,
    audio_files: list[Path],
    target_sample_rate: int | None,
    seconds_per_chunk: int | None,
    debug_mode: bool,
    debug_dir: str,
    height_min: float | None,
) -> dict[str, Any]:
    pattern_name = _pattern_name_from_path(pattern_file)
    per_file_results: list[dict[str, Any]] = []
    all_events: list[DetectionEvent] = []

    for index, audio_file in enumerate(audio_files, start=1):
        print(
            f"[{index}/{len(audio_files)}] Processing {audio_file.name}",
            file=sys.stderr,
        )
        peak_times, total_time = match_pattern(
            str(audio_file),
            [str(pattern_file)],
            debug_mode=debug_mode,
            seconds_per_chunk=seconds_per_chunk,
            target_sample_rate=target_sample_rate,
            debug_dir=debug_dir,
            height_min=height_min,
        )

        detections = []
        if peak_times is not None:
            detections = sorted(peak_times.get(pattern_name, []))

        file_start_utc = _parse_file_start_utc(audio_file)
        per_file_results.append(
            {
                "file": audio_file.name,
                "file_start_utc": _format_utc(file_start_utc, include_microseconds=False),
                "duration_seconds": _round_millis(total_time),
                "detections_seconds": [_round_millis(ts) for ts in detections],
                "detections_formatted": [seconds_to_time(ts) for ts in detections],
            }
        )

        for detection_seconds in detections:
            all_events.append(
                DetectionEvent(
                    source_file=audio_file.name,
                    timestamp_utc=file_start_utc + timedelta(seconds=detection_seconds),
                    seconds_from_file_start=detection_seconds,
                )
            )

    all_events.sort(key=lambda event: event.timestamp_utc)
    all_detections: list[dict[str, Any]] = []
    previous_event: DetectionEvent | None = None
    for event in all_events:
        delta = None
        if previous_event is not None:
            delta = _round_millis(
                (event.timestamp_utc - previous_event.timestamp_utc).total_seconds()
            )
        all_detections.append(
            {
                "source_file": event.source_file,
                "timestamp_utc": _format_utc(event.timestamp_utc, include_microseconds=True),
                "seconds_from_file_start": _round_millis(event.seconds_from_file_start),
                "delta_from_previous_seconds": delta,
            }
        )
        previous_event = event

    return {
        "generated_at_utc": _format_utc(datetime.now(UTC), include_microseconds=True),
        "pattern_file": str(pattern_file.resolve()),
        "source_dir": str(source_dir.resolve()),
        "file_count": len(audio_files),
        "per_file_results": per_file_results,
        "all_detections": all_detections,
    }


def _parse_chunk_seconds(raw_value: str) -> int | None:
    if raw_value == "auto":
        return None
    return int(raw_value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="Directory containing audio files to scan")
    parser.add_argument(
        "--pattern-file",
        required=True,
        type=Path,
        help="Single pattern file to match against each audio file",
    )
    parser.add_argument(
        "--pattern",
        default="*.m4a",
        help="Filename glob inside source_dir (default: *.m4a)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search source_dir recursively instead of only one level",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: ./tmp/<pattern>_<date>_detection_results.json)",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=None,
        help="Override target sample rate for processing",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=_parse_chunk_seconds,
        default=60,
        help='Seconds per chunk, or "auto" to let the detector derive it',
    )
    parser.add_argument(
        "--height-min",
        type=float,
        default=None,
        help="Override minimum correlation peak height",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable detector debug output",
    )
    parser.add_argument(
        "--debug-dir",
        default="./tmp",
        help="Base directory for detector debug output",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    pattern_file = args.pattern_file
    if not source_dir.is_dir():
        parser.error(f"Source directory not found: {source_dir}")
    if not pattern_file.is_file():
        parser.error(f"Pattern file not found: {pattern_file}")

    audio_files = _collect_audio_files(source_dir, args.pattern, args.recursive)
    if not audio_files:
        parser.error(
            f"No files matched pattern {args.pattern!r} in {source_dir}"
        )

    output_path = args.output or _default_output_path(pattern_file, audio_files)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_detection_report(
        pattern_file=pattern_file,
        source_dir=source_dir,
        audio_files=audio_files,
        target_sample_rate=args.target_sample_rate,
        seconds_per_chunk=args.chunk_seconds,
        debug_mode=args.debug,
        debug_dir=args.debug_dir,
        height_min=args.height_min,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
