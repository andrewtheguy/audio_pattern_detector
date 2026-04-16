#!/usr/bin/env python3
"""Trim 881 hourly captures into two ~30-min show segments via ffmpeg.

Consumes JSONL detection results produced by `run_all.sh` and writes two
stream-copied .m4a files per hour under ./tmp/trimmed/881/<date>/...
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

HOURLY_DIR = Path("/mnt/dasdata/capture881903/output/hourly/881")
RESULTS_DIR = Path("./tmp/results/881")
OUTPUT_DIR = Path("./tmp/trimmed/881")

SHOW_END_ANCHORS = {"newsreport", "881903nationalhymnintro"}
RESIDUAL_THRESHOLD_MS = 5 * 60 * 1000
MIDPOINT_FALLBACK_MS = 30 * 60 * 1000
# Non-end anchors within this window of a show end are spurious (they fall
# inside the show-end transition) and must not be picked as the show start.
MIN_SHOW_LENGTH_MS = 15 * 60 * 1000


@dataclass(frozen=True)
class Detection:
    clip_name: str
    timestamp_ms: int


@dataclass(frozen=True)
class Segments:
    show1: tuple[int, int]
    show2: tuple[int, int]


class JsonlError(Exception):
    """Raised for any problem reading/parsing a detection JSONL."""


def parse_jsonl(path: Path) -> tuple[list[Detection], int]:
    """Return (detections, total_time_ms). Raises JsonlError on failure."""
    if not path.is_file():
        raise JsonlError(f"missing JSONL at {path}")
    detections: list[Detection] = []
    total_time_ms: int | None = None
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise JsonlError(
                    f"malformed JSON at {path}:{lineno}: {e.msg}"
                ) from e
            t = obj.get("type")
            if t == "pattern_detected":
                try:
                    detections.append(
                        Detection(
                            clip_name=str(obj["clip_name"]),
                            timestamp_ms=int(obj["timestamp_ms"]),
                        )
                    )
                except (KeyError, TypeError, ValueError) as e:
                    raise JsonlError(
                        f"malformed pattern_detected at {path}:{lineno}: {e}"
                    ) from e
            elif t == "end":
                try:
                    total_time_ms = int(obj["total_time_ms"])
                except (KeyError, TypeError, ValueError) as e:
                    raise JsonlError(
                        f"malformed end event at {path}:{lineno}: {e}"
                    ) from e
    if total_time_ms is None:
        raise JsonlError(f"no end event in {path}")
    return detections, total_time_ms


def compute_segments(detections: list[Detection], total_time_ms: int) -> Segments:
    """Apply the show-segmentation rules from the plan."""
    end_anchors = [d for d in detections if d.clip_name in SHOW_END_ANCHORS]
    non_end_anchors = [d for d in detections if d.clip_name not in SHOW_END_ANCHORS]

    # show1_end: first show-end anchor past the residual threshold; else 30:00.
    show1_end = MIDPOINT_FALLBACK_MS
    for d in end_anchors:
        if d.timestamp_ms >= RESIDUAL_THRESHOLD_MS:
            show1_end = d.timestamp_ms
            break

    # show2_end: first show-end anchor strictly after show1_end; else file end.
    show2_end = total_time_ms
    for d in end_anchors:
        if d.timestamp_ms > show1_end:
            show2_end = d.timestamp_ms
            break

    # show1_start: latest non-end anchor strictly before show1_end and at
    # least MIN_SHOW_LENGTH_MS before it; else 0.
    show1_cutoff = show1_end - MIN_SHOW_LENGTH_MS
    show1_start = 0
    for d in non_end_anchors:
        if d.timestamp_ms < show1_cutoff:
            show1_start = d.timestamp_ms  # keep overwriting to get the latest

    # show2_start: latest non-end anchor in (show1_end, show2_end - 15min];
    # else show1_end.
    show2_cutoff = show2_end - MIN_SHOW_LENGTH_MS
    show2_start = show1_end
    for d in non_end_anchors:
        if show1_end < d.timestamp_ms < show2_cutoff:
            show2_start = d.timestamp_ms

    return Segments(show1=(show1_start, show1_end), show2=(show2_start, show2_end))


def format_ms(ms: int) -> str:
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def ffmpeg_copy(src: Path, dst: Path, start_ms: int, end_ms: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_ms / 1000:.3f}",
        "-to",
        f"{end_ms / 1000:.3f}",
        "-i",
        str(src),
        "-c",
        "copy",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def iter_hourly_m4as() -> Iterator[Path]:
    yield from sorted(HOURLY_DIR.rglob("*.m4a"))


_HOURLY_TS_RE = re.compile(r"_(\d{14})$")


def parse_hourly_base_dt(stem: str) -> datetime | None:
    """Extract the YYYYMMDDHHMMSS suffix from the hourly filename stem."""
    m = _HOURLY_TS_RE.search(stem)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")


def show_output_name(base_dt: datetime, start_ms: int, end_ms: int) -> str:
    begin = base_dt + timedelta(milliseconds=start_ms)
    end = base_dt + timedelta(milliseconds=end_ms)
    return (
        f"segments_881_{begin:%Y%m%d}_{begin:%H%M%S}_{end:%H%M%S}.m4a"
    )


def process_hour(m4a: Path, dry_run: bool) -> None:
    rel = m4a.relative_to(HOURLY_DIR)
    jsonl_path = RESULTS_DIR / rel.with_suffix(".jsonl")
    try:
        detections, total_time_ms = parse_jsonl(jsonl_path)
    except JsonlError as e:
        print(f"SKIP {rel}: {e}", file=sys.stderr)
        return
    segments = compute_segments(detections, total_time_ms)

    base_dt = parse_hourly_base_dt(rel.stem)
    if base_dt is None:
        print(f"SKIP {rel}: cannot parse YYYYMMDDHHMMSS from filename", file=sys.stderr)
        return

    s1_start, s1_end = segments.show1
    s2_start, s2_end = segments.show2

    show1_out = OUTPUT_DIR / rel.parent / show_output_name(base_dt, s1_start, s1_end)
    show2_out = OUTPUT_DIR / rel.parent / show_output_name(base_dt, s2_start, s2_end)

    print(
        f"{rel}\n"
        f"  show1: {format_ms(s1_start)} - {format_ms(s1_end)}  -> {show1_out}\n"
        f"  show2: {format_ms(s2_start)} - {format_ms(s2_end)}  -> {show2_out}"
    )

    if dry_run:
        return

    if not m4a.is_file():
        print(f"SKIP {rel}: source audio missing", file=sys.stderr)
        return

    if s1_end > s1_start:
        ffmpeg_copy(m4a, show1_out, s1_start, s1_end)
    else:
        print(f"SKIP show1 for {rel}: non-positive duration", file=sys.stderr)

    if s2_end > s2_start:
        ffmpeg_copy(m4a, show2_out, s2_start, s2_end)
    else:
        print(f"SKIP show2 for {rel}: non-positive duration", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    _ = ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print computed segments without invoking ffmpeg.",
    )
    args = ap.parse_args()
    dry_run: bool = bool(args.dry_run)

    for m4a in iter_hourly_m4as():
        process_hour(m4a, dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
