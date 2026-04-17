#!/usr/bin/env python3
"""Trim RTHK radio1/radio2 hourly captures into two ~30-min show segments.

Consumes JSONL detection results produced by `match_all.py` and writes two
stream-copied .m4a files per hour under
/mnt/dasdata/andrewdata/radio_shows/trimmed/m4a/rthk/<station>/<date>/...
Pass --opus to instead re-encode as Opus (8 kbps, voip profile) .opus files
under /mnt/dasdata/andrewdata/radio_shows/trimmed/opus/rthk/<station>/<date>/...
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

HOURLY_ROOT = Path("/mnt/dasdata/andrewdata/ftp/rthk/hourly")
RESULTS_ROOT = Path("./tmp/results/rthk")
OUTPUT_ROOT = Path("/mnt/dasdata/andrewdata/radio_shows/trimmed")
OUTPUT_SUBDIR = "rthk"
STATIONS = ("radio1", "radio2")

SHOW_END_ANCHORS = {"rthknewsreportcan", "rthknationalhymnintro"}

HALF_HOUR_MS = 30 * 60 * 1000
ONE_HOUR_MS = 60 * 60 * 1000
# RTHK shows align to the half-hour / hour marks; accept end-anchors within
# this window around each boundary, otherwise fall back to the mark itself.
END_ANCHOR_TOLERANCE_MS = 5 * 60 * 1000
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
    """Apply RTHK's half-hour/hour-anchored segmentation rules."""
    end_anchors = [d for d in detections if d.clip_name in SHOW_END_ANCHORS]
    non_end_anchors = [d for d in detections if d.clip_name not in SHOW_END_ANCHORS]

    # show1_end: earliest end-anchor within ±TOLERANCE of the 30-min mark;
    # else fall back to exactly 30:00. Detections are already in timestamp
    # order, so the first match is the earliest — which naturally picks
    # whichever of rthknewsreportcan / rthknationalhymnintro comes first.
    show1_end = HALF_HOUR_MS
    w1_lo = HALF_HOUR_MS - END_ANCHOR_TOLERANCE_MS
    w1_hi = HALF_HOUR_MS + END_ANCHOR_TOLERANCE_MS
    for d in end_anchors:
        if w1_lo <= d.timestamp_ms <= w1_hi:
            show1_end = d.timestamp_ms
            break

    # show2_end: earliest end-anchor within ±TOLERANCE of the 60-min mark
    # and strictly after show1_end; else fall back to min(total, 60:00).
    show2_end = min(total_time_ms, ONE_HOUR_MS)
    w2_lo = ONE_HOUR_MS - END_ANCHOR_TOLERANCE_MS
    w2_hi = ONE_HOUR_MS + END_ANCHOR_TOLERANCE_MS
    for d in end_anchors:
        if d.timestamp_ms > show1_end and w2_lo <= d.timestamp_ms <= w2_hi:
            show2_end = d.timestamp_ms
            break

    # show1_start: latest non-end anchor strictly before show1_end and at
    # least MIN_SHOW_LENGTH_MS before it; else 0.
    show1_cutoff = show1_end - MIN_SHOW_LENGTH_MS
    show1_start = 0
    for d in non_end_anchors:
        if d.timestamp_ms < show1_cutoff:
            show1_start = d.timestamp_ms

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


def ffmpeg_trim_copy(src: Path, dst: Path, start_ms: int, end_ms: int) -> None:
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


def ffmpeg_trim_opus(src: Path, dst: Path, start_ms: int, end_ms: int) -> None:
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
        "-vn",
        "-c:a",
        "libopus",
        "-application",
        "voip",
        "-b:a",
        "8k",
        "-ac",
        "1",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def iter_rthk_m4as(station: str) -> Iterator[Path]:
    yield from sorted((HOURLY_ROOT / station).rglob("*.m4a"))


_RTHK_TS_RE = re.compile(r"_(\d{4})-(\d{2})-(\d{2})_(\d{2})_to_\d{2}$")


def parse_rthk_base_dt(stem: str) -> datetime | None:
    """Extract the capture-hour start datetime from an RTHK filename stem."""
    m = _RTHK_TS_RE.search(stem)
    if not m:
        return None
    y, mo, d, h = (int(g) for g in m.groups())
    return datetime(y, mo, d, h, 0, 0)


def show_output_name(
    station: str, base_dt: datetime, start_ms: int, end_ms: int, ext: str
) -> str:
    begin = base_dt + timedelta(milliseconds=start_ms)
    end = base_dt + timedelta(milliseconds=end_ms)
    return (
        f"segments_{station}_{begin:%Y%m%d}_{begin:%H%M%S}_{end:%H%M%S}.{ext}"
    )


def process_hour(station: str, m4a: Path, dry_run: bool, opus: bool) -> None:
    station_hourly = HOURLY_ROOT / station
    station_results = RESULTS_ROOT / station

    rel = m4a.relative_to(station_hourly)
    jsonl_path = station_results / rel.with_suffix(".jsonl")
    try:
        detections, total_time_ms = parse_jsonl(jsonl_path)
    except JsonlError as e:
        print(f"SKIP {station}/{rel}: {e}", file=sys.stderr)
        return
    segments = compute_segments(detections, total_time_ms)

    base_dt = parse_rthk_base_dt(rel.stem)
    if base_dt is None:
        print(
            f"SKIP {station}/{rel}: cannot parse YYYY-MM-DD_HH_to_HH from filename",
            file=sys.stderr,
        )
        return

    s1_start, s1_end = segments.show1
    s2_start, s2_end = segments.show2

    ext = "opus" if opus else "m4a"
    trim = ffmpeg_trim_opus if opus else ffmpeg_trim_copy
    output_dir = OUTPUT_ROOT / ext / OUTPUT_SUBDIR / station / rel.parent
    show1_out = output_dir / show_output_name(
        station, base_dt, s1_start, s1_end, ext
    )
    show2_out = output_dir / show_output_name(
        station, base_dt, s2_start, s2_end, ext
    )

    print(
        f"{station}/{rel}\n"
        f"  show1: {format_ms(s1_start)} - {format_ms(s1_end)}  -> {show1_out}\n"
        f"  show2: {format_ms(s2_start)} - {format_ms(s2_end)}  -> {show2_out}"
    )

    if dry_run:
        return

    if not m4a.is_file():
        print(f"SKIP {station}/{rel}: source audio missing", file=sys.stderr)
        return

    if s1_end > s1_start:
        trim(m4a, show1_out, s1_start, s1_end)
    else:
        print(
            f"SKIP show1 for {station}/{rel}: non-positive duration",
            file=sys.stderr,
        )

    if s2_end > s2_start:
        trim(m4a, show2_out, s2_start, s2_end)
    else:
        print(
            f"SKIP show2 for {station}/{rel}: non-positive duration",
            file=sys.stderr,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    _ = ap.add_argument(
        "--station",
        required=True,
        choices=STATIONS,
        help="Which RTHK station to process.",
    )
    _ = ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print computed segments without invoking ffmpeg.",
    )
    _ = ap.add_argument(
        "--opus",
        action="store_true",
        help="Re-encode as Opus (8 kbps, voip profile) instead of stream-copying to .m4a.",
    )
    args = ap.parse_args()
    station: str = str(args.station)
    dry_run: bool = bool(args.dry_run)
    opus: bool = bool(args.opus)

    for m4a in iter_rthk_m4as(station):
        process_hour(station, m4a, dry_run, opus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
