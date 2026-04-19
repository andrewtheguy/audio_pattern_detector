#!/usr/bin/env python3
"""Run audio-pattern-detector match over hourly captures for a given source."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    hourly_dir: Path
    pattern_folder: Path
    output_base: Path
    path_filter: str | None = None


SOURCES: dict[str, Source] = {
    "881": Source(
        hourly_dir=Path("/mnt/dasdata/capture881903/output/hourly/881"),
        pattern_folder=Path("/mnt/dasdata/andrewdata/audio_clips/881903"),
        output_base=Path("./tmp/results/881"),
        #path_filter="/903/",
    ),
    "903": Source(
        hourly_dir=Path("/mnt/dasdata/capture881903/output/hourly/903"),
        pattern_folder=Path("/mnt/dasdata/andrewdata/audio_clips/881903"),
        output_base=Path("./tmp/results/903"),
        #path_filter="/903/",
    ),
    "rthk-radio1": Source(
        hourly_dir=Path("/mnt/dasdata/andrewdata/ftp/rthk/hourly/radio1"),
        pattern_folder=Path("/mnt/dasdata/andrewdata/audio_clips/rthk/radio1"),
        output_base=Path("./tmp/results/rthk/radio1"),
        path_filter="/2026/04/17/",
    ),
    "rthk-radio2": Source(
        hourly_dir=Path("/mnt/dasdata/andrewdata/ftp/rthk/hourly/radio2"),
        pattern_folder=Path("/mnt/dasdata/andrewdata/audio_clips/rthk/radio2"),
        output_base=Path("./tmp/results/rthk/radio2"),
        path_filter="/2026/04/17/",
    ),
}


def iter_m4as(source: Source) -> Iterator[Path]:
    files = sorted(source.hourly_dir.rglob("*.m4a"))
    if source.path_filter is not None:
        files = [p for p in files if source.path_filter in str(p)]
    yield from files


def process(m4a: Path, source: Source) -> None:
    rel = m4a.relative_to(source.hourly_dir)
    out_file = source.output_base / rel.with_suffix(".jsonl")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {m4a}")
    with out_file.open("w", encoding="utf-8") as f:
        subprocess.run(
            [
                "uv",
                "run",
                "audio-pattern-detector",
                "match",
                str(m4a),
                "--pattern-folder",
                str(source.pattern_folder),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=f,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    _ = ap.add_argument(
        "--source",
        choices=sorted(SOURCES.keys()),
        required=True,
        help="Which station to process.",
    )
    args = ap.parse_args()
    source = SOURCES[str(args.source)]

    for m4a in iter_m4as(source):
        process(m4a, source)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
