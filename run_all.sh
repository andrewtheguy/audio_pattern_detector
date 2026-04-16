#!/usr/bin/env bash
set -euo pipefail

HOURLY_DIR="/mnt/dasdata/capture881903/output/hourly"
PATTERN_FOLDER="/mnt/dasdata/andrewdata/audio_clips"
OUTPUT_BASE="./tmp/results"

find "$HOURLY_DIR" -name '*.m4a' -type f | sort | while IFS= read -r m4a; do
    rel="${m4a#$HOURLY_DIR/}"
    dir="$(dirname "$rel")"
    base="$(basename "$rel" .m4a)"

    out_dir="$OUTPUT_BASE/$dir"
    out_file="$out_dir/${base}.jsonl"

    mkdir -p "$out_dir"
    echo "Processing: $m4a"
    uv run audio-pattern-detector match "$m4a" \
        --pattern-folder "$PATTERN_FOLDER" \
        > "$out_file" < /dev/null
done

echo "Done."
