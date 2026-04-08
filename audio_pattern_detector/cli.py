import argparse
import sys


def _lazy_cmd_match(args: argparse.Namespace) -> None:
    """Import match command lazily to speed up CLI startup."""
    from audio_pattern_detector.match import cmd_match

    return cmd_match(args)


def _lazy_cmd_show_config(args: argparse.Namespace) -> None:
    """Import show_config command lazily to speed up CLI startup."""
    from audio_pattern_detector.match import cmd_show_config

    return cmd_show_config(args)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='audio-pattern-detector',
        description='Audio pattern detection tools'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Add match subcommand
    match_parser = subparsers.add_parser('match', help='Find pattern matches in audio files')
    match_parser.add_argument('--pattern-file', metavar='pattern file', required=False, type=str, action='append',
                              help='pattern file (can be specified multiple times)')
    match_parser.add_argument('--pattern-folder', metavar='pattern folder', required=False, type=str, help='folder with pattern audio clips')
    match_parser.add_argument('--audio-file', metavar='audio file', type=str, required=False, help='audio file to find pattern')
    match_parser.add_argument('--audio-folder', metavar='audio folder', type=str, required=False, help='audio folder to find pattern in files')
    match_parser.add_argument('--stdin', action='store_true', help='read audio from stdin in WAV format (always outputs JSONL)')
    match_parser.add_argument('--multiplexed-stdin', action='store_true',
                              help='read patterns and audio from stdin using multiplexed protocol (always outputs JSONL). '
                                   'Protocol: [uint32 num_patterns] then for each pattern [uint32 name_len][name][uint32 data_len][wav_data], '
                                   'followed by audio stream (WAV or raw PCM with --raw-pcm)')
    match_parser.add_argument('--target-sample-rate', metavar='rate', type=int, required=False, help='target sample rate for processing in Hz (default: 8000)')
    match_parser.add_argument('--jsonl', action='store_true', help='output JSONL events (start, pattern_detected, end) as they occur')
    match_parser.add_argument('--timestamp-format', choices=['ms', 'formatted'], default='ms',
                              help='timestamp format in JSONL output: "ms" for integer milliseconds (default), "formatted" for HH:MM:SS.mmm strings')
    match_parser.add_argument('--chunk-seconds', metavar='seconds', type=str, default='60',
                              help='seconds per chunk for sliding window (default: 60, use "auto" to auto-compute based on pattern length)')
    match_parser.add_argument('--debug', action=argparse.BooleanOptionalAction, help='debug mode (audio file only)', default=False)
    match_parser.add_argument('--debug-dir', metavar='dir', type=str, default='./tmp',
                              help='base directory for debug output (default: ./tmp)')
    match_parser.add_argument('--height-min', metavar='height', type=float, default=None,
                              help='override minimum correlation peak height (default: 0.25, lower to find weak matches)')
    match_parser.set_defaults(func=_lazy_cmd_match)

    # Add show-config subcommand
    show_config_parser = subparsers.add_parser('show-config', help='Show computed configuration for pattern files')
    show_config_parser.add_argument('--pattern-file', metavar='pattern file', required=False, type=str, action='append',
                                    help='pattern file (can be specified multiple times)')
    show_config_parser.add_argument('--pattern-folder', metavar='pattern folder', required=False, type=str, help='folder with pattern audio clips')
    show_config_parser.add_argument('--target-sample-rate', metavar='rate', type=int, required=False, help='target sample rate for processing in Hz (default: 8000)')
    show_config_parser.set_defaults(func=_lazy_cmd_show_config)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Call the appropriate subcommand handler
    args.func(args)


if __name__ == '__main__':
    main()
