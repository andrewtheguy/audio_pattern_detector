import argparse
import sys


def _lazy_cmd_convert(args):
    """Import convert command lazily to speed up CLI startup."""
    from audio_pattern_detector.convert import cmd_convert

    return cmd_convert(args)


def _lazy_cmd_match(args):
    """Import match command lazily to speed up CLI startup."""
    from audio_pattern_detector.match import cmd_match

    return cmd_match(args)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='audio-pattern-detector',
        description='Audio pattern detection and conversion tools'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Add convert subcommand
    convert_parser = subparsers.add_parser('convert', help='Convert audio files to clip format')
    convert_parser.add_argument('--audio-file', metavar='audio file', type=str, help='audio file to convert')
    convert_parser.add_argument('--dest-file', metavar='audio file', type=str, help='dest saved file')
    convert_parser.set_defaults(func=_lazy_cmd_convert)

    # Add match subcommand
    match_parser = subparsers.add_parser('match', help='Find pattern matches in audio files')
    match_parser.add_argument('--pattern-file', metavar='pattern file', required=False, type=str, help='pattern file')
    match_parser.add_argument('--pattern-folder', metavar='pattern folder', required=False, type=str, help='folder with pattern audio clips')
    match_parser.add_argument('--audio-file', metavar='audio file', type=str, required=False, help='audio file to find pattern')
    match_parser.add_argument('--audio-folder', metavar='audio folder', type=str, required=False, help='audio folder to find pattern in files')
    match_parser.add_argument('--audio-url', metavar='audio url', type=str, required=False, help='audio URL to find pattern (must not be a live stream)')
    match_parser.add_argument('--stdin', action='store_true', help='read audio from stdin (pipe)')
    match_parser.add_argument('--input-format', metavar='format', type=str, required=False, help='input format for stdin (e.g., mp3, wav, flac)')
    match_parser.add_argument('--debug', metavar='debug', action=argparse.BooleanOptionalAction, help='debug mode (audio file only)', default=False)
    match_parser.set_defaults(func=_lazy_cmd_match)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Call the appropriate subcommand handler
    args.func(args)


if __name__ == '__main__':
    main()
