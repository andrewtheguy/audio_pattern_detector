
def scrape():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-file', metavar='audio file', type=str, help='audio file to find pattern')
    parser.add_argument('--match-method', metavar='pattern match method', type=str, help='pattern match method',default="correlation")
    #parser.add_argument('--window', metavar='seconds', type=int, default=10, help='Only use first n seconds of the audio file')
    args = parser.parse_args()
    #print(args.method)

    # Find clip occurrences in the full audio
    peak_times = find_clip_in_audio_in_chunks('./audio_clips/rthk_beep.wav', args.audio_file, method=args.match_method)
    print(peak_times)

    for offset in peak_times:
        print(f"Clip occurs at the following times (in seconds): {str(datetime.timedelta(seconds=offset))}" )
    #    #print(f"Offset: {offset}s" )
    
