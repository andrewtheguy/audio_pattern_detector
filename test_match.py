from match import match_pattern

if __name__ == '__main__':

    pairs = (
        ("/Volumes/andrewdata/ftp/grabradiostreamed/am1430/multiple/日落大道/日落大道20240529_1600_s_1.m4a",
         "./audio_clips/am1430/programsponsoredby.wav"),
        ("/Volumes/andrewdata/ftp/grabradiostreamed/am1430/multiple/日落大道/日落大道20240529_1600_s_1.m4a",
         "./audio_clips/am1430/programsponsoredby2.wav"),
        ("/Volumes/andrewdata/ftp/grabradiostreamed/am1430/multiple/日落大道/日落大道20240529_1600_s_1.m4a",
         "./audio_clips/am1430/programsponsoredby3.wav"),
        ("/Volumes/andrewdata/ftp/grabradiostreamed/am1430/multiple/日落大道/日落大道20240529_1600_s_1.m4a",
         "./audio_clips/am1430/programsponsoredby4.wav"),

        ("/Volumes/andrewdata/ftp/grabradiostreamed/am1430/single/漫談法律/漫談法律20240519_1000_s_1.m4a",
         "./audio_clips/am1430/漫談法律intro.wav"),
        ("/Volumes/andrewdata/ftp/grabradiostreamed/am1430/single/繼續有心人friday/繼續有心人friday20240510_1200_s_1.m4a",
         "./audio_clips/am1430/繼續有心人intro.wav"),
        ("/Volumes/andrewdata/ftp/rthk/original/KnowledgeCo/KnowledgeCo20240518.m4a",
         "./audio_clips/rthk_news_report_theme.wav"),
        ("/Volumes/andrewdata/ftp/grabradiostreamed/am1430/single/置業興家/置業興家20240518_1330_s_1.m4a",
         "./audio_clips/am1430/置業興家intro.wav"),
        ("/Volumes/andrewdata/ftp/grabradiostreamed/am1430/single/法律天地/法律天地20240518_1230_s_1.m4a",
         "./audio_clips/am1430/法律天地intro.wav"),
        ("/Volumes/andrewdata/ftp/grabradiostreamed/am1430/multiple/受之有道/受之有道20240520_1800_s_1.m4a",
         "./audio_clips/am1430/受之有道outro.wav"),
        ("/Volumes/andrewdata/ftp/grabradiostreamed/am1430/multiple/日落大道/日落大道20240523_1600_s_1.m4a",
         "./audio_clips/am1430/日落大道smallinterlude.wav"),
        ("/Volumes/andrewdata/ftp/grabradiostreamed/am1430/multiple/日落大道/日落大道20240523_1600_s_1.m4a",
         "./audio_clips/am1430/日落大道interlude.wav"),
        ("/Volumes/andrewdata/ftp/rthk/original/morningsuite/morningsuite20240531.m4a",
         "./audio_clips/morningsuitebababa.wav"),
    )

    for pair in pairs:
        match_pattern(audio_file=pair[0], pattern_file=pair[1], method="correlation")
