"""Tests for the Python bindings.

Run against an installed wheel (`maturin develop` or `pip install`), or
against the extension built with `cargo build --features python --lib`.
"""

import importlib.machinery
import importlib.util
import io
import os
import pathlib
import shutil
import subprocess
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLES = REPO_ROOT / "sample_audios"

RTHK_BEEP_PATTERN = SAMPLES / "clips" / "rthk_beep.apd.toml"
CBS_NEWS_PATTERN = SAMPLES / "clips" / "cbs_news.wav"
RAINBOW_INTRO_PATTERN = SAMPLES / "clips" / "天空下的彩虹intro.wav"
ALL_PATTERNS = [RTHK_BEEP_PATTERN, CBS_NEWS_PATTERN, RAINBOW_INTRO_PATTERN]

RTHK_BEEP_AUDIO = SAMPLES / "rthk_section_with_beep.wav"
CBS_NEWS_AUDIO = SAMPLES / "cbs_news_audio_section.wav"
RAINBOW_INTRO_AUDIO = SAMPLES / "am1430_section_with_rainbow_intro.wav"

RTHK_BEEP_NAME = "rthk_beep"
CBS_NEWS_NAME = "cbs_news"
RAINBOW_INTRO_NAME = "天空下的彩虹intro"

RTHK_BEEP_EXPECTED_TIMES = [1.407875, 2.419625]
CBS_NEWS_EXPECTED_TIME = 25.89875
RAINBOW_INTRO_EXPECTED_TIME = 13.847999999999999

RTHK_BEEP_DURATION = 4.077625
CBS_NEWS_DURATION = 32.12175
RAINBOW_INTRO_DURATION = 30.00125

MODULE_ENV = "AUDIO_PATTERN_DETECTOR_PYTHON_MODULE"


def _built_module_path() -> pathlib.Path:
    override = os.environ.get(MODULE_ENV)
    if override:
        path = pathlib.Path(override)
        if path.exists():
            return path
        raise FileNotFoundError(f"{MODULE_ENV} points to a missing file: {path}")

    suffixes = list(importlib.machinery.EXTENSION_SUFFIXES)
    for fallback_suffix in (".so", ".dylib", ".pyd", ".dll"):
        if fallback_suffix not in suffixes:
            suffixes.append(fallback_suffix)

    for profile in ("debug", "release"):
        target_dir = REPO_ROOT / "target" / profile
        for stem in ("libaudio_pattern_detector", "audio_pattern_detector"):
            for suffix in suffixes:
                path = target_dir / f"{stem}{suffix}"
                if path.exists():
                    return path

    raise FileNotFoundError(
        "Could not find the audio_pattern_detector extension. Install it with "
        "`maturin develop` or build it with `cargo build --features python --lib`."
    )


def _load_module():
    if not os.environ.get(MODULE_ENV):
        try:
            import audio_pattern_detector

            return audio_pattern_detector
        except ImportError:
            pass

    module_path = _built_module_path()
    spec = importlib.util.spec_from_file_location("audio_pattern_detector", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


apd = _load_module()


class ModuleTest(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(apd.APD_EXTENSION, ".apd.toml")
        self.assertEqual(apd.DEFAULT_TARGET_SAMPLE_RATE, 8000)
        self.assertEqual(apd.DEFAULT_SECONDS_PER_CHUNK, 60)
        self.assertRegex(apd.__version__, r"^\d+\.\d+\.\d+$")

    def test_clip_name(self):
        self.assertEqual(apd.clip_name(RTHK_BEEP_PATTERN), RTHK_BEEP_NAME)
        self.assertEqual(apd.clip_name(str(RAINBOW_INTRO_PATTERN)), RAINBOW_INTRO_NAME)
        self.assertEqual(apd.clip_name("dir/903_beep.APD.toml"), "903_beep")
        self.assertEqual(apd.clip_name("dir/intro.v2.wav"), "intro.v2")


class DetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = apd.Detector(ALL_PATTERNS)

    def test_properties_and_config(self):
        self.assertEqual(self.detector.clip_names, [RTHK_BEEP_NAME, CBS_NEWS_NAME, RAINBOW_INTRO_NAME])
        self.assertEqual(self.detector.seconds_per_chunk, 60)
        self.assertEqual(self.detector.target_sample_rate, 8000)
        self.assertEqual(
            self.detector.config(),
            {
                "default_seconds_per_chunk": 60,
                "min_chunk_size_seconds": 8,
                "sample_rate": 8000,
                "clips": {
                    RTHK_BEEP_NAME: {"duration_seconds": 0.228375, "sliding_window_seconds": 1},
                    CBS_NEWS_NAME: {"duration_seconds": 0.9965, "sliding_window_seconds": 1},
                    RAINBOW_INTRO_NAME: {"duration_seconds": 3.686, "sliding_window_seconds": 4},
                },
            },
        )

    def test_auto_seconds_per_chunk(self):
        detector = apd.Detector([str(CBS_NEWS_PATTERN)], seconds_per_chunk=None)
        self.assertEqual(detector.seconds_per_chunk, 2)

    def test_match_file_reuses_detector(self):
        result = self.detector.match_file(RTHK_BEEP_AUDIO)
        self.assertEqual(
            result.detections,
            {RTHK_BEEP_NAME: RTHK_BEEP_EXPECTED_TIMES, CBS_NEWS_NAME: [], RAINBOW_INTRO_NAME: []},
        )
        self.assertEqual(list(result.detections), self.detector.clip_names)
        self.assertEqual(result.duration_seconds, RTHK_BEEP_DURATION)

        result = self.detector.match_file(str(CBS_NEWS_AUDIO))
        self.assertEqual(
            result.detections,
            {RTHK_BEEP_NAME: [], CBS_NEWS_NAME: [CBS_NEWS_EXPECTED_TIME], RAINBOW_INTRO_NAME: []},
        )
        self.assertEqual(result.duration_seconds, CBS_NEWS_DURATION)

        result = self.detector.match_file(RAINBOW_INTRO_AUDIO)
        self.assertEqual(
            result.detections,
            {RTHK_BEEP_NAME: [], CBS_NEWS_NAME: [], RAINBOW_INTRO_NAME: [RAINBOW_INTRO_EXPECTED_TIME]},
        )
        self.assertEqual(result.duration_seconds, RAINBOW_INTRO_DURATION)
        self.assertEqual(
            repr(result),
            f"MatchResult(detections={result.detections!r}, duration_seconds={RAINBOW_INTRO_DURATION!r})",
        )

    def test_on_detected_callback(self):
        events = []
        result = self.detector.match_file(RTHK_BEEP_AUDIO, on_detected=lambda name, t: events.append((name, t)))
        self.assertEqual(events, [(RTHK_BEEP_NAME, t) for t in RTHK_BEEP_EXPECTED_TIMES])
        self.assertEqual(result.detections[RTHK_BEEP_NAME], RTHK_BEEP_EXPECTED_TIMES)

    def test_callback_exception_propagates(self):
        def fail(name, timestamp):
            raise KeyError(name)

        with self.assertRaises(KeyError) as raised:
            self.detector.match_file(CBS_NEWS_AUDIO, on_detected=fail)
        self.assertEqual(raised.exception.args, (CBS_NEWS_NAME,))

    def test_match_wav_stream(self):
        with open(CBS_NEWS_AUDIO, "rb") as stream:
            result = self.detector.match_wav_stream(stream)
        self.assertEqual(
            result.detections,
            {RTHK_BEEP_NAME: [], CBS_NEWS_NAME: [CBS_NEWS_EXPECTED_TIME], RAINBOW_INTRO_NAME: []},
        )
        self.assertEqual(result.duration_seconds, CBS_NEWS_DURATION)

        events = []
        result = self.detector.match_wav_stream(
            io.BytesIO(RAINBOW_INTRO_AUDIO.read_bytes()),
            on_detected=lambda name, t: events.append((name, t)),
        )
        self.assertEqual(events, [(RAINBOW_INTRO_NAME, RAINBOW_INTRO_EXPECTED_TIME)])
        self.assertEqual(result.duration_seconds, RAINBOW_INTRO_DURATION)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not available")
    def test_match_wav_stream_from_ffmpeg_pipe(self):
        command = ["ffmpeg", "-v", "error", "-i", str(CBS_NEWS_AUDIO), "-f", "wav", "-ac", "1", "-ar", "8000", "pipe:"]
        with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
            result = self.detector.match_wav_stream(process.stdout)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(result.detections[CBS_NEWS_NAME], [CBS_NEWS_EXPECTED_TIME])
        self.assertEqual(result.duration_seconds, CBS_NEWS_DURATION)

    def test_stream_errors(self):
        with self.assertRaises(ValueError) as raised:
            self.detector.match_wav_stream(io.BytesIO(b"junkjunkjunk"))
        self.assertEqual(str(raised.exception), 'Not a WAV file: expected RIFF, got "junk"')

        with self.assertRaises(TypeError) as raised:
            self.detector.match_wav_stream(io.StringIO("x" * 100))
        self.assertEqual(str(raised.exception), "stream.read() must return bytes")

        class Broken:
            def read(self, size):
                raise ConnectionResetError("stream went away")

        with self.assertRaises(ConnectionResetError) as raised:
            self.detector.match_wav_stream(Broken())
        self.assertEqual(str(raised.exception), "stream went away")

    def test_invalid_input_raises_value_error(self):
        cases = [
            (lambda: apd.Detector([]), "No pattern clips passed"),
            (lambda: apd.Detector(["nonexistent.wav"]), "Pattern nonexistent.wav does not exist"),
            (lambda: self.detector.match_file("nonexistent.wav"), "Audio nonexistent.wav does not exist"),
            (
                lambda: apd.Detector([CBS_NEWS_PATTERN], target_sample_rate=0),
                "target sample rate must be greater than 0",
            ),
            (
                lambda: apd.Detector([CBS_NEWS_PATTERN], seconds_per_chunk=1),
                "seconds_per_chunk 1 is too small for clip 'cbs_news' "
                "(duration: 1.00s, sliding_window: 1s, minimum chunk size: 2s)",
            ),
        ]
        for call, message in cases:
            with self.assertRaises(ValueError) as raised:
                call()
            self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
