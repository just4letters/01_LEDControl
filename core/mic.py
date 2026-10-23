"""
core/mic.py -- a microphone for SpeechRecognition that works without PyAudio.

SpeechRecognition's own sr.Microphone needs PyAudio, and PyAudio needs the
PortAudio system library, which on a Mac means Homebrew (and Homebrew has been
winding down Intel-Mac support). The `sounddevice` package ships PortAudio
inside its macOS wheel and is already installed as a MediaPipe dependency, so
it can do the same job with nothing installed at the system level.

    from core.mic import open_microphone
    with open_microphone(sample_rate=16000) as source:
        audio = recognizer.listen(source, phrase_time_limit=5)

open_microphone() returns the stock sr.Microphone when PyAudio is importable --
so a machine where voice already works is untouched -- and the sounddevice
version only when it is not.

SpeechRecognition only asks a source for four things: CHUNK, SAMPLE_RATE,
SAMPLE_WIDTH, and a `.stream` whose read(n) returns n mono 16-bit frames as
bytes. That is all this implements.
"""
import speech_recognition as sr


class _Stream:
    """The `.stream` object SpeechRecognition reads from."""

    def __init__(self, raw):
        self._raw = raw

    def read(self, size):
        data, _overflowed = self._raw.read(size)     # overflow just drops frames; same as PyAudio's exception_on_overflow=False
        return bytes(data)

    def close(self):
        try:
            self._raw.stop()
        finally:
            self._raw.close()


class SoundDeviceMicrophone(sr.AudioSource):
    SAMPLE_WIDTH = 2          # int16

    def __init__(self, device_index=None, sample_rate=16000, chunk_size=1024):
        assert device_index is None or isinstance(device_index, int), "Device index must be None or an integer"
        assert isinstance(sample_rate, int) and sample_rate > 0, "Sample rate must be a positive integer"
        assert isinstance(chunk_size, int) and chunk_size > 0, "Chunk size must be a positive integer"
        try:
            import sounddevice  # noqa: F401
        except (ImportError, OSError) as e:
            raise RuntimeError(
                "No microphone backend: PyAudio is not installed and sounddevice failed to import "
                f"({e}). Try: pip install sounddevice") from e
        self.device_index = device_index
        self.SAMPLE_RATE = sample_rate
        self.CHUNK = chunk_size
        self.stream = None

    @staticmethod
    def list_microphone_names():
        import sounddevice as sd
        return [d["name"] if d["max_input_channels"] > 0 else None for d in sd.query_devices()]

    def __enter__(self):
        assert self.stream is None, "This audio source is already inside a context manager"
        import sounddevice as sd
        try:
            raw = sd.RawInputStream(samplerate=self.SAMPLE_RATE, blocksize=self.CHUNK, device=self.device_index,
                                    channels=1, dtype="int16")
            raw.start()
        except Exception as e:
            # sr.Microphone swallows this and fails later with a confusing assert; say what happened.
            raise RuntimeError(f"Could not open the microphone ({e}). On macOS, check System Settings > "
                               "Privacy & Security > Microphone for your terminal.") from e
        self.stream = _Stream(raw)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.stream is not None:
                self.stream.close()
        finally:
            self.stream = None


def open_microphone(device_index=None, sample_rate=16000, chunk_size=1024):
    """sr.Microphone if PyAudio is available, otherwise the sounddevice-backed equivalent."""
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        return SoundDeviceMicrophone(device_index, sample_rate, chunk_size)
    return sr.Microphone(device_index=device_index, sample_rate=sample_rate, chunk_size=chunk_size)
