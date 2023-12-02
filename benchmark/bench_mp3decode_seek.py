import warnings
from pathlib import Path
from functools import partial

IT = 16
LENGTH = 10
OFFSET = 300
print("Benchmarking seek")

def _benchmark(func, iterations=100) -> (float, float):
    for i in range(iterations):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            func(Path(__file__).parent.parent / "data" / "rain.mp3")


def benchmark_torch():
    import torchaudio
    _benchmark(partial(torchaudio.load, frame_offset=int(OFFSET * 32000), num_frames=int(LENGTH * 32000)), IT)


def benchmark_librosa():
    import librosa
    _benchmark(partial(librosa.load, sr=None, duration=LENGTH, offset=OFFSET), IT)


def benchmark_fastmp3():
    from fastmp3 import decode_mp3
    import numpy as np

    def mp3decode_wrapper(filename: str):
        arr_in = np.fromfile(filename, dtype='uint8')
        return decode_mp3(arr_in, offset=OFFSET, length=LENGTH)

    _benchmark(mp3decode_wrapper, IT)


__benchmarks__ = [
    (benchmark_torch, benchmark_fastmp3, "Seek: FastMP3 vs. torchaudio"),
    (benchmark_librosa, benchmark_fastmp3, "Seek: FastMP3 vs. librosa"),
]
