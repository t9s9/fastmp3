import warnings
from pathlib import Path
from functools import partial

IT = 16
print("Benchmarking torch")

def _benchmark(func, iterations=100) -> (float, float):
    root = Path("/home/t9s9/Datasets/AudioCaps/mp3/train")
    for file in list(root.glob("*.mp3"))[:iterations]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            func(file)




def benchmark_torch():
    import torchaudio
    _benchmark(torchaudio.load, IT)


def benchmark_librosa():
    import librosa
    _benchmark(partial(librosa.load, sr=None), IT)


def benchmark_fastmp3():
    from fastmp3 import decode_mp3
    import numpy as np

    def mp3decode_wrapper(filename: str):
        arr_in = np.fromfile(filename, dtype='uint8')
        return decode_mp3(arr_in)

    _benchmark(mp3decode_wrapper, IT)


__benchmarks__ = [
    (benchmark_torch, benchmark_fastmp3, "FastMP3 vs. torchaudio"),
    (benchmark_librosa, benchmark_fastmp3, "FastMP3 vs. librosa"),
]
