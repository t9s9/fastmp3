import librosa
import numpy as np
import pytest
from fastmp3 import encode_mp3, decode_mp3
from fastmp3.libmp3 import MP3DecodingError, _decode_mp3
from fastmp3.utils import open_wav


@pytest.mark.parametrize('encoding,audio_bitrate', [
    ('vbr', 4),
    ('vbr', 5),
    ('cbr', 64),
    ('cbr', 192)
])
@pytest.mark.parametrize('mono', [True, False])
@pytest.mark.parametrize('sample_rate', [44100, 32000])
@pytest.mark.parametrize('duration', [1, 3])
@pytest.mark.parametrize('offset', [0.0, 0.5, 1.0])
@pytest.mark.filterwarnings('ignore:PySoundFile failed')
def test_decode(tmp_path, input_filename, encoding, audio_bitrate, mono, sample_rate, duration, offset):
    filename = (tmp_path / input_filename.name).with_suffix('.mp3')

    success, message = encode_mp3(input_filename, filename, encoding=encoding, audio_bitrate=audio_bitrate, mono=mono,
                                  sample_rate=sample_rate)
    assert success, "FFMPEG error: " + message

    lib_mp3, lib_sr = librosa.load(filename, sr=sample_rate, mono=mono, dtype=np.float32, duration=duration,
                                   offset=offset)
    lib_mp3 = lib_mp3.reshape(-1, 1) if lib_mp3.ndim == 1 else lib_mp3.T

    arr_in = np.fromfile(filename, dtype='uint8')
    arr_out, sr = decode_mp3(arr_in, length=duration, offset=offset)
    match = np.isclose(arr_out, lib_mp3, rtol=1e-3, atol=1e-3).sum() / arr_out.size

    assert sr == sample_rate
    assert sr == lib_sr
    assert match > 0.98


@pytest.mark.parametrize('mono', [True, False])
def test_decode2(tmp_path, input_filename, mono):
    filename = (tmp_path / input_filename.name).with_suffix('.mp3')

    success, message = encode_mp3(input_filename, filename, encoding='cbr', audio_bitrate=320, mono=mono)
    assert success, "FFMPEG error: " + message

    arr_in = np.fromfile(filename, dtype='uint8')
    arr_out, mp3_sr = decode_mp3(arr_in)

    wav, wav_sr = open_wav(input_filename, mono=mono)
    assert mp3_sr == wav_sr

    max_int16 = np.iinfo(np.int16).max
    wav = wav.astype(np.float32) / max_int16

    match = np.isclose(arr_out, wav, rtol=1e-3, atol=1e-3).sum() / arr_out.size
    assert match > 0.9


@pytest.mark.parametrize('encoding,audio_bitrate', [('vbr', 4), ('cbr', 192)])
@pytest.mark.parametrize('mono', [True, False])
@pytest.mark.parametrize('sample_rate', [44100, 32000])
@pytest.mark.parametrize('duration', [1, 3])
@pytest.mark.parametrize('offset', [0.0, 0.5])
@pytest.mark.filterwarnings('ignore:PySoundFile failed')
def test_decode_file(tmp_path, input_filename, encoding, audio_bitrate, mono, sample_rate, duration, offset):
    filename = (tmp_path / input_filename.name).with_suffix('.mp3')

    success, message = encode_mp3(input_filename, filename, encoding=encoding, audio_bitrate=audio_bitrate, mono=mono,
                                  sample_rate=sample_rate)
    assert success, "FFMPEG error: " + message

    lib_mp3, lib_sr = librosa.load(filename, sr=sample_rate, mono=mono, dtype=np.float32, duration=duration,
                                   offset=offset)
    lib_mp3 = lib_mp3.reshape(-1, 1) if lib_mp3.ndim == 1 else lib_mp3.T

    arr_out, sr = decode_mp3(filename, length=duration, offset=offset)
    match = np.isclose(arr_out, lib_mp3, rtol=1e-3, atol=1e-3).sum() / arr_out.size

    assert sr == sample_rate
    assert sr == lib_sr
    assert match > 0.98


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        _decode_mp3('nonexistent.mp3', np.empty(shape=(32000,), dtype='uint8'))


def test_wrong_dtype():
    with pytest.raises(ValueError):
        _decode_mp3(np.zeros(shape=(128000,), dtype='float32'), np.empty(shape=(32000,), dtype='uint8'))


def test_wrong_input():
    with pytest.raises(TypeError):
        _decode_mp3(1, np.empty(shape=(32000,), dtype='uint8'))


def test_corrupted():
    with pytest.raises(MP3DecodingError):
        _decode_mp3(np.zeros(shape=(128000,), dtype='uint8'), np.empty(shape=(32000,), dtype='uint8'))
