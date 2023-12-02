import numpy as np
import pytest
from fastmp3 import probe_mp3, encode_mp3
from fastmp3.libmp3 import MP3DecodingError


@pytest.mark.parametrize('encoding,audio_bitrate', [('vbr', 4), ('vbr', 6), ('cbr', 64), ('cbr', 192)])
@pytest.mark.parametrize('mono', [True, False])
@pytest.mark.parametrize('sample_rate', [44100, 32000])
@pytest.mark.parametrize('duration', [2, 5])
@pytest.mark.parametrize('from_file', [False, True])
def test_probe(tmp_path, input_filename, encoding, audio_bitrate, mono, sample_rate, duration, from_file):
    filename = (tmp_path / input_filename.name).with_suffix('.mp3')

    success, message = encode_mp3(input_filename, filename, encoding=encoding, audio_bitrate=audio_bitrate, mono=mono,
                                  sample_rate=sample_rate, duration=duration)
    assert success, "FFMPEG error: " + message

    if from_file:
        inputs = filename
    else:
        inputs = np.fromfile(filename, dtype='uint8')

    probe = probe_mp3(inputs)
    assert probe.channel == 1 if mono else 2
    assert probe.sample_rate == sample_rate
    assert probe.samples == duration * sample_rate
    if encoding == 'cbr':
        assert probe.bitrate_kbps == audio_bitrate
    assert str(probe) == (f"ProbeOutput(samples={probe.samples}, channel={probe.channel}, "
                          f"sample_rate={probe.sample_rate}, bitrate_kbps={probe.bitrate_kbps})")


def test_corrupted():
    with pytest.raises(MP3DecodingError):
        probe_mp3(np.zeros(shape=(128000,), dtype='uint8'))


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        probe_mp3('nonexistent.mp3')


def test_wrong_format():
    with pytest.raises(TypeError):
        # raise a TypeError if the input is not a string or a numpy array
        probe_mp3(1)


def test_wrong_dtype():
    with pytest.raises(ValueError):
        probe_mp3(np.zeros(shape=(128000,), dtype='float32'))
