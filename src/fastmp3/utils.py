import subprocess
import wave
from pathlib import Path
from typing import Tuple, List, Union, Optional

import numpy as np


def open_wav(filename: Union[Path, str], mono: bool = False) -> (np.ndarray, int):
    """
    Open a wave file and return the PCM data and sample rate.
    :param filename: Filename of the wave file
    :param mono: If True, convert to mono by averaging the channels
    :return: PCM of shape (samples, channels), sample rate
    """
    with wave.open(str(Path(filename).with_suffix('.wav')), mode='r') as f:
        samples = f.getnframes()
        sample_rate = f.getframerate()
        channels = f.getnchannels()
        data = f.readframes(-1)
    pcm = np.frombuffer(data, dtype=np.int16).reshape(samples, channels)
    if mono:
        pcm = pcm.mean(axis=1, keepdims=True)
    return pcm, sample_rate


def apply_subprocess(command: Union[List[str], str]) -> Tuple[bool, str]:
    if isinstance(command, list):
        command = ' '.join(command)
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:  # pragma: no cover
        return False, error.output.decode('utf-8')
    return True, 'success'


def encode_mp3(input_filename: Union[str, Path],
               output_filename: Optional[Union[str, Path]] = None,
               sample_rate: int = 44100,
               encoding: str = 'vbr',
               audio_bitrate: int = 4,
               offset: int = 0,
               duration: Optional[int] = None,
               pad: Optional[str] = None,
               mono: bool = True,
               ) -> Tuple[bool, str]:
    """
    Convert an audio file to mp3 using FFMPEG with libmp3lame encoder.
    :param input_filename: Name of the input file
    :param output_filename: Name of the output file. If None, the input filename will be used with suffix .mp3
    :param sample_rate: Sample rate of the output file
    :param encoding: Encoding type, either 'cbr' (constant bitrate) or 'vbr' (variable bitrate)
    :param audio_bitrate:
    :param offset: Start time of the audio file in seconds
    :param duration: Optional duration of the audio file in seconds
    :param pad: Pad the audio to the duration. Either 'zero' or 'repeat'
    :param mono: Number of audio channels, if True, 1 channel, if False, 2 channels
    :return: Tuple of (success, message)
    """
    input_filename = Path(input_filename)
    output_filename = output_filename or input_filename
    output_filename = Path(output_filename).with_suffix('.mp3')
    if output_filename.exists():  # pragma: no cover
        return True, 'exists'

    if encoding == 'cbr':
        bitrate_encoding = '-b:a'
        bitrate = f'{audio_bitrate}k'
        c_bitrates = [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
        assert audio_bitrate in c_bitrates, f'CBR bitrate must be one of {c_bitrates}'
    elif encoding == 'vbr':
        bitrate_encoding = '-qscale:a'
        bitrate = f'{audio_bitrate}'
        v_bitrate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert audio_bitrate in v_bitrate, f'VBR bitrate must be one of {v_bitrate}'
    else:  # pragma: no cover
        raise ValueError('Encoding must be cbr (constant bitrate) or vbr (variable bitrate) '
                         '(See: https://trac.ffmpeg.org/wiki/Encode/MP3 )')

    duration_flag = ['-t', f'{int(duration)}'] if duration is not None else []

    filter_flag = []
    if pad is not None:  # pragma: no cover
        filter_flag.append('-filter_complex')
        if pad == 'zero':
            filer_cmd = f'[0]apad=whole_dur={int(duration)}[s0]'  # pad audio to duration
        elif pad == 'repeat':
            filer_cmd = f'[0]aloop=-1:size={int(duration) * sample_rate}[s0]'  # repeat audio to duration
        else:
            raise ValueError('Pad must be zero or repeat')
        filter_flag.append(filer_cmd)
        filter_flag.extend(['-map', '[s0]'])

    command = [
        'ffmpeg',
        '-y',
        '-ss', f'{int(offset)}',
        *duration_flag,
        '-threads', '0',
        '-vn',  # disable video
        '-i', f'"{input_filename}"',
        *filter_flag,
        *duration_flag,
        '-f', 'mp3',
        '-ac', f'{1 if mono else 2}',  # audio channels
        '-acodec', 'libmp3lame',
        '-ar', f'{sample_rate}',  # audio sample rate
        f'{bitrate_encoding}', f'{bitrate}',
        f'"{output_filename}"'
    ]
    return apply_subprocess(command)


def encode_wav(input_filename: Union[str, Path],
               output_filename: Optional[Union[str, Path]] = None,
               sample_rate: int = 44100,
               encoding: str = 'pcm_s16le',
               offset: int = 0,
               duration: Optional[int] = None,
               mono: bool = True,
               ) -> Tuple[bool, str]:
    """
    Convert an audio file to wav using FFMPEG.
    :param input_filename: Name of the input file
    :param output_filename: Name of the output file. If None, the input filename will be used with suffix .mp3
    :param sample_rate: Sample rate of the output file
    :param encoding: Audio codec
    :param offset: Start time of the audio file in seconds
    :param duration: Optional duration of the audio file in seconds
    :param mono: Number of audio channels, if True, 1 channel, if False, 2 channels
    :return: Tuple of (success, message)
    """
    input_filename = Path(input_filename)
    output_filename = output_filename or input_filename
    output_filename = Path(output_filename).with_suffix('.wav')
    if output_filename.exists():
        return True, 'exists'

    duration_flag = ['-t', f'{int(duration)}'] if duration is not None else []

    command = [
        'ffmpeg',
        '-y',
        '-ss', f'{int(offset)}',
        *duration_flag,
        '-threads', '0',
        '-vn',  # disable video
        '-i', f'"{input_filename}"',
        '-ac', f'{1 if mono else 2}',  # audio channels
        '-acodec', f'{encoding}',
        '-ar', f'{sample_rate}',  # audio sample rate
        f'"{output_filename}"'
    ]
    return apply_subprocess(command)
