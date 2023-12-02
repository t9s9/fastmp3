import ctypes as ct
from pathlib import Path
from typing import Union, Optional, Tuple

import fastmp3._libmp3
import numpy as np
import numpy.ctypeslib as npct

lib = ct.CDLL(fastmp3._libmp3.__file__)

uint_1d_type = npct.ndpointer(dtype=np.uint8, ndim=1, flags=['C_CONTIGUOUS'])


class MP3DecodingError(Exception):
    pass


minimp3_errors = {-1: "Parameter error",
                  -2: "Memory allocation error",
                  -3: "IO error",
                  -4: "User error",
                  -5: "Decoding error"}

ctypes_mp3_decode_buffer = lib.mp3_decode_buffer
ctypes_mp3_decode_buffer.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_int, ct.c_long, ct.c_long]
ctypes_mp3_decode_buffer.restype = ct.c_int

ctypes_mp3_decode_file = lib.mp3_decode_file
ctypes_mp3_decode_file.argtypes = [ct.c_char_p, ct.c_void_p, ct.c_int, ct.c_long, ct.c_long]
ctypes_mp3_decode_file.restype = ct.c_int


def _decode_mp3_array(arr_in: np.ndarray,
                      arr_out: np.ndarray,
                      offset: int = 0,
                      length: int = 0) -> int:
    """
    Decode mp3 data from a numpy array.
    :param arr_in: Input array of type uint8
    :param arr_out: Output array of type float32
    :param offset: Optional offset in samples
    :param length: Optional length in samples
    :return: Number of samples decoded
    """
    return ctypes_mp3_decode_buffer(arr_in.ctypes.data, arr_in.size,
                                    arr_out.ctypes.data, arr_out.size,
                                    offset, length)


def _decode_mp3_file(filename: str,
                     arr_out: np.ndarray,
                     offset: int = 0,
                     length: int = 0) -> int:
    """
    Decode mp3 data from a file.
    :param filename: Path to the file
    :param arr_out: Output array of type float32
    :param offset: Optional offset in samples
    :param length: Optional length in samples
    :return: Number of samples decoded
    """
    return ctypes_mp3_decode_file(filename.encode('utf-8'),
                                  arr_out.ctypes.data, arr_out.size,
                                  offset, length)


def _decode_mp3(inputs: Union[np.ndarray, str, Path],
                arr_out: np.ndarray,
                offset: int = 0,
                length: Optional[int] = None) -> int:
    """
    Decode mp3 data from a numpy array or a file.
    """
    if isinstance(inputs, np.ndarray):
        if inputs.dtype != np.uint8:
            raise ValueError("Input array must be of type uint8.")
        out = _decode_mp3_array(inputs, arr_out, offset, length or 0)
    elif isinstance(inputs, (str, Path)):
        if not Path(inputs).exists():
            raise FileNotFoundError(f"File {inputs} does not exist.")
        out = _decode_mp3_file(str(inputs), arr_out, offset, length or 0)
    else:
        raise TypeError("Input must be a numpy array or a path to a file.")
    if out >= 0:
        return out
    elif out == -100:
        raise MP3DecodingError("Cannot decode MP3 buffer.")
    elif out == -200:  # pragma: no cover
        raise MP3DecodingError("Cannot seek in MP3 buffer.")
    else:  # pragma: no cover
        raise MP3DecodingError("Cannot read MP3 buffer.")


class ProbeOutput(ct.Structure):
    _fields_ = [
        ('samples', ct.c_int),
        ('channel', ct.c_int),
        ('sample_rate', ct.c_int),
        ('bitrate_kbps', ct.c_int),
    ]

    def __repr__(self):
        return f"ProbeOutput(samples={self.samples}, channel={self.channel}, sample_rate={self.sample_rate}, " \
               f"bitrate_kbps={self.bitrate_kbps})"

    def __str__(self):
        return repr(self)


ctypes_mp3_probe_buffer = lib.mp3_probe_buffer
ctypes_mp3_probe_buffer.argtypes = [uint_1d_type, ct.c_int]
ctypes_mp3_probe_buffer.restype = ProbeOutput

ctypes_mp3_probe_file = lib.mp3_probe_file
ctypes_mp3_probe_file.argtypes = [ct.c_char_p]
ctypes_mp3_probe_file.restype = ProbeOutput


def _probe_mp3_array(arr_in: np.ndarray) -> ProbeOutput:
    return ctypes_mp3_probe_buffer(arr_in, arr_in.size)


def _probe_mp3_file(filename: str) -> ProbeOutput:
    return ctypes_mp3_probe_file(filename.encode('utf-8'))


def probe_mp3(inputs: Union[np.ndarray, str, Path]) -> ProbeOutput:
    """
    Probe MP3 buffer.
    :param inputs: Numpy array of type uint8.
    :return: ProbeOutput containing information about the MP3 buffer:
        - samples: Number of samples in the MP3 buffer.
        - channel: Number of channels in the MP3 buffer.
        - sample_rate: Sample rate of the MP3 buffer.
        - bitrate_kbps: Average bitrate of the MP3 buffer.
    """
    if isinstance(inputs, np.ndarray):
        if inputs.dtype != np.uint8:
            raise ValueError("Input array must be of type uint8.")
        out = _probe_mp3_array(inputs)
    elif isinstance(inputs, (str, Path)):
        if not Path(inputs).exists():
            raise FileNotFoundError(f"File {inputs} does not exist.")
        out = _probe_mp3_file(str(inputs))
    else:
        raise TypeError("Input must be a numpy array or a path to a file.")

    if out.samples == -1:
        raise MP3DecodingError("Cannot decode MP3 buffer.")
    return out


def decode_mp3(inputs: Union[np.ndarray, str, Path],
               offset: float = 0.0,
               length: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """
    Decode MP3 buffer to float32 array.
    :param inputs: Numpy array of type uint8.
    :param offset: Offset in seconds.
    :param length: Length in seconds.
    :return: Numpy array of type float32 containing the decoded MP3 buffer as well as the sample rate. The shape of the
        array is (samples, channels).
    """
    probe = probe_mp3(inputs)

    offset = int(offset * probe.sample_rate)
    max_samples = max(0, probe.samples - offset)
    if length is not None:
        length = int(length * probe.sample_rate)
        max_samples = min(max_samples, length)

    arr_out = np.empty(shape=(max_samples, probe.channel), dtype=np.float32)

    _decode_mp3(inputs, arr_out, offset, length)
    return arr_out, probe.sample_rate


ctypes_unpackbits = lib.unpackbits
ctypes_unpackbits.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]
ctypes_unpackbits.restype = ct.c_int


def unpackbits(arr_in: np.ndarray) -> np.ndarray:
    """
    Unpacks elements of a uint8 array into a binary-valued output array. The bitorder is big-endian.
    :param arr_in: Input array of type uint8 of shape (N, )
    :return: Output array of type uint8 with binary values of shape (N * 8, )
    """
    arr_out = np.empty(arr_in.size * 8, dtype=np.uint8)
    ctypes_unpackbits(arr_in.ctypes.data, arr_in.size, arr_out.ctypes.data)
    return arr_out


def _unpackbits(arr_in: np.ndarray, arr_out: np.ndarray) -> int:
    """
    Unpacks elements of a uint8 array into a binary-valued output array. The bitorder is big-endian.
    The output array must be allocated before calling this function.
    :param arr_in: Input array of type uint8 of shape (N, )
    :param arr_out: Output array of type uint8 with binary values of shape (N * 8, )
    :return: 0 if successful
    """
    return ctypes_unpackbits(arr_in.ctypes.data, arr_in.size, arr_out.ctypes.data)
