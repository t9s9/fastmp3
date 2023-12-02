import numpy as np
import pytest

from fastmp3.libmp3 import unpackbits, _unpackbits


@pytest.mark.parametrize('size', [8, 16, 32, 64, 128, 256])
def test_unpack(tmp_path, size):
    arr = np.random.randint(2, size=size, dtype=bool)
    arr = np.packbits(arr, bitorder='big')

    true = np.unpackbits(arr)

    test = unpackbits(arr)
    assert np.all(true == test)


@pytest.mark.parametrize('size', [8, 16, 32, 64, 128, 256])
def test_unpack_raw(tmp_path, size):
    arr = np.random.randint(2, size=size, dtype=bool)
    arr = np.packbits(arr, bitorder='big')

    true = np.unpackbits(arr)

    test = np.empty(size, dtype=np.uint8)
    _unpackbits(arr, test)
    assert np.allclose(true, test)
