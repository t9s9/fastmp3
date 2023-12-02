from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def root_path() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def dataset_path(root_path) -> Path:
    return root_path / "data"


@pytest.fixture(params=[
    'robin_chirp.wav',
    'alarm.wav',
    'people_talking.wav'
])
def input_filename(dataset_path, request):
    return dataset_path / request.param
