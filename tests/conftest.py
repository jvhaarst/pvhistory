from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def loaded(data_dir):
    from pvnight.loader import load

    return load(data_dir)
