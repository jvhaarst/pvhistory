from pathlib import Path

import pytest

from pvnight.config import DATA_SUBDIR

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """The folder holding the PVOutput exports, not the repository root."""
    return REPO_ROOT / DATA_SUBDIR


@pytest.fixture(scope="session")
def loaded(data_dir):
    from pvnight.loader import load

    return load(data_dir)
