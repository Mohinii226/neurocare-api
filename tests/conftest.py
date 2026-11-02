import pytest

from .helpers import build_dummy_artifacts


@pytest.fixture()
def artifact_dir(tmp_path):
    build_dummy_artifacts(tmp_path)
    return tmp_path
