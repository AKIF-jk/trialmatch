import pytest

from trialmatch.data.load import load_rows


@pytest.fixture(scope="session")
def rows():
    return load_rows()
