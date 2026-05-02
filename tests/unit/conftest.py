"""Override the parent autouse hub-and-node fixture for pure unit tests."""

import pytest


@pytest.fixture(scope="module", autouse=True)
def hub_and_node():
    yield
