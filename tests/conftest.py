"""
Module-scoped pytest fixtures that start the hub and a fake node before the
test session and tear them down after.
"""
import pytest
from tests.test_anthropic_api import FakeNode, start_hub, stop_hub


@pytest.fixture(scope="module", autouse=True)
def hub_and_node():
    start_hub()
    node = FakeNode()
    node.start()
    import time
    for _ in range(20):
        if node._registered:
            break
        time.sleep(0.2)
    assert node._registered, "Fake node failed to register within timeout"
    yield
    node.stop()
    stop_hub()
