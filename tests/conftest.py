"""Pytest configuration for multi-agent-harness tests."""


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (slower, may use network)",
    )
    config.addinivalue_line(
        "markers",
        "unit: mark test as unit test (fast, no network)",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running",
    )
