"""Regression tests for the Ozsoy 2026 reproduction.

They pin the numbers the report rests on, so a later change that breaks the
match with the paper, or with the independent checks, fails loudly.
    cd ozsoy2026 && python -m pytest tests            # fast set (~1 min)
    cd ozsoy2026 && python -m pytest tests --runslow  # adds the time-domain boundaries
"""
import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run the slow tests too")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than a few seconds")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def paper_damping():
    """Every test starts from the default damping (1.34 %) and restores it."""
    import params as P
    P.set_damping(P.ZETA_PAPER_CURVES)
    yield
    P.set_damping(P.ZETA_PAPER_CURVES)


def path(*parts):
    return os.path.join(ROOT, *parts)


def load_points(name):
    return np.loadtxt(path("digitized", name), delimiter=",", skiprows=1)
