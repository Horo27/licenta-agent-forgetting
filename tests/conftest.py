from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def in_repo_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the test from the repo root, so relative config paths resolve as in the CLI."""
    monkeypatch.chdir(ROOT)
    return ROOT
