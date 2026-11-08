"""Pytest fixtures. Databases created here are temporary synthetic files."""

from __future__ import annotations

from pathlib import Path

import pytest

from database.initialize import initialize_database


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic_stage1.db"
    initialize_database(path)
    return path
