"""Helpers for running scripts directly from the scripts directory."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> None:
    """Add the repository root to sys.path when a script is run directly."""

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
