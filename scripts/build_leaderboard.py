"""
Compatibility wrapper for the legacy leaderboard builder script.

The implementation now lives in the `fmbench.leaderboard` module so that it can
be reused by the fmbench CLI (`fmbench build-leaderboard`). This script is kept
for users who still run:

    python scripts/build_leaderboard.py

To make this robust in environments where the package is not yet installed
(`pip install -e .`), we add the project root to `sys.path` so that the local
`fmbench` package can be imported when running directly from the source tree
or in CI.
"""

from __future__ import annotations

import os
import sys

# Ensure the project root (containing the `fmbench` package) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fmbench.leaderboard import build_leaderboard


def main() -> None:
    build_leaderboard()


if __name__ == "__main__":
    main()

