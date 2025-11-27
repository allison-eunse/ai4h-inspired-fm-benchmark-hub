"""
Compatibility wrapper for the legacy leaderboard builder script.

The implementation now lives in the `fmbench.leaderboard` module so that it can
be reused by the fmbench CLI (`fmbench build-leaderboard`). This script is kept
for users who still run:

    python scripts/build_leaderboard.py
"""

from fmbench.leaderboard import build_leaderboard


def main() -> None:
    build_leaderboard()


if __name__ == "__main__":
    main()

