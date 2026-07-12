#!/usr/bin/env python3
"""Build the checked-in CA-LRU evidence tables from raw result files."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calru_paper.evidence import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
