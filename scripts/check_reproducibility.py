#!/usr/bin/env python3
"""Check that committed CA-LRU tables reproduce byte-for-byte."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calru_paper.evidence import reproducibility_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(reproducibility_main())
