from __future__ import annotations

"""Pytest configuration.

This file ensures that the project root is on *sys.path* so that local
packages (e.g. *audio_mapper*) can be imported when running the tests
without installing the project as a site-package.
"""

import sys
from pathlib import Path

# Add the repository root directory to Python's import search path.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
