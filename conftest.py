"""
conftest.py
-----------
pytest configuration: adds src/ to sys.path so every test can
``import data``, ``import models``, etc. without installing the package.
"""

import sys
from pathlib import Path

# Insert src/ at the front of the path once for all test modules
sys.path.insert(0, str(Path(__file__).parent / "src"))
