"""Pytest configuration for stock predictor tests."""
import sys
from pathlib import Path

# Ensure the src directory is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
