"""App package bootstrap.

This makes 'app' importable as a package and also adds its parent directory to
sys.path when needed so that absolute imports like `from app import schema` work
even if execution starts from within the app folder.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PARENT = APP_DIR.parent
if str(PARENT) not in sys.path:
	sys.path.insert(0, str(PARENT))
