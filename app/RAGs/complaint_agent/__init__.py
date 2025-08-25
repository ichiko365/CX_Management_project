import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PARENT = APP_DIR.parent
if str(PARENT) not in sys.path:
	sys.path.insert(0, str(PARENT))