import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aiconls import start_language_server

if __name__ == "__main__":
    start_language_server(sys.argv[1:])
