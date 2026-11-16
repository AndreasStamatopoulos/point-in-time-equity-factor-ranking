from pathlib import Path

from finance.sec import main

ROOT = Path(__file__).resolve().parent
REFRESH_CACHE = False

if __name__ == "__main__":
    main(ROOT, refresh=REFRESH_CACHE)
