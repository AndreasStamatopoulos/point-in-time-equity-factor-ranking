from pathlib import Path
from finance.revisions import import_revisions

if __name__ == "__main__":
    import_revisions(Path(__file__).resolve().parent)
