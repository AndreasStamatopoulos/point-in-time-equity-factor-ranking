from finance.config import ROOT, START_DATE
from finance.features import build_features
from finance.prices import collect_prices

COLLECT_PRICES = True


def main() -> None:
    if COLLECT_PRICES:
        collect_prices(ROOT, start=START_DATE)
    build_features(ROOT)


if __name__ == "__main__":
    main()
