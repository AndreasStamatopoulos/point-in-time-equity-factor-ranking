from pathlib import Path

from finance.eps import collect_eps

ROOT = Path(__file__).resolve().parent
PROBE_TICKER = "IBM"
REFRESH_CACHE = False
EXTERNAL_REQUESTS_LAST_24H = 0


def main() -> None:
    report = collect_eps(
        ROOT,
        PROBE_TICKER,
        refresh=REFRESH_CACHE,
        external_requests=EXTERNAL_REQUESTS_LAST_24H,
    )
    print(report["limitation"])
    endpoint_reports = report["endpoint_reports"]
    if isinstance(endpoint_reports, list):
        for item in endpoint_reports:
            if isinstance(item, dict):
                print(f"{item['endpoint']}: {item['status']}")
    print("Saved results/eps_availability.json")


if __name__ == "__main__":
    main()
