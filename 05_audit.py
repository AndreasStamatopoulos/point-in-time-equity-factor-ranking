from pathlib import Path

from finance.audit import audit_project

if __name__ == "__main__":
    report = audit_project(Path(__file__).resolve().parent)
    print(f"Audit status: {report['status']}")
    print("See results/audit.md and results/audit.json")
    if report["status"] == "failed":
        raise SystemExit(
            "Scientific audit failed; correct reported violations."
        )
