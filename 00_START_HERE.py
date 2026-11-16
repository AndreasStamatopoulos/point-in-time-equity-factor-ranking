"""Offline check: show which files exist before running API collectors."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
print('ML FINANCE — Spyder project')
print('Directory:', ROOT)
for item in ('data/universe.csv', 'data/prices.csv', 'data/sec_facts.csv',
             'data/features.csv', 'data/features_metadata.json'):
    p = ROOT / item
    print(('READY ' if p.exists() else 'MISSING'), item)
print('\nOrder: 01_collect_sec.py -> 02_collect_eps.py (optional probe)')
print('       -> 03_build_features.py -> 04_run_ols.py -> 05_audit.py')
print('EPS revisions remain unverified in the supplied source.')
print('Read README_GR.md and MATHIMA_VIMA_VIMA_GR.md before live API calls.')
