"""Build PIT 100+ stock features from *historical* vendor CSV inputs, not today's names."""
import argparse
from pathlib import Path
from finance.pit_panel import run_from_directory

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,default=Path('data/vendor'))
    parser.add_argument('--output-dir',type=Path,default=Path('data'))
    parser.add_argument('--asof',required=True,help='YYYY-MM-DD, reproducible data cutoff')
    parser.add_argument('--start',default='2017-01-01')
    a=parser.parse_args()
    run_from_directory(a.input_dir,asof=a.asof,start=a.start,out_dir=a.output_dir)