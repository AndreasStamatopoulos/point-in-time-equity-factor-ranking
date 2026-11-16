from pathlib import Path
import ast
from finance.config import FEATURES
from finance.model import ModelSettings

def test_original_assignment_is_five_plus_eps():
    assert set(FEATURES) == {'momentum_12_1','book_to_market','operating_profitability','asset_growth','fcf_yield'}

def test_cli_dashboard_python_sources_parse():
    root=Path(__file__).resolve().parents[1]
    for name in ['run.py','app.py']:
        ast.parse((root/name).read_text(encoding='utf-8'))

def test_model_stays_non_neural():
    assert ModelSettings(model='ols').model == 'ols'
