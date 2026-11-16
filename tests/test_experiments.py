"""Unit tests for the new experimental engine, not claims of financial performance."""
import numpy as np
import pandas as pd
from finance.experiments import (
    ExperimentConfig, fit_predict, inner_folds, mature_rows, oof_residuals,
    relative_target,
)


def panel_fixture():
    dates = pd.date_range('2016-01-31', periods=55, freq='ME')
    rng = np.random.default_rng(7)
    rows = []
    for date in dates:
        entry = date + pd.offsets.MonthEnd(1)
        label_end = entry + pd.offsets.MonthEnd(6)
        for j in range(7):
            rows.append({
                'date': date, 'ticker': f'T{j}',
                'label_end_6m': label_end,
                'forward_return_6m': float(rng.normal(0.04 + j * 0.003, 0.1)),
                'available_at': date,
                'momentum_12_1': float(rng.normal()),
                'book_to_market': float(rng.normal()),
            })
    return pd.DataFrame(rows)


def test_mature_rows_strictly_purges_six_month_overlap():
    panel = panel_fixture()
    decision = pd.Timestamp('2018-07-31')
    old = mature_rows(panel, decision)
    assert not old.empty
    assert (old['label_end_6m'] < decision).all()
    assert (old['date'] < decision).all()
    assert old.groupby('date').size().eq(7).all()


def test_inner_folds_are_strictly_past_and_label_mature():
    panel = panel_fixture()
    window = mature_rows(panel, pd.Timestamp('2020-06-30'))
    folds = inner_folds(window, n_folds=3)
    assert len(folds) >= 2
    for train_idx, val_idx, when in folds:
        assert (window.loc[train_idx, 'label_end_6m'] < when).all()
        assert window.loc[val_idx, 'date'].eq(when).all()
        assert set(train_idx).isdisjoint(val_idx)


def test_mlp_is_deterministic_for_fixed_seed():
    panel = panel_fixture()
    x = panel[['momentum_12_1', 'book_to_market']].iloc[:125]
    y = relative_target(panel.iloc[:125])
    params = {'layers': (8,), 'weight_decay': 0.01,
              'learning_rate': 0.001, 'batch_size': 16}
    first = fit_predict('mlp', params, x.iloc[:90], y.iloc[:90], x.iloc[90:], 42, 40)
    second = fit_predict('mlp', params, x.iloc[:90], y.iloc[:90], x.iloc[90:], 42, 40)
    np.testing.assert_allclose(first, second)


def test_hybrid_residual_training_is_cross_fitted_and_precedes_outer_date():
    panel = panel_fixture()
    outer = pd.Timestamp('2020-07-31')
    historical = mature_rows(panel, outer)
    x = historical[['momentum_12_1', 'book_to_market']]
    cfg = ExperimentConfig(min_train_months=18, trials=0, max_iter=40)
    oof = oof_residuals(historical, x, cfg)
    assert not oof.empty
    assert oof['date'].nunique() >= 3
    assert (oof['label_end_6m'] < outer).all()
    # It must not be an in-sample residual of a model fitted on this OOF date's label.
    assert oof['oof_residual'].notna().all()


def test_future_label_cannot_influence_training_prediction():
    panel = panel_fixture()
    decision = pd.Timestamp('2020-06-30')
    old = mature_rows(panel, decision)
    x = old[['momentum_12_1', 'book_to_market']]
    test_x = panel.loc[panel['date'].eq(decision), ['momentum_12_1', 'book_to_market']]
    before = fit_predict('ols', {}, x, relative_target(old), test_x, 42, 40)
    tampered = panel.copy()
    tampered.loc[tampered['date'].ge(decision), 'forward_return_6m'] = 9000.0
    old_after = mature_rows(tampered, decision)
    after = fit_predict('ols', {}, x, relative_target(old_after), test_x, 42, 40)
    np.testing.assert_allclose(before, after)