"""Offline software tests only. These are NOT empirical financial results."""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from finance.sec import validate_user_agent, SecConfigurationError, asof_facts
from finance.model import ModelSettings, net_return, portfolio_weights, prepare_features
from finance.features import monthly_schedule


class TestFundamentals(unittest.TestCase):
    def test_sec_agent_requires_contact(self):
        self.assertEqual(validate_user_agent('University Research person@real-domain.org'),
                         'University Research person@real-domain.org')
        with self.assertRaises(SecConfigurationError):
            validate_user_agent('my-script')

    def test_asof_sec_uses_only_prior_filing(self):
        base = {'ticker': 'XYZ', 'taxonomy': 'us-gaap', 'concept': 'Assets',
                'unit': 'USD', 'start': pd.NaT,
                'end': pd.Timestamp('2019-12-31'), 'form': '10-K'}
        facts = pd.DataFrame([
            {**base, 'value': 100.0, 'filed': pd.Timestamp('2020-02-01'),
             'available_at': pd.Timestamp('2020-02-02'), 'accession': '0000000001-20-000001'},
            {**base, 'value': 130.0, 'filed': pd.Timestamp('2021-02-01'),
             'available_at': pd.Timestamp('2021-02-02'), 'accession': '0000000001-21-000001'},
        ])
        early = asof_facts(facts, pd.Timestamp('2020-07-01'))
        late = asof_facts(facts, pd.Timestamp('2021-07-01'))
        self.assertEqual(early.iloc[0]['value'], 100)
        self.assertEqual(late.iloc[0]['value'], 130)

    def test_portfolio_weights_choose_global_top(self):
        data = pd.DataFrame({'sector': ['A','A','A','A','B','B','B','B'],
                             'ticker': [f'T{i}' for i in range(8)],
                             'predicted_excess': [0.9,0.6,0.3,0.1,0.8,0.5,0.2,0.0]})
        weights = portfolio_weights(data, 0.25)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertEqual(int((weights > 0).sum()), 5)
        self.assertEqual(set(data.loc[weights.gt(0), "ticker"]), {"T0", "T1", "T2", "T4", "T5"})

    def test_cost_reduces_return(self):
        self.assertLess(net_return(0.10, 5), 0.10)
        self.assertAlmostEqual(net_return(0.10, 0), 0.10)

    def test_min_sector_constraint_is_optional(self):
        panel = pd.DataFrame({
            'date': pd.to_datetime(['2022-01-31']*4),
            'ticker': ['A','B','C','D'], 'sector':['Tech','Tech','Tech','Energy'],
            'momentum_12_1': [0.1,0.2,0.4,0.5],
            'book_to_market': [0.3,0.2,0.1,0.4],
            'operating_profitability':[0.2,0.1,0.4,0.3],
            'asset_growth': [0.2,0.4,0.1,0.3],
            'fcf_yield': [0.05,0.03,0.06,0.04],
            'available_at': pd.to_datetime(['2022-01-31']*4),
        })
        features = ('momentum_12_1','book_to_market','operating_profitability','asset_growth','fcf_yield')
        eligible, normalized, excluded = prepare_features(panel, features, min_sector_stocks=1)
        self.assertEqual(len(eligible), 4)
        self.assertTrue(excluded.empty)
        self.assertAlmostEqual(normalized['momentum_12_1'].mean(), 0.0)

    def test_schedule_delays_entry(self):
        dates = pd.to_datetime(['2020-01-31','2020-02-28','2020-03-31'])
        out = monthly_schedule(pd.DataFrame({'date': dates}))
        self.assertEqual(out.loc[0,'date'], dates[0])
        self.assertEqual(out.loc[0,'entry_date'], dates[1])
        self.assertTrue(pd.isna(out.loc[0,'label_end_6m']))


if __name__ == '__main__':
    unittest.main()
