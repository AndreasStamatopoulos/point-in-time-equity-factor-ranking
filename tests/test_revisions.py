import unittest
import pandas as pd
from finance.revisions import revision_at

class RevisionTests(unittest.TestCase):
    def test_same_fiscal_period_and_prior_asof(self):
        data = pd.DataFrame({
            "ticker":["ABC","ABC","ABC"],
            "observed_at": pd.to_datetime(["2022-01-20","2022-04-20","2022-05-05"]),
            "target_fiscal_end": pd.to_datetime(["2022-12-31"]*3),
            "eps_consensus":[2.0,2.2,99.0],"source":["vendor"]*3})
        value = revision_at(data, "ABC", pd.Timestamp("2022-04-30"))
        self.assertAlmostEqual(value["earnings_revisions"],0.1)
        self.assertEqual(value["latest_observed_at"],"2022-04-20")
    def test_future_fiscal_end(self):
        d = pd.DataFrame({"ticker":["ABC","ABC"],
            "observed_at": pd.to_datetime(["2022-01-10","2022-04-10"]),
            "target_fiscal_end": pd.to_datetime(["2022-03-31"]*2),
            "eps_consensus":[1.0,2.0],"source":["vendor"]*2})
        self.assertIsNone(revision_at(d,"ABC",pd.Timestamp("2022-04-30")))
