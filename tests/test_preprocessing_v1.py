"""Regression tests for financial labels, date matching, and leakage barriers."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'p2_revision1' / 'src'))
from preprocessing_v1 import (financial_panel, exact_lags, train_zscores,
                              import_metadata, align_filings, FinancialTransform,
                              CONTROLS, tokenize_tfidf)


def annual():
    return pd.DataFrame({'gvkey': ['000001'] * 4, 'cik': ['1'] * 4,
        'fyear': [2017, 2018, 2019, 2020],
        'datadate': pd.to_datetime(['2017-12-31', '2018-12-31', '2019-12-31', '2020-12-31']),
        'at': [100., 200., 400., 500.], 'capx': [5., 10., 30., 20.],
        'ni': [10., 20., 30., 40.], 'sale': [100., 120., 140., 150.],
        'che': [5.] * 4, 'lt': [20.] * 4, 'industry_sic': [2000] * 4})


def filings():
    return pd.DataFrame({'gvkey': ['000001'] * 4, 'cik': ['1'] * 4,
        'accession_key': [str(i).zfill(18) for i in range(1, 5)],
        'filing_year': [2018, 2019, 2020, 2021],
        'filing_date': pd.to_datetime(['2018-03-01', '2019-03-01', '2020-03-01', '2021-03-01']),
        'report_period_end': annual().datadate,
        'filename': ['a.txt'] * 4, 'form_type': ['10-K'] * 4,
        'metadata_ready': [True] * 4, 'metadata_source': ['test fixture'] * 4,
        'period_match_status': ['metadata_available'] * 4})


class PreprocessingTests(unittest.TestCase):
    def test_y_and_average_assets_roa(self):
        d = financial_panel(annual()).set_index('fyear')
        self.assertAlmostEqual(d.loc[2018, 'd_capex_at'], .15 - .10)
        self.assertAlmostEqual(d.loc[2018, 'droa'], 30 / 300 - 20 / 150)
        self.assertTrue(pd.isna(d.loc[2017, 'd_capex_at']))

    def test_missing_year_not_bridged(self):
        d = financial_panel(annual().query('fyear != 2018')).set_index('fyear')
        self.assertTrue(pd.isna(d.loc[2019, 'capex_int']))

    def test_negative_capx_nonpositive_asset_and_missing_classes(self):
        a = annual(); a.loc[1, 'capx'] = -1; a.loc[2, 'at'] = 0
        d = financial_panel(a)
        self.assertTrue(pd.isna(d.loc[1, 'd_capex_at']))
        self.assertTrue(pd.isna(d.loc[2, 'd_capex_at_up']))

    def test_stub_period_invalid(self):
        a = annual(); a.loc[1, 'datadate'] = pd.Timestamp('2018-03-31')
        self.assertTrue(pd.isna(financial_panel(a).loc[1, 'capex_int']))

    def test_ambiguous_financial_key_rejected(self):
        a = annual()
        with self.assertRaises(ValueError):
            financial_panel(pd.concat([a, a.iloc[:1]]))

    def test_lag_is_past_not_future_no_bridge(self):
        d = filings().rename(columns={'filing_year': 'fyear'})
        d['score'] = [10., 20., 30., 40.]
        out = exact_lags(d, ['score'])
        self.assertTrue(pd.isna(out.loc[0, 'lag_score']))
        self.assertEqual(out.loc[1, 'lag_score'], 10.)
        out = exact_lags(d.drop(index=1), ['score']).set_index('fyear')
        self.assertTrue(pd.isna(out.loc[2020, 'lag_score']))

    def test_delayed_prior_filing_not_available(self):
        d = filings().rename(columns={'filing_year': 'fyear'})
        d['score'] = 1.; d.loc[0, 'filing_date'] = pd.Timestamp('2022-01-01')
        self.assertTrue(pd.isna(exact_lags(d, ['score']).loc[1, 'lag_score']))

    def test_training_only_zscore(self):
        d = pd.DataFrame({'split': ['train', 'train', 'test'], 'x': [1., 3., 10000.]})
        z, state = train_zscores(d, ['x'])
        self.assertEqual(state['mean']['x'], 2.)
        self.assertEqual(z.loc[0, 'x'], -1.)

    def test_train_only_imputation_industry_and_y_untouched(self):
        train = pd.DataFrame({c: [1., 3., np.nan] for c in CONTROLS})
        train['split'] = 'train'; train['sic2'] = '20'; train['accession_key'] = ['a', 'b', 'c']
        transform = FinancialTransform().fit(train)
        test = train.iloc[:1].copy(); test.loc[:, CONTROLS] = np.nan
        test['sic2'] = '99'; test['split'] = 'test'; test['d_capex_at'] = np.nan
        out = transform.transform(test)
        self.assertEqual(out.loc[0, 'capex_int_missing'], 1.)
        self.assertEqual(out.loc[0, 'industry_UNKNOWN'], 1.)
        self.assertTrue(pd.isna(test.loc[0, 'd_capex_at']))
        with self.assertRaises(ValueError):
            FinancialTransform().fit(test)

    def test_no_metadata_no_guessed_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            idx = import_metadata(filings(), Path(folder) / 'missing.csv')
        d, audit = align_filings(idx, financial_panel(annual()))
        self.assertTrue(d.empty)
        self.assertEqual(audit.period_match_status.eq('pending_metadata').sum(), 4)

    def test_actual_dates_and_label_purge(self):
        d, audit = align_filings(filings(), financial_panel(annual()))
        d = d.set_index('fyear')
        self.assertEqual(d.loc[2017, 'split'], 'train')
        # 2019-filed report's label only public in 2020, so exclude from training.
        self.assertEqual(d.loc[2018, 'split'], 'excluded')
        self.assertEqual(d.loc[2019, 'split'], 'validation')
        self.assertEqual(audit.period_match_status.eq('matched').sum(), 4)

    def test_conflicting_gvkey_not_overridden_by_cik(self):
        idx = filings(); idx.loc[0, 'gvkey'] = '999999'
        d, audit = align_filings(idx, financial_panel(annual()))
        self.assertEqual(len(d), 3)
        self.assertEqual(audit.loc[0, 'period_match_status'], 'unmatched_or_ambiguous')

    def test_negation_numbers_and_bigrams(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        tokens = tokenize_tfidf('We will not invest 25 million in 2025.')
        self.assertIn('not', tokens); self.assertIn('25', tokens)
        vec = TfidfVectorizer(tokenizer=tokenize_tfidf, token_pattern=None, ngram_range=(1, 2))
        vec.fit(['We will not invest 25 million in 2025.'])
        self.assertIn('not invest', vec.vocabulary_)

    def test_feature_and_cps_pipeline_with_synthetic_clean_shard(self):
        import hashlib
        import json
        from unittest.mock import patch
        import feature_prep_v1
        import preprocessing_v1
        import capex_plan
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'clean'; run = root / 'run'
            (source / 'data' / 'text_clean').mkdir(parents=True)
            (run / 'data').mkdir(parents=True)
            (root / 'p2_revision1').mkdir()
            (root / 'p2_revision1' / 'latest_run.json').write_text(json.dumps({'run_dir': str(run)}))
            (run / 'status.json').write_text(json.dumps({'status': 'completed', 'aligned_rows': 8, 'source_run': str(source)}))
            (run / '运行记录.md').write_text('# Synthetic integration test\n')
            texts = []
            for i in range(8):
                text = ('We will not abandon the capital investment plan to build a factory for $25 million in 2025. '
                        'The weather and market conditions remain uncertain for our customers.')
                if i == 5:
                    text = 'Operating results reflect changing demand and customer retention across regions.'
                texts.append(text)
            d = pd.DataFrame({c: [1., 2., 3., 4., 5., np.nan, 100000., 200000.] for c in CONTROLS})
            d['accession_key'] = [str(i).zfill(18) for i in range(1, 9)]
            d['index'] = range(8); d['gvkey'] = '000001'; d['fyear'] = range(2015, 2023)
            d['filing_date'] = pd.date_range('2016-03-01', periods=8, freq='365D')
            d['split'] = ['train'] * 6 + ['validation', 'test']
            d['sic2'] = ['20'] * 6 + ['99', '99']
            d['main_eligible'] = True; d['secondary_eligible'] = False
            d['clean_sha256'] = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
            d.to_parquet(run / 'data' / 'analytical_sample.parquet', index=False)
            pd.DataFrame({'accession_key': d.accession_key, 'text_natural': texts}).to_parquet(source / 'data' / 'text_clean' / 'part.parquet', index=False)
            with patch.object(feature_prep_v1, 'FORMAL', root), patch.object(preprocessing_v1, 'FORMAL', root):
                result = feature_prep_v1.build()
                self.assertEqual(len(result), 8)
                self.assertTrue((run / 'data' / 'tfidf_validation_keys.csv').exists())
                state = json.loads((run / 'data' / 'feature_transform_state.json').read_text())
                self.assertEqual(len(state['fit_accessions']), 6)
                self.assertNotIn('99', state['financial']['industries'])
                scored = capex_plan.build()
                self.assertTrue(pd.isna(scored.loc[5, 'cps_spec']))
                sheet = pd.read_csv(run / 'data' / 'capex_plan_verification_sample.csv')
                self.assertIn('accession_key', sheet)
                self.assertIn('rule_amount_present', sheet)
                before = (run / 'data' / 'capex_plan_verification_sample.csv').read_bytes()
                with self.assertRaises(RuntimeError):
                    capex_plan.build()
                self.assertEqual(before, (run / 'data' / 'capex_plan_verification_sample.csv').read_bytes())


if __name__ == '__main__':
    unittest.main()
