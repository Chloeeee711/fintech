"""Auditable annual preprocessing. No guessed fiscal years or fitted test data."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

FORMAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORMAL / 'src'))
from cleaning import normalize_accession, normalize_identifier, tokenize_tfidf

CONTROLS = ['capex_int', 'roa', 'sales_growth', 'cash_at', 'lev', 'size']


def financial_panel(annual):
    """Exact fiscal-key joins: an absent year never becomes the preceding row."""
    d = annual.copy()
    if d.duplicated(['gvkey', 'fyear']).any():
        raise ValueError('Ambiguous company/fiscal-year keys; quarantine upstream.')
    for c in ['at', 'ni', 'capx', 'sale', 'che', 'lt']:
        d[c] = pd.to_numeric(d[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    cols = ['at', 'ni', 'capx', 'sale', 'datadate']
    for shift, suffix in [(1, 'prev'), (-1, 'next')]:
        other = d[['gvkey', 'fyear'] + cols].copy()
        other['fyear'] += shift
        other = other.rename(columns={c: c + '_' + suffix for c in cols})
        d = d.merge(other, on=['gvkey', 'fyear'], how='left', validate='one_to_one')
    # A changed fiscal year-end / stub period is not a comparable full year.
    d['days_prev'] = (d.datadate - d.datadate_prev).dt.days
    d['days_next'] = (d.datadate_next - d.datadate).dt.days
    prior = d.days_prev.between(330, 400)
    future = d.days_next.between(330, 400)
    d['capex_int'] = (d.capx / d.at_prev).where(prior & d.at_prev.gt(0) & d.capx.ge(0))
    d['capex_int_fwd'] = (d.capx_next / d['at']).where(future & d['at'].gt(0) & d.capx_next.ge(0))
    d['d_capex_at'] = d.capex_int_fwd - d.capex_int
    d['roa'] = (d.ni / ((d['at'] + d.at_prev) / 2)).where(prior & d['at'].gt(0) & d.at_prev.gt(0))
    d['roa_fwd'] = (d.ni_next / ((d.at_next + d['at']) / 2)).where(future & d['at'].gt(0) & d.at_next.gt(0))
    d['droa'] = d.roa_fwd - d.roa
    d['sales_growth'] = (d.sale / d.sale_prev - 1).where(prior & d.sale_prev.gt(0) & d.sale.ge(0))
    d['cash_at'] = (d.che / d['at']).where(d['at'].gt(0) & d.che.ge(0))
    d['lev'] = (d['lt'] / d['at']).where(d['at'].gt(0) & d['lt'].ge(0))
    d['size'] = np.log(d['at'].where(d['at'].gt(0)))
    d['sic2'] = (pd.to_numeric(d.industry_sic, errors='coerce') // 100).astype('Int64').astype('string')
    d['main_y_valid'] = d.d_capex_at.notna()
    d['secondary_y_valid'] = d.droa.notna()
    # Missing Y stays missing, including its directional classification.
    for target in ['d_capex_at', 'droa']:
        d[target + '_up'] = d[target].gt(0).astype('Int64').where(d[target].notna())
    return d.replace([np.inf, -np.inf], np.nan)


def import_metadata(index, path):
    """Only externally documented, accession-specific dates can unlock matching."""
    d = index.drop(columns=['filing_date', 'report_period_end', 'form_type',
                            'metadata_source', 'metadata_ready', 'period_match_status'], errors='ignore').copy()
    cols = ['accession_key', 'filing_date', 'report_period_end', 'form_type', 'metadata_source']
    if Path(path).exists():
        m = pd.read_csv(path, dtype=str)
        if not set(cols).issubset(m.columns):
            raise ValueError('Metadata requires: ' + ', '.join(cols))
        m = m[cols].copy()
        m.accession_key = m.accession_key.map(normalize_accession)
        if m.accession_key.isna().any() or m.accession_key.duplicated().any():
            raise ValueError('Invalid or duplicate metadata accession; resolve explicitly.')
    else:
        m = pd.DataFrame(columns=cols)
    d = d.merge(m, on='accession_key', how='left', validate='one_to_one')
    for col in ['filing_date', 'report_period_end']:
        d[col] = pd.to_datetime(d[col], errors='coerce')
    d['form_type'] = d.form_type.astype('string').str.strip().str.upper()
    d['metadata_ready'] = (d.filing_date.notna() & d.report_period_end.notna()
                           & d.form_type.notna() & d.metadata_source.fillna('').str.strip().ne(''))
    d['period_match_status'] = np.where(d.metadata_ready, 'metadata_available', 'pending_metadata')
    return d


def align_filings(index, panel):
    """Exact report date + verified identifier; CIK fallback only when unique."""
    d = index.copy()
    if d.accession_key.duplicated().any():
        raise ValueError('Accession is not unique.')
    valid = (d.metadata_ready & d.form_type.eq('10-K')
             & d.filing_date.ge(d.report_period_end)
             & d.filing_date.dt.year.eq(d.filing_year)).fillna(False)
    d.loc[d.metadata_ready, 'period_match_status'] = 'invalid_dates_or_not_original_10K'
    p = panel.copy()
    p['cik_n'] = p.cik.map(normalize_identifier)
    by_gv = p.set_index(['gvkey', 'datadate'], drop=False)
    cik_groups = {k: g for k, g in p.dropna(subset=['cik_n']).groupby(['cik_n', 'datadate'])}
    records = []
    for row in d.loc[valid].itertuples():
        gv = normalize_identifier(row.gvkey, 6)
        cik = normalize_identifier(row.cik)
        key = (gv, row.report_period_end)
        hit = None
        if gv is not None:
            if key in by_gv.index:
                hit = by_gv.loc[key]
                if isinstance(hit, pd.DataFrame):
                    hit = None
                elif cik is not None and hit.cik_n != cik:
                    hit = None
            # Do not silently override a supplied but conflicting GVKEY.
        elif cik is not None:
            group = cik_groups.get((cik, row.report_period_end))
            if group is not None and len(group) == 1:
                hit = group.iloc[0]
        if hit is None:
            d.loc[d.accession_key.eq(row.accession_key), 'period_match_status'] = 'unmatched_or_ambiguous'
            continue
        record = hit.to_dict()
        record.update(accession_key=row.accession_key, filing_date=row.filing_date,
                      report_period_end=row.report_period_end, file_year=int(row.filing_year),
                      filename=row.filename, metadata_source=row.metadata_source)
        records.append(record)
        d.loc[d.accession_key.eq(row.accession_key), 'period_match_status'] = 'matched'
    if not records:
        return pd.DataFrame(), d
    out = pd.DataFrame(records).sort_values(['filing_date', 'accession_key'])
    duplicate = out.duplicated(['gvkey', 'fyear'], keep='first')
    d.loc[d.accession_key.isin(out.loc[duplicate, 'accession_key']), 'period_match_status'] = 'later_duplicate_10K'
    out = out.loc[~duplicate].copy()
    # Label availability is the first original 10-K for the *next* fiscal year.
    labels = out[['gvkey', 'fyear', 'filing_date']].copy()
    labels.fyear -= 1
    labels = labels.rename(columns={'filing_date': 'label_available_date'})
    out = out.merge(labels, on=['gvkey', 'fyear'], how='left', validate='one_to_one')
    out['split'] = 'excluded'
    # Train labels known before validation begins; validation labels before test.
    out.loc[out.file_year.between(2016, 2019) & out.label_available_date.lt(pd.Timestamp('2020-01-01')), 'split'] = 'train'
    out.loc[out.file_year.eq(2020) & out.label_available_date.lt(pd.Timestamp('2022-01-01')), 'split'] = 'validation'
    out.loc[out.file_year.between(2022, 2023), 'split'] = 'test'
    out['main_eligible'] = out.main_y_valid & out.label_available_date.notna() & out.label_available_date.gt(out.filing_date)
    out['secondary_eligible'] = out.secondary_y_valid & out.label_available_date.notna() & out.label_available_date.gt(out.filing_date)
    return out.reset_index(drop=True), d


def exact_lags(frame, columns):
    """Only t-1 from the same GVKEY, and only if it was already published."""
    if frame.duplicated(['gvkey', 'fyear']).any():
        raise ValueError('Resolve original filings before lagging.')
    prev = frame[['gvkey', 'fyear', 'filing_date'] + columns].copy()
    prev.fyear += 1
    prev = prev.rename(columns={c: 'lag_' + c for c in columns + ['filing_date']})
    out = frame.merge(prev, on=['gvkey', 'fyear'], how='left', validate='one_to_one', sort=False)
    unavailable = out.lag_filing_date.ge(out.filing_date) | out.lag_filing_date.isna()
    out.loc[unavailable, ['lag_' + c for c in columns]] = np.nan
    return out.drop(columns='lag_filing_date')


def train_zscores(frame, columns):
    tr = frame.loc[frame.split.eq('train'), columns]
    if tr.empty:
        raise ValueError('No training rows; cannot fit scaling.')
    mean = tr.mean().fillna(0)
    std = tr.std(ddof=0).replace(0, 1).fillna(1)
    z = (frame[columns] - mean) / std
    return z, {'mean': mean.to_dict(), 'std': std.to_dict(), 'fit_rows': len(tr),
               'all_missing_columns': tr.columns[tr.isna().all()].tolist()}


class FinancialTransform:
    """Train-only 1/99% X winsorisation, median fill, missing flags, scaling."""
    def fit(self, train):
        if train.empty or not train.split.eq('train').all():
            raise ValueError('fit() accepts a nonempty training-only table.')
        x = train[CONTROLS].replace([np.inf, -np.inf], np.nan)
        self.lower, self.upper = x.quantile(.01), x.quantile(.99)
        x = x.clip(self.lower, self.upper, axis=1)
        self.median = x.median().fillna(0)  # all-missing training field -> zero + flag
        filled = x.fillna(self.median)
        self.mean, self.std = filled.mean(), filled.std(ddof=0).replace(0, 1).fillna(1)
        self.industries = sorted(train.sic2.dropna().astype(str).unique())
        self.fit_accessions = train.accession_key.astype(str).tolist()
        self.all_missing_columns = x.columns[x.isna().all()].tolist()
        return self

    def transform(self, data):
        x = data[CONTROLS].replace([np.inf, -np.inf], np.nan)
        missing = x.isna().astype(float).add_suffix('_missing')
        x = (x.clip(self.lower, self.upper, axis=1).fillna(self.median) - self.mean) / self.std
        codes = data.sic2.astype('string').fillna('UNKNOWN')
        for industry in self.industries:
            x['industry_' + industry] = codes.eq(industry).astype(float)
        x['industry_UNKNOWN'] = (~codes.isin(self.industries)).astype(float)
        return pd.concat([x, missing], axis=1)

    def state(self):
        return {k: (v.to_dict() if isinstance(v, pd.Series) else v) for k, v in vars(self).items()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
