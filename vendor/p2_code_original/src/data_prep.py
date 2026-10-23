"""
data_prep.py — Build the analytical sample.

Merges the SEC 10-K Item 7 index (CIK / filing year / filename) with the
WRDS Compustat annual Fundamentals file, aligns the fiscal year, and derives:
  * forward 1-year and 2-year stock returns (prcc_f based, no CRSP needed)
  * Fama–French-style control variables (size, book-to-market, momentum,
    ROA, leverage, investment, firm age, industry)

Merge key: CIK normalised by stripping leading zeros (CIK is complete in the
SEC index; gvkey is partially missing). Fiscal-year alignment: a 10-K filed in
calendar year Y most often covers fiscal year Y-1 (Apr–Dec fiscal year-ends);
firms with Jan–Mar fiscal year-ends are captured by a second pass matching
fyear = Y. Forward prices are looked up from the same firm's next fiscal year.

Output: data/analytical_sample.parquet
"""
import os, time, numpy as np, pandas as pd
import config

FIN = ['prcc_f','at','ceq','lt','sale','ni','mkvalt','csho','capx']

def _load_compustat():
    cols = ['gvkey','cik','datadate','fyear','fyr','sic','ipodate'] + FIN + \
           ['indfmt','datafmt','consol']
    parts = []
    t0 = time.time()
    for ch in pd.read_stata(config.COMPUSTAT_ANNUAL_ZIP, columns=cols, chunksize=100000):
        ch = ch[(ch['indfmt'] == 'INDL') & (ch['datafmt'] == 'STD') & (ch['consol'] == 'C')]
        parts.append(ch)
    comp = pd.concat(parts, ignore_index=True)
    print(f"  [comp] loaded {len(comp):,} rows in {time.time()-t0:.1f}s")
    comp['cik_n'] = comp['cik'].astype(str).str.lstrip('0')
    comp['gvkey'] = comp['gvkey'].astype(str).str.zfill(6)
    comp['fyear'] = comp['fyear'].astype(int)
    for c in FIN:
        comp[c] = pd.to_numeric(comp[c], errors='coerce')
    comp['sic2'] = comp['sic'].astype(str).str.zfill(4).str[:2]
    comp['datadate'] = pd.to_datetime(comp['datadate'], errors='coerce')
    # ROA only where total assets are positive (at<=0 -> Inf, invalid).
    comp['roa'] = np.where(comp['at'] > 0, comp['ni'] / comp['at'], np.nan)
    # one row per (cik_n, fyear): keep latest datadate
    comp = comp.sort_values('datadate').drop_duplicates(['cik_n','fyear'], keep='last')
    return comp

def _load_sec_index():
    idx = pd.read_csv(config.SEC_INDEX,
                      usecols=['cik','gvkey','year','filename','ipodate'])
    idx['cik_n'] = idx['cik'].astype(str).str.lstrip('0')
    idx['gvkey'] = idx['gvkey'].apply(lambda x: f"{int(x):06d}" if pd.notna(x) else np.nan)
    idx['file_year'] = idx['year'].astype(int)
    idx['text_path'] = idx['file_year'].astype(str) + '/' + idx['filename']
    return idx

def build():
    t0 = time.time()
    print("Loading Compustat annual …")
    comp = _load_compustat()
    print("Loading SEC 10-K index …")
    idx = _load_sec_index()
    print(f"  [sec] {len(idx):,} filings, {idx['cik_n'].nunique():,} CIKs")

    # price + roa look-ups: (cik_n, fyear) -> value
    price = comp.dropna(subset=['prcc_f']).set_index(['cik_n','fyear'])['prcc_f'].to_dict()
    roa_map = comp.dropna(subset=['roa']).set_index(['cik_n','fyear'])['roa'].to_dict()
    at_map  = comp.dropna(subset=['at']).set_index(['cik_n','fyear'])['at'].to_dict()
    capx_map= comp.dropna(subset=['capx']).set_index(['cik_n','fyear'])['capx'].to_dict()
    def fwd(cik, fy, k, m):
        return m.get((cik, fy + k), np.nan)

    # ---- vectorised two-pass fiscal-year alignment ----
    comp_v = comp[['cik_n','fyear'] + FIN + ['sic2','fyr','gvkey','datadate','roa']].copy()
    idx['fy1'] = idx['file_year'] - 1
    m1 = idx.merge(comp_v, left_on=['cik_n','fy1'], right_on=['cik_n','fyear'], how='left')
    unmatched = m1['fyear'].isna()
    m2 = idx[unmatched].merge(comp_v, left_on=['cik_n','file_year'],
                              right_on=['cik_n','fyear'], how='left')
    df = pd.concat([m1[~unmatched], m2], ignore_index=True)
    print(f"  [merge] matched {df['fyear'].notna().sum():,} of {len(idx):,} filings")

    df = df[df['fyear'].notna()].copy()
    df['fyear'] = df['fyear'].astype(int)
    df = df.rename(columns={'ipodate':'ipo_sec'})

    # forward / past prices
    cik = df['cik_n'].values; fy = df['fyear'].values
    df['ret_1y'] = [ (fwd(c, y, 1, price)/fwd(c,y,0,price)-1) if (fwd(c,y,0,price)>0 and pd.notna(fwd(c,y,1,price))) else np.nan
                    for c,y in zip(cik,fy) ]
    df['ret_2y'] = [ (fwd(c, y, 2, price)/fwd(c,y,0,price)-1) if (fwd(c,y,0,price)>0 and pd.notna(fwd(c,y,2,price))) else np.nan
                    for c,y in zip(cik,fy) ]
    df['mom']    = [ (fwd(c, y, 0, price)/fwd(c,y,-1,price)-1) if (fwd(c,y,-1,price)>0 and pd.notna(fwd(c,y,0,price))) else np.nan
                    for c,y in zip(cik,fy) ]
    # investment: ΔAT / AT_{-1}  (lagged TOTAL ASSETS, not price)
    at0 = df['at'].values
    atm1= np.array([fwd(c,y,-1,at_map) for c,y in zip(cik,fy)])
    df['invest'] = np.where((atm1>0)&pd.notna(atm1), (at0-atm1)/atm1, np.nan)
    # current ROA (requires ni & at present) — guard at<=0 -> NaN, then dropped.
    df['roa']    = np.where(df['at'] > 0, df['ni'] / df['at'], np.nan)
    # future profitability (primary outcome): ROA at fyear+1 and change in ROA
    df['roa_fwd'] = np.array([fwd(c,y,1,roa_map) for c,y in zip(cik,fy)])
    df['roa_fwd'] = df['roa_fwd'].where(np.isfinite(df['roa_fwd']))
    df['droa']    = df['roa_fwd'] - df['roa']

    # ---- secondary outcome: capital expenditure (CAPEX/AT) ----
    # capex_int  = current-period CAPEX / AT (investment intensity)
    capx_t = np.array([fwd(c,y,0,capx_map) for c,y in zip(cik,fy)])
    at_t   = df['at'].values
    df['capex_int'] = np.where((at_t>0)&pd.notna(capx_t), capx_t/at_t, np.nan)
    # d_capex    = next-year CAPEX minus current CAPEX (real action vs plan, H5/H6)
    capx_tp1 = np.array([fwd(c,y,1,capx_map) for c,y in zip(cik,fy)])
    df['d_capex'] = capx_tp1 - capx_t
    # d_capex_at = change in CAPEX normalised by current AT (standardised, H5)
    df['d_capex_at'] = np.where(at_t>0, df['d_capex']/at_t, np.nan)

    # ---- controls ----
    df['size']   = np.where(df['at']  > 0, np.log(df['at']),  np.nan)
    df['bm']     = np.where(df['mkvalt'] > 0, df['ceq'] / df['mkvalt'], np.nan)
    df['lev']    = np.where(df['at'] > 0, df['lt'] / df['at'], np.nan)
    df['log_mkvalt'] = np.where(df['mkvalt'] > 0, np.log(df['mkvalt']), np.nan)
    d = pd.to_datetime(df['ipo_sec'], errors='coerce')
    df['age'] = df['fyear'] - d.dt.year
    df['age'] = df['age'].where((df['age'] > 0) & (df['age'] < 120), np.nan)

    # ---- winsorise continuous variables to economically sensible ranges ----
    # ROA = NI/AT explodes to ±thousands for firms with near-zero total assets;
    # clip to ±100% of assets. Other controls are clipped to standard ranges.
    df['roa']     = df['roa'].clip(-1, 1)
    df['roa_fwd'] = df['roa_fwd'].clip(-1, 1)
    df['droa']    = df['roa_fwd'] - df['roa']
    df['bm']      = df['bm'].clip(-5, 10)
    df['lev']     = df['lev'].clip(0, 3)
    df['mom']     = df['mom'].clip(-0.9, 5)
    df['invest']  = df['invest'].clip(-1, 5)
    df['capex_int'] = df['capex_int'].clip(-1, 1)
    df['d_capex_at'] = df['d_capex_at'].clip(-2, 2)

    # ---- sample window & cleaning ----
    df = df[(df['file_year'] >= config.FILE_YEAR_MIN) &
            (df['file_year'] <= config.FILE_YEAR_MAX)]
    df = df.dropna(subset=['ret_1y','roa_fwd','roa','at','prcc_f','size'])
    df = df[(df['ret_1y'] > -0.99) & (df['ret_1y'] < 5)]
    df['split'] = np.where(df['file_year'] >= config.TEST_FILE_YEAR_MIN, 'test', 'train')
    df = df.reset_index(drop=True)

    out = os.path.join(config.DATA, 'analytical_sample.parquet')
    df.to_parquet(out, index=False)
    print(f"  [save] {out}")
    print(f"  rows={len(df):,}  train={ (df.split=='train').sum():,}  "
          f"test={ (df.split=='test').sum():,}")
    print(f"  file-year {df['file_year'].min()}–{df['file_year'].max()}  "
          f"unique firms(cik)={df['cik_n'].nunique():,}")
    print(f"  mean ret_1y={df['ret_1y'].mean():.4f}  std={df['ret_1y'].std():.4f}")
    return df

if __name__ == '__main__':
    build()
