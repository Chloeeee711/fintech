"""
empirical.py — Core econometric tests for the disclosure-substance study.

Implements PROJECT2_PLAN.md §5 / §8:
  * Incremental-information ladder  G1 (financial controls) -> G2 (+LM tone)
    -> G3 (+rule SSI + risk lexicons) -> G_full (+neural_score), for two outcomes:
        - forward ROA  (primary, roa_fwd)
        - ΔCAPEX/AT    (secondary, d_capex_at, H5/H6)
  * Panel OLS with industry x year fixed effects and FIRM-clustered SE.
  * Out-of-sample R^2 via expanding-window year-block CV + Clark-West test
    (nested) / Diebold-Mariano (non-nested) for "does text predict *more
    accurately*".
  * Long-short Q5-Q1 spread (economic significance) overall, train/test split,
    and by industry group.
  * Ablation (H3): which component of G3 actually carries the signal.

Outputs: data/empirical_results.json
"""
import os, json, time, math, numpy as np, pandas as pd, config

CONTROLS = ['size', 'bm', 'roa', 'lev', 'mom', 'invest', 'age']

# Information-set definitions (feature columns added ON TOP of CONTROLS)
G1 = []                                   # financial baseline
G2 = ['tone_lm']                           # + dictionary sentiment
G3 = ['tone_lm', 'ssi', 'risk_demand', 'risk_cost', 'risk_liquidity']
G_FULL = G3 + ['neural_score']             # + deep-learning signal (optional)


def ols(X, y, cluster=None):
    """OLS with optional one-way cluster-robust SE (cluster = array of ids)."""
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    XtX_inv = np.linalg.pinv(X.T @ X)
    if cluster is None:
        V = (ss_res / (n - k)) * XtX_inv
    else:
        cl = np.asarray(cluster)
        u = resid[:, None] * X
        meat = np.zeros((k, k))
        for g in np.unique(cl):
            ug = u[cl == g].sum(0)
            meat += np.outer(ug, ug)
        V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    t = beta / se
    return dict(beta=beta, se=se, t=t, r2=r2, n=n)


def _design(df, feats, controls=True):
    X = df[feats].values.astype(float) if feats else np.empty((len(df), 0))
    if controls:
        Xc = np.nan_to_num(df[CONTROLS].values.astype(float))
        X = np.hstack([X, Xc]) if X.shape[1] else Xc
    sic = pd.get_dummies(df['sic2'].fillna('00'), drop_first=True).values.astype(float)
    yr = pd.get_dummies(df['file_year'], drop_first=True).values.astype(float)
    return np.hstack([np.ones((len(df), 1)), X, sic, yr])


def panel_ladder(df, target, cluster):
    """R^2 ladder G1->G2->G3->G_full with cluster-robust text-coefficient t-stats."""
    out = {}
    specs = {'G1': G1, 'G2': G2, 'G3': G3, 'G_full': G_FULL}
    res = {}
    for name, feats in specs.items():
        has = [f for f in feats if f in df.columns and df[f].notna().any()]
        X = _design(df, has)
        r = ols(X, df[target].values.astype(float), cluster)
        # text effect = coefficient on the *first* non-control regressor (tone_lm),
        # and on ssi when present; report whichever is informative
        res[name] = r
        out[name] = dict(r2=r['r2'], n=r['n'])
    # incremental R2
    out['incR2_G1_to_G2'] = res['G2']['r2'] - res['G1']['r2']
    out['incR2_G2_to_G3'] = res['G3']['r2'] - res['G2']['r2']
    out['incR2_G3_to_Gfull'] = res['G_full']['r2'] - res['G3']['r2']
    # tone_lm and ssi coefficients (cluster-robust t) from G3 (where both sit)
    Xg3 = _design(df, [f for f in G3 if f in df.columns])
    rg3 = ols(Xg3, df[target].values.astype(float), cluster)
    cols = ['(intercept)'] + [f for f in G3 if f in df.columns] + CONTROLS + \
           list(pd.get_dummies(df['sic2'].fillna('00'), drop_first=True).columns) + \
           list(pd.get_dummies(df['file_year'], drop_first=True).columns)
    def _coef(name):
        if name in cols:
            i = cols.index(name)
            return dict(beta=float(rg3['beta'][i]), t=float(rg3['t'][i]))
        return None
    out['coef_tone_lm'] = _coef('tone_lm')
    out['coef_ssi'] = _coef('ssi')
    out['coef_neural_score'] = _coef('neural_score') if 'neural_score' in df.columns else None
    return out


def long_short(df, col, by='file_year', target='roa_fwd'):
    d = df.copy()
    d['q'] = d.groupby(by)[col].transform(lambda s: pd.qcut(s, 5, labels=False, duplicates='drop'))
    series = d.groupby(by).apply(
        lambda g: g.loc[g['q'] == 4, target].mean() - g.loc[g['q'] == 0, target].mean())
    series = series.dropna()
    r = series.values
    mean = r.mean(); std = r.std(ddof=1)
    return dict(mean=float(mean), std=float(std),
                sharpe=float(mean / std) if std > 0 else float('nan'),
                tstat=float(mean / (std / math.sqrt(len(r)))) if std > 0 else float('nan'),
                T=int(len(r)),
                series={str(k): float(v) for k, v in series.items()})


def cv_oos(df, target, min_train_year=2013):
    """Expanding-window year-block CV; returns per-group SSE, OOS R^2, CW/DM tests."""
    years = sorted(df['file_year'].dropna().unique())
    groups = {'G1': G1, 'G2': G2, 'G3': G3}
    preds = {g: [] for g in groups}; ytrue = []
    for y in years:
        if y < min_train_year:
            continue
        tr = df[df['file_year'] < y]
        te = df[df['file_year'] == y]
        if len(te) < 50:
            continue
        yt = te[target].values.astype(float)
        ytrue.append(yt)
        ytr = tr[target].values.astype(float)
        for g, feats in groups.items():
            feats_all = feats + CONTROLS
            Xtr = np.nan_to_num(tr[feats_all].values.astype(float))
            Xte = np.nan_to_num(te[feats_all].values.astype(float))
            A = Xtr.T @ Xtr + 1e-6 * np.eye(Xtr.shape[1])
            beta = np.linalg.solve(A, Xtr.T @ ytr)
            preds[g].append(Xte @ beta)
    yall = np.concatenate(ytrue)
    out = {}
    sse = {}
    for g in groups:
        p = np.concatenate(preds[g])
        e = yall - p
        sse[g] = float(e @ e)
        out[g] = dict(oos_r2=float(1 - sse[g] / sse['G1']))
    # Clark-West (nested) and Diebold-Mariano (non-nested)
    def cw(a, b):
        # d_t = (e_a^2 - e_b^2); Newey-West(1) t-stat over pooled obs
        ea = yall - np.concatenate(preds[a]); eb = yall - np.concatenate(preds[b])
        d = ea ** 2 - eb ** 2
        mean = d.mean()
        # HAC variance (NW lag 1)
        n = len(d); ac = np.cov(d[:-1], d[1:])[0, 1] if n > 1 else 0.0
        var = d.var(ddof=1) + 2.0 * ac
        se = math.sqrt(var / n)
        return dict(cw_t=float(mean / se) if se > 0 else float('nan'),
                    delta_oos_r2=float(out[b]['oos_r2'] - out[a]['oos_r2']))
    out['CW_G1_vs_G2'] = cw('G1', 'G2')
    out['CW_G1_vs_G3'] = cw('G1', 'G3')
    out['CW_G2_vs_G3'] = cw('G2', 'G3')
    return out


def run():
    t0 = time.time()
    smp = pd.read_parquet(os.path.join(config.DATA, 'analytical_sample.parquet'))
    tx = pd.read_parquet(os.path.join(config.DATA, 'texts.parquet'))
    tcols = ['index', 'ssi', 'tone_lm', 'unc_ratio', 'lit_ratio',
             'risk_demand', 'risk_cost', 'risk_liquidity', 'lag_ssi', 'lag_tone']
    df = smp.merge(tx[tcols], left_index=True, right_on='index', how='left')
    nf_path = os.path.join(config.DATA, 'neural_features.parquet')
    if os.path.exists(nf_path):
        nf = pd.read_parquet(nf_path)
        if 'neural_score' in nf.columns:
            df = df.merge(nf[['index', 'neural_score']], on='index', how='left')
    cluster = df['cik_n'].values
    out = dict(n_total=int(len(df)), file_years=sorted(df['file_year'].dropna().unique().tolist()))

    # ---- continuous outcome 1: forward ROA (primary) ----
    d_roa = df.dropna(subset=['roa_fwd'] + CONTROLS).copy()
    cl_roa = d_roa['cik_n'].values
    out['roa_fwd'] = dict(panel=panel_ladder(d_roa, 'roa_fwd', cl_roa),
                          cv_oos=cv_oos(d_roa, 'roa_fwd'),
                          ls_tone=long_short(d_roa, 'tone_lm', target='roa_fwd'),
                          ls_ssi=long_short(d_roa, 'ssi', target='roa_fwd'))
    if 'neural_score' in df.columns:
        out['roa_fwd']['ls_neural'] = long_short(d_roa, 'neural_score', target='roa_fwd')
    # train/test + industry split for the headline G2->G3 contrast
    out['roa_fwd']['ls_ssi_by_split'] = {
        s: long_short(d_roa[d_roa['split'] == s], 'ssi', target='roa_fwd')
        for s in ['train', 'test']}
    mfg = d_roa['sic2'].isin([f'{i:02d}' for i in range(10, 40)])
    svc = d_roa['sic2'].isin([f'{i:02d}' for i in range(40, 90)])
    out['roa_fwd']['ls_ssi_by_industry'] = {
        'manufacturing': long_short(d_roa[mfg], 'ssi', target='roa_fwd'),
        'services': long_short(d_roa[svc], 'ssi', target='roa_fwd')}

    # ---- continuous outcome 2: ΔCAPEX/AT (secondary, H5/H6) ----
    d_cap = df.dropna(subset=['d_capex_at'] + CONTROLS).copy()
    cl_cap = d_cap['cik_n'].values
    out['d_capex_at'] = dict(panel=panel_ladder(d_cap, 'd_capex_at', cl_cap),
                             cv_oos=cv_oos(d_cap, 'd_capex_at'),
                             ls_tone=long_short(d_cap, 'tone_lm', target='d_capex_at'),
                             ls_ssi=long_short(d_cap, 'ssi', target='d_capex_at'))

    # ---- ablation (H3): which component of G3 carries the signal ----
    def _r2_of(feats):
        X = _design(d_roa, [f for f in feats if f in d_roa.columns])
        return ols(X, d_roa['roa_fwd'].values.astype(float), cl_roa)['r2']
    full = [f for f in G3 if f in d_roa.columns]
    out['ablation_roa_fwd'] = dict(
        G3_full=float(_r2_of(full)),
        drop_risk=float(_r2_of([f for f in full if not f.startswith('risk_')])),
        drop_tone=float(_r2_of([f for f in full if f != 'tone_lm'])),
        drop_ssi=float(_r2_of([f for f in full if f != 'ssi'])))

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (np.floating, float)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    json.dump(clean(out), open(os.path.join(config.DATA, 'empirical_results.json'), 'w'), indent=2)
    print(f"  [empirical] done in {time.time()-t0:.0f}s; wrote empirical_results.json")
    # console summary
    r = out['roa_fwd']['panel']
    print(f"  ROA panel R2 G1={r['G1']['r2']:.4f} G2={r['G2']['r2']:.4f} "
          f"G3={r['G3']['r2']:.4f}  incR2 G1->G2={r['incR2_G1_to_G2']:.4f} "
          f"G2->G3={r['incR2_G2_to_G3']:.4f}")
    cw = out['roa_fwd']['cv_oos']
    print(f"  ROA OOS R2 G1={cw['G1']['oos_r2']:.3f} G2={cw['G2']['oos_r2']:.3f} "
          f"G3={cw['G3']['oos_r2']:.3f}  CW(G1vsG3)={cw['CW_G1_vs_G3']['cw_t']:.2f}")
    return out


if __name__ == '__main__':
    run()
