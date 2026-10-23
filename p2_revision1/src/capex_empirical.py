"""
capex_empirical.py — Focused core tests (C1-C4).

Core topic: "Can the specificity and implementation stage of management's
capital-investment plans predict future investment changes (DCAPEX/AT) and
profitability improvement (DROA)?"

Implements PROJECT2_PLAN.md section 2.5:
  C1 (follow-through): plan specificity -> realised investment change
  C2 (value):           plan specificity -> profitability improvement
  C3 (stage gradient):  signal concentrated in later-stage plans
  C4 (empty promise):   high talk + low walk -> DROA penalty
Plus the incremental ladder G1 (financial) -> G2 (+tone) -> G3 (+generic SSI)
-> G4 (+capital-plan CPS) on BOTH outcomes, with expanding-window year-block
CV and Clark-West OOS accuracy tests.

Merge hygiene: result variables come ONLY from analytical_sample; the text
frame contributes ONLY cps_* columns (avoids the _x/_y suffix trap).

Outputs: data/capex_empirical_results.json
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd

import config

CONTROLS = ['size', 'bm', 'roa', 'lev', 'mom', 'invest', 'age']
CPS_COLS = ['cps_composite', 'cps_spec', 'cps_stage_mean', 'cps_late_share',
            'cps_amt_log1p', 'cps_stage_max', 'cps_n_sent',
            'lag_cps_composite', 'lag_cps_spec', 'lag_cps_stage_max']

G1, G2, G3, G4 = [], ['tone_lm'], ['tone_lm', 'ssi'], \
                 ['tone_lm', 'ssi', 'cps_composite', 'cps_stage_mean']


def ols(X, y, cluster=None):
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
    return dict(beta=beta, se=se, t=beta / se, r2=r2, n=n)


def _design(df, feats):
    X = df[feats].values.astype(float) if feats else np.empty((len(df), 0))
    Xc = np.nan_to_num(df[CONTROLS].values.astype(float))
    X = np.hstack([X, Xc]) if X.shape[1] else Xc
    sic = pd.get_dummies(df['sic2'].fillna('00'), drop_first=True).values.astype(float)
    yr = pd.get_dummies(df['file_year'], drop_first=True).values.astype(float)
    return np.hstack([np.ones((len(df), 1)), X, sic, yr])


def _colnames(df, feats):
    return (['(intercept)'] + feats + CONTROLS +
            list(pd.get_dummies(df['sic2'].fillna('00'), drop_first=True).columns) +
            list(pd.get_dummies(df['file_year'], drop_first=True).columns))


def _coef(res, cols, name):
    if name in cols:
        i = cols.index(name)
        return dict(beta=float(res['beta'][i]), t=float(res['t'][i]),
                    se=float(res['se'][i]))
    return None


def ladder(df, target, cluster):
    """G1->G4 R^2 ladder + cluster-robust CPS/stage coefficients (from G4)."""
    out, res = {}, {}
    for name, feats in [('G1', G1), ('G2', G2), ('G3', G3), ('G4', G4)]:
        has = [f for f in feats if f in df.columns and df[f].notna().any()]
        r = ols(_design(df, has), df[target].values.astype(float), cluster)
        res[name] = r
        out[name] = dict(r2=r['r2'], n=r['n'])
    out['incR2_G1_to_G2'] = res['G2']['r2'] - res['G1']['r2']
    out['incR2_G2_to_G3'] = res['G3']['r2'] - res['G2']['r2']
    out['incR2_G3_to_G4'] = res['G4']['r2'] - res['G3']['r2']
    has4 = [f for f in G4 if f in df.columns]
    r4 = ols(_design(df, has4), df[target].values.astype(float), cluster)
    cols4 = _colnames(df, has4)
    out['coef_cps_composite'] = _coef(r4, cols4, 'cps_composite')
    out['coef_cps_stage_mean'] = _coef(r4, cols4, 'cps_stage_mean')
    out['coef_tone_lm'] = _coef(r4, cols4, 'tone_lm')
    out['coef_ssi'] = _coef(r4, cols4, 'ssi')
    return out


def cv_oos(df, target, min_train_year=2013):
    """Expanding-window year-block CV + Clark-West (nested) tests."""
    years = sorted(df['file_year'].dropna().unique())
    groups = {'G1': G1, 'G2': G2, 'G3': G3, 'G4': G4}
    preds = {g: [] for g in groups}; ytrue = []
    for y in years:
        if y < min_train_year:
            continue
        tr, te = df[df['file_year'] < y], df[df['file_year'] == y]
        if len(te) < 50:
            continue
        ytrue.append(te[target].values.astype(float))
        ytr = tr[target].values.astype(float)
        for g, feats in groups.items():
            fa = [f for f in feats if f in df.columns] + CONTROLS
            Xtr = np.nan_to_num(tr[fa].values.astype(float))
            Xte = np.nan_to_num(te[fa].values.astype(float))
            beta = np.linalg.solve(Xtr.T @ Xtr + 1e-6 * np.eye(Xtr.shape[1]),
                                   Xtr.T @ ytr)
            preds[g].append(Xte @ beta)
    yall = np.concatenate(ytrue)
    out, sse = {}, {}
    for g in groups:
        e = yall - np.concatenate(preds[g])
        sse[g] = float(e @ e)
        out[g] = dict(oos_r2=float(1 - sse[g] / sse['G1']))

    def cw(a, b):
        ea = yall - np.concatenate(preds[a]); eb = yall - np.concatenate(preds[b])
        d = ea ** 2 - eb ** 2
        n = len(d)
        ac = np.cov(d[:-1], d[1:])[0, 1] if n > 1 else 0.0
        var = d.var(ddof=1) + 2.0 * ac
        se = math.sqrt(var / n)
        return dict(cw_t=float(d.mean() / se) if se > 0 else float('nan'),
                    delta_oos_r2=float(out[b]['oos_r2'] - out[a]['oos_r2']))
    out['CW_G1_vs_G2'] = cw('G1', 'G2')
    out['CW_G1_vs_G3'] = cw('G1', 'G3')
    out['CW_G1_vs_G4'] = cw('G1', 'G4')
    out['CW_G3_vs_G4'] = cw('G3', 'G4')
    return out


def long_short(df, col, target, by='file_year'):
    """Q5-Q1 spread per year. Rank-based quintiles with method='first' so the
    mass point at zero (non-disclosing firms) still yields five populated
    bins (ties broken by row order, which is outcome-neutral)."""
    d = df.copy()
    d['q'] = d.groupby(by)[col].transform(lambda s: s.rank(pct=True, method='first'))
    d['q'] = np.floor(d['q'] * 5).clip(0, 4).astype(int)
    s = d.groupby(by)[['q', target]].apply(
        lambda g: g.loc[g['q'] == 4, target].mean() - g.loc[g['q'] == 0, target].mean()
    ).dropna()
    r = s.values
    if len(r) == 0:
        return dict(mean=float('nan'), std=float('nan'), sharpe=float('nan'),
                    tstat=float('nan'), T=0, series={})
    mean, std = r.mean(), r.std(ddof=1)
    return dict(mean=float(mean), std=float(std),
                sharpe=float(mean / std) if std > 0 else float('nan'),
                tstat=float(mean / (std / math.sqrt(len(r)))) if std > 0 else float('nan'),
                T=int(len(r)))


def _t(cluster_arr, resid, X):
    """cluster-robust t for a single contrast is handled via ols(); helper
    here just wraps a two-group difference test with firm-clustered SE."""
    pass


def group_diff(d, mask_a, target, cluster):
    """Difference in means of target between mask_a and the rest, with
    firm-clustered SE via OLS on a group dummy + controls + FE."""
    dd = d.copy()
    dd['_ga'] = mask_a.astype(float)
    X = _design(dd, ['_ga'])
    r = ols(X, dd[target].values.astype(float), cluster)
    cols = _colnames(dd, ['_ga'])
    c = _coef(r, cols, '_ga')
    return c


def run():
    t0 = time.time()
    smp = pd.read_parquet(os.path.join(config.DATA, 'analytical_sample.parquet'))
    tx = pd.read_parquet(os.path.join(config.DATA, 'texts.parquet'))
    tcols = (['index', 'tone_lm', 'ssi', 'risk_demand', 'risk_cost',
              'risk_liquidity'] + CPS_COLS)
    df = smp.merge(tx[tcols], left_index=True, right_on='index', how='left')
    if 'droa' not in df.columns:
        df['droa'] = df['roa_fwd'] - df['roa']
    df['d_cps'] = df['cps_composite'] - df['lag_cps_composite']
    cluster = df['cik_n'].values
    out = dict(n_total=int(len(df)),
               file_years=sorted(df['file_year'].dropna().unique().tolist()),
               cps_coverage=float(df['cps_composite'].notna().mean()))

    base = ['droa', 'd_capex_at'] + CONTROLS + CPS_COLS
    d = df.dropna(subset=base).copy()
    cl = d['cik_n'].values
    out['n_complete_case'] = int(len(d))

    # ---------- ladders on both outcomes ----------
    out['droa'] = dict(panel=ladder(d, 'droa', cl),
                       cv_oos=cv_oos(d, 'droa'),
                       ls_cps=long_short(d, 'cps_composite', 'droa'),
                       ls_tone=long_short(d, 'tone_lm', 'droa'),
                       ls_ssi=long_short(d, 'ssi', 'droa'))
    out['d_capex_at'] = dict(panel=ladder(d, 'd_capex_at', cl),
                             cv_oos=cv_oos(d, 'd_capex_at'),
                             ls_cps=long_short(d, 'cps_composite', 'd_capex_at'))

    # ---------- C1: specificity -> realised investment (full + flagged) ----------
    def _simple_reg(dd, x, y):
        X = _design(dd, [x])
        r = ols(X, dd[y].values.astype(float), dd['cik_n'].values)
        return _coef(r, _colnames(dd, [x]), x)

    out['C1'] = dict(
        full=dict(cps_spec_to_dcapex=_simple_reg(d, 'cps_spec', 'd_capex_at'),
                  cps_composite_to_dcapex=_simple_reg(d, 'cps_composite', 'd_capex_at'),
                  dcps_to_dcapex=_simple_reg(d, 'd_cps', 'd_capex_at'),
                  lag_cps_to_dcapex=_simple_reg(d, 'lag_cps_composite', 'd_capex_at')))
    # flagged subsample: among firm-years that actually disclose a capex plan
    d_fl = d[d['cps_composite'].notna() & (d['cps_n_sent'] > 0)].copy()
    out['C1']['flagged'] = dict(
        n=int(len(d_fl)),
        cps_spec_to_dcapex=_simple_reg(d_fl, 'cps_spec', 'd_capex_at'),
        cps_composite_to_dcapex=_simple_reg(d_fl, 'cps_composite', 'd_capex_at'))

    # ---------- C2: specificity -> profitability improvement ----------
    out['C2'] = dict(
        cps_spec_to_droa=_simple_reg(d, 'cps_spec', 'droa'),
        cps_composite_to_droa=_simple_reg(d, 'cps_composite', 'droa'),
        dcps_to_droa=_simple_reg(d, 'd_cps', 'droa'),
        flagged=dict(n=int(len(d_fl)),
                     cps_spec_to_droa=_simple_reg(d_fl, 'cps_spec', 'droa'),
                     cps_composite_to_droa=_simple_reg(d_fl, 'cps_composite', 'droa')))
    mfg = d['sic2'].isin([f'{i:02d}' for i in range(10, 40)])
    out['C2']['by_industry'] = dict(
        manufacturing=_simple_reg(d[mfg], 'cps_composite', 'droa'),
        services=_simple_reg(d[~mfg], 'cps_composite', 'droa'))

    # ---------- discloser vs non-discloser contrast (levels) ----------
    disc = d['cps_n_sent'] > 0
    out['discloser_contrast'] = {
        'n_discloser': int(disc.sum()), 'n_nondiscloser': int((~disc).sum()),
        'droa_discloser': float(d.loc[disc, 'droa'].mean()),
        'droa_nondiscloser': float(d.loc[~disc, 'droa'].mean()),
        'dcapex_discloser': float(d.loc[disc, 'd_capex_at'].mean()),
        'dcapex_nondiscloser': float(d.loc[~disc, 'd_capex_at'].mean())}

    # ---------- C3: stage gradient ----------
    out['C3'] = dict(
        by_stage_group={}, interaction_spec_x_late=None)
    for st in [2, 3, 4, 5]:
        ds = d[d['cps_stage_max'] == st]
        if len(ds) > 100:
            out['C3']['by_stage_group'][str(st)] = dict(
                n=int(len(ds)),
                mean_droa=float(ds['droa'].mean()),
                mean_dcapex=float(ds['d_capex_at'].mean()),
                mean_spec=float(ds['cps_spec'].mean()),
                cps_spec_to_droa=_simple_reg(ds, 'cps_spec', 'droa'),
                cps_spec_to_dcapex=_simple_reg(ds, 'cps_spec', 'd_capex_at'))
    # interaction: spec x late_share
    di = d.copy()
    di['spec_x_late'] = di['cps_spec'] * di['cps_late_share']
    X = _design(di, ['cps_spec', 'spec_x_late'])
    r = ols(X, di['droa'].values.astype(float), di['cik_n'].values)
    cols = _colnames(di, ['cps_spec', 'spec_x_late'])
    out['C3']['interaction_spec_x_late'] = dict(
        cps_spec=_coef(r, cols, 'cps_spec'),
        spec_x_late=_coef(r, cols, 'spec_x_late'))

    # ---------- C4: talk-vs-walk (empty promise) ----------
    d4 = d.copy()
    med_spec = d4['cps_spec'].median()
    med_walk = d4['d_capex_at'].median()
    d4['hi_talk'] = (d4['cps_spec'] > med_spec).astype(int)
    d4['lo_walk'] = (d4['d_capex_at'] <= med_walk).astype(int)
    d4['quad'] = np.where(d4['hi_talk'] & d4['lo_walk'], 'talk_nowalk',
                  np.where(d4['hi_talk'] & ~d4['lo_walk'].astype(bool), 'talk_walk',
                  np.where(~d4['hi_talk'].astype(bool) & d4['lo_walk'].astype(bool),
                           'notalk_nowalk', 'notalk_walk')))
    quads = {}
    for q in ['talk_walk', 'talk_nowalk', 'notalk_walk', 'notalk_nowalk']:
        m = d4['quad'] == q
        quads[q] = dict(n=int(m.sum()),
                        mean_droa=float(d4.loc[m, 'droa'].mean()),
                        mean_dcapex=float(d4.loc[m, 'd_capex_at'].mean()))
    tnw = group_diff(d4, (d4['quad'] == 'talk_nowalk').values, 'droa', d4['cik_n'].values)
    gap = d4['cps_spec'].std() if d4['cps_spec'].std() > 0 else 1.0
    d4['talk_walk_gap'] = ((d4['cps_spec'] - d4['cps_spec'].mean()) / gap -
                           (d4['d_capex_at'] - d4['d_capex_at'].mean()) /
                           (d4['d_capex_at'].std() + 1e-12))
    Xg = _design(d4, ['talk_walk_gap'])
    rg = ols(Xg, d4['droa'].values.astype(float), d4['cik_n'].values)
    out['C4'] = dict(quadrants=quads,
                     talk_nowalk_vs_rest=tnw,
                     gap_coef_on_droa=_coef(rg, _colnames(d4, ['talk_walk_gap']),
                                            'talk_walk_gap'))
    np.savez(os.path.join(config.DATA, 'fig_cps_quadrant.npz'),
             talk=d4['cps_spec'].values.astype(float),
             walk=d4['d_capex_at'].values.astype(float),
             droa=d4['droa'].values.astype(float),
             quad=d4['quad'].values.astype('U16'))

    # ---------- economic significance: 1 SD of cps_composite ----------
    sd = float(d['cps_composite'].std())
    for tgt in ['droa', 'd_capex_at']:
        c = out[tgt]['panel'].get('coef_cps_composite')
        if c:
            c['per_1sd_pp'] = c['beta'] * sd * 100.0

    with open(os.path.join(config.DATA, 'capex_empirical_results.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f"wrote data/capex_empirical_results.json in {time.time()-t0:.0f}s")

    # ---------- console summary ----------
    for tgt in ['droa', 'd_capex_at']:
        p = out[tgt]['panel']
        print(f"\n=== {tgt} ladder ===  n={out['n_complete_case']:,}")
        for g in ['G1', 'G2', 'G3', 'G4']:
            print(f"  {g}: R2={p[g]['r2']:.4f}")
        print(f"  incR2 G2->G3={p['incR2_G2_to_G3']:.4f}  G3->G4={p['incR2_G3_to_G4']:.4f}")
        cw = out[tgt]['cv_oos']
        print(f"  OOS R2: G1={cw['G1']['oos_r2']:.4f} G3={cw['G3']['oos_r2']:.4f} "
              f"G4={cw['G4']['oos_r2']:.4f} | CW G1vsG4 t={cw['CW_G1_vs_G4']['cw_t']:.2f} "
              f"CW G3vsG4 t={cw['CW_G3_vs_G4']['cw_t']:.2f}")
        cc = p.get('coef_cps_composite')
        if cc:
            print(f"  coef cps_composite: beta={cc['beta']:.5f} t={cc['t']:.2f} "
                  f"(1SD={cc.get('per_1sd_pp', float('nan')):.3f}pp)")
        ls = out[tgt]['ls_cps']
        print(f"  LS Q5-Q1: mean={ls['mean']:.4f} t={ls['tstat']:.2f}")
    print("\n=== C1 (specificity -> dcapex) ===")
    print(f"  full: spec beta={out['C1']['full']['cps_spec_to_dcapex']['beta']:.4f} "
          f"t={out['C1']['full']['cps_spec_to_dcapex']['t']:.2f} | "
          f"flagged n={out['C1']['flagged']['n']:,} "
          f"t={out['C1']['flagged']['cps_spec_to_dcapex']['t']:.2f}")
    print("=== C2 (specificity -> droa) ===")
    c2 = out['C2']
    print(f"  full: spec beta={c2['cps_spec_to_droa']['beta']:.5f} "
          f"t={c2['cps_spec_to_droa']['t']:.2f} | "
          f"flagged t={c2['flagged']['cps_spec_to_droa']['t']:.2f}")
    print(f"  mfg t={c2['by_industry']['manufacturing']['t']:.2f} vs "
          f"svc t={c2['by_industry']['services']['t']:.2f}")
    print("=== C3 (stage groups: spec -> droa t-stat) ===")
    for st, v in out['C3']['by_stage_group'].items():
        print(f"  stage {st}: n={v['n']:6,} t={v['cps_spec_to_droa']['t']:.2f}")
    ix = out['C3']['interaction_spec_x_late']
    print(f"  spec x late: t={ix['spec_x_late']['t']:.2f}")
    print("=== C4 (talk vs walk) ===")
    for q, v in out['C4']['quadrants'].items():
        print(f"  {q:14s} n={v['n']:6,} mean_droa={v['mean_droa']:.4f}")
    tnw = out['C4']['talk_nowalk_vs_rest']
    print(f"  talk_nowalk vs rest: beta={tnw['beta']:.4f} t={tnw['t']:.2f}")
    gp = out['C4']['gap_coef_on_droa']
    print(f"  gap coef: beta={gp['beta']:.4f} t={gp['t']:.2f}")


if __name__ == '__main__':
    run()
