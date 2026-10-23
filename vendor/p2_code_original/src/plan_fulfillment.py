"""
plan_fulfillment.py — H6: the "talk vs walk" plan-fulfillment gap.

Hypothesis (PROJECT2_PLAN.md §2 H6): firms that *disclose* ambitious forward
investment intent (high disclosure-substance + optimistic tone) but then
*under-deliver* on realised capital expenditure exhibit weaker forward
profitability than firms whose words and actions align.

We operationalise:
    plan_intent  = z(ssi) + z(tone_lm) + flag_plan_disclosed      (TALK)
    walk         = z(d_capex_at)                                  (WALK)
    gap          = plan_intent  -  walk        (>0 = talk > walk)

and (a) run a panel regression  roa_fwd ~ gap + controls + FE, and
(b) build the 2x2 TALK x WALK quadrant table (mean forward ROA per cell,
    cluster-robust t on the "talk-no-walk" worst cell vs the rest).

Outputs: data/plan_fulfillment.json  +  data/fig_h6_quadrant.npz (for the report)
"""
import os, json, time, numpy as np, pandas as pd, config

CONTROLS = ['size', 'bm', 'roa', 'lev', 'mom', 'invest', 'age']


def _z(s):
    s = s.astype(float)
    m, sd = s.mean(), s.std()
    return (s - m) / sd if sd > 0 else s * 0.0


def _ols(X, y, cluster=None):
    """OLS; cluster-robust (one-way) SE when cluster ids given."""
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    if cluster is None:
        V = (resid @ resid) / (n - k) * XtX_inv
    else:
        cl = np.asarray(cluster)
        u = resid[:, None] * X
        meat = np.zeros((k, k))
        for g in np.unique(cl):
            ug = u[cl == g].sum(0)
            meat += np.outer(ug, ug)
        V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return beta, se, beta / se


def run():
    t0 = time.time()
    smp = pd.read_parquet(os.path.join(config.DATA, 'analytical_sample.parquet'))
    tx = pd.read_parquet(os.path.join(config.DATA, 'texts.parquet'))
    # NB: roa_fwd / d_capex_at come from analytical_sample (smp) itself —
    # pulling them from texts too would collide into _x/_y suffixed columns.
    tcols = ['index', 'ssi', 'tone_lm', 'flag_plan_disclosed']
    df = smp.merge(tx[tcols], left_index=True, right_on='index', how='left')
    d = df.dropna(subset=['roa_fwd', 'd_capex_at'] + CONTROLS).copy()

    # ---- construct TALK / WALK / GAP ----
    d['plan_intent'] = _z(d['ssi']) + _z(d['tone_lm']) + d['flag_plan_disclosed'].fillna(0)
    d['walk'] = _z(d['d_capex_at'])
    d['gap'] = d['plan_intent'] - d['walk']

    # ---- (a) panel regression of gap on forward ROA ----
    Xc = np.nan_to_num(d[CONTROLS].values.astype(float))
    sic = pd.get_dummies(d['sic2'].fillna('00'), drop_first=True).values.astype(float)
    yr = pd.get_dummies(d['file_year'], drop_first=True).values.astype(float)
    X = np.hstack([np.ones((len(d), 1)), d[['gap']].values.astype(float), Xc, sic, yr])
    beta, se, t = _ols(X, d['roa_fwd'].values.astype(float), d['cik_n'].values)
    gap_coef, gap_t = float(beta[1]), float(t[1])

    # ---- (b) 2x2 quadrant ----
    talk_hi = d['plan_intent'] > d['plan_intent'].median()
    walk_hi = d['walk'] > d['walk'].median()
    d['quad'] = np.where(talk_hi & walk_hi, 'talk_walk',
                 np.where(talk_hi & ~walk_hi, 'talk_nowalk',
                  np.where(~talk_hi & walk_hi, 'notalk_walk', 'notalk_nowalk')))
    quad = {}
    for q in ['talk_walk', 'talk_nowalk', 'notalk_walk', 'notalk_nowalk']:
        sub = d[d['quad'] == q]
        quad[q] = dict(n=int(len(sub)),
                       mean_roa_fwd=float(sub['roa_fwd'].mean()),
                       mean_gap=float(sub['gap'].mean()))
    # worst cell (talk_nowalk) vs the rest
    worst = d[d['quad'] == 'talk_nowalk']['roa_fwd']
    rest = d[d['quad'] != 'talk_nowalk']['roa_fwd']
    diff = worst.mean() - rest.mean()
    pooled_sd = np.sqrt((worst.var(ddof=1) * (len(worst) - 1) +
                         rest.var(ddof=1) * (len(rest) - 1)) /
                        (len(worst) + len(rest) - 2))
    t_worst = diff / (pooled_sd * np.sqrt(1 / len(worst) + 1 / len(rest)))

    out = dict(
        gap_coef_on_roa_fwd=gap_coef, gap_t_stat=gap_t,
        n=int(len(d)),
        talk_nowalk_vs_rest_diff=diff, talk_nowalk_vs_rest_t=t_worst,
        quadrants=quad,
        corr_plan_intent_walk=float(np.corrcoef(d['plan_intent'], d['walk'])[0, 1]),
    )
    json.dump(out, open(os.path.join(config.DATA, 'plan_fulfillment.json'), 'w'), indent=2)

    # persist compact arrays for the report figure
    np.savez(os.path.join(config.DATA, 'fig_h6_quadrant.npz'),
             plan_intent=d['plan_intent'].values.astype(float),
             walk=d['walk'].values.astype(float),
             roa_fwd=d['roa_fwd'].values.astype(float),
             quad=d['quad'].astype('U16').values)
    print(f"  [H6 plan-fulfillment] gap_coef={gap_coef:.4f} (t={gap_t:.2f})  "
          f"talk_nowalk_vs_rest diff={diff:.4f} (t={t_worst:.2f})  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == '__main__':
    run()
