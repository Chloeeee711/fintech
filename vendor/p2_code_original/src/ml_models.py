"""
ml_models.py — Incremental-Information Model Matrix (PROJECT2_PLAN.md §5/§8).

For each information set G (rows) and each model family M (columns) we train on
the in-sample window (file-years < 2020) and evaluate OUT-OF-SAMPLE on the
held-out test period (file-years 2020–2022):

  Information sets (rows):
    G1  financial controls only
    G2  + dictionary tone (tone_lm)
    G3  + generic rule SSI + 3 risk-subtype lexicons
    G4  + CPS — the capital-plan construct (sentence-level specificity & stage;
         the v2 focused core)
    G5  + neural_score (deep-learning probe; extended ladder, off the main table)

  Model families (columns):
    Ridge  (linear / logistic)        — interpretable baseline
    ENet   (Elastic Net, L1+L2)       — sparse linear: which text features survive
    RF     (random forest)
    XGB    (gradient-boosted trees)
    SVM    (RBF-kernel support vector) — capped training sample (see NOTE)

Targets:
  (1) droa         regression  -> OOS R^2, MAE ;  classification (ΔROA>0) -> AUC/F1/Brier
  (2) d_capex_at   regression  -> OOS R^2, MAE   (secondary, C1/H5/H6)
  (3) roa_fwd      regression  -> OOS R^2, MAE   (robustness, legacy ladder)

Evaluation protocol:
  * OOS ΔR^2 reported RELATIVE to that family's own G1 (isolates the incremental
    information contribution of each text layer within a fixed model family).
  * Diebold–Mariano (predictive-loss) test G_k vs G1 per family, two-sided
    p-values, then Benjamini–Hochberg FDR across the family x comparison grid.
  * 1-SD economic significance: top vs bottom tercile of fitted ROA score.

NOTE on SVM: the RBF kernel scales ~O(n^2); to keep the run tractable we train
the SVM family on a 20k stratified subsample (documented); all G-sets use the
SAME subsample so the G1->G4 contrast remains valid. Other families use full.
"""
import os, time, json, math, numpy as np, pandas as pd, config
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier

CONTROLS = ['size', 'bm', 'roa', 'lev', 'mom', 'invest', 'age']

# Information sets: each is the EXTRA feature block on top of CONTROLS.
G3_BLOCK = ['tone_lm', 'ssi', 'risk_demand', 'risk_cost', 'risk_liquidity']
CPS_BLOCK = ['cps_composite', 'cps_stage_mean']
GROUPS = {
    'G1': [],                                    # controls only
    'G2': ['tone_lm'],                           # + dictionary tone
    'G3': G3_BLOCK,                              # + generic SSI + risk lexicons
    'G4': G3_BLOCK + CPS_BLOCK,                  # + capital-plan construct (v2 core)
    'G5': G3_BLOCK + CPS_BLOCK + ['neural_score'],  # + DL probe (extended ladder)
}
MODELS = ['Ridge', 'ENet', 'RF', 'XGB', 'SVM']   # 4 sets x 5 families = 20 cells
SVM_CAP = 20000
MAIN_GROUPS = ['G1', 'G2', 'G3', 'G4']           # G5 reported separately


def _load():
    smp = pd.read_parquet(os.path.join(config.DATA, 'analytical_sample.parquet'))
    tx = pd.read_parquet(os.path.join(config.DATA, 'texts.parquet'))
    tcols = ['index', 'ssi', 'tone_lm', 'risk_demand', 'risk_cost', 'risk_liquidity',
             'cps_composite', 'cps_stage_mean']
    if os.path.exists(os.path.join(config.DATA, 'neural_features.parquet')):
        nf = pd.read_parquet(os.path.join(config.DATA, 'neural_features.parquet'))
        if 'neural_score' in nf.columns:
            tx = tx.merge(nf[['index', 'neural_score']], on='index', how='left')
            tcols.append('neural_score')
    df = smp.merge(tx[tcols], left_index=True, right_on='index', how='left')
    # G5 only meaningful if neural_score present
    g5_ok = 'neural_score' in df.columns and df['neural_score'].notna().any()
    groups = {k: v for k, v in GROUPS.items() if k != 'G5' or g5_ok}
    df['split'] = df['split'].astype(str)
    tr = df[df['split'] == 'train'].reset_index(drop=True)
    te = df[df['split'] == 'test'].reset_index(drop=True)
    return df, tr, te, groups


def _complete(tr, te, target, groups):
    """Keep rows with non-null target + all features used by any G-set."""
    need = CONTROLS + [target] + sorted({f for g in groups.values() for f in g})
    keep = [c for c in need if c in tr.columns]
    tr = tr.dropna(subset=keep).reset_index(drop=True)
    te = te.dropna(subset=keep).reset_index(drop=True)
    return tr, te


def _fit_predict(reg, clf, Xtr, ytr_reg, ytr_cls, Xte, family):
    """Return (pred_reg, pred_cls_prob, pred_cls). For SVM we may cap training."""
    if family == 'SVM':
        n = len(Xtr)
        if n > SVM_CAP:
            rng = np.random.default_rng(config.SEED)
            if ytr_cls is not None:
                idx = np.concatenate([rng.choice(np.where(ytr_cls == c)[0],
                                                 size=min(SVM_CAP // 2, (ytr_cls == c).sum()),
                                                 replace=False) for c in np.unique(ytr_cls)])
            else:
                idx = rng.choice(n, size=SVM_CAP, replace=False)
            Xtr, ytr_reg, ytr_cls = Xtr[idx], (ytr_reg[idx] if ytr_reg is not None else None), \
                (ytr_cls[idx] if ytr_cls is not None else None)
    # regression
    if ytr_reg is not None:
        reg.fit(Xtr, ytr_reg)
        pred_reg = reg.predict(Xte)
    else:
        pred_reg = None
    # classification
    if ytr_cls is not None:
        clf.fit(Xtr, ytr_cls)
        pred_prob = clf.predict_proba(Xte)[:, 1]
        pred_cls = (pred_prob >= 0.5).astype(int)
    else:
        pred_prob = pred_cls = None
    return pred_reg, pred_prob, pred_cls


def _make_models(family):
    if family == 'Ridge':
        return Ridge(alpha=1.0), LogisticRegression(max_iter=3000)
    if family == 'ENet':
        # Elastic Net (L1+L2): the sparse-linear angle on the text block —
        # which text features survive shrinkage is itself interpretable.
        return (ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000,
                           random_state=config.SEED),
                LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=1.0,
                                   solver='saga', max_iter=5000,
                                   random_state=config.SEED))
    if family == 'RF':
        return (RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=config.SEED),
                RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=config.SEED))
    if family == 'XGB':
        return (XGBRegressor(n_estimators=200, n_jobs=-1, random_state=config.SEED,
                             tree_method='hist', verbosity=0),
                XGBClassifier(n_estimators=200, n_jobs=-1, random_state=config.SEED,
                              eval_metric='logloss', tree_method='hist', verbosity=0))
    if family == 'SVM':
        return (SVR(kernel='rbf', C=1.0, cache_size=2000),
                SVC(kernel='rbf', C=1.0, probability=True, cache_size=2000))


def _metrics_reg(y, p):
    sst = float(((y - y.mean()) ** 2).sum())
    sse = float(((y - p) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else 0.0
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return r2, mae, rmse, sse


def _metrics_cls(y, prob):
    from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
    auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float('nan')
    pred = (prob >= 0.5).astype(int)
    f1 = float(f1_score(y, pred, zero_division=0))
    br = float(brier_score_loss(y, prob))
    return auc, f1, br


def _dm_pvalue(loss_a, loss_b):
    """Two-sided Diebold-Mariano (equal-weighted, HAC lag-1) for mean(loss_a-loss_b)."""
    d = loss_a - loss_b          # >0 means b has lower loss (b better)
    n = len(d)
    if n < 4:
        return float('nan')
    mean = d.mean()
    var = d.var(ddof=1)
    if n > 2:
        ac = np.cov(d[:-1], d[1:])[0, 1]
        var = var + 2.0 * ac
    se = math.sqrt(var / n)
    if se <= 0:
        return float('nan')
    t = mean / se
    return float(2 * (1 - stats.norm.cdf(abs(t))))


def _bh_fdr(pvals):
    """Benjamini-Hochberg: returns list of (raw_p, q_value) for sorted input."""
    items = sorted(pvals, key=lambda x: (x[1] if x[1] == x[1] else 1.0))
    m = len(items)
    out = []
    prev_q = 0.0
    for i, (name, p) in enumerate(items):
        q = (p * m / (i + 1)) if (p == p) else 1.0
        q = min(max(q, prev_q), 1.0)
        prev_q = q
        out.append((name, p, q))
    return out


def _run_target(df, tr, te, target, groups, with_cls=False, cls_thresh=0.0):
    """Full G x family OOS evaluation for one target.

    Returns (matrix, bh_fdr_list, pred_store, y_test).
    with_cls=True additionally fits a classifier on (target > cls_thresh).
    """
    y = te[target].values.astype(float)
    y_cls_tr = (tr[target].values > cls_thresh).astype(int) if with_cls else None
    y_cls_te = (te[target].values > cls_thresh).astype(int) if with_cls else None
    matrix, pred_store = {}, {}
    for fam in MODELS:
        reg, clf = _make_models(fam)
        fam_res, fam_se = {}, {}
        for g in groups:
            feats = [f for f in CONTROLS + groups[g] if f in df.columns]
            sc = StandardScaler().fit(tr[feats].values.astype(float))
            Xtr = np.nan_to_num(sc.transform(tr[feats].values.astype(float)))
            Xte = np.nan_to_num(sc.transform(te[feats].values.astype(float)))
            ytr_reg = tr[target].values.astype(float)
            pred_reg, pred_prob, _ = _fit_predict(reg, clf, Xtr, ytr_reg,
                                                  y_cls_tr, Xte, fam)
            r2, mae, rmse, _ = _metrics_reg(y, pred_reg)
            fam_res[g] = {'oos_r2': r2, 'mae': mae, 'rmse': rmse}
            fam_se[g] = ((y - pred_reg) ** 2)
            if pred_prob is not None:
                auc, f1, br = _metrics_cls(y_cls_te, pred_prob)
                fam_res[g].update({'auc': auc, 'f1': f1, 'brier': br})
            pred_store[(fam, g)] = pred_reg
        base = fam_se['G1'].sum()
        for g in groups:
            fam_res[g]['delta_r2_vs_G1'] = float(1 - fam_se[g].sum() / base) if base > 0 else 0.0
        dm, pvals = {}, []
        for g in groups:
            if g == 'G1':
                continue
            p = _dm_pvalue(fam_se['G1'], fam_se[g])   # loss_G1 - loss_Gk
            dm[f'DM_G1_vs_{g}'] = p
            pvals.append((f'{fam}|{g}', p))
        fam_res['dm_pvalues'] = dm
        matrix[fam] = fam_res
    all_p = [(f'{fam}|{n}', p) for fam in MODELS
             for n, p in matrix[fam]['dm_pvalues'].items()]
    return matrix, [{'comparison': n, 'p': p, 'q': q}
                    for n, p, q in _bh_fdr(all_p)], pred_store, y


def run():
    t0 = time.time()
    df, tr, te, groups = _load()
    results = {'groups_used': list(groups.keys()),
               'n_train': int(len(tr)), 'n_test': int(len(te)), 'models': MODELS}

    # ---------- (1) PRIMARY target: droa (profitability improvement) ----------
    tr, te = _complete(tr, te, 'droa', groups)
    results['n_train_droa'] = int(len(tr)); results['n_test_droa'] = int(len(te))
    matrix, fdr, pred_store, y_droa = _run_target(df, tr, te, 'droa', groups,
                                                  with_cls=True)
    results['droa_matrix'] = matrix
    results['droa_bh_fdr'] = fdr

    # ---------- 1-SD economic significance (representative: RF on G4) ----------
    key = ('RF', 'G4') if ('RF', 'G4') in pred_store else ('RF', 'G3')
    rep = pred_store.get(key)
    if rep is not None:
        s = pd.Series(rep).rank(pct=True)
        top, bot = y_droa[s > 2/3], y_droa[s < 1/3]
        results['economic_significance'] = {
            'model': f'RF_{key[1]}', 'target': 'droa',
            'top_tercile_mean_droa': float(top.mean()),
            'bottom_tercile_mean_droa': float(bot.mean()),
            'spread': float(top.mean() - bot.mean()),
            'n_top': int(len(top)), 'n_bottom': int(len(bot))}

    # ---------- ElasticNet sparsity diagnostic (which text features survive L1?) ----------
    # A flat ENet row would be ambiguous; this records exactly which coefficients
    # the L1 keeps on the G4 (full text stack) design.
    try:
        feats = [f for f in CONTROLS + groups['G4'] if f in df.columns]
        sc = StandardScaler().fit(tr[feats].values.astype(float))
        Xtr = np.nan_to_num(sc.transform(tr[feats].values.astype(float)))
        en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000,
                        random_state=config.SEED)
        en.fit(Xtr, tr['droa'].values.astype(float))
        nz = [f for f, c in zip(feats, en.coef_) if abs(c) > 1e-8]
        results['enet_sparsity'] = {
            'group': 'G4', 'n_features': int(len(feats)),
            'n_nonzero': int(len(nz)), 'nonzero': nz,
            'nonzero_text': [f for f in nz if f not in CONTROLS]}
    except Exception as e:                                   # pragma: no cover
        results['enet_sparsity'] = {'error': str(e)}

    # ---------- (2) SECONDARY target: d_capex_at (investment change) ----------
    tr_c, te_c = _complete(tr, te, 'd_capex_at', groups)
    results['n_train_capex'] = int(len(tr_c)); results['n_test_capex'] = int(len(te_c))
    cap_matrix, cap_fdr, _, _ = _run_target(df, tr_c, te_c, 'd_capex_at', groups)
    results['d_capex_at_matrix'] = cap_matrix
    results['d_capex_at_bh_fdr'] = cap_fdr

    # ---------- (3) ROBUSTNESS target: roa_fwd level ----------
    tr_r, te_r = _complete(tr, te, 'roa_fwd', groups)
    results['n_train_roa'] = int(len(tr_r)); results['n_test_roa'] = int(len(te_r))
    roa_matrix, roa_fdr, _, _ = _run_target(df, tr_r, te_r, 'roa_fwd', groups)
    results['roa_fwd_matrix'] = roa_matrix
    results['roa_fwd_bh_fdr'] = roa_fdr

    # ---------- diagnostics: CPS (rule construct) vs neural (fitted probe) ----------
    diag = {}
    if 'cps_composite' in df.columns:
        diag['corr_cps_roa_test'] = float(te['cps_composite'].corr(te['roa']))
        diag['corr_cps_droa_test'] = float(te['cps_composite'].corr(te['droa']))
        diag['corr_cps_dcapex_test'] = float(te['cps_composite'].corr(te['d_capex_at']))
        diag['note_cps'] = ('CPS is a rule-based sentence-level construct (specificity + '
                            'stage), not a fitted probe of the outcome, so unlike the '
                            'neural probe it does not mechanically overlap with the G1 '
                            'control roa (persistence).')
    if 'neural_score' in df.columns:
        diag['corr_neural_roa_test'] = float(te['neural_score'].corr(te['roa']))
        diag['corr_neural_roafwd_test'] = float(te['neural_score'].corr(te['roa_fwd']))
        yt = te['roa_fwd'].values.astype(float)
        p = te['neural_score'].values.astype(float)
        sst = float(((yt - yt.mean()) ** 2).sum())
        diag['neural_univariate_oos_r2'] = float(1 - ((yt - p) ** 2).sum() / sst) if sst > 0 else None
        diag['note_neural'] = ('neural_score is a Ridge probe fit on train-period ΔROA '
                               '(droa); the v2 retargeting removes most of the mechanical '
                               'overlap with the G1 control roa (persistence) that a '
                               'forward-ROA-level probe suffered, so the nested G5 block '
                               'is now a fair like-for-like comparison against the CPS '
                               'construct.')
    results['g4_diagnostic'] = diag

    # ---------- console summary ----------
    print(f"\n[ml_models] PRIMARY target droa — OOS R^2 by information set:")
    hdr = "family " + " ".join(f"{g:>8s}" for g in groups)
    print("  " + hdr)
    for fam in MODELS:
        row = " ".join(f"{matrix[fam][g]['oos_r2']:8.4f}" for g in groups)
        print(f"  {fam:>6s} {row}")
    if 'economic_significance' in results:
        e = results['economic_significance']
        print(f"  1-SD econ sig ({e['model']}): top-bottom droa spread = {e['spread']:.4f}")
    print(f"  BH-FDR significant (q<0.05): "
          f"{sum(1 for r in results['droa_bh_fdr'] if (r['q'] or 1) < 0.05)}/"
          f"{len(results['droa_bh_fdr'])} comparisons")
    print(f"\n[ml_models] SECONDARY target d_capex_at — OOS R^2:")
    for fam in MODELS:
        row = " ".join(f"{cap_matrix[fam][g]['oos_r2']:8.4f}" for g in groups)
        print(f"  {fam:>6s} {row}")
    print(f"\n[ml_models] ROBUSTNESS target roa_fwd — OOS R^2:")
    for fam in MODELS:
        row = " ".join(f"{roa_matrix[fam][g]['oos_r2']:8.4f}" for g in groups)
        print(f"  {fam:>6s} {row}")
    print(f"  wrote ml_models_results.json  ({time.time()-t0:.0f}s)")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, float):
            return o if (o == o and abs(o) < 1e15) else None
        return o
    json.dump(clean(results), open(os.path.join(config.DATA, 'ml_models_results.json'), 'w'),
              indent=2)
    return results


if __name__ == '__main__':
    run()
