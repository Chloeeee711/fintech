"""
text_shootout.py — Unified-Ridge benchmark across text representations.

Step 5 of the seven-step pipeline ("先以 Ridge 为统一基准比较 TF-IDF、FinBERT、
MPNet、SSI/CPS、Qwen 因子和摘要向量的预测效果"): holding the estimator (Ridge),
the time split, and the complete-case sample FIXED, we compare what each TEXT
REPRESENTATION adds over the financial baseline. The G1 controls are included
in every specification, so every number below is an incremental-information
number (text BEYOND financials), directly comparable across rows:

    G1 financial      controls only — the reference every spec must beat
    TF-IDF (SVD-200)  classic bag-of-words; SVD fit on TRAIN texts only (leak-free)
    Tone (LM)         dictionary sentiment
    SSI (rule)        generic disclosure-substance index
    CPS (rule)        capital-plan specificity & stage — the v2 core construct
    neural_score      Ridge probe on the from-scratch Transformer embedding
    G4 combo          SSI + CPS + tone (best structured stack)

Offline substitution (disclosed, §3.3 of the report): FinBERT / MPNet /
Qwen-summary vectors cannot be computed in this sandbox (no weights, no API).
The from-scratch risk-aware Transformer probe is the end-to-end proxy, and the
sentence-level CPS is the structured proxy for Qwen extraction.

Targets: droa (primary) and d_capex_at (secondary).
Metrics: OOS R2 / MAE / RMSE; Diebold-Mariano vs G1, Benjamini-Hochberg FDR
across the representation x target grid. Same test rows for every spec.
"""
import os, json, time, numpy as np, pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
import config
import ml_models as M

REPS = [
    ('G1 financial', []),
    ('TF-IDF (SVD-200)', 'TFIDF'),
    ('Tone (LM)', ['tone_lm']),
    ('SSI (rule)', ['ssi']),
    ('CPS (rule)', ['cps_composite', 'cps_stage_mean']),
    ('neural_score (DL probe)', ['neural_score']),
    ('G4 combo (SSI+CPS+tone)', M.G3_BLOCK + M.CPS_BLOCK),
]
TARGETS = [('droa', 'ΔROA (primary)'), ('d_capex_at', 'ΔCAPEX/AT (secondary)')]


def _tfidf_svd():
    """SVD-200 of the train-fitted TF-IDF, mapped to analytical row ids."""
    tx = pd.read_parquet(os.path.join(config.DATA, 'texts.parquet'))
    tr_ids = tx.loc[tx['split'] == 'train', 'index'].values
    te_ids = tx.loc[tx['split'] == 'test', 'index'].values
    Xtr = sparse.load_npz(os.path.join(config.DATA, 'tfidf_train.npz'))
    Xte = sparse.load_npz(os.path.join(config.DATA, 'tfidf_test.npz'))
    assert Xtr.shape[0] == len(tr_ids) and Xte.shape[0] == len(te_ids), \
        'tfidf matrix / texts.parquet row misalignment'
    svd = TruncatedSVD(n_components=200, random_state=config.SEED)
    Ztr = svd.fit_transform(Xtr)
    Zte = svd.transform(Xte)
    return ({i: k for k, i in enumerate(tr_ids)}, Ztr), \
           ({i: k for k, i in enumerate(te_ids)}, Zte)


def run():
    t0 = time.time()
    df, tr, te, groups = M._load()
    (tr_map, Ztr), (te_map, Zte) = _tfidf_svd()
    results = {'n_train': int(len(tr)), 'n_test': int(len(te)),
               'reps': [r[0] for r in REPS],
               'note': ('Ridge(alpha=1) on standardized features; G1 financial '
                        'controls included in EVERY spec, so each row is the '
                        'incremental information of that text representation '
                        'beyond financials.')}

    for target, tlabel in TARGETS:
        # like-for-like: same complete-case rows across ALL representations
        allf = sorted({f for _, fs in REPS for f in fs if isinstance(fs, list)})
        need = [c for c in M.CONTROLS + [target] + allf if c in df.columns]
        trc = tr.dropna(subset=need).reset_index(drop=True)
        tec = te.dropna(subset=need).reset_index(drop=True)
        rows, losses = {}, {}
        for name, feats in REPS:
            if feats == 'TFIDF':
                Xtr = np.hstack([trc[M.CONTROLS].values.astype(float),
                                 Ztr[[tr_map[i] for i in trc['index'].values]]])
                Xte = np.hstack([tec[M.CONTROLS].values.astype(float),
                                 Zte[[te_map[i] for i in tec['index'].values]]])
            else:
                use = [f for f in M.CONTROLS + feats if f in df.columns]
                sc = StandardScaler().fit(trc[use].values.astype(float))
                Xtr = np.nan_to_num(sc.transform(trc[use].values.astype(float)))
                Xte = np.nan_to_num(sc.transform(tec[use].values.astype(float)))
            y = tec[target].values.astype(float)
            reg = Ridge(alpha=1.0).fit(Xtr, trc[target].values.astype(float))
            pred = reg.predict(Xte)
            r2, mae, rmse, sse = M._metrics_reg(y, pred)
            rows[name] = {'oos_r2': r2, 'mae': mae, 'rmse': rmse}
            losses[name] = (y - pred) ** 2
        # Diebold-Mariano vs G1 + Benjamini-Hochberg across representations
        base = losses['G1 financial']
        pvals = []
        for name, _ in REPS:
            if name == 'G1 financial':
                continue
            p = M._dm_pvalue(base, losses[name])
            rows[name]['dm_p_vs_G1'] = p
            pvals.append((name, p))
        fdr = [{'rep': n, 'p': p, 'q': q} for n, p, q in M._bh_fdr(pvals)]
        results[target] = {'label': tlabel, 'n_train': int(len(trc)),
                           'n_test': int(len(tec)), 'specs': rows,
                           'bh_fdr': fdr}
        print(f"\n=== {tlabel}  (train {len(trc):,} / test {len(tec):,}) ===")
        for name, _ in REPS:
            r = rows[name]
            dm = r.get('dm_p_vs_G1')
            dm_s = f"  DM p={dm:.3f}" if dm is not None else ''
            print(f"  {name:26s} OOS R2={r['oos_r2']:+.4f}  "
                  f"MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}{dm_s}")

    out = os.path.join(config.DATA, 'text_shootout.json')
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\n  [save] {out}   ({time.time()-t0:.0f}s)")
    return results


if __name__ == '__main__':
    run()
