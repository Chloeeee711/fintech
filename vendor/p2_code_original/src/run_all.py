"""
run_all.py — Reproduce the entire Project-2 pipeline end-to-end.

Run from the src/ directory:  python run_all.py   (optionally: python run_all.py --fresh
to clear stale Transformer artifacts before retraining).
Fully offline. Writes all intermediate artifacts and the final report PDF.
"""
import runpy, os, sys, time, config


def _clear_transformer():
    for fn in ['enc_ids_train.npy', 'enc_ids_test.npy', 'enc_meta_train.parquet',
               'enc_meta_test.parquet', 'emb_train.npy', 'emb_test.npy',
               'encoder.pt', 'neural_features.parquet', 'vocab.json']:
        p = os.path.join(config.DATA, fn)
        if os.path.exists(p):
            os.remove(p)
            print(f"  cleared {fn}")


def _transformer():
    import numpy as np
    import transformer_encoder as te
    ids_path = os.path.join(config.DATA, 'enc_ids_train.npy')
    stale = True
    if os.path.exists(ids_path):
        try:
            stale = np.load(ids_path, mmap_mode='r').shape[1] != config.ENC_MAX_LEN
        except Exception:
            stale = True
    if stale:
        print("enc_ids missing or tokenised at a different length -> re-preparing")
        te.prepare()
    te.train()


STAGES = [
    ('data_prep.py',        'Build analytical sample (SEC + Compustat merge, capex)'),
    ('features_text.py',    'Read texts + lexicon + SSI + TF-IDF + lagged SSI'),
    ('capex_plan.py',       'Sentence-level capital-plan construct (CPS) + lagged version'),
    ('transformer',         'Train risk-aware Transformer encoder -> neural_score'),
    ('ml_models.py',        '4x4 incremental-information ML matrix (OOS R2/AUC/DM/FDR)'),
    ('empirical.py',        'Panel ladder, OOS Clark-West, long-short, H3 ablation'),
    ('capex_empirical.py',  'C1-C4 focused capex-plan tests (core v2 analysis)'),
    ('plan_fulfillment.py', 'H6 talk-vs-walk plan-fulfillment gap'),
    ('rl_portfolio.py',     'Reinforcement-learning dynamic allocation'),
    ('make_result_figures.py', 'Result exhibits (SSI dist, ablation, H6, H5, 4x4, CPS)'),
    ('visualize.py',        'Main report figures (tone over time, quintiles, PCA, RL)'),
]


def main():
    if '--fresh' in sys.argv:
        _clear_transformer()
    t0 = time.time()
    for stage, desc in STAGES:
        print(f"\n=== {desc} ===", flush=True)
        if stage == 'transformer':
            _transformer()
        else:
            runpy.run_path(os.path.join(config.SRC, stage), run_name='__main__')
    print("\n=== Build report PDF ===", flush=True)
    # runpy swaps sys.path[0] to the script's own dir; keep src/ importable
    # so build_report.py can `import config`.
    sys.path.insert(0, config.SRC)
    runpy.run_path(os.path.join(config.ROOT, 'report', 'build_report.py'),
                   run_name='__main__')
    print(f"\nALL DONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
