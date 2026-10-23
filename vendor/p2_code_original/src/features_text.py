"""
features_text.py — Text reading, LM-style lexicon features, and TF-IDF.

Reads the SEC 10-K Item-7 plain text for every filing in the analytical
sample, computes the traditional Loughran–McDonald-style tone measures, and
builds a TF-IDF representation used by the ML / transformer stages.

Outputs:
  data/texts.parquet            consolidated table (text + lexicon features + target)
  data/tfidf_vectorizer.joblib  fitted TfidfVectorizer
  data/tfidf_train.npz          sparse TF-IDF matrix (train)
  data/tfidf_test.npz           sparse TF-IDF matrix (test)
"""
import os, re, time, numpy as np, pandas as pd, config
from concurrent.futures import ProcessPoolExecutor
from lexicon import score_text
from specificity import score_text_ssi
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

SEC = config.SEC_DIR
TOKEN = re.compile(r"[a-z']+")

def _read_one(path):
    fp = os.path.join(SEC, path)
    try:
        with open(fp, 'r', errors='ignore') as f:
            return f.read(config.MAX_CHARS)
    except Exception:
        return ''

def _read_and_score(args):
    row_id, path = args
    text = _read_one(path)
    feats = score_text(text)
    feats.update(score_text_ssi(text))   # append SSI raw dimensions + risk densities
    feats['row_id'] = row_id
    feats['text'] = text
    return feats

def build():
    t0 = time.time()
    df = pd.read_parquet(os.path.join(config.DATA, 'analytical_sample.parquet'))
    print(f"Reading {len(df):,} Item-7 texts (serial) …")
    feats = []
    for i, p in zip(df.index, df['text_path']):
        r = _read_and_score((i, p))
        feats.append(r)
        if (i + 1) % 10000 == 0:
            print(f"  {i+1:,} read  ({time.time()-t0:.0f}s)")
    fx = pd.DataFrame(feats).set_index('row_id')
    # z-score the five density-style SSI dims across the corpus, then average
    # into the disclosure-substance index `ssi` (equal-weight, transparent).
    dims = ['ssi_d1', 'ssi_d2', 'ssi_d3', 'ssi_d4_dens', 'ssi_d5_ratio']
    ssi_z = fx[dims].sub(fx[dims].mean()).div(fx[dims].std().replace(0, 1.0))
    fx['ssi'] = ssi_z.mean(axis=1)
    texts = df.join(fx)
    # keep only rows that actually had text
    texts = texts[texts['wc'].notna() & (texts['wc'] > 0)]
    # Persist the original analytical-sample row id as an explicit key so the
    # transformer / empirical / RL stages can join on 'index' downstream.
    texts['index'] = texts.index
    # Lagged SSI / tone (same firm, prior fiscal year) for ΔSSI / Δtone (H5).
    # Aggregate to ONE row per (cik_n, py): a firm may have several Compustat
    # records in the prior year (restatements), which would otherwise fan-out
    # the merge and duplicate rows. Mean of prior-year ssi/tone is deterministic.
    prev = texts[['cik_n', 'fyear', 'ssi', 'tone_lm']].copy()
    prev['py'] = prev['fyear'] - 1
    prev = prev[['cik_n', 'py', 'ssi', 'tone_lm']].rename(
        columns={'ssi': 'lag_ssi', 'tone_lm': 'lag_tone'})
    prev = prev.groupby(['cik_n', 'py'], as_index=False)[['lag_ssi', 'lag_tone']].mean()
    texts = texts.merge(prev, left_on=['cik_n', 'fyear'], right_on=['cik_n', 'py'],
                        how='left').drop(columns=['py'])
    # Safety: guarantee a 1:1 mapping with analytical_sample.index (no dup rows).
    texts = texts.drop_duplicates(subset='index', keep='first')
    out_texts = os.path.join(config.DATA, 'texts.parquet')
    texts.to_parquet(out_texts, index=False)
    print(f"  [texts] saved {out_texts}  rows={len(texts):,}  "
          f"in {time.time()-t0:.1f}s")

    # ---- TF-IDF (fit on train only to avoid leakage) ----
    print("Fitting TF-IDF …")
    train_txt = texts.loc[texts['split'] == 'train', 'text'].tolist()
    test_txt  = texts.loc[texts['split'] == 'test', 'text'].tolist()
    vec = TfidfVectorizer(max_features=config.TFIDF_MAX_FEATURES,
                          ngram_range=config.TFIDF_NGRAM,
                          min_df=5, max_df=0.5, stop_words='english',
                          sublinear_tf=True)
    Xtr = vec.fit_transform(train_txt)
    Xte = vec.transform(test_txt)
    joblib.dump(vec, os.path.join(config.DATA, 'tfidf_vectorizer.joblib'))
    from scipy import sparse
    sparse.save_npz(os.path.join(config.DATA, 'tfidf_train.npz'), Xtr)
    sparse.save_npz(os.path.join(config.DATA, 'tfidf_test.npz'), Xte)
    print(f"  [tfidf] train={Xtr.shape} test={Xte.shape}  "
          f"vocab={len(vec.vocabulary_)}")
    return texts

if __name__ == '__main__':
    build()
