"""Clean full text and training-only transformations, keyed by accession."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing_v1 import (FORMAL, FinancialTransform, exact_lags, sha256,
                              tokenize_tfidf, train_zscores)
from lexicon import score_text
from specificity import score_text_ssi


def build():
    latest = json.loads((FORMAL / 'p2_revision1' / 'latest_run.json').read_text())
    run = Path(latest['run_dir'])
    status = json.loads((run / 'status.json').read_text())
    if status['status'] != 'completed' or not status['aligned_rows']:
        raise RuntimeError('No verified filing matches. Complete filing_metadata.csv first; no guessed dates.')
    data = run / 'data'
    if (data / 'texts.parquet').exists():
        raise RuntimeError('Features already exist; create a new preprocessing run instead of overwriting.')
    df = pd.read_parquet(data / 'analytical_sample.parquet')
    # Freeze the main-task population across all representation comparisons.
    # A missing secondary Y never removes a valid main-task row.
    df = df[df.main_eligible & df.split.ne('excluded')].copy()
    if not {'train', 'validation', 'test'}.issubset(set(df.split)):
        raise RuntimeError('Verified train/validation/test rows required before fitting any transforms.')
    source = Path(status['source_run'])
    wanted = set(df.accession_key)
    pieces, hashes = [], {}
    for shard in sorted((source / 'data' / 'text_clean').glob('*.parquet')):
        part = pd.read_parquet(shard, columns=['accession_key', 'text_natural'])
        part = part[part.accession_key.isin(wanted)]
        if not part.empty:
            pieces.append(part)
            hashes[str(shard)] = sha256(shard)
    if not pieces:
        raise RuntimeError('No clean text shards for the matched accessions.')
    df = df.merge(pd.concat(pieces), on='accession_key', how='left', validate='one_to_one')
    if df.text_natural.isna().any() or df.text_natural.str.strip().eq('').any():
        raise RuntimeError('Missing/empty text; quarantine, never substitute an empty string.')
    actual_hash = df.text_natural.map(lambda t: __import__('hashlib').sha256(t.encode('utf-8')).hexdigest())
    if not actual_hash.eq(df.clean_sha256).all():
        raise RuntimeError('Clean text differs from manifest hashes.')
    df = df.rename(columns={'text_natural': 'text'})
    # No 12,000-character truncation, no stemming/stopword removal for encoders.
    raw = pd.DataFrame([{**score_text(t), **score_text_ssi(t)} for t in df.text], index=df.index)
    df = pd.concat([df, raw], axis=1)
    dims = ['ssi_d1', 'ssi_d2', 'ssi_d3', 'ssi_d4_dens', 'ssi_d5_ratio']
    z, ssi_state = train_zscores(df, dims)
    df['ssi'] = z.mean(axis=1)
    df = exact_lags(df, ['ssi', 'tone_lm'])
    # Do not require a lag to keep a company with only one usable report.
    df.to_parquet(data / 'texts.parquet', index=False)
    transform = FinancialTransform().fit(df[df.split.eq('train')])
    f = transform.transform(df)
    f.insert(0, 'accession_key', df.accession_key.values)
    f.to_parquet(data / 'financial_features.parquet', index=False)
    # Callable tokenizer would bypass sklearn's built-in n-gram generation.
    # tokenizer=... keeps negations/numbers AND lets ngram_range=(1,2) work.
    vec = TfidfVectorizer(tokenizer=tokenize_tfidf, token_pattern=None,
                          lowercase=False, ngram_range=(1, 2), min_df=5,
                          max_df=.95, max_features=20000, sublinear_tf=True)
    tr = df.split.eq('train')
    vec.fit(df.loc[tr, 'text'])
    for split in ['train', 'validation', 'test']:
        rows = df[df.split.eq(split)]
        matrix = vec.transform(rows.text)
        sparse.save_npz(data / ('tfidf_' + split + '.npz'), matrix)
        rows[['accession_key', 'index', 'gvkey', 'fyear']].to_csv(data / ('tfidf_' + split + '_keys.csv'), index=False)
    joblib.dump(vec, data / 'tfidf_vectorizer.joblib')
    joblib.dump(transform, data / 'financial_transform.joblib')
    state = {'financial': transform.state(), 'ssi': ssi_state,
             'fit_accessions': df.loc[tr, 'accession_key'].tolist(),
             'text_shard_sha256': hashes,
             'note': 'No Y winsorisation/imputation; raw financial/text features preserved.'}
    (data / 'feature_transform_state.json').write_text(json.dumps(state, indent=2))
    (run / 'feature_status.json').write_text(json.dumps({
        'status': 'completed', 'fit_transform_executed': True,
        'fit_split': 'train', 'rows': len(df),
        'split_counts': df.split.value_counts().to_dict(),
        'training_ready': False, 'downstream_models_require_repairs': True,
    }, indent=2))
    with (run / '运行记录.md').open('a') as out:
        out.write('\n## 文本与财务特征\n\n读取完整清洗文本并核对哈希；仅训练集拟合X缩尾、填补、标准化、行业词表、SSI与TF-IDF；验证/测试仅转换。\n')
        out.write('\n各split数量：`' + json.dumps(df.split.value_counts().to_dict()) + '`；TF-IDF词数：' + str(len(vec.vocabulary_)) + '。\n')
    hashes = {str(f.relative_to(run)): sha256(f) for f in data.rglob('*') if f.is_file()}
    (run / 'feature_output_sha256.json').write_text(json.dumps(hashes, indent=2))
    return df
