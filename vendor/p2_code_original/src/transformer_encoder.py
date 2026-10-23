"""
transformer_encoder.py — A from-scratch Transformer financial-language-model
encoder (the deep-learning / "LLM-era" textual representation).

We train a small BERT-style encoder (learned token + positional embeddings,
multi-head self-attention, mean pooling) directly on the 10-K Item-7 corpus
with a RISK-AWARE MULTI-TASK objective (PROJECT2_PLAN.md §6, v2 focus):

    * regress the standardised 1-year ROA CHANGE  (ΔROA = ROA_{t+1} − ROA_t,
      the v2 primary outcome — retargeted away from the forward-ROA *level*,
      which is dominated by ROA persistence and mechanically overlaps the
      G1 control `roa`)
    * classify the ΔROA sign (up/down)
    * 3 auxiliary risk-subtype BCE heads: demand / cost / liquidity
      (these force the encoder to learn risk-relevant representations and
       feed the transformer's auxiliary risk signal used in G4 / H3 / H6)

The pooled document vector is our dense "neural tone" embedding; a LINEAR
PROBE on the frozen embedding (5-fold OUT-OF-FOLD on train, full-fit for test)
yields a single comparable `neural_score` used in the incremental-R² test
(H2: LLM signal beyond the static dictionary) and as the G5 block in the 4x4
matrix (offline proxy for the LLM/Qwen risk-extraction layer).

Everything runs locally on CPU with PyTorch — no external model weights, no API.
"""
import os, re, time, json, numpy as np, pandas as pd, torch, torch.nn as nn
import config

DEVICE = torch.device('cpu')
TOK = re.compile(r"[a-z']+")
SEED = config.SEED
RISK_TYPES = ['risk_demand', 'risk_cost', 'risk_liquidity']


def _load_split(split):
    t = pd.read_parquet(os.path.join(config.DATA, 'texts.parquet'))
    return t[t['split'] == split].reset_index(drop=False)   # keep orig 'index'


def build_vocab(texts, max_vocab=config.ENC_VOCAB_SIZE):
    from collections import Counter
    cnt = Counter()
    for s in texts:
        cnt.update(TOK.findall(s.lower()))
    words = [w for w, _ in cnt.most_common(max_vocab - 2)]
    vocab = {w: i + 2 for i, w in enumerate(words)}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    return vocab


def prepare():
    """Build vocabulary and tokenised, length-padded id matrices."""
    t0 = time.time()
    train = _load_split('train')
    test = _load_split('test')
    print(f"build_vocab on {len(train):,} train docs …", flush=True)
    vocab = build_vocab(train['text'].tolist(), config.ENC_VOCAB_SIZE)
    json.dump(vocab, open(os.path.join(config.DATA, 'vocab.json'), 'w'))
    print(f"  vocab size={len(vocab)}  ({time.time()-t0:.1f}s)", flush=True)

    def tok_to_ids(text, max_len=config.ENC_MAX_LEN):
        ids = [vocab.get(w, 1) for w in TOK.findall(text.lower())][:max_len]
        ids += [0] * (max_len - len(ids))
        return np.array(ids, dtype=np.int16)

    meta_cols = ['index', 'cik_n', 'fyear', 'file_year', 'ret_1y', 'roa_fwd',
                 'roa', 'split', 'tone_lm', 'unc_ratio', 'lit_ratio'] + RISK_TYPES
    for name, df in [('train', train), ('test', test)]:
        X = np.stack([tok_to_ids(s) for s in df['text'].tolist()])
        np.save(os.path.join(config.DATA, f'enc_ids_{name}.npy'), X)
        df[meta_cols].to_parquet(
            os.path.join(config.DATA, f'enc_meta_{name}.parquet'), index=False)
        print(f"  {name}: X{X.shape}  meta saved", flush=True)
    return vocab


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nlayers, dim_feed, max_len,
                 n_risk=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feed, dropout=dropout,
                                           batch_first=True)
        self.tfm = nn.TransformerEncoder(layer, nlayers)
        self.ret_head = nn.Linear(d_model, 1)                 # ΔROA regression
        self.sign_head = nn.Linear(d_model, 1)                # ΔROA sign
        self.risk_heads = nn.ModuleList(
            [nn.Linear(d_model, 1) for _ in range(n_risk)])   # risk subtypes BCE
        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.ret_head.weight)
        nn.init.xavier_uniform_(self.sign_head.weight)
        for h in self.risk_heads:
            nn.init.xavier_uniform_(h.weight)

    def forward(self, x):
        mask = (x != 0)
        h = self.drop(self.embed(x) + self.pos[:, :x.size(1)])
        h = self.tfm(h, src_key_padding_mask=~mask)
        lens = mask.sum(1, keepdim=True).clamp(min=1)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / lens
        ret = self.ret_head(pooled).squeeze(-1)
        sign = self.sign_head(pooled).squeeze(-1)
        risks = [hd(pooled).squeeze(-1) for hd in self.risk_heads]
        return pooled, ret, sign, risks


def train():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    Xtr = np.load(os.path.join(config.DATA, 'enc_ids_train.npy'))
    Xte = np.load(os.path.join(config.DATA, 'enc_ids_test.npy'))
    meta_tr = pd.read_parquet(os.path.join(config.DATA, 'enc_meta_train.parquet'))
    meta_te = pd.read_parquet(os.path.join(config.DATA, 'enc_meta_test.parquet'))

    # ---- targets ----
    # v2 focus: the regression head (and the probe below) target the 1-year ROA
    # CHANGE, not the forward-ROA level. The level is dominated by ROA
    # persistence, so a level-targeted probe mechanically correlates with the
    # G1 control `roa`; the change target is the construct-level question
    # ("does disclosure predict profitability improvement").
    roa_tr = meta_tr['roa_fwd'].values.astype(np.float32)
    droa_tr = (meta_tr['roa_fwd'].values - meta_tr['roa'].values).astype(np.float32)
    mu, sd = droa_tr.mean(), droa_tr.std()
    ret_tr_z = (droa_tr - mu) / sd
    sign_tr = (droa_tr > 0).astype(np.float32)
    # risk binary labels from training-median split (avoid leakage of test)
    risk_labels_tr = {}
    for r in RISK_TYPES:
        med = np.nanmedian(meta_tr[r].values.astype(np.float32))
        risk_labels_tr[r] = (meta_tr[r].values.astype(np.float32) > med).astype(np.float32)

    model = Encoder(len(json.load(open(os.path.join(config.DATA, 'vocab.json')))),
                    config.ENC_EMBED, config.ENC_HEADS, config.ENC_LAYERS,
                    config.ENC_FF, config.ENC_MAX_LEN, n_risk=len(RISK_TYPES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config.ENC_LR, weight_decay=1e-5)
    mse = nn.MSELoss(); bce = nn.BCEWithLogitsLoss()
    Xtr_t = torch.from_numpy(Xtr).long()
    y_r = torch.from_numpy(ret_tr_z); y_s = torch.from_numpy(sign_tr)
    y_rk = {r: torch.from_numpy(risk_labels_tr[r]) for r in RISK_TYPES}

    n = len(Xtr_t); bs = config.ENC_BATCH
    perm = np.random.permutation(n); nval = n // 10
    val_idx, tr_idx = perm[:nval], perm[nval:]
    best_val, wait = 1e9, 0
    for ep in range(config.ENC_EPOCHS):
        model.train()
        for i in range(0, len(tr_idx), bs):
            idx = tr_idx[i:i + bs]
            xb = Xtr_t[idx].to(DEVICE)
            _, rp, sp, rk_p = model(xb)
            loss = (mse(rp, y_r[idx].to(DEVICE))
                    + 0.5 * bce(sp, y_s[idx].to(DEVICE))
                    + 0.3 * sum(bce(rk_p[k], y_rk[RISK_TYPES[k]][idx].to(DEVICE))
                                for k in range(len(RISK_TYPES))))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            xv = Xtr_t[val_idx].to(DEVICE)
            _, rv, _, _ = model(xv)
            val_mse = mse(rv, y_r[val_idx].to(DEVICE)).item()
        print(f"  epoch {ep+1}/{config.ENC_EPOCHS}  val_droa_MSE={val_mse:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if val_mse < best_val - 1e-4:
            best_val = val_mse; wait = 0
            torch.save(model.state_dict(), os.path.join(config.DATA, 'encoder.pt'))
        else:
            wait += 1
            if wait >= 3:
                print("  early stop", flush=True); break
    if not os.path.exists(os.path.join(config.DATA, 'encoder.pt')):
        torch.save(model.state_dict(), os.path.join(config.DATA, 'encoder.pt'))

    # ---- embeddings + neural_score (linear probe, OOF on train) ----
    model.eval()
    def embed(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                xb = torch.from_numpy(X[i:i + 512]).long().to(DEVICE)
                p, _, _, _ = model(xb); out.append(p.cpu().numpy())
        return np.concatenate(out, 0)
    E_tr = embed(Xtr); E_te = embed(Xte)
    np.save(os.path.join(config.DATA, 'emb_train.npy'), E_tr)
    np.save(os.path.join(config.DATA, 'emb_test.npy'), E_te)

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    score_tr = np.zeros(len(E_tr))
    for tr_i, va_i in kf.split(E_tr):
        p = Ridge(alpha=1.0).fit(E_tr[tr_i], droa_tr[tr_i])
        score_tr[va_i] = p.predict(E_tr[va_i])
    probe = Ridge(alpha=1.0).fit(E_tr, droa_tr)
    score_te = probe.predict(E_te)
    meta_tr['neural_score'] = score_tr
    meta_te['neural_score'] = score_te
    out = pd.concat([meta_tr, meta_te], ignore_index=True)
    out.to_parquet(os.path.join(config.DATA, 'neural_features.parquet'), index=False)

    # ---- probe evaluation on the HELD-OUT test window ----
    y_te = meta_te['roa_fwd'].values.astype(np.float32)
    droa_te = (y_te - meta_te['roa'].values.astype(np.float32))
    r2 = 1 - np.sum((droa_te - score_te) ** 2) / np.sum((droa_te - droa_te.mean()) ** 2)
    corr_droa = np.corrcoef(score_te, droa_te)[0, 1]
    corr_roa = np.corrcoef(score_te, meta_te['roa'].values)[0, 1]   # persistence overlap
    corr_ret = np.corrcoef(score_te, meta_te['ret_1y'].values)[0, 1]
    # risk-head AUC on test (sanity: did the encoder learn risk signal?)
    risk_auc = {}
    with torch.no_grad():
        _, _, _, rk_te = model(torch.from_numpy(Xte).long().to(DEVICE))
    for k, r in enumerate(RISK_TYPES):
        from sklearn.metrics import roc_auc_score
        lab = (meta_te[r].values.astype(np.float32) >
               np.nanmedian(meta_te[r].values.astype(np.float32))).astype(int)
        if lab.sum() > 0 and lab.sum() < len(lab):
            risk_auc[r] = float(roc_auc_score(lab, rk_te[k].cpu().numpy()))
    print(f"  [done] embeddings train{E_tr.shape} test{E_te.shape}", flush=True)
    print(f"  neural_score OOS R² on ΔROA (test) = {r2:.4f}", flush=True)
    print(f"  neural_score vs ΔROA corr (test) = {corr_droa:.4f}", flush=True)
    print(f"  neural_score vs current-ROA corr (test) = {corr_roa:.4f}  "
          f"(persistence overlap with G1 control)", flush=True)
    print(f"  neural_score vs stock return corr (test) = {corr_ret:.4f}", flush=True)
    print(f"  risk-head test AUC = {risk_auc}", flush=True)
    json.dump({'neural_score_oos_r2_droa': float(r2),
               'neural_score_corr_droa': float(corr_droa),
               'neural_score_corr_roa': float(corr_roa),
               'neural_score_corr_ret': float(corr_ret),
               'risk_head_auc': risk_auc},
              open(os.path.join(config.DATA, 'transformer_metrics.json'), 'w'), indent=2)
    return out


if __name__ == '__main__':
    import sys
    # Always re-prepare when explicitly asked; otherwise prepare if artifacts
    # are missing OR stale (token length changed since they were written).
    ids_path = os.path.join(config.DATA, 'enc_ids_train.npy')
    stale = True
    if os.path.exists(ids_path):
        try:
            stale = np.load(ids_path, mmap_mode='r').shape[1] != config.ENC_MAX_LEN
        except Exception:
            stale = True
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        prepare()
    else:
        if stale:
            print(f"enc_ids missing or tokenised at a different length "
                  f"(config ENC_MAX_LEN={config.ENC_MAX_LEN}) -> re-preparing",
                  flush=True)
            prepare()
        train()
