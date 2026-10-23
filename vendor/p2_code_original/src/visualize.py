"""
visualize.py — All figures for the Project-2 report.

Reads cached results and renders (everything is primary-outcome aware):
  1. fig_tone_time      Average disclosure tone (LM dictionary vs neural) over time.
  2. fig_quintile_roa   Mean forward ROA by tone quintile (PRIMARY monotonicity).
  3. fig_quintile_ret   Mean forward stock return by tone quintile (robustness).
  4. fig_netvalue       Long–short net-value curve (return-based trading view).
  5. fig_pca            PCA of transformer embeddings, coloured by ΔROA sign.
  6. fig_ml_auc         Out-of-sample AUC of ΔROA (profitability) sign prediction.
  7. fig_rl / exposure  RL dynamic-allocation vs naïve long–short.
"""
import os, json, numpy as np, pandas as pd, config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi':130, 'font.size':10, 'axes.grid':True,
                     'grid.alpha':0.3, 'figure.autolayout':True})
BLUE='#1f4e79'; ORANGE='#c55a11'; GREEN='#2e7d32'; GREY='#666666'

def _load():
    t = pd.read_parquet(os.path.join(config.DATA,'texts.parquet'))
    nf = pd.read_parquet(os.path.join(config.DATA,'neural_features.parquet'))
    # neural_features carries index, file_year, ret_1y, roa_fwd, roa, tone_lm,
    # neural_score, split; pull sic2 + ssi (absent there) from texts.
    df = nf.merge(t[['index','sic2','ssi']], on='index', how='left')
    df['droa'] = df['roa_fwd'] - df['roa']
    return t, nf, df

def fig_tone_time(df):
    g = df.groupby('file_year').agg(tone_lm=('tone_lm','mean'),
                                   neural=('neural_score','mean')).reset_index()
    # df already carries file_year / tone_lm / neural_score from neural_features.
    fig, ax = plt.subplots(figsize=(7,3.2))
    ax.plot(g['file_year'], g['tone_lm'], '-o', color=BLUE, label='LM dictionary tone')
    ax2 = ax.twinx()
    ax2.plot(g['file_year'], g['neural'], '-s', color=ORANGE, label='Neural (DL) score')
    ax.set_xlabel('Filing year'); ax.set_ylabel('LM tone (pos−neg)/(pos+neg)', color=BLUE)
    ax2.set_ylabel('Neural score (standardised)', color=ORANGE)
    ax.set_title('Average disclosure tone over time')
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(config.FIG,'fig_tone_time.png')); plt.close(fig)

def _quintile(df, col, target, title, fname, ylabel):
    d = df.copy()
    d['q'] = d.groupby('file_year')[col].transform(lambda s: pd.qcut(s,5,labels=False,duplicates='drop'))
    g = d.groupby('q')[target].mean()
    fig, ax = plt.subplots(figsize=(5,3.2))
    ax.bar(g.index+1, g.values, color=BLUE)
    ax.axhline(0,color='k',lw=0.8)
    ax.set_xlabel(f'{title} quintile (1=low, 5=high)')
    ax.set_ylabel(ylabel); ax.set_title(f'{title}: mean {target}')
    fig.tight_layout(); fig.savefig(os.path.join(config.FIG,fname)); plt.close(fig)

def fig_quintile_roa(df):
    _quintile(df, 'tone_lm', 'roa_fwd', 'LM dictionary tone', 'fig_quintile_roa.png',
              'Mean forward ROA')
    _quintile(df, 'ssi', 'roa_fwd', 'Rule SSI (disclosure substance)', 'fig_quintile_ssi_roa.png',
              'Mean forward ROA')
    _quintile(df, 'neural_score', 'roa_fwd', 'Neural (DL) score', 'fig_quintile_roa_neural.png',
              'Mean forward ROA')

def fig_quintile_ret(df):
    _quintile(df,'tone_lm','ret_1y','LM dictionary tone','fig_quintile_ret.png',
              'Mean next-year return')
    _quintile(df,'neural_score','ret_1y','Neural (DL) score','fig_quintile_ret_neural.png',
              'Mean next-year return')

def fig_netvalue(df):
    for col, name in [('ssi','Rule SSI (disclosure substance)'),
                      ('tone_lm','LM dictionary tone'),('neural_score','Neural (DL) score')]:
        if col not in df.columns:
            continue
        d = df.copy()
        d['q']=d.groupby('file_year')[col].transform(lambda s: pd.qcut(s,5,labels=False,duplicates='drop'))
        ser = d.groupby('file_year').apply(
            lambda g: g.loc[g['q']==4,'ret_1y'].mean()-g.loc[g['q']==0,'ret_1y'].mean())
        nv = (1+ser.fillna(0)).cumprod()
        fig, ax = plt.subplots(figsize=(7,3.2))
        ax.plot(nv.index, nv.values, '-o', color=GREEN, label=f'{name} Q5−Q1')
        ax.axhline(1,color='k',lw=0.8); ax.set_ylabel('Growth of $1')
        ax.set_xlabel('Filing year'); ax.set_title(f'Long–short net value: {name}')
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(config.FIG,f'fig_netvalue_{col}.png')); plt.close(fig)

def fig_pca(df):
    E = np.load(os.path.join(config.DATA,'emb_train.npy'))
    from sklearn.decomposition import PCA
    p = PCA(n_components=2).fit_transform(E)
    y = df[df['split']=='train']['droa'].values
    sign = (y>0).astype(int)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(p[sign==1,0], p[sign==1,1], s=3, alpha=0.25, color=GREEN, label='ΔROA > 0 (profitability improves)')
    ax.scatter(p[sign==0,0], p[sign==0,1], s=3, alpha=0.25, color=ORANGE, label='ΔROA < 0')
    ax.set_title('PCA of transformer document embeddings'); ax.legend()
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    fig.tight_layout(); fig.savefig(os.path.join(config.FIG,'fig_pca.png')); plt.close(fig)

def fig_ml_auc():
    """OOS AUC (ΔROA sign) per family x information set from the 4x4 matrix."""
    p = os.path.join(config.DATA, 'ml_models_results.json')
    if not os.path.exists(p):
        return
    res = json.load(open(p))
    mat = res.get('roa_fwd_matrix') or {}
    if not mat:
        return
    fams = list(mat.keys())
    groups = [g for g in ['G1', 'G2', 'G3', 'G4'] if g in mat[fams[0]]]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.8 / len(groups)
    colors = [GREY, BLUE, ORANGE, GREEN]
    for j, g_ in enumerate(groups):
        vals = [mat[f].get(g_, {}).get('auc', float('nan')) for f in fams]
        ax.bar(np.arange(len(fams)) + j * w, vals, width=w,
               color=colors[j % len(colors)], label=g_)
    ax.axhline(0.5, color='k', ls='--', lw=0.8)
    ax.set_xticks(np.arange(len(fams)) + 0.4 - w / 2); ax.set_xticklabels(fams)
    ax.set_ylim(0.45, 0.75)
    ax.set_ylabel('Out-of-sample AUC (ΔROA>0, 2020–2022)')
    ax.set_title('Classification accuracy by information set × model family')
    ax.legend(fontsize=8, ncol=4); fig.tight_layout()
    fig.savefig(os.path.join(config.FIG, 'fig_ml_auc.png')); plt.close(fig)

def fig_rl():
    """RL agent vs naive long-short, per signal book (SSI primary, neural comparison)."""
    p = os.path.join(config.DATA, 'rl_results.json')
    if not os.path.exists(p):
        return
    R = json.load(open(p))
    books = {k: v for k, v in R.items() if isinstance(v, dict) and 'years' in v}
    for sig, r in books.items():
        yrs = r['years']; exps = r['agent_exps']; ls = r['ls_year']
        cum = 0; nv_a = [1]; nv_n = [1]; ca = 0; cn = 0
        for i in range(len(yrs)):
            ca += exps[i] * ls[i]; cn += ls[i]
            nv_a.append(1 + ca); nv_n.append(1 + cn)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.plot(yrs, nv_a[1:], '-o', color=GREEN, label='RL agent (dynamic exposure)')
        ax.plot(yrs, nv_n[1:], '-s', color=GREY, label='Naïve full long–short')
        ax.set_ylabel('Growth of $1'); ax.set_xlabel('Filing year')
        ax.set_title(f'RL dynamic allocation vs naïve long–short — {r.get("description", sig)}')
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(config.FIG, f'fig_rl_{sig}.png')); plt.close(fig)
        # exposure
        fig2, ax2 = plt.subplots(figsize=(7, 2.6))
        ax2.bar(yrs, exps, color=BLUE); ax2.set_ylabel('Exposure')
        ax2.set_xlabel('Filing year'); ax2.set_title(f'RL agent annual exposure — {sig} book')
        ax2.set_ylim(0, 1.05); fig2.tight_layout()
        fig2.savefig(os.path.join(config.FIG, f'fig_rl_exposure_{sig}.png')); plt.close(fig2)

def run():
    t, nf, df = _load()
    fig_tone_time(df)
    fig_quintile_roa(df)
    fig_quintile_ret(df)
    fig_netvalue(df)
    fig_pca(df)
    fig_ml_auc()
    fig_rl()
    print("figures written to", config.FIG)

if __name__ == '__main__':
    run()
