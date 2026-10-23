"""
make_result_figures.py — Result exhibits for the report (PROJECT2_PLAN.md §9/I).

Produces, into outputs/figures/:
  fig_ssi_dist.png        SSI distribution (and by tone sign) — shows the new
                          disclosure-substance index is well-behaved / informative
  fig_ablation.png        H3 ablation — R^2 of G3 full vs dropping risk / tone / SSI
  fig_quadrant_h6.png     H6 plan-fulfillment (talk vs walk) 2x2 quadrant
  fig_delta_ssi_capex.png H5 — ΔSSI (this year minus prior) vs ΔCAPEX/AT
  fig_4x4_matrix.png      4x4 incremental-information OOS-R^2 heatmap (G1..G4 x models)
"""
import os, json, numpy as np, pandas as pd, config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = config.FIG
D = config.DATA
plt.rcParams.update({'figure.dpi': 130, 'font.size': 10})


def _load_texts():
    t = pd.read_parquet(os.path.join(D, 'texts.parquet'))
    smp = pd.read_parquet(os.path.join(D, 'analytical_sample.parquet'))
    # only text columns come from texts.parquet — roa_fwd / d_capex_at live in
    # analytical_sample and would collide into _x/_y suffixes if pulled twice.
    cols = ['index', 'ssi', 'tone_lm', 'lag_ssi']
    return smp.merge(t[[c for c in cols if c in t.columns]],
                     left_index=True, right_on='index', how='left')


def fig_ssi_dist(df):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(df['ssi'].dropna(), bins=60, color='#3b6ea5', alpha=0.85)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Disclosure-Substance Index (SSI, z-scored)')
    ax.set_ylabel('Number of filings')
    ax.set_title('Distribution of the rule-based SSI')
    fig.tight_layout(); p = os.path.join(FIG, 'fig_ssi_dist.png'); fig.savefig(p); plt.close(fig)
    return p


def fig_ablation():
    try:
        R = json.load(open(os.path.join(D, 'empirical_results.json')))
        ab = R['ablation_roa_fwd']
    except Exception:
        return None
    labels = ['G3\n(full)', 'drop\nrisk', 'drop\ntone', 'drop\nSSI']
    vals = [ab['G3_full'], ab['drop_risk'], ab['drop_tone'], ab['drop_ssi']]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(labels, vals, color=['#2e7d32', '#f9a825', '#ef6c00', '#c62828'])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('In-sample R² (forward ROA)')
    ax.set_title('H3 Ablation — which G3 component carries the signal?')
    ax.set_ylim(0, max(vals) * 1.15)
    fig.tight_layout(); p = os.path.join(FIG, 'fig_ablation.png'); fig.savefig(p); plt.close(fig)
    return p


def fig_quadrant_h6():
    try:
        z = np.load(os.path.join(D, 'fig_h6_quadrant.npz'), allow_pickle=True)
        plan = z['plan_intent']; walk = z['walk']; roa = z['roa_fwd']; quad = z['quad']
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = {'talk_walk': '#2e7d32', 'talk_nowalk': '#c62828',
            'notalk_walk': '#1565c0', 'notalk_nowalk': '#888888'}
    # display winsorisation only (quadrant assignment uses the unclipped median split)
    plan_c = np.clip(plan, np.nanpercentile(plan, 0.5), np.nanpercentile(plan, 99.5))
    walk_c = np.clip(walk, -4, 4)
    for q in cmap:
        m = quad == q.encode() if isinstance(quad[0], bytes) else quad == q
        if m.sum() == 0:
            continue
        ax.scatter(plan_c[m], walk_c[m], s=6, alpha=0.25, color=cmap[q], label=q)
    ax.axvline(0, color='k', lw=0.7, ls='--'); ax.axhline(0, color='k', lw=0.7, ls='--')
    ax.set_xlabel('TALK  (plan_intent = z(SSI)+z(tone)+plan-flag)')
    ax.set_ylabel('WALK  (z(ΔCAPEX/AT), display-winsorised at ±4)')
    ax.set_title('H6 Plan-fulfillment gap: talk vs walk')
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout(); p = os.path.join(FIG, 'fig_quadrant_h6.png'); fig.savefig(p); plt.close(fig)
    return p


def fig_delta_ssi_capex(df):
    d = df.dropna(subset=['lag_ssi', 'ssi', 'd_capex_at'])
    d_dssi = (d['ssi'] - d['lag_ssi'])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d_dssi, d['d_capex_at'], s=5, alpha=0.2, color='#6a1b9a')
    # trend line
    b, a = np.polyfit(d_dssi, d['d_capex_at'], 1)
    xs = np.linspace(d_dssi.min(), d_dssi.max(), 50)
    ax.plot(xs, a + b * xs, color='k', lw=1.5,
            label=f'slope={b:.4f}')
    ax.set_xlabel('ΔSSI  (this year − prior year)')
    ax.set_ylabel('ΔCAPEX / AT')
    ax.set_title('H5 — Change in disclosure substance vs change in capex')
    ax.legend(fontsize=9)
    fig.tight_layout(); p = os.path.join(FIG, 'fig_delta_ssi_capex.png'); fig.savefig(p); plt.close(fig)
    return p


def fig_4x4_matrix():
    try:
        M = json.load(open(os.path.join(D, 'ml_models_results.json')))
        mat = M['droa_matrix']
    except Exception:
        return None
    models = M.get('models', ['Ridge', 'ENet', 'RF', 'XGB', 'SVM'])
    groups = M.get('groups_used', ['G1', 'G2', 'G3', 'G4'])
    A = np.array([[mat[m][g]['oos_r2'] for g in groups] for m in models])
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(A, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(groups)):
            ax.text(j, i, f'{A[i,j]:.3f}', ha='center', va='center',
                    color='white' if A[i, j] < np.nanmax(A) * 0.7 else 'black', fontsize=9)
    ax.set_title(f'{len(groups)}x{len(models)} incremental-information OOS R² '
                 f'(ΔROA, primary target)')
    ax.set_xlabel('Information set  (G1→G4 adds text layers)')
    ax.set_ylabel('Model family')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); p = os.path.join(FIG, 'fig_4x4_matrix.png'); fig.savefig(p); plt.close(fig)
    return p


def fig_cps_stage():
    """Mean ΔROA / ΔCAPEX by implementation-stage group (C3)."""
    try:
        R = json.load(open(os.path.join(D, 'capex_empirical_results.json')))
        sg = R['C3']['by_stage_group']
    except Exception:
        return None
    stages = sorted(sg.keys())
    labels = [f'stage {s}\n(n={sg[s]["n"]:,})' for s in stages]
    droa = [sg[s]['mean_droa'] for s in stages]
    dcap = [sg[s]['mean_dcapex'] for s in stages]
    x = np.arange(len(stages)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    b1 = ax.bar(x - w / 2, droa, w, color='#2e7d32', label='mean ΔROA')
    b2 = ax.bar(x + w / 2, dcap, w, color='#1565c0', label='mean ΔCAPEX/AT')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Mean (complete-case sample)')
    ax.set_title('C3 — Outcomes by capital-plan implementation stage')
    ax.legend(fontsize=9)
    fig.tight_layout(); p = os.path.join(FIG, 'fig_cps_stage.png'); fig.savefig(p); plt.close(fig)
    return p


def fig_cps_ladder():
    """OOS R² ladder G1→G4 for ΔROA with Clark–West t annotations."""
    try:
        R = json.load(open(os.path.join(D, 'capex_empirical_results.json')))
        cv = R['droa']['cv_oos']
    except Exception:
        return None
    groups = ['G1', 'G2', 'G3', 'G4']
    vals = [cv[g]['oos_r2'] for g in groups]
    cw = [None, cv['CW_G1_vs_G2']['cw_t'], cv['CW_G1_vs_G3']['cw_t'],
          cv['CW_G1_vs_G4']['cw_t']]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(groups, vals, color=['#888888', '#f9a825', '#ef6c00', '#2e7d32'])
    for b, v, t in zip(bars, vals, cw):
        lab = f'{v:.4f}' if v == v else 'n/a'
        if t is not None and t == t:
            lab += f'\nCW t={t:.2f}'
        ax.text(b.get_x() + b.get_width() / 2, v, lab, ha='center', va='bottom', fontsize=8.5)
    ax.set_ylabel('OOS R² (expanding-window year-block CV)')
    ax.set_title('Incremental OOS accuracy for ΔROA (G1→G4)')
    ax.set_ylim(0, max(v for v in vals if v == v) * 1.35)
    fig.tight_layout(); p = os.path.join(FIG, 'fig_cps_ladder.png'); fig.savefig(p); plt.close(fig)
    return p


def fig_cps_quadrant():
    """C4 talk-vs-walk: quadrant mean ΔROA bar chart (+ scatter inset for
    disclosers, where TALK actually varies)."""
    try:
        R = json.load(open(os.path.join(D, 'capex_empirical_results.json')))
        quads = R['C4']['quadrants']
        tnw = R['C4']['talk_nowalk_vs_rest']
    except Exception:
        return None
    order = ['talk_walk', 'talk_nowalk', 'notalk_walk', 'notalk_nowalk']
    labels = ['TALK & WALK\n(specific plan,\ninvested)',
              'TALK, no WALK\n(specific plan,\nno investment)',
              'no TALK & WALK\n(vague mention,\ninvested)',
              'no TALK, no WALK\n(vague mention,\nno investment)']
    vals = [quads[q]['mean_droa'] for q in order]
    ns = [quads[q]['n'] for q in order]
    colors = ['#2e7d32', '#c62828', '#1565c0', '#888888']
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(4), vals, color=colors)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(i, v, f'{v:+.4f}\nn={n:,}', ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=8.5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Mean ΔROA (forward ROA − current ROA)')
    t_txt = f"t={tnw['t']:.2f}" if tnw and tnw.get('t') == tnw.get('t') else 'n/a'
    ax.set_title(f'C4 — Talk vs walk: the "empty promise" quadrant underperforms '
                 f'(vs rest: {t_txt})', fontsize=10)
    fig.tight_layout(); p = os.path.join(FIG, 'fig_cps_quadrant.png'); fig.savefig(p); plt.close(fig)
    return p


def run():
    df = _load_texts()
    out = {}
    out['ssi_dist'] = fig_ssi_dist(df)
    out['ablation'] = fig_ablation()
    out['quadrant_h6'] = fig_quadrant_h6()
    out['delta_ssi_capex'] = fig_delta_ssi_capex(df)
    out['matrix_4x4'] = fig_4x4_matrix()
    out['cps_stage'] = fig_cps_stage()
    out['cps_ladder'] = fig_cps_ladder()
    out['cps_quadrant'] = fig_cps_quadrant()
    print("result figures:", {k: (os.path.basename(v) if v else None) for k, v in out.items()})
    return out


if __name__ == '__main__':
    run()
