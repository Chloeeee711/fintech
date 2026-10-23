"""Generate design/architecture diagrams for PROJECT2_PLAN.md (PDF-safe PNGs)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC',
                                   'Arial Unicode MS', 'Noto Sans CJK SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'outputs', 'design_figures')
os.makedirs(OUT, exist_ok=True)

C = {
    'blue':   ('#E6F1FB', '#185FA5', '#0C447C'),
    'teal':   ('#E1F5EE', '#0F6E56', '#085041'),
    'amber':  ('#FAEEDA', '#854F0B', '#633806'),
    'purple': ('#EEEDFE', '#534AB7', '#3C3489'),
    'coral':  ('#FAECE7', '#993C1D', '#712B13'),
    'green':  ('#EAF3DE', '#3B6D11', '#27500A'),
    'pink':   ('#FBEAF0', '#993556', '#72243E'),
    'gray':   ('#F1EFE8', '#5F5E5A', '#444441'),
    'red':    ('#FCEBEB', '#A32D2D', '#791F1F'),
}


def new_fig(H):
    fig = plt.figure(figsize=(6.8, H / 100.0), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 680)
    ax.set_ylim(H, 0)
    ax.axis('off')
    return fig, ax


def box(ax, x, y, w, h, title, sub=None, color='gray', fs_t=13, fs_s=10.5):
    fill, edge, tc = C[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=6",
                 fc=fill, ec=edge, lw=1.3))
    ax.text(x + w / 2, y + h / 2 - (7 if sub else 0), title, ha='center', va='center',
            fontsize=fs_t, color=tc, weight='bold')
    if sub:
        ax.text(x + w / 2, y + h / 2 + 11, sub, ha='center', va='center',
                fontsize=fs_s, color='#5F5E5A')


def arrow(ax, x1, y1, x2, y2, color='#5F5E5A', rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.6,
                                connectionstyle=f'arc3,rad={rad}'))


def txt(ax, x, y, s, size=12, color='#444441', weight='normal', ha='left', va='center'):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, weight=weight)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, transparent=True)
    plt.close(fig)
    print('wrote', name)


# ---------------------------------------------------------------- Fig 1
def fig1():
    H = 540
    fig, ax = new_fig(H)
    txt(ax, 340, 22, 'End-to-End Research Pipeline', size=15, weight='bold', ha='center', color='#1E3A8A')
    box(ax, 40, 46, 285, 52, 'SEC 10-K Item 7 (MD&A)', '≈110k filings · 2010-2022 · ≤12k chars', 'blue')
    box(ax, 355, 46, 285, 52, 'Compustat Annual', 'ROA · CAPEX · leverage · momentum', 'teal')
    arrow(ax, 340, 98, 340, 118)
    box(ax, 40, 120, 600, 58, 'Preprocessing', 'CIK normalize · 2-pass fiscal-year align · winsorize ROA [-1,1]', 'gray')
    txt(ax, 620, 110, 'Train 2010-2019 then Test 2020-2022 (COVID)', size=10, color='#854F0B', weight='bold', ha='right')
    arrow(ax, 340, 178, 340, 198)
    box(ax, 10, 200, 124, 62, 'LM Tone', 'lexicon sentiment', 'blue', fs_t=11)
    box(ax, 144, 200, 124, 62, 'SSI Rule', '5-dim substance', 'purple', fs_t=11)
    box(ax, 278, 200, 124, 62, 'CPS Rule', 'capex-plan sentences', 'pink', fs_t=11)
    box(ax, 412, 200, 124, 62, 'TF-IDF', '20k features', 'amber', fs_t=11)
    box(ax, 546, 200, 124, 62, 'Transformer', 'from-scratch encoder', 'coral', fs_t=11)
    arrow(ax, 340, 262, 340, 282)
    box(ax, 40, 284, 600, 56, 'Model Matrix G1 → G5', 'incremental feature blocks · Ridge / RandomForest / XGBoost / SVM · time-series CV', 'green')
    arrow(ax, 340, 340, 340, 360)
    box(ax, 40, 362, 190, 54, 'ΔROA', 'primary outcome (ROA t+1 - ROA t)', 'pink')
    box(ax, 245, 362, 190, 54, 'Δ(CAPEX/AT)', 'secondary outcome', 'amber')
    box(ax, 450, 362, 190, 54, '1-yr Return', 'robustness check', 'gray')
    arrow(ax, 340, 416, 340, 436)
    box(ax, 40, 438, 600, 52, 'Validation & Robustness', 'industry x year FE · cluster-robust SE · incremental R2 · long-short Q5-Q1 · ablation · COVID', 'gray', fs_s=10)
    save(fig, 'fig1_pipeline.png')


# ---------------------------------------------------------------- Fig 2
def fig2():
    H = 500
    fig, ax = new_fig(H)
    txt(ax, 340, 22, 'SSI Five-Dimension Construction', size=15, weight='bold', ha='center', color='#3C3489')
    box(ax, 230, 44, 220, 42, 'MD&A text (Item 7)', 'input document', 'gray')
    arrow(ax, 340, 86, 340, 104)
    dims = [
        ('D1 Quantification density', '#numeric tokens / #tokens'),
        ('D2 Project specificity', 'project nouns / keywords'),
        ('D3 Temporal explicitness', 'date / quarter patterns'),
        ('D4 Execution stage', 'plan → progress verbs'),
        ('D5 Evidence markers', 'achieved vs intend verbs'),
    ]
    y = 110
    for i, (t, s) in enumerate(dims):
        box(ax, 40, y, 310, 50, t, s, 'purple' if i % 2 == 0 else 'blue')
        y += 58
    arrow(ax, 350, 235, 410, 235, color='#27500A')
    fill, edge, tc = C['green']
    ax.add_patch(FancyBboxPatch((410, 175), 230, 120, boxstyle="round,pad=0.6,rounding_size=6", fc=fill, ec=edge, lw=1.3))
    txt(ax, 525, 200, 'SSI Index', size=13, color=tc, weight='bold', ha='center')
    txt(ax, 525, 226, 'normalized weighted sum of 5 dims', size=10, color='#5F5E5A', ha='center')
    txt(ax, 525, 246, 'feeds M2 feature and', size=10, color='#5F5E5A', ha='center')
    txt(ax, 525, 266, 'Transformer aux heads', size=10, color='#5F5E5A', ha='center')
    box(ax, 40, 420, 190, 44, 'Demand risk lexicon', 'demand/order/backlog', 'coral', fs_t=10.5)
    box(ax, 245, 420, 190, 44, 'Cost risk lexicon', 'cost/input/margin', 'coral', fs_t=10.5)
    box(ax, 450, 420, 190, 44, 'Liquidity risk lexicon', 'cash/debt/covenant', 'coral', fs_t=10.5)
    txt(ax, 340, 488, 'Risk subtypes → H3 ablation & Transformer risk heads', size=11, color='#5F5E5A', ha='center')
    save(fig, 'fig2_ssi.png')


# ---------------------------------------------------------------- Fig 3
def fig3():
    H = 432
    fig, ax = new_fig(H)
    txt(ax, 340, 20, '4 x 5 Incremental Information Design', size=15, weight='bold', ha='center', color='#27500A')
    txt(ax, 340, 42, '20 core configurations = 4 information sets x 5 model families',
        size=10.5, ha='center', color='#5F5E5A')

    rows = [('G1  Financial baseline', 'size/bm/roa/lev/mom/invest/age', 'gray'),
            ('G2  + Dictionary tone', 'LM lexicon sentiment', 'blue'),
            ('G3  + Rule SSI', '5-dim substance + risk lexicons', 'purple'),
            ('G4  + CPS (v2 core)', 'capital-plan specificity & stage', 'pink')]
    cols = ['Ridge', 'ElasticNet', 'RandomForest', 'XGBoost', 'SVM-RBF']

    x_lab, w_lab = 20, 190
    cx, cw = [216, 308, 400, 492, 584], 88
    y = 60
    ax.add_patch(FancyBboxPatch((x_lab, y), w_lab, 30, boxstyle='round,pad=0',
                                fc='#F1EFE8', ec='#5F5E5A', lw=0.8))
    txt(ax, x_lab + 10, y + 15, 'Information set', size=10.5, weight='bold', color='#444441')
    for x, c in zip(cx, cols):
        ax.add_patch(FancyBboxPatch((x, y), cw, 30, boxstyle='round,pad=0',
                                    fc='#F1EFE8', ec='#5F5E5A', lw=0.8))
        txt(ax, x + cw / 2, y + 15, c, size=8.5, weight='bold', color='#444441', ha='center')
    y = 94
    for lab, sub, col in rows:
        fill, edge, tc = C[col]
        ax.add_patch(FancyBboxPatch((x_lab, y), w_lab, 34, boxstyle='round,pad=0.4,rounding_size=5',
                                    fc=fill, ec=edge, lw=1.2))
        txt(ax, x_lab + 10, y + 13, lab, size=10, weight='bold', color=tc)
        txt(ax, x_lab + 10, y + 26, sub, size=7.5, color='#5F5E5A')
        for x in cx:
            ax.add_patch(FancyBboxPatch((x, y), cw, 34, boxstyle='round,pad=0.3,rounding_size=4',
                                        fc='#FFFFFF', ec='#C9C6BC', lw=0.8))
            ax.add_patch(Circle((x + cw / 2, y + 17), 5, fc=edge, ec='none'))
        y += 40

    # like-for-like note (v2 protocol: full sample, shared folds and budget)
    ax.add_patch(FancyBboxPatch((20, 266), 640, 40, boxstyle='round,pad=0.5,rounding_size=5',
                                fc='#FAEEDA', ec='#854F0B', lw=1.1))
    txt(ax, 340, 280, 'Like-for-like: every cell shares the same complete-case observations, the same year-block folds',
        size=9.5, ha='center', color='#633806', weight='bold')
    txt(ax, 340, 296, 'and the same tuning budget — G1 is the null anchor, so each step isolates the added information.',
        size=9, ha='center', color='#633806')

    # reading guide
    ax.add_patch(FancyBboxPatch((20, 316), 310, 100, boxstyle='round,pad=0.5,rounding_size=5',
                                fc='#E1F5EE', ec='#0F6E56', lw=1.1))
    txt(ax, 32, 332, 'How to read the grid', size=10.5, weight='bold', color='#085041')
    txt(ax, 32, 350, 'row effect = information content', size=9, color='#085041')
    txt(ax, 32, 366, 'column effect = model capacity', size=9, color='#085041')
    txt(ax, 32, 382, 'interaction = does text need nonlinearity?', size=9, color='#085041')
    txt(ax, 32, 400, 'G1->G2 = H1 · G2->G3 = H2 · G3->G4 = C1-C4', size=8.5, color='#085041')

    # extended ladder
    ax.add_patch(FancyBboxPatch((350, 316), 310, 100, boxstyle='round,pad=0.5,rounding_size=5',
                                fc='#FBEAF0', ec='#993556', lw=1.1))
    txt(ax, 362, 332, 'Extended ladder (outside the 20)', size=10.5, weight='bold', color='#72243E')
    txt(ax, 362, 350, 'M5 dynamics: dSSI / dtone (H5)', size=9, color='#72243E')
    txt(ax, 362, 366, 'DL neural_score: end-to-end vs structured', size=9, color='#72243E')
    txt(ax, 362, 382, 'H4 evidence-backed optimism interaction', size=9, color='#72243E')
    txt(ax, 362, 398, 'H6 plan-fulfillment quadrant', size=9, color='#72243E')
    save(fig, 'fig3_matrix.png')


# ---------------------------------------------------------------- Fig 4
def fig4():
    H = 470
    fig, ax = new_fig(H)
    txt(ax, 340, 22, 'Risk-Aware Multi-Task Transformer', size=15, weight='bold', ha='center', color='#712B13')
    box(ax, 220, 40, 240, 38, 'Input token ids (length 256)', None, 'gray', fs_t=12.5)
    arrow(ax, 340, 78, 340, 92)
    box(ax, 170, 94, 340, 44, 'Token + Positional Embedding (96-d)', 'vocab 30k · learned', 'blue')
    arrow(ax, 340, 138, 340, 152)
    box(ax, 150, 154, 380, 44, 'TransformerEncoder ×2 (4 heads, FF 256)', 'self-attention · layer norm · dropout', 'purple')
    arrow(ax, 340, 198, 340, 212)
    box(ax, 200, 214, 280, 44, 'Mean Pooling → 96-d vector', 'contextual representation', 'teal')
    arrow(ax, 480, 236, 520, 236, color='#3B6D11')
    box(ax, 520, 214, 140, 42, 'Ridge OOF probe', 'leak-free', 'green', fs_t=11)
    arrow(ax, 590, 256, 590, 272, color='#3B6D11')
    box(ax, 520, 274, 140, 42, 'neural_score', 'DL signal', 'green', fs_t=11)
    arrow(ax, 340, 258, 340, 274)
    box(ax, 210, 276, 260, 38, 'Multi-Task Heads', 'five outputs', 'coral', fs_t=13)
    hx, hw, hy, hh = [24, 150, 276, 402, 528], 112, 330, 48
    heads = [('ΔROA reg', 'regression'), ('ΔROA sign', 'BCE'), ('risk demand', 'BCE'),
             ('risk cost', 'BCE'), ('risk liq', 'BCE')]
    for x, (t, s) in zip(hx, heads):
        box(ax, x, hy, hw, hh, t, s, 'coral', fs_t=11)
    for x in hx:
        arrow(ax, x + hw / 2, hy + hh, 340, 410, color='#993C1D')
    box(ax, 140, 412, 400, 44, 'Loss = MSE(ΔROA) + 0.5·BCE(sign) + Σ 0.3·BCE(risk_i)', 'multi-task objective', 'gray', fs_t=11.5)
    save(fig, 'fig4_transformer.png')


# ---------------------------------------------------------------- Fig 5
def fig5():
    H = 462
    fig, ax = new_fig(H)
    txt(ax, 340, 20, 'Hypothesis Map & Identification', size=15, weight='bold', ha='center', color='#0C447C')

    # --- left: focused-core hypothesis chips (v2) ---
    hs = [('C1', 'capex-plan specificity -> ΔCAPEX', 'blue'),
          ('C2', 'capex-plan specificity -> ΔROA', 'purple'),
          ('C3', 'implementation-stage gradient', 'coral'),
          ('C4', 'empty-promise penalty (talk≠walk)', 'pink'),
          ('H1-H6', 'legacy SSI / tone benchmark ladder', 'gray'),
          ('DL · RL', 'Transformer probe + REINFORCE layer', 'green')]
    y = 44
    for hid, desc, col in hs:
        box(ax, 24, y, 340, 32, hid + '    ' + desc, None, col, fs_t=11)
        y += 40

    # --- right: identification panel (explicit layout) ---
    fill, edge, tc = C['gray']
    ax.add_patch(FancyBboxPatch((388, 44), 268, 232, boxstyle="round,pad=0.6,rounding_size=6",
                                fc=fill, ec=edge, lw=1.3))
    txt(ax, 522, 66, 'Identification', size=12.5, color=tc, weight='bold', ha='center')
    lines = ['panel OLS', 'industry x year FE', 'firm cluster-robust SE',
             'incremental R2', 'long-short Q5 - Q1', 'ablation', 'COVID stress window']
    ly = 96
    for s in lines:
        txt(ax, 522, ly, s, size=10, color='#5F5E5A', ha='center')
        ly += 24

    # --- bottom: train / test timeline ---
    txt(ax, 340, 302, 'Train / Test Timeline (file_year)', size=12.5, weight='bold', ha='center', color='#444441')
    x0, nyr, dx = 60, 13, 560 / 12.0
    xtr_end, xte0 = x0 + 9.5 * dx, x0 + 9.5 * dx
    ax.add_patch(FancyBboxPatch((x0, 322), xtr_end - x0, 24, boxstyle='round,pad=0', fc='#E6F1FB', ec='#185FA5', lw=1))
    ax.add_patch(FancyBboxPatch((xte0, 322), 626 - xte0, 24, boxstyle='round,pad=0', fc='#FCEBEB', ec='#A32D2D', lw=1))
    txt(ax, (x0 + xtr_end) / 2, 334, 'Train 2010-2019', size=10.5, ha='center', color='#0C447C', weight='bold')
    txt(ax, (xte0 + 626) / 2, 334, 'Test 2020-2022', size=10, ha='center', color='#791F1F', weight='bold')
    for i in range(nyr):
        xx = x0 + i * dx
        ax.plot([xx, xx], [320, 326], color='#444441', lw=0.8)
        txt(ax, xx, 352, str(2010 + i), size=8.5, ha='center', color='#5F5E5A')
    txt(ax, (xte0 + 626) / 2, 370, 'COVID stress', size=9.5, ha='center', color='#A32D2D')
    txt(ax, 340, 404, 'In-sample time-series CV inside train; test is held-out OOS (never used for fitting).',
        size=10, ha='center', color='#5F5E5A')
    save(fig, 'fig5_hypothesis.png')


# ---------------------------------------------------------------- Fig 6
def fig6():
    H = 300
    fig, ax = new_fig(H)
    txt(ax, 340, 22, 'RL Decision Loop (REINFORCE)', size=15, weight='bold', ha='center', color='#185FA5')
    nodes = [('State', 'prev cum, dd, mkt,\nmean SSI, unc, t', 'gray', 28),
             ('Policy net', '2-layer MLP', 'purple', 152),
             ('Action', 'exposure w in [0,1]', 'blue', 276),
             ('Portfolio', 'w x neural long-short', 'teal', 400),
             ('Reward', 'r = ret - 2 x dd^1.5', 'amber', 524)]
    nw, nh, ny = 110, 84, 110
    for t, s, col, x in nodes:
        box(ax, x, ny, nw, nh, t, s, col, fs_t=12)
    for i in range(len(nodes) - 1):
        x1 = nodes[i][3] + nw
        x2 = nodes[i + 1][3]
        arrow(ax, x1, ny + nh / 2, x2, ny + nh / 2)
    arrow(ax, 524 + 55, ny, 28 + 55, ny, color='#3B6D11', rad=-0.32)
    txt(ax, 340, 70, 'REINFORCE update (policy gradient) loops back to next-year state', size=11, ha='center', color='#3B6D11', weight='bold')
    save(fig, 'fig6_rl.png')


# ---------------------------------------------------------------- Fig 7
def fig7():
    H = 480
    fig, ax = new_fig(H)
    txt(ax, 340, 22, 'Three Signature Analyses', size=15, weight='bold', ha='center', color='#72243E')
    # Panel A quadrant
    ax.add_patch(FancyBboxPatch((95, 80), 250, 180, boxstyle='round,pad=0', fc='#FBEAF0', ec='#993556', lw=1))
    ax.plot([220, 220], [80, 260], color='#993556', lw=1)
    ax.plot([95, 345], [170, 170], color='#993556', lw=1)
    txt(ax, 220, 72, 'Plan optimism (SSI plan)', size=11, ha='center', color='#72243E', weight='bold')
    txt(ax, 90, 170, 'Actual\ncapex Δ', size=9, ha='right', va='center', color='#72243E', weight='bold')
    txt(ax, 157, 120, '闷声投资', size=11, ha='center', color='#444441')
    txt(ax, 283, 120, '说到做到', size=11, ha='center', color='#444441', weight='bold')
    txt(ax, 157, 220, '无为', size=11, ha='center', color='#444441')
    txt(ax, 283, 220, '空头支票', size=11, ha='center', color='#A32D2D', weight='bold')
    txt(ax, 220, 300, 'A. Plan-Fulfillment Quadrant (H6)', size=12, ha='center', weight='bold', color='#444441')
    # Panel B interaction
    bx, by, bw, bh = 380, 90, 260, 160
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle='round,pad=0', fc='#E6F1FB', ec='#185FA5', lw=1))
    ax.plot([bx, bx + bw], [by, by], color='#444441', lw=1)
    ax.plot([bx, bx], [by, by + bh], color='#444441', lw=1)
    xl, xr = bx + 20, bx + bw - 20
    ax.plot([xl, xr], [by + bh - 30, by + 30], color='#185FA5', lw=2.5)
    ax.plot([xl, xr], [by + bh - 60, by + bh - 72], color='#993C1D', lw=2.5, linestyle='--')
    txt(ax, xl, by + bh - 8, 'optimism low', size=9, ha='left', color='#5F5E5A')
    txt(ax, xr, by + bh - 8, 'optimism high', size=9, ha='right', color='#5F5E5A')
    txt(ax, bx + 10, by + 10, 'ROA↑', size=9, ha='left', color='#5F5E5A')
    txt(ax, (bx + bx + bw) / 2, by - 12, 'evidence-backed (solid) vs not (dashed)', size=10, ha='center', color='#0C447C')
    txt(ax, 510, 300, 'B. Evidence-Backed Optimism (H4)', size=12, ha='center', weight='bold', color='#444441')
    # Panel C scatter
    cy = 350
    ax.add_patch(FancyBboxPatch((50, cy), 580, 90, boxstyle='round,pad=0', fc='#EAF3DE', ec='#3B6D11', lw=1))
    rng = np.random.default_rng(7)
    pts = rng.uniform(-1.2, 1.2, size=(42, 2))
    pts[:, 1] = 0.6 * pts[:, 0] + 0.28 * rng.normal(0, 1, 42)
    for px, py in pts:
        X = 340 + px * 215
        Y = cy + 45 - py * 22
        ax.scatter(X, Y, s=12, color='#3B6D11', alpha=0.6, zorder=3)
    ax.plot([120, 600], [cy + 58, cy + 30], color='#854F0B', lw=2, zorder=4)
    txt(ax, 340, cy - 12, 'C. ΔSSI → Δ(CAPEX/AT): positive strategic-shift signal (H5)', size=12, ha='center', weight='bold', color='#444441')
    save(fig, 'fig7_signature.png')


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print('ALL DONE ->', OUT)
