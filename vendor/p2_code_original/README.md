# Project 2 — Textual Analysis using LLMs: Capital-Investment Plans & Future Performance

**Topic (v2 focused).** Can the *specificity* and *implementation stage* of
management's **capital-investment plans** disclosed in 10-K *Item 7* (MD&A)
predict **future investment changes (ΔCAPEX)** and **profitability improvement
(ΔROA = ROA_{t+1} − ROA_t)** — and how? The "how" is a transparent,
fully offline, sentence-level rule-extraction pipeline with manual verification
and incremental-information testing against financial, dictionary and generic
rule-based baselines.

**Why this dataset.** The assignment ships a large, real-world corpus:
* SEC 10-K Item-7 plain text (~110k filings, 2010–2025)
* WRDS Compustat Fundamentals (annual)

Everything runs **fully offline** (no API key, no model download) on the
Anaconda Python that already has `torch`, `sklearn`, `xgboost`, `pandas`,
`pyarrow`. The deep-learning block is a from-scratch Transformer (an offline
proxy for an LLM/Qwen risk-extraction layer that cannot run in this sandbox).

## Methods — information ladder + focused construct
1. **Financial controls (G1)** — size, leverage, profitability, growth, capex
   intensity (analytical sample only; result variables never patched).
2. **Dictionary baseline (G2)** — Loughran–McDonald-style finance lexicon
   (`src/lexicon.py`): net tone, uncertainty & litigious ratios.
3. **Generic rule SSI (G3)** — `src/specificity.py`: five transparency
   dimensions (quantification density, project specificity, temporal
   explicitness, execution stage, evidence markers) + three risk-subtype
   lexicons (demand / cost / liquidity).
4. **CPS — Capital-Plan Specificity & stage (G4, the v2 core)** —
   `src/capex_plan.py`: sentence-level extraction restricted to
   *capital-investment-plan* sentences (two-tier topic filter: unconditional
   capital nouns, or construction/expand verbs only when a capital noun follows
   within a 3-word window; exclusion layer removes acquisitions, investment
   securities, IPO/underwriting and SEC-reference false positives). Each hit
   sentence is scored on five dimensions — S1 amount+unit, S2 project count,
   S3 timeline, S4 commitment strength (will/plan=1.0, may/could=0.5),
   S5 ordinal implementation stage (2 vague intent → 3 specific plan →
   4 underway → 5 commissioned/ramping) — aggregated to firm-year
   (`cps_composite`, `cps_spec`, `cps_stage_mean`, `cps_late_share`, …) plus
   lagged versions. A 30-sentence verification sample is written out for
   manual re-checking (`data/capex_plan_verification_sample.csv`).
5. **Deep-learning representation (G5, extended)** — a from-scratch risk-aware
   Transformer (`src/transformer_encoder.py`): multi-task objective (ΔROA
   regression + ΔROA sign + 3 risk BCE heads); an out-of-fold Ridge probe on the
   pooled embedding yields `neural_score`.
6. **Incremental-information matrix (4 sets × 5 families)** — `src/ml_models.py`:
   G1 → G2 → G3 → G4 × {Ridge, ElasticNet, RandomForest, XGBoost, SVM-RBF} = 20
   cells (G5 +neural reported separately), with OOS R²/MAE, AUC/F1/Brier,
   Diebold–Mariano + Benjamini–Hochberg FDR, and 1-SD economic significance.

## Focused hypotheses (`src/capex_empirical.py`)
| # | Hypothesis | Test |
|---|-----------|------|
| C1 | Plan specificity → realised investment (ΔCAPEX) | panel slope + OOS |
| C2 | Plan specificity → profitability improvement (ΔROA) | panel slope + OOS |
| C3 | Stage gradient: signal concentrated in later-stage plans | by-stage cells + interaction |
| C4 | Empty-promise penalty: talk without walk | discloser contrast + gap |

Panel OLS with industry×year FE + firm-clustered SE; expanding-window
year-block CV; Clark–West (nested); DM + BH-FDR; rank-based Q5−Q1 spreads.

## Reproduce
```bash
cd src
python run_all.py          # runs the whole pipeline end-to-end (order below)
```
Individual stages (each writes into `data/`):
```bash
python data_prep.py                 # merge SEC + Compustat -> analytical_sample.parquet
python features_text.py             # read texts, lexicon + SSI + risk features -> texts.parquet
python capex_plan.py                # sentence-level CPS construct + verification sample
python transformer_encoder.py prepare   # vocab + tokenisation
python transformer_encoder.py train     # train encoder -> neural_features.parquet
python empirical.py                 # panel ladder, Clark-West, long-short, ablation
python capex_empirical.py           # C1-C4 focused tests -> capex_empirical_results.json
python ml_models.py                 # 4x4 matrix, DM/BH-FDR, economic significance
python plan_fulfillment.py          # talk-vs-walk
python rl_portfolio.py              # RL agent (SSI + neural books)
python visualize.py                 # main report figures
python make_result_figures.py       # result exhibits
python make_design_figures.py       # design/schematic figures
cd ../report && PYTHONPATH=../src python build_report.py   # -> report.pdf
```

## Submission layout
```
P2_Submission/
  PROJECT2_PLAN.md     research plan (hypotheses, design, identification)
  src/                 all source code (submitted as p2_code.zip)
  data/                intermediate parquet/npz + result JSONs
  outputs/figures/     generated figures
  outputs/design_figures/  design schematics
  report/              report.md -> report.pdf
  p2_code.zip          zipped code + plan + report (the submitted program)
```

## Key results (see report.pdf) — reported honestly
**Focused core (CPS on ΔROA):**
* **Core ladder:** OOS R² G4=0.0021 vs G1=0.0000; Clark–West G1vsG4 t=2.46,
  G3vsG4 t=2.31 — the CPS block adds out-of-sample accuracy beyond financials
  *and* beyond the generic SSI block. Tone alone: t=0.86 n.s.
* **Stage, not verbosity, carries the signal:** panel cps_composite −0.0058
  (t=−1.81) vs cps_stage_mean +0.0038 (t=1.89).
* **C1 not supported** (t=0.75). **C2 not supported as stated** (t=−0.28;
  positive in manufacturing t=1.15, significantly *negative* in services
  t=−2.26); disclosers beat non-disclosers in levels (+0.0028 vs −0.0017).
* **C3 weakly supported** — only the stage-4 cell is positive (t=1.66);
  raw stage means are non-monotonic (stage-5 firms commissioned *after* the
  investment peak), so the controlled panel coefficient is the right object.
* **C4 SUPPORTED (headline):** talk-no-walk vs rest β=−0.0160, t=−3.32;
  continuous gap t=−2.80.
* **Dictionary sentiment runs the wrong way** on ΔROA (spread −0.0155,
  t=−2.44) — a cautionary result for lexicon-based screens.
* **4×5 (ΔROA):** G4 > G1 in 3/5 model families; XGBoost gains +0.0227 OOS R²
  from G3→G4 (nonlinear stage×specificity interaction signature);
  ElasticNet's L1 keeps exactly one feature (`roa`) and zeroes every text
  coefficient — the signal is not linearly recoverable;
  BH-FDR 4/20 significant; economic spread 13.6pp.

**Legacy blocks (robustness / boundaries):**
* Generic rule SSI (H2) supported at block level (G3 vs G1 CW t=2.44);
  ΔSSI→ΔCAPEX (H5) not supported; talk-vs-walk (H6) partially supported.
* Transformer: risk heads AUC up to 0.83 (demand head improved vs the
  level-targeted version), but once the probe is retargeted off ROA persistence
  (corr with current ROA 0.72 → 0.07) it carries **no** OOS ΔROA signal
  (R² ≈ −0.16) — reported as a boundary, not tuned away.
* RL: Sharpe unchanged, maximum drawdown roughly halved — a risk-management
  layer, not a return-enhancement layer.
