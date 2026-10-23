"""
verification_metrics.py — Rule-extractor quality gate (step 4 of the pipeline).

Reads data/capex_plan_verification_sample.csv (100 sentences: 50 rule-positive
+ 50 rule-negative, dual-annotator columns a1_* / a2_*) and computes:

  * topic filter   precision / recall / F1 + TP/FP/FN/TN vs the human label
  * amount check   accuracy of the extracted dollar amount (rows where the
                   rule claims an amount and the annotator judged it)
  * stage rubric   exact-match rate + quadratic-weighted Cohen's kappa
  * inter-annotator agreement (Cohen's kappa) once a2_* is filled in

Label semantics (annotator instructions):
  a*_topic       1 = sentence genuinely describes THIS FIRM's own capital-
                  investment plan (project / amount / timeline / stage of a
                  capex-like investment); 0 = it does not (acquisition,
                  investment securities, industry description, opex, …)
  a*_amount_ok   1 = rule_amount correctly reflects a dollar amount stated in
                  the sentence; 0 = wrong number / wrong unit / no amount;
                  blank when rule_amount is empty
  a*_stage       human stage 2-5 per the CPS rubric (topic=1 rows only):
                  2 vague intent · 3 specific plan · 4 underway · 5 commissioned

NOTE: a Qwen-vs-rule consistency metric is NOT computable in this offline
sandbox (no API); rule-vs-human is the operative quality gate and is reported
honestly as such.
"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
import config


def _num(s):
    return pd.to_numeric(s.replace('', np.nan), errors='coerce')


def run():
    path = os.path.join(config.DATA, 'capex_plan_verification_sample.csv')
    vs = pd.read_csv(path, dtype={'a1_topic': str, 'a2_topic': str,
                                  'a1_amount_ok': str, 'a2_amount_ok': str,
                                  'a1_stage': str, 'a2_stage': str})
    out = {'n_sentences': int(len(vs)),
           'n_rule_positive': int(vs['rule_topic'].sum()),
           'n_rule_negative': int((vs['rule_topic'] == 0).sum())}

    for who in ['a1', 'a2']:
        topic = _num(vs[f'{who}_topic'])
        filled = int(topic.notna().sum())
        if filled == 0:
            out[who] = {'n_labeled': 0,
                        'note': 'annotator columns empty — not yet labelled'}
            continue
        y = topic.dropna().astype(int).values
        r = vs.loc[topic.notna(), 'rule_topic'].astype(int).values
        tp = int(((r == 1) & (y == 1)).sum()); fp = int(((r == 1) & (y == 0)).sum())
        fn = int(((r == 0) & (y == 1)).sum()); tn = int(((r == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float('nan')
        rec = tp / (tp + fn) if tp + fn else float('nan')
        f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec
                                               and prec + rec) else float('nan')
        d = {'n_labeled': filled, 'topic': {
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'precision': prec, 'recall': rec, 'F1': f1}}

        # amount accuracy (only where the rule claims an amount)
        am = _num(vs[f'{who}_amount_ok'])
        m = am.notna() & vs['rule_amount'].notna()
        if m.sum():
            ok = am[m].astype(int).values
            d['amount'] = {'n_judged': int(m.sum()),
                           'accuracy': float(ok.mean()),
                           'n_wrong': int((ok == 0).sum())}
        # stage rubric (topic=1 rows only)
        st = _num(vs[f'{who}_stage'])
        m2 = st.notna() & vs['rule_stage'].notna() & (topic == 1)
        if m2.sum() >= 5:
            h = st[m2].astype(int).values
            ru = vs.loc[m2, 'rule_stage'].astype(int).values
            d['stage'] = {'n_judged': int(m2.sum()),
                          'exact_match': float((h == ru).mean()),
                          'kappa_quadratic': float(cohen_kappa_score(
                              h, ru, weights='quadratic'))}
        out[who] = d

    # inter-annotator agreement (both filled)
    t1, t2 = _num(vs['a1_topic']), _num(vs['a2_topic'])
    both = t1.notna() & t2.notna()
    if both.sum() >= 10:
        out['inter_annotator'] = {
            'n_both': int(both.sum()),
            'topic_agreement': float((t1[both].astype(int) ==
                                      t2[both].astype(int)).mean()),
            'topic_kappa': float(cohen_kappa_score(t1[both].astype(int).values,
                                                   t2[both].astype(int).values))}
    else:
        out['inter_annotator'] = {'n_both': int(both.sum()),
                                  'note': 'second annotator not yet filled'}

    op = os.path.join(config.DATA, 'verification_metrics.json')
    json.dump(out, open(op, 'w'), indent=2)
    print(json.dumps(out, indent=2))
    print(f"  [save] {op}")
    return out


if __name__ == '__main__':
    run()
