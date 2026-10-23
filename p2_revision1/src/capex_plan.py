"""
Phase B2: Capital-investment-plan (CAPEX-PLAN) construct.

Core topic: "Can the specificity and implementation stage of management's
capital-investment plans predict future investment changes and
profitability improvement?"

Design (sentence-level, fully offline, rule-based):
  1. Keep Item-7 sentences whose topic is *organic* capital investment
     (construction, expansion, new facilities, capacity additions).
     Exclude M&A / financial-investment language ("invested $X to acquire",
     "investment securities") so the construct is about capex plans, not
     deals.
  2. Score each kept sentence on:
       - amount specificity    ($ figure with explicit unit)
       - count specificity     (N facilities / plants / lines)
       - timeline specificity  (fiscal year / quarter horizon)
       - commitment strength   (will / plan to = 1.0; may / could = 0.5)
       - implementation stage  (2 vague intent, 3 specific plan,
                                4 underway, 5 completed / ramping)
  3. Aggregate to firm-year, then build lagged versions (same firm, prior
     fiscal year) for delta scores.

Data rules (project convention, per spec):
  - Never patch Y; never drop a whole firm for a one-year gap.
  - Lag merge aggregates the prior-year frame to one row per
    (cik_n, prior_year) BEFORE merging -> no fan-out duplicate index.
  - Delta features are NaN whenever either year's text is missing.

Output: appends cps_* / lag_cps_* columns to data/texts.parquet and writes
a 100-sentence manual-verification sample (50 topic-positive + 50 topic-negative,
dual-annotator columns) to data/.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ------------------------------------------------------------------ #
#  Sentence splitter
# ------------------------------------------------------------------ #
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# C-speed character classes for the table-dump filter (a Python char loop
# over ~50k documents is the hot spot of this stage)
_ALPHA_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")

# ------------------------------------------------------------------ #
#  Topic filter: organic capital-investment language (deliberately narrow)
#  Two tiers:
#   TIER A - unambiguous capex nouns (capital expenditure, new plant, ...)
#   TIER B - action verbs (construct / expand) ONLY when a capex noun
#            follows within a 3-word window, so generic business language
#            ("expand our business", "construction companies") is excluded.
# ------------------------------------------------------------------ #
_CAPEX_NOUN = (r"(?:capacit\w*|facilit\w*|plants?|fabs?|factor\w+|production|"
               r"manufactur\w*|footprint|warehous\w*|campuses|campus|"
               r"(?:data|distribution)\s+centers?|retail\s+stores?|stores?|"
               r"operations?|infrastructure)")
_TOPIC = re.compile(r"""
    capital\s+(?:expenditure|investment|spending|project|improvement|plan|outlay|resource) |
    new\s+(?:facilit\w+|plants?|fabs?|factor\w+|warehous\w+|campuses|campus|sites?) |
    (?:manufacturing|production|fabrication)\s+(?:facilit\w+|plants?|capacit\w*) |
    capacity\s+(?:expansion|addition|additions|increase) |
    (?:add|additional|increase|increasing)\s+(?:\w+\s+){0,2}capacit\w+ |
    build[\s-]*out | groundbreak\w* |
    (?:will|to)\s+commission\b | \bcommissioned\b |
    \bcommissioning\s+(?:of|for)\b |
    ramped?\s+(?:up|production) | placed\s+(?:in|into)\s+service |
    fabrication\s+facilit\w+ | \bfab\b |
    (?:construction|construct\w*|expand\w*|expansion)\W+(?:\w+\W+){0,3}?""" + _CAPEX_NOUN + r""" |
    under\s+construction
""", re.I | re.X)

#  Sentences matching these are NOT organic capex plans (M&A / financial
#  investment / regulatory references / IPO proceeds) even if they contain
#  "investment" or "commission".
_EXCLUDE = re.compile(r"""
    acqui(?:re|red|ring|sition|sitions) | merger | divest\w* |
    available[\s-]for[\s-]sale | trading\s+securit\w+ |
    investment\s+securit\w+ | equity\s+method | goodwill |
    securities\s+and\s+exchange\s+commission | exchange\s+commission |
    initial\s+public\s+offering | \bIPO\b | underwrit\w+
""", re.I | re.X)

# ------------------------------------------------------------------ #
#  Specificity markers
#  NOTE: the text pipeline strips '$' and '%' symbols (e.g. "$1.2 billion"
#  -> "1.2 billion", "30%" -> "30"), so amount patterns must NOT require a
#  dollar sign; the unit word is required to avoid matching bare years.
# ------------------------------------------------------------------ #
_AMOUNT = re.compile(r"\b\d[\d,.]*\s*(?:billion|million|thousand)\b"
                     r"(?!\s+(?:units?|products?|customers?|shares?|employees|"
                     r"people|square|stores?|locations?|devices?|vehicles?|"
                     r"barrels?|tons?|gallons?|users?|subscribers?|members?))", re.I)
_AMOUNT_VAL = re.compile(r"\b(\d[\d,.]*)\s*(billion|million|thousand)\b"
                         r"(?!\s+(?:units?|products?|customers?|shares?|employees|"
                         r"people|square|stores?|locations?|devices?|vehicles?|"
                         r"barrels?|tons?|gallons?|users?|subscribers?|members?))", re.I)
_COUNT = re.compile(r"\b(?:two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
                    r"(?:new\s+)?(?:facilit\w+|plants?|fabs?|lines?|sites?|"
                    r"campuses|warehouses?|projects?|units?)\b", re.I)
_COUNT_MULTI = re.compile(r"\b(?:several|multiple|numerous|various)\s+"
                          r"(?:new\s+)?(?:facilit\w+|plants?|fabs?|sites?|projects?)\b", re.I)
_TIMELINE = re.compile(r"""
    \b(?:by|in|during|over|through|throughout|starting|beginning)\s+
    (?:the\s+)?(?:Q[1-4]|first|second|third|fourth)(?:\s+quarter)?\s*
    (?:of\s+)?(?:fiscal\s+)?20\d\d |
    \b(?:by|in|during|over|through)\s+(?:the\s+)?(?:fiscal\s+)?20\d\d |
    \b(?:by|in|during|over|through)\s+the\s+(?:next|coming|following)\s+
    (?:fiscal\s+)?(?:year|two|three|four|five)\s+years? |
    \b(?:by|in|during|over|through)\s+the\s+(?:next|coming|following)\s+\w+\s+months?
""", re.I | re.X)
_COMMIT_STRONG = re.compile(r"\b(?:will|plans?\s+to|planning\s+to|intends?\s+to|"
                             r"expects?\s+to|anticipates?\s+to|is\s+committed|"
                             r"are\s+committed|committed\s+to)\b", re.I)
_COMMIT_WEAK = re.compile(r"\b(?:may|might|could|would|seeks?\s+to|hopes?\s+to|"
                          r"targets?|aims?\s+to)\b", re.I)

# ------------------------------------------------------------------ #
#  Implementation stage
# ------------------------------------------------------------------ #
_STAGE5 = re.compile(r"""
    \bcommissioned\b | \bcommissioning\s+(?:of|for)\b |
    completed\s+(?:the\s+)?(?:construction|build[\s-]*out|expansion|facility|
                            plant|project|investment) |
    began\s+(?:production|operations|manufacturing|construction|ramp\w*) |
    ramped?\s+up | placed\s+(?:in|into)\s+service | started\s+production |
    commercial\s+production\s+began | entered\s+(?:production|service)
""", re.I | re.X)
_STAGE4 = re.compile(r"""
    \bin\s+progress | under\s+(?:construction|way|development|build) |
    phase\s+[IVX\d]+ | \b\d{1,3}\s+(?:percent\s+)?(?:complete|constructed|built) |
    continuing\s+to\s+(?:build|construct|expand|invest|ramp) |
    remains?\s+on\s+(?:track|schedule|budget) |
    currently\s+(?:building|constructing|expanding)
""", re.I | re.X)

_UNIT = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


def _sentence_stage(s, amt=None, cnt=None, tl=None):
    """2 = vague intent, 3 = specific plan, 4 = underway, 5 = completed.

    amt/cnt/tl are the marker flags already computed by `_sentence_spec`
    (avoids re-running the same regex searches on every hit sentence).
    """
    if _STAGE5.search(s):
        return 5
    if _STAGE4.search(s):
        return 4
    if amt is None:
        amt = _AMOUNT.search(s) is not None
    if cnt is None:
        cnt = (_COUNT.search(s) or _COUNT_MULTI.search(s)) is not None
    if tl is None:
        tl = _TIMELINE.search(s) is not None
    if amt or cnt or tl:
        return 3
    return 2


def _sentence_spec(s):
    """Mean of [amount, count, timeline, commitment] markers in [0,1]."""
    amt = 1.0 if _AMOUNT.search(s) else 0.0
    cnt = 1.0 if (_COUNT.search(s) or _COUNT_MULTI.search(s)) else 0.0
    tl = 1.0 if _TIMELINE.search(s) else 0.0
    if _COMMIT_STRONG.search(s):
        com = 1.0
    elif _COMMIT_WEAK.search(s):
        com = 0.5
    else:
        com = 0.0
    return (amt + cnt + tl + com) / 4.0, amt, cnt, tl, com


def _sentence_amount_value(s):
    """Largest explicit unit amount in the sentence (0 if none)."""
    best = 0.0
    for m in _AMOUNT_VAL.finditer(s):
        val = float(m.group(1).replace(",", "")) * _UNIT[m.group(2).lower()]
        best = max(best, val)
    return best


def _artefact_ok(sent):
    """False for page-header / table-of-contents / table-dump artefacts."""
    n = len(sent)
    alpha = len(_ALPHA_RE.findall(sent))
    digits = len(_DIGIT_RE.findall(sent))
    return not (alpha < 0.55 * n or "table of contents" in sent.lower()
                or digits > 0.25 * n or "---" in sent)


def score_text_capex_plan(text):
    """Score one MD&A text; return flat dict of firm-year capex-plan features."""
    out = dict(cps_flag=0, cps_n_sent=0, cps_spec=0.0, cps_spec_max=0.0,
               cps_stage_max=0, cps_stage_mean=0.0, cps_late_share=0.0,
               cps_amt_log1p=0.0, cps_amt_n=0, cps_commit_share=0.0,
               cps_tl_share=0.0)
    if not isinstance(text, str) or not text:
        return out
    specs, stages, amts, commits, tls = [], [], [], [], []
    for sent in _SENT_SPLIT.split(text):
        if len(sent) < 25 or not _artefact_ok(sent):
            continue
        if not _TOPIC.search(sent):
            continue
        if _EXCLUDE.search(sent):
            continue
        spec, amt, cnt, tl, com = _sentence_spec(sent)
        specs.append(spec)
        stages.append(_sentence_stage(sent, amt=amt, cnt=cnt, tl=tl))
        amts.append(_sentence_amount_value(sent))
        commits.append(com)
        tls.append(tl)
    if not specs:
        return out
    specs = np.asarray(specs)
    stages = np.asarray(stages)
    amts = np.asarray(amts)
    commits = np.asarray(commits)
    tls = np.asarray(tls)
    out.update(
        cps_flag=1,
        cps_n_sent=int(len(specs)),
        cps_spec=float(specs.mean()),
        cps_spec_max=float(specs.max()),
        cps_stage_max=int(stages.max()),
        cps_stage_mean=float(stages.mean()),
        cps_late_share=float((stages >= 4).mean()),
        cps_amt_log1p=float(np.log1p(amts.max())),
        cps_amt_n=int((amts > 0).sum()),
        cps_commit_share=float((commits > 0).mean()),
        cps_tl_share=float(tls.mean()),
    )
    return out



def build():
    """Revision 1: preserve raw scores, fit composites on train, exact prior FY."""
    import json
    from pathlib import Path
    from preprocessing_v1 import FORMAL, train_zscores, exact_lags, sha256
    latest = json.loads((FORMAL / 'p2_revision1' / 'latest_run.json').read_text())
    run = Path(latest['run_dir'])
    path = run / 'data' / 'texts.parquet'
    if not path.exists():
        raise RuntimeError('Verified text preprocessing must finish first.')
    if (run / 'data' / 'cps_transform_state.json').exists():
        raise RuntimeError('CPS already generated; use a fresh run to preserve results.')
    tx = pd.read_parquet(path)
    stale = [c for c in tx if c.startswith('cps_') or c.startswith('lag_cps')]
    tx = tx.drop(columns=stale)
    fx = pd.DataFrame([score_text_capex_plan(t) for t in tx.text], index=tx.index)
    # Rule non-detection is not evidence of zero actual investment or zero budget.
    conditional = [c for c in fx if c not in ['cps_flag', 'cps_n_sent']]
    fx.loc[fx.cps_flag.eq(0), conditional] = np.nan
    tx = pd.concat([tx, fx], axis=1)
    dims = ['cps_spec', 'cps_stage_mean', 'cps_late_share', 'cps_amt_n']
    z, state = train_zscores(tx, dims)
    for c in dims:
        tx[c + '_z'] = z[c]
    tx['cps_composite'] = z.mean(axis=1)
    tx = exact_lags(tx, ['cps_composite', 'cps_spec', 'cps_stage_max',
                         'cps_amt_log1p', 'cps_flag'])
    tx.to_parquet(path, index=False)
    (run / 'data' / 'cps_transform_state.json').write_text(json.dumps(state, indent=2))
    # Reservoir sample from ALL documents, including rule-negative documents.
    # Equal stratified counts are for error diagnosis, not population recall.
    target = run / 'data' / 'capex_plan_verification_sample.csv'
    if target.exists():
        raise RuntimeError('Never overwrite an annotation sheet.')
    rng = np.random.default_rng(config.SEED)
    pools, seen = {0: [], 1: []}, {0: 0, 1: 0}
    for row in tx.itertuples():
        for sentence_id, sent in enumerate(_SENT_SPLIT.split(row.text)):
            if len(sent) < 25 or not _artefact_ok(sent):
                continue
            topic = int(bool(_TOPIC.search(sent)) and not bool(_EXCLUDE.search(sent)))
            record = dict(accession_key=row.accession_key, gvkey=row.gvkey,
                          fyear=row.fyear, sentence_id=sentence_id, sentence=sent,
                          rule_topic=topic)
            if topic:
                spec, amount_present, cnt, tl, com = _sentence_spec(sent)
                record.update(rule_spec=spec, rule_amount_present=amount_present,
                              rule_amount_value_heuristic=_sentence_amount_value(sent),
                              rule_stage=_sentence_stage(sent, amt=amount_present, cnt=cnt, tl=tl))
            seen[topic] += 1
            if len(pools[topic]) < 50:
                pools[topic].append(record)
            else:
                pos = int(rng.integers(seen[topic]))
                if pos < 50:
                    pools[topic][pos] = record
    records = pools[0] + pools[1]
    rng.shuffle(records)
    annotation = pd.DataFrame(records)
    for col in ['a1_topic', 'a1_amount_ok', 'a1_stage', 'a1_note',
                'a2_topic', 'a2_amount_ok', 'a2_stage', 'a2_note']:
        annotation[col] = ''
    annotation.to_csv(target, index=False)
    (run / 'data' / 'annotation_sampling.json').write_text(json.dumps({
        'sentence_population': seen, 'sampling': 'stratified reservoir up to 50 per rule class',
        'warning': 'Do not report unweighted population recall from this diagnostic sample.'}, indent=2))
    with (run / '运行记录.md').open('a') as out:
        out.write('\n## CPS预处理\n\n复合分数仅训练集标准化，上一年精确滞后；未检出主题的条件分数设缺失。标注从全部文档分层抽句，保留完整句子、来源与总体数量；金额启发值尚未人工核实。\n')
    files = {str(f.relative_to(run)): sha256(f) for f in (run / 'data').glob('*') if f.is_file()}
    (run / 'cps_output_sha256.json').write_text(json.dumps(files, indent=2))
    return tx

if __name__ == '__main__':
    build()
