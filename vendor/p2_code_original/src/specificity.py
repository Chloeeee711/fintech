"""
specificity.py — Rule-based SSI (disclosure-substance index) and risk lexicons.

Offline, deterministic, no model. Implements the five dimensions of
"substantive, verifiable, concrete" disclosure from PROJECT2_PLAN.md §4.2:
  D1  quantification density      (numbers with explicit units)
  D2  project specificity          (concrete asset / project / location nouns)
  D3  temporal explicitness        (dated, quarter-anchored horizons)
  D4  execution stage              (plan -> approve -> construct -> near -> complete)
  D5  evidence markers             (achieved past vs forward-looking language)

Plus three risk-subtype lexicons (H3 ablation + transformer auxiliary heads)
and a few semantic-disambiguation guard flags (§4.6). All regexes are
pre-compiled once at import time (this matters: the scorer runs over ~53k
documents, and on-the-fly recompilation was the bottleneck).

Each dimension returns a per-document RAW score; features_text.py z-scores the
five density-style dims across the corpus and averages them into `ssi`.
"""
import re

TOKEN_RE = re.compile(r"[a-z']+")

# --------------------------------------------------------------------------
# D1 — quantification density
# --------------------------------------------------------------------------
NUM_RE = re.compile(
    r"(?:\$?\s*\d[\d,\.]*\s*(?:million|billion|trillion|%|percent|percentage|"
    r"years?|months?|days?|units?|shares?|hours?)\b)", re.I)

# --------------------------------------------------------------------------
# D2 — project specificity: concrete asset / project / location nouns
# --------------------------------------------------------------------------
PROJECT_TERMS = [
    "factory", "factories", "plant", "plants", "facility", "facilities",
    "manufacturing site", "mill", "refinery", "warehouse", "distribution center",
    "data center", "datacenter", "headquarters", "campus", "site", "sites",
    "production line", "assembly line", "line of business", "product line",
    "new product", "product portfolio", "equipment", "machinery", "rig", "vessel",
    "store", "stores", "retail location", "branch", "subsidiary", "acquisition",
    "joint venture", "plant expansion", "capacity", "expansion project",
    "capital project", "greenfield", "brownfield", "lease", "leased facility",
    "customer", "customers", "supplier", "suppliers", "contract", "contracts",
    "agreement", "program", "initiative", "platform", "pipeline", "region",
    "project", "projects",
]
PROJECT_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in PROJECT_TERMS) + r")\b", re.I)

# --------------------------------------------------------------------------
# D3 — temporal explicitness
# --------------------------------------------------------------------------
TEMP_TERMS = [
    r"next fiscal year", r"following (?:two|three|four|five) (?:fiscal )?years",
    r"for the year ending", r"\bq[1-4]\b", r"first quarter", r"second quarter",
    r"third quarter", r"fourth quarter", r"\b20\d\d\b", r"through 20\d\d",
    r"by (?:the )?(?:first|second|third|fourth) quarter", r"next (?:two|three|four|five) years",
    r"over the (?:next|coming) (?:twelve|eighteen|twenty-four) months",
    r"during (?:fiscal )?20\d\d",
]
TEMP_RE = re.compile(r"(?:" + "|".join(TEMP_TERMS) + r")")

# --------------------------------------------------------------------------
# D4 — execution stage (ordinal 1..5)  [pre-compiled]
# --------------------------------------------------------------------------
STAGE_VERBS = {
    1: [r"\bplan\b", r"plans to", r"expect to", r"intend", r"anticipate",
        r"project to", r"propose", r"subject to"],
    2: [r"\bapprov", r"authoriz", r"committed to", r"board approved", r"obtained approval"],
    3: [r"\bconstruct", r"\bbuild\b", r"building", r"\bdevelop\b", r"developing",
        r"underway", r"in progress", r"commenc", r"breaking ground"],
    4: [r"near completion", r"nearly complete", r"substantially complete",
        r"on track to complete", r"final stages"],
    5: [r"\bcompleted\b", r"\bcompleted the\b", r"finished", r"\bimplemented\b",
        r"\bachieved\b", r"placed in service", r"ramped up", r"operational",
        r"\bdelivered\b", r"\bcommercially available\b"],
}
STAGE_RES = {s: [re.compile(p) for p in pats] for s, pats in STAGE_VERBS.items()}
STAGE_ALL_RE = re.compile(r"(?:" + "|".join(p for pats in STAGE_VERBS.values() for p in pats) + r")")

# --------------------------------------------------------------------------
# D5 — evidence markers: achieved (past) vs forward-looking  [pre-compiled]
# --------------------------------------------------------------------------
EVID_PAST = [
    r"\bachieved\b", r"\brealized\b", r"\bsigned\b", r"\bcompleted\b",
    r"\bimplemented\b", r"\bdelivered\b", r"\bgenerated\b", r"\brecorded\b",
    r"\brecognized\b", r"\blaunched\b", r"\bsecured\b", r"\bexceeded\b",
]
EVID_FWD = [
    r"\bexpect\b", r"\bmay\b", r"\banticipate\b", r"\bintend\b", r"\bplan\b",
    r"\bshould\b", r"\bwill\b", r"\boutlook\b", r"\bguidance\b", r"\btarget\b",
    r"\bprojected\b", r"\bestimate\b", r"\bseek to\b", r"\bgoal\b",
]
EVID_PAST_RE = re.compile(r"(?:" + "|".join(EVID_PAST) + r")")
EVID_FWD_RE = re.compile(r"(?:" + "|".join(EVID_FWD) + r")")

# --------------------------------------------------------------------------
# Risk subtype lexicons (H3 ablation + transformer auxiliary heads)  [pre-compiled]
# --------------------------------------------------------------------------
RISK_LEX = {
    'risk_demand': [
        r"\b(demand|orders?|backlog|bookings?|customer[s]?|sales volume|units? sold|reorder)\b"],
    'risk_cost': [
        r"\b(cost[s]?|input[s]?|margin[s]?|wage[s]?|raw material[s]?|commodity|commodities|"
        r"supply|supplier[s]?|inflation|price (?:of|for) (?:raw|input|energy))\b"],
    'risk_liquidity': [
        r"\b(cash|liquidity|debt|covenant[s]?|borrowing[s]?|refinanc\w*|interest rate|"
        r"leverage|working capital|credit facility|revolving)\b"],
}
RISK_RES = {k: re.compile(r"(?:" + "|".join(pats) + r")") for k, pats in RISK_LEX.items()}

# --------------------------------------------------------------------------
# Semantic-disambiguation guard flags (§4.6)  [pre-compiled]
# --------------------------------------------------------------------------
# "artificial cost" vs "headcount" — distinct events, never conflated
HEADCOUNT_RE = re.compile(r"\b(headcount|employees?|workforce|staff|layoff[s]?|hiring)\b", re.I)
# "plan not disclosed" flag — absence of forward capex language != no investment
PLAN_DISCLOSED_RE = re.compile(
    r"(?:plan|expect|intend|anticipate|will|target|outlook|guidance|committ)", re.I)


def _density(text, pattern):
    low = text.lower()
    toks = TOKEN_RE.findall(low)
    return len(pattern.findall(low)) / max(len(toks), 1)


def d1_quant(text):
    return _density(text, NUM_RE)


def d2_project(text):
    return _density(text, PROJECT_RE)


def d3_temporal(text):
    return _density(text, TEMP_RE)


def d4_stage(text):
    low = text.lower()
    stage = 0
    for s, res in STAGE_RES.items():
        if any(r.search(low) for r in res):
            stage = s
    return float(stage), _density(text, STAGE_ALL_RE)


def d5_evidence(text):
    low = text.lower()
    past = len(EVID_PAST_RE.findall(low))
    fwd = len(EVID_FWD_RE.findall(low))
    tot = past + fwd
    if tot == 0:
        return 0.0, 0.0
    return past / tot, (past - fwd)


def risk_density(text):
    return {k: _density(text, r) for k, r in RISK_RES.items()}


def score_text_ssi(text):
    """Raw per-document SSI features. Returns a flat dict."""
    if not text:
        return dict(ssi_d1=0.0, ssi_d2=0.0, ssi_d3=0.0, ssi_d4_stage=0.0,
                    ssi_d4_dens=0.0, ssi_d5_ratio=0.0, ssi_d5_net=0.0,
                    risk_demand=0.0, risk_cost=0.0, risk_liquidity=0.0,
                    flag_headcount=0.0, flag_plan_disclosed=0.0)
    out = {}
    out['ssi_d1'] = d1_quant(text)
    out['ssi_d2'] = d2_project(text)
    out['ssi_d3'] = d3_temporal(text)
    out['ssi_d4_stage'], out['ssi_d4_dens'] = d4_stage(text)
    out['ssi_d5_ratio'], out['ssi_d5_net'] = d5_evidence(text)
    out.update(risk_density(text))
    out['flag_headcount'] = float(bool(HEADCOUNT_RE.search(text)))
    out['flag_plan_disclosed'] = float(bool(PLAN_DISCLOSED_RE.search(text)))
    return out
