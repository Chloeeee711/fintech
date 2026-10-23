"""
lexicon.py — Loughran–McDonald-style finance sentiment lexicon (offline).

We reproduce the four canonical LM (2011) word categories — Negative, Positive,
Uncertainty, Litigious — as a transparent, fully offline module. The official
LM master dictionary (pysentiment / sraf) can be dropped in for exact
replication; this curated reconstruction captures the high-signal finance
vocabulary that drives the tone measure and requires no network access.

Public-domain finance word lists in the LM tradition; assembled for the
course project "Textual Analysis using LLMs".
"""
import re

# ---------------------------------------------------------------------------
# NEGATIVE  (bears, losses, deterioration, distress, litigation, risk)
# ---------------------------------------------------------------------------
NEGATIVE = {
 'loss','losses','lost','losing','impairment','impairments','impaired','decline',
 'declines','declined','declining','decrease','decreases','decreased','decreasing',
 'down','downside','downward','weak','weakness','weakened','weakening','adverse',
 'adversely','deteriorate','deteriorated','deterioration','deteriorating','deficit',
 'deficits','default','defaults','defaulted','defaulting','fail','fails','failed',
 'failure','failures','shortfall','shortfalls','bankrupt','bankruptcy','bankruptcies',
 'restructure','restructuring','restructured','write-down','writedowns','write-off',
 'write-offs','writeoff','writeoffs','terminate','terminated','termination',
 'penalty','penalties','fraud','fraudulent','investigation','investigations',
 'investigated','risk','risks','risky','unfavorable','unfavourable','negative',
 'damage','damages','claim','claims','contingent','contingency','doubt','doubtful',
 'doubts','obsolete','obsolescence','volatility','volatile','downturn','recession',
 'slowdown','discontinued','discontinue','disposal','dispose','disposition','layoff',
 'layoffs','charge','charges','casualty','catastrophe','catastrophes','crisis',
 'delinquent','delinquencies','distressed','distress','foreclosure','insolvency',
 'insolvent','malpractice','misstatement','misstatements','nonperforming','overdue',
 'plaintiff','recall','recalled','sue','sued','suing','suspension','suspended',
 'underperform','underperformed','underperformance','unpaid','violation','violations',
 'violate','violated','difficulty','difficulties','disruption','disrupt','disrupted',
 'hurt','hurts','hurting','harm','harmed','harmful','threat','threats','threatened',
 'burden','burdens','burdensome','deficiency','deficiencies','depressed','drop',
 'dropped','falls','falling','loss-making','lossmaking','negatively','offset',
 'impair','disappointing','disappointment','worsened','worsening','slowed','slump',
 'contraction','contract','contracted','strike','struck','curtail','curtailed',
 'diminished','eroded','erosion','hinder','hindered','hampered','impeded','setback',
 'setbacks','stagnant','stagnation','weaknesses','weaker','worst','adversely',
 'nonpayment','undercapitalized','underfunded','write-downs','writedown','cease',
 'ceased','abandoned','abandon','withdraw','withdrawn','withdrawal','lose','loses',
 'losing','unprofitable','deflation','deflate','correction','corrected','crash',
 'downgrade','downgraded','downgrades','lawsuit','lawsuits','litigation','litigate',
 'liquidation','liquidated','insolvency','contagion','impairing','defaults',
}

# ---------------------------------------------------------------------------
# POSITIVE  (gains, growth, strength, profitability, opportunity)
# ---------------------------------------------------------------------------
POSITIVE = {
 'gain','gains','gained','profit','profits','profitable','profitability','growth',
 'grow','grew','growing','grown','improve','improved','improvement','improving',
 'increase','increases','increased','increasing','strong','strength','strengths',
 'strengthen','strengthened','strengthening','success','successful','successfully',
 'succeed','succeeded','favorable','favourable','benefit','benefits','beneficial',
 'opportunity','opportunities','positive','positively','outperform','outperformed',
 'outperformance','exceed','exceeded','exceeding','exceeds','robust','resilient',
 'resilience','advantage','advantages','advantageous','record','milestone',
 'achievement','achievements','achieve','achieved','achieving','optimism','optimistic',
 'prosper','prospered','prosperity','healthy','solid','efficient','efficiently',
 'efficiency','effective','effectively','innovation','innovative','innovations',
 'expand','expanded','expansion','enhance','enhanced','enhancement','enhancing',
 'lead','leading','leader','leaders','leadership','win','won','winning','upturn',
 'upward','upside','surplus','surpluses','thrive','thrived','thriving','flourish',
 'accomplish','accomplished','progress','progressed','improved','stabilized',
 'stabilise','stable','stability','appreciate','appreciated','appreciation','boost',
 'boosted','momentum','recovery','recovered','rebound','rebounds','rebounded',
 'strengths','competitive','competitiveness','superior','premium','premiums',
 'dividend','dividends','value','valuable','growthe','gains','profited','profiting',
 'outpace','outpaced','advantaged','well-positioned','wellpositioned','promising',
 'accelerate','accelerated','acceleration','upgrade','upgraded','upgrades','best',
 'better','best-in-class','industry-leading','market-leading','increased','high-quality',
 'quality','stronger','strongest','boosting','delivered','deliver','delivering',
 'enhances','improves','growing','grew','expanded','expanding','robust','resilient',
 'attractive','attract','attracted','opportune','prosperous','sound','prudent',
 'favorable','favourable','success','succeeded','excellent','outstanding',
}

# ---------------------------------------------------------------------------
# UNCERTAINTY  (modal / hedging / estimation language)
# ---------------------------------------------------------------------------
UNCERTAINTY = {
 'may','might','possible','possibility','possibly','potential','potentially',
 'uncertain','uncertainty','uncertainties','uncertainly','approximate','approximately',
 'estimate','estimates','estimated','estimating','expect','expects','expected',
 'expectation','expectations','believe','believes','assume','assumption','assumptions',
 'depend','depends','depending','reflect','reflects','regarding','relative','some',
 'somewhat','tentative','tentatively','unclear','unlikely','probable','probability',
 'could','should','would','intend','intend','intends','anticipate','anticipates',
 'anticipated','anticipating','contingent','contingency','preliminary','speculative',
 'variable','varies','vary','varied','varying','fluctuation','fluctuations','doubt',
 'doubts','doubtful','indeterminate','indefinite','ambiguous','ambiguity','perhaps',
 'envisage','envisaged','forecast','forecasts','project','projects','projected',
 'projection','projections','predict','predicts','prediction','predictions','if',
 'whether','maybe','presumably','seem','seems','appear','appears','risk','risks',
 'indicate','indicates','indicated','suggest','suggests','suggested','likely',
 'unlikely','probable','improbable','conceivable','question','questions','subject',
 'subject to','pending','provisional','estimation','estimations','extrapolate',
 'extrapolated','hypothetical','hypothesis','uncertain','uncertainties',
}

# ---------------------------------------------------------------------------
# LITIGIOUS  (legal / regulatory / court language)
# ---------------------------------------------------------------------------
LITIGIOUS = {
 'litigation','lawsuit','lawsuits','litigant','litigants','litigate','litigated',
 'litigating','court','courts','judicial','legal','legally','legality','statute',
 'statutes','statutory','regulation','regulations','regulatory','regulate','regulated',
 'compliance','compliant','noncompliance','violation','violations','violate','violated',
 'plaintiff','plaintiffs','defendant','defendants','claim','claims','claimed','damage',
 'damages','liability','liabilities','suit','suits','sued','subpoena','subpoenas',
 'settlement','settlements','adjudicate','adjudicated','arbitration','arbitrate',
 'attorney','attorneys','counsel','dispute','disputes','disputed','appeal','appealed',
 'appeals','ruling','rulings','verdict','penalty','penalties','prosecutor','prosecution',
 'indemnify','indemnification','indemnified','law','laws','lawful','unlawful','misdemeanor',
 'felony','conviction','convicted','testify','testimony','witness','jurisdiction',
 'jurisdictions','ordinance','ordinances','decree','decrees','injunction','injunctions',
 'compliance','breach','breached','breaches','contractual','tort','torts','plea',
 'plead','pleaded','deposition','depositions','lawsuits','litigation','litigate',
 'regulator','regulators','legislation','legislative','enforcement','enforced',
 'enforce','sue','suing','ordered','order','orders','writ','writs','complied',
 'noncompliant','infringe','infringement','infringed','patent','patents','trademark',
 'trademarks','copyright','copyrights','disclosure','disclosures','subpoenas',
 'indictment','indicted','compliance','legal','regulatory','court','courts',
}

# compile regex for fast tokenisation
_TOKEN = re.compile(r"[a-z']+")

def score_text(text):
    """Return dict of LM-style counts and tone measures for one document."""
    if not isinstance(text, str):
        text = ''
    toks = _TOKEN.findall(text.lower())
    wc = len(toks)
    if wc == 0:
        return dict(wc=0, pos=0, neg=0, unc=0, lit=0,
                    tone_lm=0.0, unc_ratio=0.0, lit_ratio=0.0)
    pos = sum(1 for t in toks if t in POSITIVE)
    neg = sum(1 for t in toks if t in NEGATIVE)
    unc = sum(1 for t in toks if t in UNCERTAINTY)
    lit = sum(1 for t in toks if t in LITIGIOUS)
    denom = (pos + neg)
    tone_lm = (pos - neg) / denom if denom > 0 else 0.0
    return dict(wc=wc, pos=pos, neg=neg, unc=unc, lit=lit,
                tone_lm=tone_lm,
                unc_ratio=unc / wc,
                lit_ratio=lit / wc)

if __name__ == '__main__':
    s = score_text("We reported a net loss and impairment charges amid litigation risk and uncertain demand.")
    print(s)
