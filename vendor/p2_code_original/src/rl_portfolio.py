"""
rl_portfolio.py — Reinforcement-learning dynamic allocation agent (Phase H / step 7).

For each SIGNAL book (ssi / cps / neural) and each BOOK construction
(long-short = top minus bottom quintile, or long-only-vs-cash = top quintile
held, remainder in cash) a contextual policy agent decides the *exposure*
(0-100%) to that book each calendar year from PRE-RETURN state features
(prior cumulative return, prior drawdown, prior market return, and the year's
aggregate disclosure uncertainty / mean signal). It is trained with REINFORCE
to maximise cumulative return while penalising drawdowns AND turnover, and is
compared against two naive benchmarks:

    * buy_hold_100  always 100% invested in the book (买入持有)
    * fixed_50      always 50% invested in the book (固定仓位)

Transaction costs are modelled as TC * |a_t - a_{t-1}| per year (10bp per unit
of exposure change, a_{-1}=0 at entry), deducted from the realised book return —
so the agent is rewarded for *not* churning and the comparison against the
static benchmarks is apples-to-apples.

NOTE (disclosed limitation): ret_1y is a PRICE-RATIO proxy derived from
Compustat prcc_f (not CRSP), so absolute performance numbers are indicative
only; the within-book RL-vs-benchmark comparison is the interpretable object.
"""
import os, json, time, numpy as np, pandas as pd, torch, torch.nn as nn
import config

SEED = config.SEED
TC = 0.001                              # 10 bp per unit of exposure turnover
SIGNALS = {'ssi': 'SSI book (rule-based disclosure substance)',
           'cps': 'CPS book (capital-plan specificity & stage, long-vs-cash focus)',
           'neural': 'neural book (deep-learning neural_score)'}
BOOKS = {'ls': 'long-short (top quintile - bottom quintile)',
         'long': 'long-only top quintile vs cash (0)'}


def _state_features(signal='ssi', book='ls'):
    """Annual book return + pre-return state frame for one signal/book combo."""
    tx = pd.read_parquet(os.path.join(config.DATA, 'texts.parquet'))
    nf_path = os.path.join(config.DATA, 'neural_features.parquet')
    if signal == 'neural':
        if not os.path.exists(nf_path):
            return None
        nf = pd.read_parquet(nf_path)
        df = tx[['index', 'file_year', 'unc_ratio', 'ret_1y']].merge(
            nf[['index', 'neural_score']], on='index', how='left')
        col = 'neural_score'
    else:
        df = tx.copy()
        col = 'cps_composite' if signal == 'cps' else 'ssi'
    if col not in df.columns:
        return None
    df = df.dropna(subset=[col, 'ret_1y'])
    # within-year quintile label (duplicates='drop' handles heavy ties, e.g. many
    # firms with a zero CPS composite). Use each year's MAX/MIN label as the
    # top/bottom quintile so the long-short book is always well-defined.
    df['q'] = df.groupby('file_year')[col].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates='drop'))
    qmax = df.groupby('file_year')['q'].transform('max')
    qmin = df.groupby('file_year')['q'].transform('min')
    top = df.loc[df['q'] == qmax].groupby('file_year')['ret_1y'].mean()
    bot = df.loc[df['q'] == qmin].groupby('file_year')['ret_1y'].mean()
    ls = (top - bot) if book == 'ls' else top          # long-short vs long-vs-cash
    agg = df.groupby('file_year').agg(
        mean_unc=('unc_ratio', 'mean'), mean_sig=(col, 'mean'),
        mkt_ret=('ret_1y', 'mean')).reindex(ls.index)
    # drop any year with a missing book return or state feature (empty quintile, etc.)
    keep = ls.notna() & agg['mean_unc'].notna() & agg['mean_sig'].notna() \
        & agg['mkt_ret'].notna()
    years = [y for y, k in zip(ls.index.tolist(), keep.tolist()) if k]
    ls = ls[keep].values.astype(float)
    agg = agg[keep]
    return years, ls, agg


def _make_states(years, ls, agg):
    # pre-return state vector per year t: [prev_cum, prev_dd, prev_mkt, mean_unc, mean_sig, t_norm]
    n = len(years)
    prev_cum = np.zeros(n); prev_dd = np.zeros(n); prev_mkt = np.zeros(n)
    cum = 0.0; peak = 0.0; dd = 0.0
    for i in range(n):
        prev_cum[i] = cum; prev_dd[i] = dd
        prev_mkt[i] = agg['mkt_ret'].values[i - 1] if i > 0 else 0.0
        cum += ls[i]; peak = max(peak, cum); dd = max(0.0, peak - cum)
    S = np.stack([prev_cum, prev_dd, prev_mkt,
                  agg['mean_unc'].values, agg['mean_sig'].values,
                  (np.arange(n) - n / 2) / n], 1)
    return S.astype(np.float32)


class Policy(nn.Module):
    def __init__(self, dim=6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.logstd = nn.Parameter(torch.tensor(-1.0))

    def forward(self, s):
        mean = torch.sigmoid(self.net(s))
        return mean, torch.exp(self.logstd)


def _portfolio(ls, actions):
    """Net P&L of a per-year exposure path, starting flat (a_{-1}=0).

    Realised return each year = a_t * ls_t - TC * |a_t - a_{t-1}| (turnover
    cost). Returns cumulative net return, max drawdown, Sharpe, total cost.
    """
    cum = 0.0; peak = 0.0; maxdd = 0.0; prev_a = 0.0
    rets = []; costs = []
    for i in range(len(ls)):
        a = float(actions[i])
        cost = TC * abs(a - prev_a)
        pr = a * ls[i] - cost
        cum += pr; peak = max(peak, cum); maxdd = max(maxdd, peak - cum)
        rets.append(pr); costs.append(cost); prev_a = a
    rets = np.array(rets)
    sharpe = rets.mean() / rets.std(ddof=1) if rets.std(ddof=1) > 0 else 0.0
    return dict(cum=cum, maxdd=maxdd, sharpe=sharpe,
                total_cost=float(sum(costs)), rets=rets)


def _rollout(policy, S, ls, deterministic=True):
    with torch.no_grad():
        means, _ = policy(torch.tensor(S))
        actions = [min(1.0, max(0.0, float(means[i]))) for i in range(len(S))]
    out = _portfolio(ls, actions)
    out['exps'] = np.array(actions)
    return out


def _train_policy(Str, ls_tr):
    torch.manual_seed(SEED); np.random.seed(SEED)
    policy = Policy(6)
    opt = torch.optim.Adam(policy.parameters(), lr=5e-3)
    for it in range(400):
        logps = []; rews = []; cum = 0.0; peak = 0.0; dd = 0.0; prev_a = 0.0
        means, std = policy(torch.tensor(Str))
        for i in range(len(Str)):
            m = float(means[i].detach()); a = min(1.0, max(0.0, m + np.random.normal(0, float(std))))
            cost = TC * abs(a - prev_a)
            pr = a * ls_tr[i] - cost
            new_cum = cum + pr; new_peak = max(peak, new_cum); new_dd = max(0.0, new_peak - new_cum)
            dd_inc = max(0.0, new_dd - dd)
            rew = pr - 2.0 * dd_inc ** 1.5
            cum = new_cum; peak = new_peak; dd = new_dd; prev_a = a
            dist = torch.distributions.Normal(m, std)
            logps.append(dist.log_prob(torch.tensor(a).float())); rews.append(rew)
        G = np.cumsum(rews[::-1])[::-1]; G = G - G.mean(); G = G / G.std() if G.std() > 0 else G
        loss = -(torch.stack(logps) * torch.tensor(G).float()).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return policy


def run():
    t0 = time.time()
    res = {}
    for sig, desc in SIGNALS.items():
        for book, bdesc in BOOKS.items():
            sf = _state_features(sig, book)
            if sf is None:
                continue
            years, ls, agg = sf
            S = _make_states(years, ls, agg)
            train_years = [y for y in years if y < config.TEST_FILE_YEAR_MIN]
            tr_idx = [years.index(y) for y in train_years]
            policy = _train_policy(S[tr_idx], ls[tr_idx])
            full = _rollout(policy, S, ls, deterministic=True)
            buy = _portfolio(ls, [1.0] * len(ls))        # 买入持有
            fix = _portfolio(ls, [0.5] * len(ls))        # 固定仓位
            key = f'{sig}_{book}'
            res[key] = dict(
                signal=sig, signal_desc=desc, book=book, book_desc=bdesc,
                years=years, ls_year=ls.tolist(), tc_bps=TC * 1e4,
                agent=dict(cum=float(full['cum']), maxdd=float(full['maxdd']),
                           sharpe=float(full['sharpe']),
                           total_cost=float(full['total_cost']),
                           exps=full['exps'].tolist()),
                buyhold100=dict(cum=float(buy['cum']), maxdd=float(buy['maxdd']),
                                sharpe=float(buy['sharpe']),
                                total_cost=float(buy['total_cost'])),
                fixed50=dict(cum=float(fix['cum']), maxdd=float(fix['maxdd']),
                             sharpe=float(fix['sharpe']),
                             total_cost=float(fix['total_cost'])),
                n_train=len(tr_idx), n_total=len(years))
            print(f"[rl:{key}] train<2020={len(tr_idx)} total={len(years)}")
            print(f"  AGENT  cum={full['cum']:.3f} sharpe={full['sharpe']:.3f} "
                  f"maxDD={full['maxdd']:.3f} cost={full['total_cost']:.3f}")
            print(f"  BUY100 cum={buy['cum']:.3f} sharpe={buy['sharpe']:.3f} "
                  f"maxDD={buy['maxdd']:.3f}")
            print(f"  FIX50  cum={fix['cum']:.3f} sharpe={fix['sharpe']:.3f} "
                  f"maxDD={fix['maxdd']:.3f}")
    json.dump(res, open(os.path.join(config.DATA, 'rl_results.json'), 'w'), indent=2)
    print(f"  ({time.time()-t0:.0f}s)")
    return res


if __name__ == '__main__':
    run()
