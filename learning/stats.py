"""
Overfitting controls for a search that tries many strategies.

When a loop evaluates hundreds of variants on the same history, the best
backtest Sharpe is inflated by selection alone. The Deflated Sharpe Ratio
(Bailey & López de Prado, 2014) asks: given N trials with this spread of
Sharpe ratios, what is the probability that the winner's TRUE Sharpe is > 0?

  psr(returns, sr_benchmark)  P[true SR > sr_benchmark], corrected for
                              sample length, skew and fat tails
  expected_max_sharpe(n, var) Sharpe the best of n zero-skill trials would
                              show by luck
  deflated_sharpe(returns, n_trials, trial_sharpe_var)
                              psr with that luck level as the benchmark

All Sharpe inputs/outputs here are PER-PERIOD (daily) unless named *_annual.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
_EULER = 0.5772156649015329


def sharpe(returns) -> float:
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1))


def psr(returns, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P[true per-period SR > sr_benchmark]."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    t = len(r)
    if t < 10 or r.std(ddof=1) == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std(ddof=0)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return 0.0
    return float(_N.cdf((sr - sr_benchmark) * math.sqrt(t - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, trial_sharpe_var: float) -> float:
    """Expected maximum per-period Sharpe among n zero-skill trials."""
    if n_trials <= 1 or trial_sharpe_var <= 0:
        return 0.0
    a = _N.inv_cdf(1 - 1 / n_trials)
    b = _N.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(trial_sharpe_var) * ((1 - _EULER) * a + _EULER * b)


def deflated_sharpe(returns, n_trials: int, trial_sharpe_var: float) -> float:
    return psr(returns, expected_max_sharpe(n_trials, trial_sharpe_var))


def annual_to_daily_var(annual_sharpes) -> float:
    """Variance of per-period Sharpe implied by a list of annualised Sharpes."""
    s = [x / math.sqrt(252) for x in annual_sharpes if x == x]
    if len(s) < 2:
        return 0.0
    return float(np.var(s, ddof=1))
