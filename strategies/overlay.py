"""
Index + leverage overlay on the ETF trend targets ("option B").

    final = leverage x (index_weight x index fund + (1 - index_weight) x trend targets)

Tested in audit/h4_netted.py and audit/h5_ladder.py (2007-2026, borrowing at
T-bill + 2.5%): 40% SPY / 60% trend at 1.5x returned ~10%/yr, Sharpe ~0.60,
worst drop ~-34% (trend alone: 4.9%, 0.55, -13%; SPY: 11.1%, 0.56, -55%).

The trend book's own T-bill (BIL) slot is used first, so the account only
borrows when risky exposure is above 100% of equity. Total long exposure is
hard-capped at max_gross. When the overlay is disabled the trend targets are
returned unchanged, so the bot behaves exactly as before.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

DEFAULTS = {
    "enabled": False,
    "index_symbol": "SPY",
    "index_weight": 0.40,     # share of the mix in the index fund (rest = trend book)
    "leverage": 1.0,          # 1.0 = no borrowing
    "max_gross": 1.6,         # hard cap on total long exposure / equity
    "borrow_rate_pct": 7.0,   # margin interest used by the model and replay (set to the broker's rate)
    "require_margin": True,   # leverage > 1 only on a margin account (Alpaca multiplier >= 2)
}
MAX_LEVERAGE = 2.0            # Reg T overnight limit; never configurable above this


def settings(cfg: Optional[dict]) -> dict:
    """Validated overlay settings from config['etf_trend']['overlay'].
    Raises ValueError on a bad config so the cycle stops instead of trading on it."""
    s = {**DEFAULTS, **((cfg or {}).get("overlay") or {})}
    s["enabled"] = bool(s["enabled"])
    for k in ("index_weight", "leverage", "max_gross", "borrow_rate_pct"):
        s[k] = float(s[k])
    if not 0.0 <= s["index_weight"] <= 1.0:
        raise ValueError(f"overlay.index_weight must be between 0 and 1, got {s['index_weight']}")
    if not 1.0 <= s["leverage"] <= MAX_LEVERAGE:
        raise ValueError(f"overlay.leverage must be between 1 and {MAX_LEVERAGE}, got {s['leverage']}")
    if not 1.0 <= s["max_gross"] <= MAX_LEVERAGE:
        raise ValueError(f"overlay.max_gross must be between 1 and {MAX_LEVERAGE}, got {s['max_gross']}")
    if not 0.0 <= s["borrow_rate_pct"] <= 25.0:
        raise ValueError(f"overlay.borrow_rate_pct looks wrong: {s['borrow_rate_pct']}")
    return s


def active(s: dict) -> bool:
    return bool(s.get("enabled")) and (s["index_weight"] > 0 or s["leverage"] > 1)


def label(cfg: Optional[dict], base: str) -> str:
    """Strategy label used for the equity curve, state and logs. A different
    label starts a new paper track record, so readiness is measured from the switch."""
    s = settings(cfg)
    if not active(s):
        return base
    return f"{base}+{s['index_symbol'].lower()}{s['index_weight'] * 100:.0f}x{s['leverage']:g}"


def apply(trend: pd.Series, s: dict, cash_symbol: str = "BIL", leverage: Optional[float] = None,
          max_gross: Optional[float] = None) -> pd.Series:
    """Final target weights (of equity) from the trend book's targets (which
    include cash_symbol and sum to 1). The result includes cash_symbol and its
    risky part may sum to more than 1 (the excess is borrowed)."""
    if not active(s):
        return trend.copy()
    L = s["leverage"] if leverage is None else float(leverage)
    cap = s["max_gross"] if max_gross is None else float(max_gross)
    wi = s["index_weight"]
    risky = trend.drop(cash_symbol, errors="ignore").astype(float)
    out = risky * (1.0 - wi) * L
    out[s["index_symbol"]] = float(out.get(s["index_symbol"], 0.0)) + wi * L
    out = out[out > 1e-12]
    gross_ = float(out.sum())
    if gross_ > cap:
        out = out * (cap / gross_)
    out[cash_symbol] = max(0.0, 1.0 - float(out.sum()))
    return out


def gross(targets: pd.Series, cash_symbol: str = "BIL") -> float:
    """Risky (non-cash) exposure as a share of equity."""
    return float(targets.drop(cash_symbol, errors="ignore").sum())


def borrow_limit(targets: pd.Series, cash_symbol: str = "BIL") -> float:
    """Share of equity the plan is allowed to borrow to reach `targets`."""
    return max(0.0, gross(targets, cash_symbol) - 1.0)


def projected_long(holdings: dict, prices: dict, orders: list) -> float:
    """Total long market value after `orders` (sells and buys) at their planned prices."""
    q = {s: float(v) for s, v in holdings.items()}
    for o in orders:
        q[o["symbol"]] = q.get(o["symbol"], 0.0) + (o["qty"] if o["side"] == "buy" else -o["qty"])
    return sum(max(0.0, v) * float(prices.get(s) or 0) for s, v in q.items())
