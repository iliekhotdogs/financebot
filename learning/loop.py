"""
Recursive learning loop — runs OUTSIDE the order path (daily, after close).

Each run:
  0. Apply a promotion that was announced earlier and whose grace period
     has elapsed (unless vetoed).
  1. Propose: mutate the best strategies (exploit) and draw fresh genomes
     from OTHER strategy families (explore / switch category).
  2. Evaluate: one causal backtest per (strategy, symbol) on ~2y of daily
     bars; an equal-weight PORTFOLIO is scored on training / validation /
     out-of-sample windows (common calendar) incl. costs, vs. the active
     strategy and equal-weight buy-and-hold. Results persisted to
     learning/evaluations.jsonl.
  3. Paper gate: beat the active strategy on validation+OOS Sharpe, enough
     trades, and pass a Deflated-Sharpe test that accounts for EVERY
     strategy ever tried (multiple-testing control).
  4. Shadow paper: candidates are then scored only on bars that arrive
     AFTER they entered paper (data the search never saw).
  5. Promotion gate: PromotionGate.assess_for_promotion (paper days,
     trades, symbols, Sharpe, drawdown, must beat active) + Deflated Sharpe
     on offline+forward returns.
  6. Promotion: with allow_auto_promotion, the winner is ANNOUNCED on
     Discord first; the switch happens after `announce_grace_hours`
     (veto: `python main.py learning-veto`). Otherwise it waits for
     `learning-approve`.

The loop has no order credentials and never places orders.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from strategies import families as fam
from .audit import LearningAuditLog
from .candidates import Candidate, CandidateConfig, CandidateStatus, CandidateStore
from .evaluation import EvaluationResult
from .model_selection import ModelSelector, SelectionConfig, SelectionMetric
from .promotion import PromotionDecision, PromotionGate, PromotionGateConfig
from . import stats

logger = logging.getLogger(__name__)

LEARNING_DIR = os.path.dirname(os.path.abspath(__file__))
BOT_DIR = os.path.dirname(LEARNING_DIR)
CANDIDATES_PATH = os.path.join(LEARNING_DIR, "candidates.csv")
AUDIT_PATH = os.path.join(LEARNING_DIR, "learning_audit_log.csv")
EVALUATIONS_PATH = os.path.join(LEARNING_DIR, "evaluations.jsonl")
ACTIVE_STRATEGY_PATH = os.path.join(LEARNING_DIR, "active_strategy.json")
PENDING_PROMOTION_PATH = os.path.join(LEARNING_DIR, "pending_promotion.json")
REPORTS_DIR = os.path.join(LEARNING_DIR, "reports")

ACTIVE_ID = "active"
PORTFOLIO = "PORTFOLIO"
_EVAL_FIELDS = {f.name for f in fields(EvaluationResult)}
_LEGACY_KEYS = ("sma_short", "sma_long", "atr_period", "atr_mult_initial", "atr_mult_tight",
                "atr_mult_tightest", "profit_floor_1", "profit_lock_1", "profit_floor_2",
                "profit_lock_2")


def _now(now: Optional[datetime]) -> datetime:
    return now or datetime.now(timezone.utc)


def _read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------- #
# Genomes ↔ candidates
# ---------------------------------------------------------------------- #

def genome_of(c: Candidate) -> dict:
    cfg = c.config
    if cfg.genes:
        return fam.normalize(cfg.genes)
    return fam.normalize({"family": "sma_cross", **{k: getattr(cfg, k) for k in _LEGACY_KEYS}})


def active_genome(config: dict) -> dict:
    """Active genome: learning/active_strategy.json, else config.yaml `strategy`."""
    rec = _read_json(ACTIVE_STRATEGY_PATH, None)
    if rec:
        if rec.get("genome"):
            return fam.normalize(rec["genome"])
        if rec.get("params"):  # legacy record
            return fam.normalize({"family": "sma_cross", **rec["params"]})
    s = dict(config.get("strategy", {}))
    s.setdefault("family", "sma_cross")
    return fam.normalize(s)


def active_params(config: dict) -> dict:  # backwards compatible name
    return active_genome(config)


def _candidate(genome: dict, hypothesis: str, parent_id: Optional[str], now: datetime) -> Candidate:
    g = fam.normalize(genome)
    cfg = CandidateConfig(
        **{k: g[k] for k in _LEGACY_KEYS if k in g and k not in ("sma_short", "sma_long")},
        family=g["family"], genes=g,
    )
    if g["family"] == "sma_cross":
        cfg.sma_short, cfg.sma_long = g["sma_short"], g["sma_long"]
    return Candidate(
        candidate_id=f"cand-{now:%Y%m%d}-{g['family']}-{fam.genome_hash(g)}",
        version=f"{g['family']}-{fam.genome_hash(g)}",
        hypothesis=hypothesis,
        intended_conditions="Daily bars, liquid US equities, long-only",
        required_data="Daily OHLCV, ~2y",
        abstain_rule="No entry without the family's entry event; flat otherwise",
        config=cfg,
        parent_candidate_id=parent_id,
        created_at=now.isoformat(),
    )


def _notes(c: Candidate) -> dict:
    try:
        return json.loads(c.notes) if c.notes else {}
    except ValueError:
        return {"text": c.notes}


def _set_notes(c: Candidate, **kw) -> None:
    n = _notes(c)
    n.update(kw)
    c.notes = json.dumps(n, default=str)


# ---------------------------------------------------------------------- #
# Persistence of evaluation results
# ---------------------------------------------------------------------- #

def save_results(results: list, run_id: str, path: Optional[str] = None) -> None:
    path = path or EVALUATIONS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in results:
            d = r.to_dict()
            d["profit_factor"] = min(d.get("profit_factor", 0.0), 1e6)  # inf isn't JSON
            f.write(json.dumps({"run_id": run_id, **d}, default=str) + "\n")


def load_results(path: Optional[str] = None, latest_only: bool = True) -> dict:
    """{candidate_id: [EvaluationResult]} — by default only each candidate's latest run."""
    path = path or EVALUATIONS_PATH
    if not os.path.exists(path):
        return {}
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    latest_run = {}
    for r in rows:
        cid = r.get("candidate_id")
        if r.get("run_id", "") >= latest_run.get(cid, ""):
            latest_run[cid] = r.get("run_id", "")
    out: dict = {}
    for r in rows:
        cid = r.get("candidate_id")
        if latest_only and r.get("run_id") != latest_run.get(cid):
            continue
        kw = {k: v for k, v in r.items() if k in _EVAL_FIELDS}
        if isinstance(kw.get("confidence_interval_95"), list):
            kw["confidence_interval_95"] = tuple(kw["confidence_interval_95"])
        out.setdefault(cid, []).append(EvaluationResult(**kw))
    return out


# ---------------------------------------------------------------------- #
# Portfolio evaluation
# ---------------------------------------------------------------------- #

def _naive_ts(x):
    t = pd.Timestamp(x)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.normalize()


def _naive_daily_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s.index = idx.normalize()
    return s[~s.index.duplicated(keep="last")]


class PortfolioRun:
    """One causal backtest per symbol; equal-weight portfolio daily returns.

    Backtests are causal (a decision at bar t uses only bars ≤ t), so slicing
    a full-history run by date equals running it up to that date.
    """

    def __init__(self, strategy, bars: dict):
        parts, self.trades = [], []
        for sym, df in bars.items():
            raw = strategy.backtest(df)
            d = raw.get("daily_returns")
            if d is None:
                continue
            parts.append(_naive_daily_index(d).rename(sym))
            for t in raw.get("trades", []):
                self.trades.append({**t, "symbol": sym, "entry_date": _naive_ts(t["entry_date"])})
        self.daily = pd.concat(parts, axis=1).fillna(0.0).mean(axis=1) if parts else pd.Series(dtype=float)

    def window(self, start, end=None) -> dict:
        d = self.daily[self.daily.index >= start]
        if end is not None:
            d = d[d.index < end]
        tr = [t for t in self.trades if t["entry_date"] >= start and (end is None or t["entry_date"] < end)]
        rets = [t["pnl_pct"] for t in tr]
        gp, gl = sum(r for r in rets if r > 0), -sum(r for r in rets if r <= 0)
        m = fam.metrics_from_daily(d)
        return {"daily": d, "trades": len(tr), "symbols": len({t["symbol"] for t in tr}),
                "sharpe": m["sharpe_ratio"], "ret": m["total_return_pct"],
                "dd": abs(m["max_drawdown_pct"]),
                "pf": (gp / gl) if gl > 0 else (10.0 if gp > 0 else 0.0),
                "win_rate": (sum(1 for r in rets if r > 0) / len(rets) * 100) if rets else 0.0}


def buy_and_hold_daily(bars: dict) -> pd.Series:
    parts = [_naive_daily_index(df["Close"].pct_change().fillna(0.0)).rename(s) for s, df in bars.items()]
    return pd.concat(parts, axis=1).fillna(0.0).mean(axis=1) if parts else pd.Series(dtype=float)


def window_bounds(bars: dict, config: dict) -> dict:
    """Common-calendar training / validation / out_of_sample start dates."""
    ev = config.get("learning", {}).get("evaluation", {})
    tr, va, oo = ev.get("train_days", 252), ev.get("validation_days", 126), ev.get("out_of_sample_days", 63)
    cal = sorted(set().union(*[set(_naive_daily_index(df["Close"]).index) for df in bars.values()]))
    n = len(cal)
    if n >= tr + va + oo + 60:
        b = n - (tr + va + oo)
        return {"training": cal[b], "validation": cal[b + tr], "out_of_sample": cal[b + tr + va]}
    logging.getLogger(__name__).warning(
        "Only %d bars for train+validation+OOS=%d (+60): falling back to proportional windows; "
        "raise learning.evaluation.history_days", n, tr + va + oo)
    return {"training": cal[int(n * 0.3)], "validation": cal[int(n * 0.6)],
            "out_of_sample": cal[int(n * 0.85)]}


def evaluate_portfolio(cid: str, run: PortfolioRun, bh: pd.Series, starts: dict) -> tuple:
    """EvaluationResult per window (symbol=PORTFOLIO) + val+OOS daily returns."""
    order = ["training", "validation", "out_of_sample"]
    results = []
    for i, p in enumerate(order):
        end = starts[order[i + 1]] if i + 1 < len(order) else None
        w = run.window(starts[p], end)
        b = bh[bh.index >= starts[p]]
        if end is not None:
            b = b[b.index < end]
        bm = fam.metrics_from_daily(b)
        r = EvaluationResult(
            candidate_id=cid, symbol=PORTFOLIO, period=p,
            start_date=str(starts[p].date()), end_date=str((end or bh.index[-1]).date()),
            net_return_pct=w["ret"], max_drawdown_pct=w["dd"], sharpe_ratio=w["sharpe"],
            total_trades=w["trades"], profit_factor=w["pf"], win_rate=w["win_rate"],
            baseline_buyhold_return_pct=bm["total_return_pct"], baseline_buyhold_sharpe=bm["sharpe_ratio"],
            period_days=len(w["daily"]),
        )
        r.passed_gates = w["pf"] >= 1.0 and w["ret"] > 0
        if not r.passed_gates:
            r.rejection_reasons.append(f"pf={w['pf']:.2f} ret={w['ret']:.2f}%")
        results.append(r)
    scored = run.daily[run.daily.index >= starts["validation"]]
    w = run.window(starts["validation"])
    sm = fam.metrics_from_daily(scored)
    results.append(EvaluationResult(
        candidate_id=cid, symbol=PORTFOLIO, period="scored",
        start_date=str(starts["validation"].date()), end_date=str(bh.index[-1].date()),
        net_return_pct=sm["total_return_pct"], max_drawdown_pct=abs(sm["max_drawdown_pct"]),
        sharpe_ratio=sm["sharpe_ratio"], total_trades=w["trades"], profit_factor=w["pf"],
        passed_gates=w["pf"] >= 1.0, period_days=len(scored),
    ))
    return results, scored


# ---------------------------------------------------------------------- #
# Proposals
# ---------------------------------------------------------------------- #

def propose_candidates(store: CandidateStore, audit: LearningAuditLog, parents: list,
                       n_exploit: int, n_explore: int, rng: random.Random, now: datetime) -> list:
    """Mutations of the best parents + fresh genomes from other families."""
    existing = {fam.genome_hash(genome_of(c)) for c in store.read_all()}
    existing |= {fam.genome_hash(g) for _, g in parents}
    created = []

    def add(genome, why, parent_id):
        if fam.validate(genome) or fam.genome_hash(genome) in existing:
            return
        c = _candidate(genome, why, parent_id, now)
        existing.add(fam.genome_hash(genome))
        store.save(c)
        audit.log_event("candidate_created", c.candidate_id, c.version, details=c.to_dict(), rationale=why)
        created.append(c)

    tries = 0
    while sum(1 for c in created if c.parent_candidate_id != "explore") < n_exploit and tries < 50:
        tries += 1
        # bias toward the best parents (index 0 = active)
        pid, pg = parents[min(len(parents) - 1, int(rng.random() ** 2 * len(parents)))]
        child = fam.mutate(pg, rng, n_genes=rng.choice([1, 2, 3]))
        add(child, f"Mutation of {pid}: {fam.describe(child)}", pid)
    active_family = parents[0][1]["family"] if parents else "sma_cross"
    tries = 0
    while sum(1 for c in created if c.parent_candidate_id == "explore") < n_explore and tries < 50:
        tries += 1
        others = [f for f in fam.FAMILIES if f != active_family]
        g = fam.random_genome(rng.choice(others), rng)
        add(g, f"Exploration (new category): {fam.describe(g)}", "explore")
    return created


# ---------------------------------------------------------------------- #
# Promotion: announce → grace → apply
# ---------------------------------------------------------------------- #

def announce_promotion(config: dict, c: Candidate, evidence: dict, now: datetime) -> bool:
    """Post the switch to Discord FIRST; only a delivered message schedules it."""
    import notify
    grace = float(config.get("learning", {}).get("promotion", {}).get("announce_grace_hours", 12))
    apply_after = now + timedelta(hours=grace)
    msg = (f"📣 Paper bot plans to change its strategy to {c.candidate_id}.\n"
           f"⏰ Planned time: {apply_after:%Y-%m-%d %H:%M} UTC.\n"
           f"🛑 To stop it, run: `python main.py learning-veto`")
    if not notify.send(config, msg, kind="promotions"):
        return False
    _write_json(PENDING_PROMOTION_PATH, {
        "candidate_id": c.candidate_id, "genome": genome_of(c), "announced_at": now.isoformat(),
        "apply_after": apply_after.isoformat(), "evidence": evidence,
    })
    _set_notes(c, announced_at=now.isoformat())
    CandidateStore(CANDIDATES_PATH).save(c)
    LearningAuditLog(AUDIT_PATH).log_event("promotion_announced", c.candidate_id, c.version,
                                           details=evidence, rationale=msg[:500])
    return True


def apply_pending_promotion(config: dict, now: Optional[datetime] = None) -> Optional[str]:
    """Switch only when auto-promotion is still enabled after the grace period."""
    if not config.get("learning", {}).get("promotion", {}).get("allow_auto_promotion", False):
        return None
    now = _now(now)
    p = _read_json(PENDING_PROMOTION_PATH, None)
    if not p or now < datetime.fromisoformat(p["apply_after"]):
        return None
    c = CandidateStore(CANDIDATES_PATH).get(p["candidate_id"])
    os.remove(PENDING_PROMOTION_PATH)
    if c is None or c.status != CandidateStatus.PAPER:
        return None
    approve_candidate(c.candidate_id, operator="auto-promotion", now=now)
    import notify
    notify.send(config, f"✅ Paper bot strategy changed to {c.candidate_id}. "
                        f"To undo: `python main.py learning-rollback`",
                kind="promotions")
    return c.candidate_id


def veto_pending(operator: str = "operator") -> Optional[str]:
    p = _read_json(PENDING_PROMOTION_PATH, None)
    if not p:
        return None
    os.remove(PENDING_PROMOTION_PATH)
    store = CandidateStore(CANDIDATES_PATH)
    c = store.get(p["candidate_id"])
    if c:
        c.status = CandidateStatus.REJECTED
        _set_notes(c, vetoed_by=operator)
        store.save(c)
    LearningAuditLog(AUDIT_PATH).log_event("promotion_vetoed", p["candidate_id"], "",
                                           details=p, rationale=f"Vetoed by {operator}")
    return p["candidate_id"]


def approve_candidate(candidate_id: str, operator: str = "operator",
                      now: Optional[datetime] = None) -> dict:
    """Make a candidate the active strategy (keeps the previous one for rollback)."""
    store = CandidateStore(CANDIDATES_PATH)
    c = store.get(candidate_id)
    if c is None:
        raise ValueError(f"Candidate {candidate_id} not found")
    g = genome_of(c)
    record = {
        "candidate_id": c.candidate_id,
        "genome": {**g, "version": c.version},
        "params": {k: g[k] for k in _LEGACY_KEYS if k in g},
        "approved_at": _now(now).isoformat(),
        "approved_by": operator,
        "previous": _read_json(ACTIVE_STRATEGY_PATH, None),
    }
    _write_json(ACTIVE_STRATEGY_PATH, record)
    c.status = CandidateStatus.PROMOTED
    store.save(c)
    LearningAuditLog(AUDIT_PATH).log_event("promoted", c.candidate_id, c.version, details=record,
                                           rationale=f"Approved by {operator}")
    return record


def rollback_active(operator: str = "operator") -> Optional[dict]:
    """Restore the previously active strategy (or config.yaml if none)."""
    current = _read_json(ACTIVE_STRATEGY_PATH, None)
    if not current:
        return None
    store = CandidateStore(CANDIDATES_PATH)
    c = store.get(current.get("candidate_id", ""))
    if c:
        c.status = CandidateStatus.ROLLED_BACK
        store.save(c)
    prev = current.get("previous")
    if prev:
        _write_json(ACTIVE_STRATEGY_PATH, prev)
    else:
        os.remove(ACTIVE_STRATEGY_PATH)
    LearningAuditLog(AUDIT_PATH).log_event("rolled_back", current.get("candidate_id", ""), "",
                                           details=current, rationale=f"Rollback by {operator}")
    return prev


# ---------------------------------------------------------------------- #
# Config helpers
# ---------------------------------------------------------------------- #

def _gate(config: dict) -> PromotionGate:
    pc = config.get("learning", {}).get("promotion", {})
    return PromotionGate(PromotionGateConfig(
        min_paper_days=pc.get("min_paper_days", 30),
        min_paper_trades=pc.get("min_paper_trades", 10),
        min_paper_symbols=pc.get("min_paper_symbols", 2),
        min_paper_sharpe=pc.get("min_paper_sharpe", 0.5),
        max_paper_drawdown_pct=pc.get("max_paper_drawdown_pct", 10.0),
        min_paper_net_return_pct=pc.get("min_paper_net_return_pct", 0.0),
        must_beat_active_on_sharpe=pc.get("must_beat_active_on_sharpe", True),
        must_beat_active_on_net_return=pc.get("must_beat_active_on_net_return", False),
        require_operator_approval=pc.get("require_operator_approval", True),
        allow_auto_promotion=pc.get("allow_auto_promotion", False),
    ))


def _selector(config: dict) -> ModelSelector:
    sc = config.get("learning", {}).get("selection", {})
    return ModelSelector(SelectionConfig(
        primary_metric=SelectionMetric(sc.get("primary_metric", "sharpe_ratio")),
        max_drawdown_pct=sc.get("max_drawdown_pct", 15.0),
        max_exposure_pct=sc.get("max_exposure_pct", 100.0),
        min_trades=sc.get("min_trades", 5),
        min_net_return_pct=sc.get("min_net_return_pct", 0.0),
        min_periods_passed=sc.get("min_periods_passed", 2),
        require_consistency=sc.get("require_consistency", True),
        max_return_volatility_across_periods=sc.get("max_return_volatility_across_periods", 50.0),
        evaluation_windows=sc.get("evaluation_windows", ["validation", "out_of_sample"]),
    ))


def _load_bars(dp, symbols: list, days: int = 760) -> dict:
    from engine import fetch_bars
    bars, _ = fetch_bars(dp, symbols, days)
    return {s: df for s, df in bars.items() if df is not None and len(df) >= 200}


def _trial_stats(store: CandidateStore, stored: dict, extra: list, series: dict) -> tuple:
    """
    (effective number of independent trials, variance of per-period Sharpe).

    Raw trial count overstates multiple testing: mutations are near-copies
    and strategies within a family move together. Following López de Prado,
    N_eff = N / (1 + (N-1)·ρ̄), with ρ̄ the average pairwise correlation of
    the scored daily returns of this run's strategies, scaled to all trials.
    """
    sharpes = list(extra)
    for rs in stored.values():
        for r in rs:
            if r.symbol == PORTFOLIO and r.period == "scored":
                sharpes.append(r.sharpe_ratio)
    n_raw = max(len(store.read_all()) + 1, len(sharpes))
    ratio = 1.0
    if len(series) >= 3:
        m = pd.concat(series.values(), axis=1).fillna(0.0)
        m = m.loc[:, m.std() > 0]
        if m.shape[1] >= 3:
            c = np.corrcoef(m.to_numpy().T)
            k = c.shape[0]
            rho = max(0.0, (c.sum() - k) / (k * (k - 1)))
            ratio = (k / (1 + (k - 1) * rho)) / k
    n_eff = max(2, int(round(n_raw * ratio)))
    # Null-hypothesis dispersion: a zero-skill strategy's per-period Sharpe
    # estimate over T days has variance ≈ 1/(T-1). (The cross-trial variance
    # mixes in REAL skill differences — e.g. a terrible incumbent next to good
    # candidates — and blocked every promotion in simulation.)
    t_len = max((len(v) for v in series.values()), default=0)
    null_var = 1.0 / (t_len - 1) if t_len > 30 else stats.annual_to_daily_var(sharpes)
    return n_eff, n_raw, null_var


# ---------------------------------------------------------------------- #
# Orchestration
# ---------------------------------------------------------------------- #

def run_learning_cycle(dp, config: dict, symbols: list, propose: Optional[int] = None,
                       seed: Optional[int] = None, now: Optional[datetime] = None,
                       bars: Optional[dict] = None) -> str:
    """One full learning iteration. Returns the report path."""
    now = _now(now)
    lcfg = config.get("learning", {})
    gcfg = lcfg.get("gates", {})
    pcfg = lcfg.get("promotion", {})
    store = CandidateStore(CANDIDATES_PATH)
    audit = LearningAuditLog(AUDIT_PATH)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    rng = random.Random(seed if seed is not None else int(now.strftime("%Y%m%d")))
    lines = [f"# Learning run {run_id}", ""]

    switched = apply_pending_promotion(config, now)
    if switched:
        lines += [f"**Applied announced promotion → `{switched}`**", ""]

    if bars is None:
        ev = lcfg.get("evaluation", {})
        bars = _load_bars(dp, symbols[:int(ev.get("max_symbols", 20))], int(ev.get("history_days", 1500)))
    if not bars:
        lines.append("**No usable price history — learning skipped.**")
        return _write_report(run_id, lines)
    lines += [f"Symbols ({len(bars)}): {', '.join(bars)}", ""]

    starts = window_bounds(bars, config)
    bh = buy_and_hold_daily(bars)
    act_g = active_genome(config)
    act_run = PortfolioRun(fam.make_strategy(act_g), bars)
    act_res, act_scored = evaluate_portfolio(ACTIVE_ID, act_run, bh, starts)
    save_results(act_res, run_id)
    act_sh = fam.metrics_from_daily(act_scored)["sharpe_ratio"]
    bh_sh = fam.metrics_from_daily(bh[bh.index >= starts["validation"]])["sharpe_ratio"]
    lines += ["## Active strategy", fam.describe(act_g),
              f"- validation+OOS Sharpe {act_sh:.2f} (equal-weight buy & hold {bh_sh:.2f})", ""]

    # --- 1. Propose ---
    stored = load_results()
    ranked = []
    for c in store.read_all():
        if c.status in (CandidateStatus.PAPER, CandidateStatus.PROMOTED):
            oos = [r for r in stored.get(c.candidate_id, []) if r.symbol == PORTFOLIO
                   and r.period in ("validation", "out_of_sample")]
            if oos:
                ranked.append((np.mean([r.sharpe_ratio for r in oos]), c.candidate_id, genome_of(c)))
    ranked.sort(key=lambda x: -x[0])
    parents = [("active", act_g)] + [(cid, g) for _, cid, g in ranked[:4]]
    n_total = propose if propose is not None else int(lcfg.get("propose_per_run", 3))
    live = [c for c in store.read_all()
            if c.status in (CandidateStatus.PROPOSED, CandidateStatus.EVALUATING, CandidateStatus.PAPER)]
    n_total = min(n_total, max(0, int(lcfg.get("max_live_candidates", 12)) - len(live)))
    n_explore = min(n_total, int(lcfg.get("explore_per_run", 1)))
    new = propose_candidates(store, audit, parents, n_total - n_explore, n_explore, rng, now) if n_total else []
    lines.append(f"## Proposed {len(new)} new candidate(s)")
    lines += [f"- `{c.candidate_id}`: {c.hypothesis}" for c in new]
    lines.append("")

    # --- 2/3. Evaluate + paper gate ---
    lines.append("## Offline evaluation (equal-weight portfolio, costs included)")
    runs, scored_by = {}, {}
    to_eval = [c for c in store.read_all()
               if c.status in (CandidateStatus.PROPOSED, CandidateStatus.EVALUATING, CandidateStatus.PAPER)]
    for c in to_eval:
        run = PortfolioRun(fam.make_strategy(genome_of(c)), bars)
        res, scored = evaluate_portfolio(c.candidate_id, run, bh, starts)
        save_results(res, run_id)
        runs[c.candidate_id], scored_by[c.candidate_id] = run, scored
    n_trials, n_raw, var_sr = _trial_stats(store, load_results(), [act_sh],
                                           {**scored_by, ACTIVE_ID: act_scored})
    luck = stats.expected_max_sharpe(n_trials, var_sr) * np.sqrt(252)
    lines.append(f"_Multiple-testing control: {n_raw} strategies tried (≈{n_trials} independent); "
                 f"the best of that many zero-skill strategies would show Sharpe ≈ {luck:.2f} by luck._")
    margin = float(gcfg.get("min_sharpe_margin_vs_active", 0.10))
    min_tr = int(gcfg.get("min_offline_trades", 10))
    psr_min = float(gcfg.get("psr_min_paper", 0.75))
    dsr_min_paper = float(gcfg.get("dsr_min_paper", 0.50))
    max_dd = float(lcfg.get("selection", {}).get("max_drawdown_pct", 15.0))
    for c in to_eval:
        if c.status not in (CandidateStatus.PROPOSED, CandidateStatus.EVALUATING):
            continue
        scored = scored_by[c.candidate_id]
        m = fam.metrics_from_daily(scored)
        w = runs[c.candidate_id].window(starts["validation"])
        dsr = stats.deflated_sharpe(scored, n_trials, var_sr)
        p = stats.psr(scored)
        reasons = []
        if w["trades"] < min_tr:
            reasons.append(f"trades {w['trades']} < {min_tr}")
        if m["sharpe_ratio"] < act_sh + margin:
            reasons.append(f"Sharpe {m['sharpe_ratio']:.2f} < active {act_sh:.2f}+{margin}")
        if abs(m["max_drawdown_pct"]) > max_dd:
            reasons.append(f"drawdown {abs(m['max_drawdown_pct']):.1f}% > {max_dd}%")
        if p < psr_min:
            reasons.append(f"PSR {p:.2f} < {psr_min}")
        if dsr < dsr_min_paper:
            reasons.append(f"deflated Sharpe {dsr:.2f} < {dsr_min_paper}")
        if reasons:
            c.status = CandidateStatus.REJECTED
            _set_notes(c, rejected="; ".join(reasons))
        else:
            c.status = CandidateStatus.PAPER
            _set_notes(c, paper_start=now.isoformat(), offline_sharpe=m["sharpe_ratio"], offline_dsr=dsr)
        store.save(c)
        audit.log_event("evaluation_complete", c.candidate_id, c.version,
                        details={"run_id": run_id, "sharpe": m["sharpe_ratio"], "dsr": dsr, "psr": p,
                                 "trades": w["trades"], "reasons": reasons},
                        rationale="; ".join(reasons) or "entered shadow paper")
        lines.append(f"- `{c.candidate_id}` {fam.describe(genome_of(c))} → **{c.status.value}** "
                     f"(Sharpe {m['sharpe_ratio']:.2f}, trades {w['trades']}, PSR {p:.2f}, DSR {dsr:.2f})"
                     + (f" — {'; '.join(reasons)}" if reasons else ""))
    lines.append("")

    # --- 4/5. Shadow paper + promotion gate ---
    lines.append("## Shadow paper (forward test on bars after paper start)")
    gate = _gate(config)
    ready = []
    max_paper_days = int(pcfg.get("max_paper_days", 120))
    dsr_min = float(gcfg.get("dsr_min_promotion", 0.80))
    for c in store.read_all():
        if c.status != CandidateStatus.PAPER:
            continue
        try:
            start = datetime.fromisoformat(_notes(c).get("paper_start"))
        except (TypeError, ValueError):
            start = now
            _set_notes(c, paper_start=start.isoformat())
            store.save(c)
        st = _naive_ts(start) + pd.Timedelta(days=1)
        run = runs.get(c.candidate_id) or PortfolioRun(fam.make_strategy(genome_of(c)), bars)
        fw, inc = run.window(st), act_run.window(st)
        days = (now - start).days
        paper = {"days_in_paper": days, "total_trades": fw["trades"], "symbols_traded": fw["symbols"],
                 "sharpe_ratio": fw["sharpe"], "max_drawdown_pct": fw["dd"], "net_return_pct": fw["ret"]}
        incumbent = {"sharpe_ratio": inc["sharpe"], "net_return_pct": inc["ret"]}
        a = gate.assess_for_promotion(c, paper, incumbent)
        combined = pd.concat([scored_by.get(c.candidate_id, pd.Series(dtype=float)), fw["daily"]])
        combined = combined[~combined.index.duplicated(keep="last")]
        dsr = stats.deflated_sharpe(combined, n_trials, var_sr)
        is_ready = a.decision in (PromotionDecision.READY_FOR_PROMOTION, PromotionDecision.APPROVED)
        if is_ready and dsr < dsr_min:
            is_ready = False
            a.gates_failed.append(f"deflated Sharpe {dsr:.2f} < {dsr_min}")
        audit.log_event("promotion_assessed", c.candidate_id, c.version,
                        details={"paper": paper, "active": incumbent, "dsr": dsr, **a.to_dict()},
                        rationale="; ".join(a.gates_failed) or "ready")
        verdict = "ready" if is_ready else "needs_more_data"
        if is_ready:
            ready.append((fw["sharpe"], c, {"paper_days": days, "paper_sharpe": fw["sharpe"],
                                            "active_sharpe": inc["sharpe"], "paper_ret": fw["ret"],
                                            "paper_trades": fw["trades"], "dsr": dsr,
                                            "n_trials": n_trials}))
        elif days >= max_paper_days or (days >= 30 and fw["sharpe"] < inc["sharpe"] - 0.5):
            c.status = CandidateStatus.REJECTED
            _set_notes(c, rejected=f"shadow paper failed after {days}d")
            store.save(c)
            verdict = "retired"
        lines.append(f"- `{c.candidate_id}` {days}d: Sharpe {fw['sharpe']:.2f} vs active {inc['sharpe']:.2f}, "
                     f"ret {fw['ret']:.2f}%, trades {fw['trades']}, DSR {dsr:.2f} → **{verdict}**"
                     + (f" ({'; '.join(a.gates_failed[:2])})" if verdict == "needs_more_data" else ""))
    lines.append("")

    # --- 6. Promotion ---
    lines.append("## Promotion")
    pending = _read_json(PENDING_PROMOTION_PATH, None)
    if pending:
        lines.append(f"- Pending switch to `{pending['candidate_id']}` after {pending['apply_after']}")
    elif ready:
        ready.sort(key=lambda x: -x[0])
        _, best, ev = ready[0]
        if pcfg.get("allow_auto_promotion", False):
            if announce_promotion(config, best, ev, now):
                lines.append(f"- Announced on Discord: switching to `{best.candidate_id}` after the grace period")
            else:
                lines.append(f"- `{best.candidate_id}` is READY but the Discord announcement failed — "
                             f"NOT switching (will retry next run)")
        else:
            lines.append(f"- `{best.candidate_id}` READY — approve with "
                         f"`python main.py learning-approve --candidate-id {best.candidate_id}`")
    else:
        lines.append("- No candidate is ready. Active strategy retained.")
    lines.append("")

    sel = _selector(config).select(
        [c for c in store.read_all() if c.status == CandidateStatus.PAPER],
        {k: [r for r in v if r.symbol == PORTFOLIO] for k, v in load_results().items()},
    )
    audit.log_event("selection_run", sel.selected_candidate_id or "", "", details=sel.to_dict(),
                    rationale=sel.selection_reason)
    lines += ["## Leaderboard (paper candidates)", f"- {sel.selection_reason or 'n/a'}", ""]
    return _write_report(run_id, lines)


def _write_report(run_id: str, lines: list) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"{run_id}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
