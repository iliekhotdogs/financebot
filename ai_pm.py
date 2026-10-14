"""
AI portfolio manager for ETF mode (config etf_trend.ai).

The trend rule + overlay still compute the base targets. Once per trading day
an LLM (Kimi K3 via the NVIDIA API by default) sees a compact briefing and may
TILT those targets (each ETF up/down by at most max_tilt of equity) or HOLD an
ETF (skip today's rule trade for it). Code enforces every limit; the model
never places orders and cannot change the risk limits.

No loops, by construction:
  - at most one model call per data date (journal/ai_pm.json records it,
    success or failure); a content error is never retried;
  - a bad or out-of-range answer, or a model failure, keeps the plan already
    in place ("carried"); after stale_days without a good answer the tilts
    expire and the bot runs the pure rules;
  - at most one Discord yes/no question per day, only for big moves, with a
    deadline after which the safe choice (no change) applies; after a "no",
    big moves are not asked again for cooldown_days.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time as dtime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger("finance-bot.ai")

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(BOT_DIR, "journal", "ai_pm.json")
PACIFIC = ZoneInfo("America/Los_Angeles")
YES, NO = "✅", "❌"

DEFAULTS = {
    "enabled": False,
    "model": None,                 # default: advisor.model
    "base_url": None,              # default: advisor.base_url
    "api_key_env": None,           # default: advisor.api_key_env
    "timeout_seconds": 120,
    "max_tokens": 4000,            # room for a reasoning model's thinking; the answer itself is short
    "decide_after": "06:00",       # Pacific; first trading cycle after this asks the model
    "max_tilt": 0.15,              # per ETF, share of equity
    "max_total_tilt": 0.40,        # sum of |tilts|
    "min_change": 0.02,            # a tilt change this big trades off-cycle
    "ask_turnover": 0.25,          # ask before moving more than this share of equity...
    "ask_exposure_increase": 0.10, # ...or raising total exposure by more than this
    "approval_deadline": "12:00",  # Pacific, on the next trading day after the data date
    "approver_user_id": "",        # only this Discord user's reaction counts (blank: any person)
    "cooldown_days": 3,            # after a "no", don't ask about big moves for this many days
    "stale_days": 5,               # tilts older than this (trading days) expire → pure rules
    "history": 10,                 # past decisions shown to the model
    "news_headlines": 5,           # market headlines (Finnhub) in the briefing; 0 = off
}

SYSTEM_PROMPT = """You manage a PAPER-money ETF portfolio once a day. A tested rule sets the base weights: each ETF
is held while above its 10-month average (4 tranches, inverse-volatility weights), plus a 40% SPY core,
all at 1.5x leverage. You may adjust the rule's weights. Code enforces every limit; you never place orders.

Reply with ONLY this JSON:
{"tilts": {"<ETF>": <percentage points of equity, e.g. 5 or -3>}, "hold": ["<ETF>"],
 "confidence": <0..1>, "summary": "<max 300 chars, plain English for the owner: what you changed and why>"}

Rules:
- tilts: omit ETFs you leave alone. Each between -MAXT and +MAXT; sum of absolute tilts at most MAXTOT.
  BIL (T-bills) cannot be tilted; it absorbs the remainder.
- hold: skip today's rule trade for that ETF (keep its current weight). Use rarely.
- Default is NO change: the rule has a 20-year record. Tilt only for a clear reason in the data given,
  and keep tilts small. Learn from "past" (bp = your result vs the rule; "ans" = the owner's yes/no).
- Use only the data given. Never invent prices or news.
Units: all numbers are percent. "rule"/"now" = weight of equity; "vs_avg" = price vs 10-month average;
r1m/r3m = 1- and 3-month return; vol = annualised volatility; "in" = tranches holding it (of 4)."""


# ---------------------------------------------------------------------- #
# Settings
# ---------------------------------------------------------------------- #

def settings(config: dict) -> dict:
    """etf_trend.ai merged with defaults; model connection defaults to the advisor's (Kimi)."""
    s = {**DEFAULTS, **(((config.get("etf_trend") or {}).get("ai")) or {})}
    adv = config.get("advisor") or {}
    s["model"] = s["model"] or adv.get("model", "moonshotai/kimi-k3")
    s["base_url"] = s["base_url"] or adv.get("base_url")
    s["api_key_env"] = s["api_key_env"] or adv.get("api_key_env", "NVIDIA_API_KEY")
    s["enabled"] = bool(s["enabled"])
    for k in ("max_tilt", "max_total_tilt", "min_change", "ask_turnover", "ask_exposure_increase"):
        s[k] = float(s[k])
    if not (0 < s["max_tilt"] <= 0.25 and 0 < s["max_total_tilt"] <= 0.6):
        raise ValueError("etf_trend.ai: max_tilt must be in (0, 0.25] and max_total_tilt in (0, 0.6]")
    return s


def active(config: dict) -> bool:
    return settings(config)["enabled"]


def label(config: dict, base: str) -> str:
    """Strategy label suffix: a new label restarts the paper track record and readiness clock."""
    return f"{base}+ai" if active(config) else base


# ---------------------------------------------------------------------- #
# Target maths (pure)
# ---------------------------------------------------------------------- #

def apply(rules: pd.Series, plan: Optional[dict], current_w: dict, cap: float,
          cash_symbol: str = "BIL") -> pd.Series:
    """Rule targets + AI tilts/holds → final targets (incl. cash symbol).
    Weights never go below 0; total risky exposure is scaled down to `cap`;
    the cash symbol takes whatever is left (0 when borrowing)."""
    risky = rules.drop(cash_symbol, errors="ignore").astype(float).copy()
    plan = plan or {}
    for s, t in (plan.get("tilts") or {}).items():
        risky[s] = float(risky.get(s, 0.0)) + float(t)
    for s in plan.get("hold") or []:
        risky[s] = float(current_w.get(s, 0.0))
    risky = risky.clip(lower=0.0)
    risky = risky[risky > 1e-9]
    g = float(risky.sum())
    if g > cap:
        risky = risky * (cap / g)
    out = risky.copy()
    out[cash_symbol] = max(0.0, 1.0 - float(risky.sum()))
    return out


def gross(t: pd.Series, cash_symbol: str = "BIL") -> float:
    return float(t.drop(cash_symbol, errors="ignore").sum())


def turnover(a: pd.Series, b: pd.Series) -> float:
    """One-way share of equity that moves between two target sets."""
    idx = a.index.union(b.index)
    return 0.5 * float((a.reindex(idx).fillna(0) - b.reindex(idx).fillna(0)).abs().sum())


def changed(plan: Optional[dict], executed: Optional[dict], min_change: float) -> bool:
    """True when the plan differs materially from what was last traded."""
    p, e = plan or {}, executed or {}
    pt, et_ = p.get("tilts") or {}, e.get("tilts") or {}
    if set(p.get("hold") or []) != set(e.get("hold") or []):
        return True
    return any(abs(float(pt.get(s, 0)) - float(et_.get(s, 0))) >= min_change - 1e-12 for s in set(pt) | set(et_))


def validate(raw: dict, symbols: list, s: dict) -> tuple:
    """Model JSON → (plan, None) or (None, error). Tilts come in percentage points."""
    if not isinstance(raw, dict):
        return None, "answer is not a JSON object"
    tilts_in, hold_in = raw.get("tilts") or {}, raw.get("hold") or []
    if not isinstance(tilts_in, dict) or not isinstance(hold_in, list):
        return None, "tilts must be an object and hold a list"
    tilts = {}
    for sym, v in tilts_in.items():
        sym = str(sym).upper().strip()
        if sym not in symbols:
            return None, f"unknown or untiltable symbol {sym}"
        try:
            v = float(v) / 100.0
        except (TypeError, ValueError):
            return None, f"tilt for {sym} is not a number"
        if not np.isfinite(v) or abs(v) > s["max_tilt"] + 1e-9:
            return None, f"tilt {sym} {v * 100:+.1f}% is outside ±{s['max_tilt'] * 100:.0f}%"
        if abs(v) >= 0.001:
            tilts[sym] = round(v, 4)
    if sum(abs(v) for v in tilts.values()) > s["max_total_tilt"] + 1e-9:
        return None, f"total tilt is above {s['max_total_tilt'] * 100:.0f}%"
    hold = []
    for sym in hold_in:
        sym = str(sym).upper().strip()
        if sym not in symbols:
            return None, f"unknown symbol in hold: {sym}"
        if sym in tilts:
            return None, f"{sym} is both tilted and held"
        hold.append(sym)
    try:
        conf = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    summary = " ".join(str(raw.get("summary") or "").split())[:300]
    return {"tilts": tilts, "hold": sorted(set(hold)), "confidence": round(min(max(conf, 0.0), 1.0), 2),
            "summary": summary}, None


# ---------------------------------------------------------------------- #
# Briefing (compact, token-efficient)
# ---------------------------------------------------------------------- #

def _pct(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x) * 100, nd)


def build_briefing(closes: pd.DataFrame, rules: pd.Series, info: dict, current_w: dict, equity: float,
                   peak: Optional[float], cash: float, symbols: list, s: dict, past: list, news: list) -> dict:
    latest = {}
    for k in sorted(info.get("signals", {}), key=lambda k: info.get("tranche_last", {}).get(k) or ""):
        for sym, v in info["signals"][k].items():
            latest[sym] = v                                   # most recently updated tranche wins
    etf = {}
    for sym in symbols + ["BIL"]:
        c = closes[sym].dropna() if sym in closes else pd.Series(dtype=float)
        row = {"rule": _pct(rules.get(sym, 0.0)), "now": _pct(current_w.get(sym, 0.0))}
        if sym != "BIL" and len(c) > 64:
            row.update(vs_avg=round(latest[sym]["pct_vs_avg"], 1) if sym in latest else None,
                       **{"in": f"{info.get('in_market', {}).get(sym, 0)}/{info.get('tranches', 4)}"},
                       r1m=_pct(c.iloc[-1] / c.iloc[-22] - 1), r3m=_pct(c.iloc[-1] / c.iloc[-64] - 1),
                       vol=_pct(c.pct_change().iloc[-63:].std() * np.sqrt(252), 0))
        etf[sym] = row
    b = {
        "date": info.get("asof"),
        "acct": {"eq": round(equity), "dd": _pct(equity / peak - 1) if peak else None,
                 "gross": _pct(gross(pd.Series(current_w))), "rule_gross": _pct(gross(rules)),
                 "cash": _pct(cash / equity) if equity else None},
        "limits": {"max_tilt": s["max_tilt"] * 100, "max_total": s["max_total_tilt"] * 100},
        "etf": etf,
        "past": past,
    }
    if "SPY" in closes:
        spy = closes["SPY"].dropna()
        b["spy_r5d"] = _pct(spy.iloc[-1] / spy.iloc[-6] - 1) if len(spy) > 6 else None
    if news:
        b["news"] = news
    return b


def _outcome_bp(delta: dict, closes: pd.DataFrame, start: str, end: str) -> Optional[float]:
    """Approximate result of an applied tilt vs the rules between two data dates, in bp."""
    if not delta:
        return 0.0
    try:
        a, b = pd.Timestamp(start), pd.Timestamp(end)
        r = 0.0
        for sym, w in delta.items():
            c = closes[sym].dropna()
            r += float(w) * (float(c.asof(b)) / float(c.asof(a)) - 1)
        return round(r * 10_000, 1)
    except Exception:
        return None


def past_summary(store: dict, closes: pd.DataFrame, asof: str, n: int) -> list:
    """Last n decisions for the briefing, scored up to `asof`. Also fills in each record's bp."""
    recs = store.get("decisions", [])
    for i, r in enumerate(recs):
        end = recs[i + 1]["asof"] if i + 1 < len(recs) else asof
        if r.get("delta") is not None and end != r["asof"]:
            r["bp"] = _outcome_bp(r["delta"], closes, r["asof"], end)
    out = []
    for r in recs[-n:]:
        p = r.get("proposal") or {}
        out.append({"d": r["asof"], "tilt": {k: round(v * 100, 1) for k, v in (p.get("tilts") or {}).items()},
                    "hold": p.get("hold") or [], "res": r.get("status"), "ans": r.get("answer"),
                    "bp": r.get("bp")})
    return out


def _headlines(n: int) -> list:
    key = os.getenv("FINNHUB_API_KEY")
    if not key or n <= 0:
        return []
    try:
        url = "https://finnhub.io/api/v1/news?" + urllib.parse.urlencode({"category": "general", "token": key})
        with urllib.request.urlopen(url, timeout=10) as r:
            items = json.loads(r.read())
        return [str(i.get("headline", ""))[:140] for i in items[:n] if i.get("headline")]
    except Exception as e:
        logger.warning("headlines unavailable: %s", type(e).__name__)
        return []


# ---------------------------------------------------------------------- #
# Model call — ONE call; a bad answer is never retried
# ---------------------------------------------------------------------- #

def _call_model(s: dict, briefing: dict) -> tuple:
    """(raw dict, None) or (None, error). Retries only when no answer arrived at all
    (connection / 5xx); never re-asks to get a different answer."""
    if os.getenv("PYTEST_CURRENT_TEST"):
        return None, "model calls are disabled under pytest"
    key = os.getenv(s["api_key_env"] or "")
    if not key:
        return None, f"{s['api_key_env']} not set"
    try:
        import openai
    except ImportError:
        return None, "openai package not installed"
    client = openai.OpenAI(api_key=key, base_url=s["base_url"], timeout=float(s["timeout_seconds"]))
    sysmsg = (SYSTEM_PROMPT.replace("MAXTOT", f"{s['max_total_tilt'] * 100:.0f}")
              .replace("MAXT", f"{s['max_tilt'] * 100:.0f}"))
    msgs = [{"role": "system", "content": sysmsg},
            {"role": "user", "content": json.dumps(briefing, separators=(",", ":"))}]
    transient = tuple(c for c in (getattr(openai, "APIConnectionError", None),
                                  getattr(openai, "InternalServerError", None)) if isinstance(c, type))
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(model=s["model"], messages=msgs, temperature=0.2,
                                                  max_tokens=int(s["max_tokens"]),
                                                  response_format={"type": "json_object"})
            break
        except transient as e:
            if attempt == 1:
                return None, f"{type(e).__name__}"
        except Exception as e:
            return None, f"{type(e).__name__}: {str(e)[:120]}"
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(raw), None
    except (ValueError, TypeError):
        return None, f"answer was not JSON (finish_reason={resp.choices[0].finish_reason})"


_REAL_CALL = _call_model


# ---------------------------------------------------------------------- #
# Discord yes/no (needs DISCORD_BOT_TOKEN; a webhook cannot read reactions)
# ---------------------------------------------------------------------- #

def _discord(method: str, path: str, body: Optional[dict] = None):
    token = os.getenv("DISCORD_BOT_TOKEN")
    req = urllib.request.Request(f"https://discord.com/api/v10{path}", method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bot {token}", "Content-Type": "application/json",
                                          "User-Agent": "DiscordBot (finance-bot, 1.0)"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = r.read()
        return json.loads(data) if data else None


def _channel(config: dict) -> str:
    return str((config.get("notifications") or {}).get("discord_channel_id") or os.getenv("DISCORD_CHANNEL_ID") or "")


def ask(config: dict, text: str) -> Optional[str]:
    """Post a question with ✅/❌ reactions; returns the message id, or None if it can't ask."""
    import notify
    if not notify._enabled(config) or not os.getenv("DISCORD_BOT_TOKEN") or not _channel(config):
        return None
    try:
        prefix = str((config.get("notifications") or {}).get("prefix", "🤖 finance-bot"))[:80]
        msg = _discord("POST", f"/channels/{_channel(config)}/messages",
                       {"content": f"**{prefix}** — {notify._compact(text, 1800)}", "allowed_mentions": {"parse": []}})
        mid = str(msg["id"])
        for e in (YES, NO):
            _discord("PUT", f"/channels/{_channel(config)}/messages/{mid}/reactions/{urllib.parse.quote(e)}/@me")
        return mid
    except Exception as e:
        logger.warning("Discord question failed: %s", type(e).__name__)
        return None


def read_answer(config: dict, message_id: str, approver: str = "") -> Optional[str]:
    """'yes' / 'no' from a person's reaction, else None. Both reactions → 'no' (safe)."""
    got = {}
    for word, e in (("yes", YES), ("no", NO)):
        try:
            users = _discord("GET", f"/channels/{_channel(config)}/messages/{message_id}"
                                    f"/reactions/{urllib.parse.quote(e)}?limit=25") or []
        except Exception as ex:
            logger.warning("Discord reactions read failed: %s", type(ex).__name__)
            return None
        got[word] = any(not u.get("bot") and (not approver or str(u.get("id")) == str(approver)) for u in users)
    if got["no"]:
        return "no"
    return "yes" if got["yes"] else None


# ---------------------------------------------------------------------- #
# Daily plan (state machine, one record per data date)
# ---------------------------------------------------------------------- #

def _load(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"decisions": []}


def _save(path: str, store: dict) -> None:
    store["decisions"] = store.get("decisions", [])[-400:]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(store, f, indent=1, default=str)
    os.replace(tmp, path)


def _hhmm(v: str) -> dtime:
    h, m = str(v).split(":")
    return dtime(int(h), int(m))


def _deadline(asof: str, s: dict) -> datetime:
    nxt = (pd.Timestamp(asof) + pd.offsets.BDay(1)).date()
    return datetime.combine(nxt, _hhmm(s["approval_deadline"]), tzinfo=PACIFIC)


def _carried(store: dict, asof: str, s: dict) -> tuple:
    """(plan, since) currently in force from earlier days, or ({}, None) when expired."""
    for r in reversed(store.get("decisions", [])):
        if r["asof"] >= asof:
            continue
        since = r.get("effective_since")
        if not r.get("effective") or not since:
            return {}, None
        age = len(pd.bdate_range(since, asof)) - 1
        return (dict(r["effective"]), since) if age <= int(s["stale_days"]) else ({}, None)
    return {}, None


def _fmt(plan: dict) -> str:
    parts = [f"{k} {v * 100:+.0f}%" for k, v in sorted((plan.get("tilts") or {}).items(), key=lambda kv: -abs(kv[1]))]
    parts += [f"hold {h}" for h in plan.get("hold") or []]
    return ", ".join(parts) or "no changes (pure rules)"


def daily_plan(config: dict, *, asof: str, rules: pd.Series, info: dict, closes: pd.DataFrame,
               current_w: dict, equity: float, cash: float, peak: Optional[float], cap: float,
               symbols: list, execute: bool, now: Optional[datetime] = None,
               store_path: Optional[str] = None, caller=None) -> dict:
    """
    The AI plan in force for this cycle: {"tilts", "hold", "status", "note"}.
    Asks the model at most once per data date (only in trading cycles, after
    decide_after), resolves yes/no questions, and never blocks the rules.
    """
    import notify
    s = settings(config)
    path = store_path or STORE_PATH
    now = (now or datetime.now(timezone.utc)).astimezone(PACIFIC)
    store = _load(path)
    recs = store.setdefault("decisions", [])
    rec = recs[-1] if recs and recs[-1]["asof"] == asof else None
    carried, since = _carried(store, asof, s)
    cash_sym = "BIL"

    model_fn = caller or _call_model
    if os.getenv("PYTEST_CURRENT_TEST") and model_fn is _REAL_CALL:
        # other tests that run a cycle with the shipped config: no model call, no record, no Discord
        return {**carried, "status": "none", "note": "disabled under pytest"}
    if rec is None and execute and now.time() >= _hhmm(s["decide_after"]):
        past = past_summary(store, closes, asof, int(s["history"]))
        brief = build_briefing(closes, rules, info, current_w, equity, peak, cash, symbols, s, past,
                               _headlines(int(s["news_headlines"])))
        raw, err = model_fn(s, brief)
        plan, verr = (None, err) if err else validate(raw, symbols, s)
        rec = {"asof": asof, "decided_at": now.isoformat(), "model": s["model"], "proposal": plan,
               "briefing_chars": len(json.dumps(brief, separators=(",", ":")))}
        if plan is None:
            rec.update(status="fallback", error=verr, effective=carried, effective_since=since)
            notify.send(config, f"🧠 **AI plan** (data through {asof}): no usable answer today ({verr}). "
                                f"Keeping the current plan: {_fmt(carried)}.", kind="ai")
        else:
            new_t = apply(rules, plan, current_w, cap, cash_sym)
            cur_t = apply(rules, carried, current_w, cap, cash_sym)
            move, up = turnover(new_t, cur_t), gross(new_t) - gross(cur_t)
            big = move > s["ask_turnover"] or up > s["ask_exposure_increase"]
            head = (f"🧠 **AI plan** (data through {asof}, {s['model'].split('/')[-1]}, "
                    f"confidence {plan['confidence']:.0%})\n{plan['summary'] or '(no summary)'}\n"
                    f"Changes vs rules: {_fmt(plan)} · exposure {gross(new_t) * 100:.0f}% "
                    f"(rules {gross(rules) * 100:.0f}%)")
            if not big:
                rec.update(status="accepted", effective=plan, effective_since=asof)
                notify.send(config, head, kind="ai")
            else:
                why = (f"moves {move * 100:.0f}% of the account" if move > s["ask_turnover"]
                       else f"raises exposure by {up * 100:.0f} points")
                last_no = next((r["asof"] for r in reversed(recs) if r.get("answer") == "no"), None)
                if last_no and len(pd.bdate_range(last_no, asof)) - 1 <= int(s["cooldown_days"]):
                    rec.update(status="declined", answer="cooldown", effective=carried, effective_since=since)
                    notify.send(config, head + f"\nThis {why}; you said no on {last_no}, so it is skipped "
                                               f"(no questions for {s['cooldown_days']} days after a no).", kind="ai")
                else:
                    dl = _deadline(asof, s)
                    mid = ask(config, head + f"\n❓ This {why}. React {YES} to approve or {NO} to skip. "
                                             f"No answer by {dl:%a %H:%M} PT = skip.")
                    if mid:
                        rec.update(status="pending", question_id=mid, deadline=dl.isoformat(),
                                   effective=carried, effective_since=since)
                    else:
                        rec.update(status="declined", answer="cannot_ask", effective=carried, effective_since=since)
                        notify.send(config, head + f"\nThis {why}, but I can't ask you (needs DISCORD_BOT_TOKEN), "
                                                   f"so it is skipped.", kind="ai")
            rec["delta"] = {k: round(float(v), 4) for k, v in
                            (apply(rules, rec["effective"], current_w, cap, cash_sym)
                             .subtract(rules, fill_value=0.0)).items() if abs(v) > 1e-6}
        recs.append(rec)
        _save(path, store)

    elif rec is not None and rec.get("status") == "pending" and execute:
        ans = read_answer(config, rec["question_id"], s["approver_user_id"])
        if ans is None and now >= datetime.fromisoformat(rec["deadline"]):
            ans = "expired"
        if ans:
            rec["answer"] = ans
            if ans == "yes":
                rec.update(status="approved", effective=rec["proposal"], effective_since=asof)
                notify.send(config, f"🧠 Approved — applying the AI plan: {_fmt(rec['proposal'])}.", kind="ai")
            else:
                rec.update(status="declined")
                notify.send(config, f"🧠 {'Skipped (no answer by the deadline)' if ans == 'expired' else 'Skipped'} "
                                    f"— keeping the current plan: {_fmt(rec.get('effective') or {})}.", kind="ai")
            rec["delta"] = {k: round(float(v), 4) for k, v in
                            (apply(rules, rec.get("effective"), current_w, cap, cash_sym)
                             .subtract(rules, fill_value=0.0)).items() if abs(v) > 1e-6}
            _save(path, store)

    if rec is None:
        return {**carried, "status": "carried" if carried else "none", "note": "no decision yet today"}
    eff = rec.get("effective") or {}
    return {"tilts": dict(eff.get("tilts") or {}), "hold": list(eff.get("hold") or []),
            "status": rec["status"], "note": rec.get("error") or rec.get("answer") or ""}


def dry_run(config: dict, *, asof: str, rules: pd.Series, info: dict, closes: pd.DataFrame,
            current_w: dict, equity: float, cash: float, peak: Optional[float], cap: float,
            symbols: list) -> dict:
    """One real model call that saves nothing and posts nothing (for `main.py ai-decide`)."""
    s = settings(config)
    store = _load(STORE_PATH)
    brief = build_briefing(closes, rules, info, current_w, equity, peak, cash, symbols, s,
                           past_summary(store, closes, asof, int(s["history"])), _headlines(int(s["news_headlines"])))
    raw, err = _call_model(s, brief)
    plan, verr = (None, err) if err else validate(raw, symbols, s)
    final = apply(rules, plan, current_w, cap) if plan else None
    return {"briefing": brief, "briefing_chars": len(json.dumps(brief, separators=(",", ":"))),
            "raw": raw, "plan": plan, "error": verr, "final": final}
