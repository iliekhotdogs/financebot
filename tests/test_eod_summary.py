"""Read-only EOD prose, exact paper data and lossless Discord delivery."""
import csv
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import main
import notify
from eod_summary import build_summary, split_message

NOW = datetime(2026, 10, 7, 17, tzinfo=ZoneInfo("America/Los_Angeles"))


def fixture_paths(tmp_path, *, trades=(), fills=(), equity=(), state=None):
    j = tmp_path / "journal"
    j.mkdir()
    for name, rows, fields in (
        ("trade_log.csv", trades, "timestamp symbol action qty price order_id status source"),
        ("etf_fills.csv", fills, "order_id symbol filled_avg_price slippage_bp filled_at status"),
        ("equity_curve.csv", equity, "date equity strategy_version"),
    ):
        with (j / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields.split(), extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    (j / "etf_state.json").write_text(json.dumps(state or {}))
    return tmp_path


def test_daily_report_includes_todays_ai_decision_summary(tmp_path):
    base = fixture_paths(tmp_path)
    (base / "journal" / "ai_pm.json").write_text(json.dumps({"decisions": [{
        "asof": "2026-10-06", "decided_at": "2026-10-07T06:30:00-07:00",
        "model": "moonshotai/kimi-k3", "status": "accepted",
        "proposal": {"summary": "Commodity momentum favors DBC over VNQ.",
                     "confidence": 0.35, "tilts": {"DBC": 0.03, "VNQ": -0.03}},
        "effective": {"tilts": {"DBC": 0.03, "VNQ": -0.03}}
    }]}))
    msg = build_summary({"strategy": {"mode": "etf_trend"}, "etf_trend": {"ai": {"enabled": True}}},
                        {}, [], {}, base=base, now=NOW)
    assert "AI portfolio manager" in msg
    assert "data through 2026-10-06" in msg
    assert "accepted" in msg
    assert "Commodity momentum favors DBC over VNQ." in msg
    assert "DBC +3%" in msg and "VNQ -3%" in msg
    assert "confidence 35%" in msg


def test_ai_summary_does_not_relabel_old_decision_as_today(tmp_path):
    base = fixture_paths(tmp_path)
    (base / "journal" / "ai_pm.json").write_text(json.dumps({"decisions": [{
        "asof": "2026-10-05", "decided_at": "2026-10-06T06:30:00-07:00",
        "status": "accepted", "proposal": {"summary": "Yesterday's thesis"}
    }]}))
    msg = build_summary({"etf_trend": {"ai": {"enabled": True}}}, {}, [], {}, base=base, now=NOW)
    assert "No new AI decision recorded today." in msg
    assert "Yesterday's thesis" not in msg


def test_ai_pending_and_failed_record_are_honest(tmp_path):
    base = fixture_paths(tmp_path)
    path = base / "journal" / "ai_pm.json"
    path.write_text(json.dumps({"decisions": [{
        "asof": "2026-10-06", "decided_at": "2026-10-07T06:30:00-07:00",
        "status": "pending", "proposal": {"summary": "Wait for approval", "tilts": {"DBC": 0.04}}
    }]}))
    config = {"etf_trend": {"ai": {"enabled": True}}}
    msg = build_summary(config, {}, [], {}, base=base, now=NOW)
    assert "pending" in msg and "Proposed tilts" in msg
    assert "applying" not in msg.lower()
    path.write_text('{broken json')
    msg = build_summary(config, {}, [], {}, base=base, now=NOW)
    assert "No new AI decision recorded today." in msg


def test_ai_generated_summary_is_bounded_and_cannot_ping_discord(tmp_path):
    base = fixture_paths(tmp_path)
    (base / "journal" / "ai_pm.json").write_text(json.dumps({"decisions": [{
        "asof": "2026-10-06", "decided_at": "2026-10-07T06:30:00-07:00",
        "status": "accepted", "proposal": {"summary": "@everyone\n" + "x" * 2000}
    }]}))
    msg = build_summary({"etf_trend": {"ai": {"enabled": True}}}, {}, [], {}, base=base, now=NOW)
    assert "@everyone" not in msg
    assert len(msg.split("Today's decision (")[1].split("\n")[0]) < 400


def test_quiet_day_opens_with_no_move_and_all_sections(tmp_path):
    base = fixture_paths(tmp_path, equity=[{"date": "2026-10-07", "equity": "100000", "strategy_version": "etf_trend_v2.0"}])
    msg = build_summary({"strategy": {"mode": "etf_trend"}},
                        {"equity": "100000", "last_equity": "100000", "cash": "100000"}, [],
                        {"status": "NOT_READY", "days": 3, "tracking_error_pct": None}, base=base, now=NOW)
    for heading in ("Today in brief", "Performance", "Trades today", "Holdings", "Trend signals", "Next up", "Alerts today"):
        assert heading in msg
    assert "No change" in msg.split("Today in brief")[1].split("Performance")[0]
    assert "No trades" in msg and "n/a" in msg
    assert "20% readiness" in msg and "40%" not in msg  # absent halt setting must not be invented


def test_trades_fills_exposure_and_benchmark(tmp_path):
    base = fixture_paths(tmp_path,
        trades=[{"timestamp": "2026-10-07T17:10:00+00:00", "symbol": "SPY", "action": "buy", "qty": "2", "price": "101", "order_id": "abc", "status": "submitted", "source": "etf_trend"},
                {"timestamp": "2026-10-07T18:00:00+00:00", "symbol": "EFA", "action": "sell", "qty": "3", "price": "55", "status": "submitted", "source": "etf_trend"}],
        fills=[{"order_id": "abc", "symbol": "SPY", "filled_avg_price": "102", "slippage_bp": "10", "filled_at": "2026-10-07T17:12:00+00:00", "status": "filled"}],
        equity=[{"date": "2026-10-06", "equity": "10000", "strategy_version": "etf_trend_v2.0"},
                {"date": "2026-10-07", "equity": "10200", "strategy_version": "etf_trend_v2.0"}],
        state={"status": "idle", "signals": {"0": {"SPY": {"pct_vs_avg": -0.3, "in": True}}},
               "in_market": {"SPY": 1}, "tranches": 4})
    msg = build_summary({"strategy": {"mode": "etf_trend"}, "etf_trend": {"max_drawdown_halt_pct": 40}},
                        {"equity": "10200", "last_equity": "10000", "cash": "-2000"},
                        [{"symbol": "SPY", "qty": "100", "current_price": "102", "market_value": "10200"}],
                        {"status": "NOT_READY", "days": 2, "tracking_error_pct": 1.2, "max_drawdown_pct": 5},
                        base=base, now=NOW, spy_closes=(100, 101))
    assert "+$200" in msg and "+1.0%" in msg
    assert "SPY: fill $102" in msg and "10.0 bp" in msg
    assert "EFA" in msg and "planned $55" in msg and "fill n/a" in msg
    assert "Borrowed $2,000" in msg and "100.0%" in msg
    assert "Watch" in msg
    assert "20% readiness limit" in msg and "40% trading halt" in msg


def test_fill_price_survives_missing_slippage(tmp_path):
    base = fixture_paths(tmp_path,
        trades=[{"timestamp": "2026-10-07T17:00:00+00:00", "symbol": "SPY", "action": "buy", "qty": "1", "price": "100", "order_id": "id", "status": "submitted"}],
        fills=[{"order_id": "id", "symbol": "SPY", "filled_avg_price": "101", "filled_at": "2026-10-07T17:05:00+00:00", "status": "filled"}])
    msg = build_summary({}, {}, [], {}, base=base, now=NOW)
    assert "fill $101" in msg and "slippage n/a" in msg


def test_preview_uses_copy_data_without_sending_or_writing(tmp_path):
    from scripts.preview_eod_summary import preview
    base = fixture_paths(tmp_path, state={"status": "idle"})
    result = preview({"strategy": {"mode": "etf_trend"}}, base=base, now=NOW)
    assert "📊 Paper bot today" in result and result.startswith("Source: ")
    assert "Equity n/a" in result
    assert sorted(p.name for p in (base / "journal").iterdir()) == ["equity_curve.csv", "etf_fills.csv", "etf_state.json", "trade_log.csv"]


def test_blocked_overrides_good_move_and_alerts(tmp_path):
    base = fixture_paths(tmp_path, state={"status": "blocked — incomplete price data (SPY)", "updated_at": "2026-10-07T19:00:00+00:00"})
    msg = build_summary({"strategy": {"mode": "etf_trend"}}, {"equity": "10200", "last_equity": "10000"}, [], {}, base=base, now=NOW)
    brief = msg.split("Today in brief")[1].split("Performance")[0]
    assert "paused" in brief.lower() and "SPY" in brief
    assert "incomplete price data" in msg.split("Alerts today")[1]


def test_missing_prices_are_na_not_zero(tmp_path):
    base = fixture_paths(tmp_path, trades=[{"timestamp": "2026-10-07T17:00:00+00:00", "symbol": "DBC", "action": "buy", "qty": "1", "status": "submitted"}])
    msg = build_summary({"strategy": {"mode": "etf_trend"}}, {}, [{"symbol": "DBC", "qty": "1"}], {}, base=base, now=NOW)
    assert "planned n/a" in msg and "fill n/a" in msg and "DBC n/a" in msg
    assert "Equity n/a" in msg and "SPY day n/a" in msg


def test_lossless_split_preserves_every_character_and_discord_limit():
    text = "📊 Paper bot today\n" + "\n".join("asset %03d " % i + "x" * 100 for i in range(75))
    parts = split_message(text, prefix="🤖 finance-bot")
    assert len(parts) > 1
    assert all(len("**🤖 finance-bot** — " + p) <= 2000 for p in parts)
    assert "".join(parts) == text


def test_daily_transport_can_use_full_discord_budget_without_truncation(monkeypatch):
    sent = {}
    class Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
    def urlopen(req, timeout=0):
        sent.update(json.loads(req.data))
        return Resp()
    monkeypatch.setattr(notify.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fake")
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("FINBOT_NOTIFY", raising=False)
    text = "x" * 1500
    assert notify.send({"notifications": {"discord_channel_id": "123"}}, text, kind="daily", max_content_len=2000)
    assert sent["content"].endswith(text)
    assert len(sent["content"]) <= 2000


def test_digest_falls_back_to_original_on_builder_error(monkeypatch):
    import eod_summary
    sent = []
    monkeypatch.setattr(eod_summary, "build_summary", lambda *a, **k: (_ for _ in ()).throw(ValueError("sensitive detail")))
    monkeypatch.setattr(notify, "send", lambda config, text, **kw: sent.append(text) or True)
    from dashboard import render
    monkeypatch.setattr(render, "write_dashboard", lambda *a, **k: "/nonexistent")
    class DP:
        trading_client = False
    main._daily_digest({"strategy": {"mode": "etf_trend"}}, DP(), "", {})
    assert sent[0].startswith("📊 Paper bot today")
    assert "ValueError" in sent[0] and "sensitive detail" not in sent[0]
    assert "(summary unavailable: ValueError)" in sent[0]

def test_digest_summary_precedes_complete_legacy_and_attachments_on_legacy(monkeypatch):
    import eod_summary
    sent = []
    monkeypatch.setattr(eod_summary, "build_summary", lambda *a, **k: "SUMMARY:" + "x" * 4100)
    monkeypatch.setattr(notify, "send", lambda config, text, **kw: sent.append((text, kw)) or True)
    from dashboard import render
    monkeypatch.setattr(render, "write_dashboard", lambda *a, **k: "/fake/dashboard.html")
    class DP:
        trading_client = False
    config = {"notifications": {"prefix": "custom-prefix"}, "strategy": {"mode": "etf_trend"}}
    main._daily_digest(config, DP(), "", {})
    assert len(sent) > 2
    assert "".join(text for text, _ in sent[:-1]) == "SUMMARY:" + "x" * 4100
    assert sent[-1][0].startswith("📊 Paper bot today")
    assert sent[-1][1]["files"] and all(not kw.get("files") for _, kw in sent[:-1])
    assert all(len("**custom-prefix** — " + text) <= 2000 for text, _ in sent)

def test_digest_short_summary_precedes_unchanged_legacy_message(monkeypatch):
    import eod_summary
    sent = []
    monkeypatch.setattr(eod_summary, "build_summary", lambda *a, **k: "📝 Today in brief\nQuiet day.")
    monkeypatch.setattr(notify, "send", lambda config, text, **kw: sent.append((text, kw)) or True)
    from dashboard import render
    monkeypatch.setattr(render, "write_dashboard", lambda *a, **k: "/fake/dashboard.html")
    class DP:
        trading_client = False
    main._daily_digest({"strategy": {"mode": "etf_trend"}}, DP(), "", {})
    assert len(sent) == 2
    assert sent[0][0].startswith("📝 Today in brief\nQuiet day.")
    assert sent[1][0].startswith("📊 Paper bot today")
    assert sent[1][1]["files"]

def test_opening_thresholds_and_stale_block_does_not_override(tmp_path):
    base = fixture_paths(tmp_path, state={"status": "blocked — stale", "updated_at": "2026-10-06T19:00:00+00:00"})
    for delta, word in ((0, "Quiet day"), (40, "Quiet day"), (50, "Up day"), (-50, "Down day"),
                        (200, "Up day"), (201, "Big up day"), (-201, "Big down day")):
        msg = build_summary({}, {"equity": 10000 + delta, "last_equity": 10000}, [], {}, base=base, now=NOW)
        assert word in msg.split("Today in brief")[1].split("Performance")[0]
        assert "stale" not in msg.split("Alerts today")[1]

def test_today_block_reason_and_failed_trade_are_visible(tmp_path):
    base = fixture_paths(tmp_path, state={"status": "blocked — incomplete price data (SPY)", "updated_at": "2026-10-07T19:00:00+00:00"},
                         trades=[{"timestamp": "2026-10-07T19:00:00+00:00", "symbol": "SPY", "action": "buy", "qty": "1", "status": "rejected", "price": "300"}])
    msg = build_summary({}, {"equity": 11000, "last_equity": 10000}, [], {}, base=base, now=NOW)
    assert "⚠️ Problem day" in msg and "incomplete price data" in msg
    assert "rejected" in msg.lower() and "planned $300" in msg

def test_holdings_weight_order_and_drawdown_from_equity_track(tmp_path):
    base = fixture_paths(tmp_path, equity=[{"date": "2026-10-05", "equity": "10000", "strategy_version": "s"},
                                           {"date": "2026-10-06", "equity": "12000", "strategy_version": "s"},
                                           {"date": "2026-10-07", "equity": "9000", "strategy_version": "s"}], state={"strategy_version": "s"})
    msg = build_summary({}, {"equity": "9000", "cash": "-1000"},
                        [{"symbol": "AAA", "market_value": "1000"}, {"symbol": "ZZZ", "market_value": "9000"}],
                        {}, base=base, now=NOW)
    holdings = msg.split("Holdings")[1].split("Trend signals")[0]
    assert holdings.index("ZZZ") < holdings.index("AAA")
    assert "111.1%" in holdings and "25.0%" in msg

def test_next_tranche_has_number_and_future_business_day(tmp_path):
    base = fixture_paths(tmp_path, state={"asof": "2026-10-06"})
    msg = build_summary({"etf_trend": {"tranche_days": [0, 5, 10, 15]}}, {}, [], {}, base=base, now=NOW)
    assert "2026-10-08" in msg and "tranche 2" in msg.lower() and "holidays" in msg

def test_preview_uses_read_only_broker_and_labels_fallback(tmp_path):
    from scripts.preview_eod_summary import preview
    base = fixture_paths(tmp_path, equity=[{"date": "2026-10-07", "equity": "12345", "strategy_version": "s"}])
    class Broker:
        trading_client = object()
        paper = True
        def get_account(self): return {"equity": "20000", "last_equity": "19000", "cash": "1000"}
        def get_positions(self): return [{"symbol": "SPY", "market_value": "19000"}]
        def place_order(self, *a, **k): raise AssertionError("order")
    assert "Equity $20,000" in preview({}, base=base, now=NOW, broker=Broker())
    assert "journal" in preview({}, base=base, now=NOW, broker=None).lower()
    assert "Equity $12,345" in preview({}, base=base, now=NOW, broker=None)

def test_preview_includes_existing_daily_message_after_summary(tmp_path):
    from scripts.preview_eod_summary import preview
    base = fixture_paths(tmp_path)
    class Broker:
        trading_client = object()
        paper = True
        def get_account(self): return {"equity": "20000", "last_equity": "19000", "cash": "1000"}
        def get_positions(self): return [{"symbol": "SPY", "market_value": "19000"}]
    msg = preview({"strategy": {"mode": "etf_trend"}}, base=base, now=NOW, broker=Broker())
    assert "📊 Paper bot today" in msg
    assert "💰 Account $20,000 · today +1,000" in msg
    assert "📦 1 holdings" in msg
    assert "📌 ETFs" in msg
    assert msg.index("📝 Today in brief") < msg.index("💰 Account")

def test_benchmark_requires_matching_strategy_start_and_real_history(tmp_path):
    base = fixture_paths(tmp_path, equity=[{"date": "2026-10-05", "equity": "100", "strategy_version": "s"},
                                           {"date": "2026-10-07", "equity": "110", "strategy_version": "s"}],
                         state={"strategy_version": "s"})
    msg = build_summary({}, {}, [], {}, base=base, now=NOW,
                        spy_history={"2026-10-05": 200, "2026-10-07": 210})
    assert "+10.0% vs +5.0%" in msg
    msg = build_summary({}, {}, [], {}, base=base, now=NOW,
                        spy_history={"2026-10-06": 200, "2026-10-07": 210})
    assert "+10.0% vs n/a" in msg

def test_preview_reads_current_readiness_state_not_synthetic_comparison(tmp_path):
    from scripts.preview_eod_summary import preview
    base = fixture_paths(tmp_path, state={"strategy_version": "s"})
    learning = base / "learning"
    learning.mkdir()
    (learning / "readiness_state.json").write_text(json.dumps({"strategy_version": "s", "days": 12,
        "tracking_error_pct": 1.4, "updated_at": "2026-10-07T20:00:00+00:00"}))
    (learning / "etf_compare.json").write_text(json.dumps({"tracking_error_pct": 99}))
    msg = preview({}, base=base, now=NOW)
    assert "Readiness clock 12/252" in msg and "Tracking error +1.4%" in msg

def test_preview_reads_fresh_unlabeled_readiness_for_current_mode(tmp_path):
    from scripts.preview_eod_summary import preview
    base = fixture_paths(tmp_path, state={"strategy_version": "s"},
                         equity=[{"date": "2026-10-07", "equity": "10000", "strategy_version": "s"}])
    learning = base / "learning"
    learning.mkdir()
    (learning / "readiness_state.json").write_text(json.dumps({
        "mode": "etf_trend", "days": 0, "targets": {"min_days": 252},
        "updated_at": "2026-10-07T20:00:00+00:00"}))
    msg = preview({"strategy": {"mode": "etf_trend"}}, base=base, now=NOW)
    assert "readiness saved today" in msg
    assert "Readiness clock 0/252" in msg

def test_dated_audit_error_marks_problem_day_and_alert(tmp_path):
    base = fixture_paths(tmp_path)
    with (base / "journal" / "audit_log.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "cycle_outcome", "execution_error"])
        writer.writeheader()
        writer.writerow({"timestamp": "2026-10-06T20:00:00+00:00", "cycle_outcome": "execution_failed", "execution_error": "old"})
        writer.writerow({"timestamp": "2026-10-07T20:00:00+00:00", "cycle_outcome": "execution_failed", "execution_error": "broker offline"})
    msg = build_summary({}, {}, [], {}, base=base, now=NOW)
    assert "⚠️ Problem day" in msg and "broker offline" in msg.split("Alerts today")[1]
    assert "old" not in msg.split("Alerts today")[1]

def test_drawdown_only_current_strategy_label(tmp_path):
    base = fixture_paths(tmp_path, equity=[{"date": "2026-10-04", "equity": "100000", "strategy_version": "old"},
                                           {"date": "2026-10-05", "equity": "10000", "strategy_version": "s"},
                                           {"date": "2026-10-07", "equity": "9900", "strategy_version": "s"}],
                         state={"strategy_version": "s"})
    msg = build_summary({}, {}, [], {}, base=base, now=NOW)
    assert "Max drawdown +1.0%" in msg

def test_single_day_track_has_real_clock_but_insufficient_drawdown(tmp_path):
    base = fixture_paths(tmp_path, equity=[{"date": "2026-10-07", "equity": "10000", "strategy_version": "s"}],
                         state={"strategy_version": "s"})
    msg = build_summary({}, {}, [], {}, base=base, now=NOW)
    assert "Readiness clock 1/252" in msg
    assert "Max drawdown n/a" in msg
