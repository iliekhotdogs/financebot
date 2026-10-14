"""Dashboard rendering and Discord attachments."""
import json
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import notify  # noqa: E402
from dashboard import render  # noqa: E402

CFG = {"strategy": {"family": "sma_cross", "version": "v1.1"}, "execution": {"auto_trade": True}}


def _fake_data(**over):
    d = {"generated_at": "2026-10-02 21:00", "mode": "paper", "auto_trade": True,
         "strategy": {"label": "v1.1", "name": "sma_cross"}, "equity": 101000.0, "day_pl": 250.0,
         "start_equity": 100000.0, "peak": 101500.0, "halted": False, "halt_reason": None,
         "curve": [("2026-09-%02d" % i, 100000 + i * 50.0) for i in range(1, 25)],
         "bench": {"2026-09-%02d" % i: 500 + i for i in range(1, 25)},
         "positions": [{"symbol": "AAPL", "qty": 10, "entry": 200.0, "price": 205.0, "pl": 50.0,
                        "pl_pct": 2.5, "stop": 190.0, "strategy": "v1.1"}],
         "trades": [], "closed": [], "closed_stats": {"n": 0, "win_rate": None, "avg": None},
         "pending": None, "leaderboard": [], "switches": [], "policy": {}, "readiness": {},
         "latest_report": "# report <script>alert(1)</script>"}
    d.update(over)
    return d


def test_render_full_page_escapes_and_contains_sections():
    page = render.render_html(_fake_data(halted=True, halt_reason="Drawdown",
                                         pending={"candidate_id": "cand-x", "apply_after": "2026-10-03T09:00"}))
    for s in ("Account vs SPY", "Open positions", "Real-money readiness", "Learning leaderboard",
              "Trading halted", "Strategy switch scheduled", "AAPL"):
        assert s in page
    assert "<script>alert(1)</script>" not in page          # report text is escaped
    assert page.count("<polyline") >= 3                     # account, SPY, drawdown


def test_render_empty_install_does_not_crash():
    page = render.render_html(_fake_data(curve=[], bench={}, positions=[], equity=None, day_pl=None,
                                         start_equity=None))
    assert "equity curve starts after the first trading cycle" in page


def test_collect_on_fresh_install(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "BOT_DIR", str(tmp_path))
    import engine
    monkeypatch.setattr(engine, "ACTIVE_STRATEGY_PATH", str(tmp_path / "none.json"))
    d = render.collect(CFG, fetch_benchmark=False)
    assert d["curve"] == [] and d["positions"] == [] and d["strategy"]["label"] == "v1.1"


def test_png_render(tmp_path):
    pytest.importorskip("matplotlib")
    out = render.render_png(_fake_data(), str(tmp_path / "eq.png"))
    assert out and os.path.getsize(out) > 5000


def test_discord_attachments_multipart(tmp_path, monkeypatch):
    f = tmp_path / "eq.png"
    f.write_bytes(b"\x89PNG fake")
    sent = {}

    class Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=0):
        sent["url"], sent["body"], sent["ctype"] = req.full_url, req.data, req.headers["Content-type"]
        return Resp()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "t")
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("FINBOT_NOTIFY", raising=False)
    ok = notify.send({"notifications": {"discord_channel_id": "123"}}, "hello", kind="daily", files=[str(f)])
    assert ok and sent["url"].endswith("/channels/123/messages")
    assert sent["ctype"].startswith("multipart/form-data")
    assert b'name="payload_json"' in sent["body"] and b'filename="eq.png"' in sent["body"]


def test_flask_requires_token_when_exposed(monkeypatch):
    pytest.importorskip("flask")
    from dashboard.app import create_app
    monkeypatch.setattr(render, "collect", lambda *a, **k: _fake_data())
    app = create_app(CFG, dp=None, token="secret").test_client()
    assert app.get("/").status_code == 401
    r = app.get("/?token=secret")
    assert r.status_code == 200 and b"Finance Bot" in r.data
    assert app.get("/api/data").status_code == 200          # cookie set by the first visit


def test_spy_nan_close_is_dropped(tmp_path, monkeypatch):
    """yfinance returned NaN for the current session (2026-10-05) → SPY line vanished."""
    from datetime import datetime, timezone
    cache = tmp_path / "spy.json"
    cache.write_text(json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(),
                                 "closes": {"2026-10-02": 769.64, "2026-10-05": float("nan")}}))
    monkeypatch.setattr(render, "SPY_CACHE", str(cache))
    closes = render.spy_closes("2026-10-04")
    assert closes == {"2026-10-02": 769.64}
    acct, spy = render._series({"curve": [("2026-10-04", 100000.0), ("2026-10-05", 100084.0)], "bench": closes})
    assert [round(v, 2) for _, v in spy] == [100.0, 100.0]     # as-of: carried forward, never NaN


def _etf_data(**over):
    positions = [{"symbol": s, "qty": 1, "entry": 100, "price": 100, "mv": v,
                  "pl": 12, "pl_pct": 1.2, "stop": None} for s, v in
                 (("SPY", 36000), ("IEF", 10000), ("BIL", 52000))]
    d = _fake_data(etf_mode=True, equity=100000, start_equity=99000, day_pl=182,
                   positions=positions, trades=[], tz="America/Los_Angeles",
                   etf={"asof": "2026-10-05", "last_executed_decision": "2026-10-01",
                        "targets": {"SPY": .36, "IEF": .10, "BIL": .52},
                        "in_market": {"SPY": 4, "IEF": 2}, "tranches": 4,
                        "updated_at": "2026-10-06T16:00:00+00:00", "status": "ok"},
                   tranche_days=[0, 5, 10, 15], cash=2000)
    d.update(over)
    return d


def test_etf_sections_and_allocation():
    page = render.render_html(_etf_data())
    for text in ("Holdings", "Next rebalance", "Bot health", "46% invested", "52% T-bills"):
        assert text in page
    for text in ("Closed trades", "Decisions today", "Room to stop", "97.8", "Target allocation"):
        assert text not in page


def test_stock_mode_retains_sections():
    page = render.render_html(_fake_data())
    assert all(s in page for s in ("Open positions", "Closed trades", "Learning leaderboard"))


def test_etf_margin_allocation():
    d = _etf_data(cash=-23000)
    d['positions'][0]['mv'] = 113000
    d['etf']['gross_target'] = 1.23
    page = render.render_html(d)
    assert 'borrowed 23%' in page and 'target 123%' in page and 'equity-marker' in page


@pytest.mark.parametrize('now,status,want', [
    ('2026-10-06T17:00:00+00:00', 'ok', 'Stale'),
    ('2026-10-06T17:00:00+00:00', 'blocked — incomplete price data (SPY)', 'Blocked'),
    ('2026-10-10T17:00:00+00:00', 'ok', 'Market closed'),
])
def test_etf_health(now, status, want):
    d = _etf_data(now=datetime.fromisoformat(now))
    d['etf']['updated_at'] = '2026-10-06T15:00:00+00:00'
    d['etf']['status'] = status
    assert want in render.render_html(d)


def test_etf_rebalance_weekday_fallback():
    page = render.render_html(_etf_data())
    assert 'Decision ≈Thu 8 Oct at close → orders ≈Fri 9 Oct morning' in page


def test_etf_collect_calendar_cached_for_day(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import engine
    monkeypatch.setattr(render, 'BOT_DIR', str(tmp_path))
    monkeypatch.setattr(render, 'CALENDAR_CACHE', str(tmp_path / 'cal.json'))
    monkeypatch.setattr(engine, 'ACTIVE_STRATEGY_PATH', str(tmp_path / 'active.json'))
    client = SimpleNamespace(get_calendar=lambda **kw: [SimpleNamespace(date='2026-10-08')])
    cfg = {'strategy': {'mode': 'etf_trend'}, 'etf_trend': {'tranche_days': [0, 5, 10, 15]}}
    d = render.collect(cfg, account={'equity': '100000', 'cash': '2000'}, positions=[],
                       fetch_benchmark=False, trading_client=client)
    assert '2026-10-08' in d['market_days'] and d['cash'] == 2000
    client.get_calendar = lambda **kw: pytest.fail('calendar cache not used')
    assert render.collect(cfg, account={'equity': '100000'}, positions=[],
                          fetch_benchmark=False, trading_client=client)['market_days'] == d['market_days']


def test_etf_report_writes_compare_json(tmp_path, monkeypatch):
    import pandas as pd
    from learning import etf_monitor, etf_fills, readiness
    import engine_etf
    from strategies import etf_trend
    ix = pd.bdate_range('2026-09-01', periods=8)
    closes = pd.DataFrame({'SPY': range(100, 108), 'IEF': range(90, 98), 'BIL': range(80, 88)}, index=ix)
    monkeypatch.setattr(engine_etf, 'load_prices', lambda *a: (closes, closes))
    monkeypatch.setattr(etf_trend, 'targets_asof', lambda *a: (pd.Series({'SPY': .5}), {'asof': '2026-09-10', 'in_market': {'SPY': 4}, 'tranches': 4}))
    monkeypatch.setattr(etf_trend, 'model_equity', lambda *a: pd.Series([1 + i * .01 for i in range(8)], index=ix))
    monkeypatch.setattr(etf_monitor, 'paper_equity', lambda: pd.Series([100000 + i * 1000 for i in range(8)], index=ix))
    monkeypatch.setattr(readiness, 'assess_etf_and_notify', lambda *a: {'status': 'NOT_READY', 'summary': 'test'})
    monkeypatch.setattr(etf_fills, 'reconcile_fills', lambda *a: [])
    monkeypatch.setattr(etf_monitor, 'REPORT_DIR', str(tmp_path / 'reports'))
    monkeypatch.setattr(etf_monitor, 'BOT_DIR', str(tmp_path))
    cfg = {'etf_trend': {'symbols': ['SPY'], 'cash_symbol': 'BIL'}}
    etf_monitor.report(None, cfg, now=datetime(2026, 9, 11, tzinfo=timezone.utc))
    obj = json.loads((tmp_path / 'learning' / 'etf_compare.json').read_text())
    assert set(('generated_at', 'tracking_error_pct', 'paper', 'model', 'spy', 'sixty_forty')) <= obj.keys()
    assert obj['series']['paper'][str(ix[0].date())] == 100
    assert len(obj['series']['model']) == 8


def test_etf_chart_falls_back_without_comparison():
    page = render.render_html(_etf_data(etf_compare=None))
    assert 'Account vs SPY' in page and 'Model comparison appears after the end-of-day job.' in page
    assert '<polyline' in page


def test_etf_chart_shows_four_series_and_ranges():
    series = {k: {'2026-10-01': 100, '2026-10-02': 101} for k in
              ('paper', 'model', 'spy', 'sixty_forty')}
    cmp_ = {'generated_at': datetime.now(timezone.utc).isoformat(),
            'tracking_error_pct': .42, 'series': series,
            **{k: {'total_return_pct': 1.2, 'max_drawdown_pct': -2.3}
               for k in series}}
    page = render.render_html(_etf_data(etf_compare=cmp_))
    for text in ('Paper account vs what it should have done', 'Model', 'Tracking error',
                 '60/40', '1M', '3M', 'YTD', 'All', 'Data table'):
        assert text in page
    assert 'lineStyle:2' in page and 'series.m' in page


def test_etf_chart_stale_comparison_falls_back():
    cmp_ = {'generated_at': '2020-01-01T00:00:00+00:00',
            'series': {'paper': {'2026-10-01': 100}, 'model': {'2026-10-01': 100}}}
    page = render.render_html(_etf_data(etf_compare=cmp_))
    assert 'Account vs SPY' in page and 'Model comparison appears after the end-of-day job.' in page


def test_etf_calendar_excludes_holiday():
    d = _etf_data(market_days=['2026-10-01', '2026-10-02', '2026-10-05',
                               '2026-10-06', '2026-10-07', '2026-10-09',
                               '2026-10-13', '2026-10-14'])
    page = render.render_html(d)
    assert 'Decision Fri 9 Oct at close → orders Tue 13 Oct morning' in page
    assert '≈' not in page.split('Next rebalance')[1].split('Bot health')[0]


def test_etf_orders_filter_fill_and_group():
    trades = [dict(timestamp='2026-10-09T15:00:00+00:00', symbol='SPY', action='buy',
                   qty='2', price='101', status='submitted', source='etf_trend', order_id='a'),
              dict(timestamp='2026-10-09T15:01:00+00:00', symbol='IEF', action='sell',
                   qty='1', price='90', status='submitted', source='etf_trend', order_id='b'),
              dict(timestamp='2026-10-09T15:02:00+00:00', symbol='XYZ', action='buy',
                   qty='1', price='1', status='submitted', source='stock_signals', order_id='c')]
    page = render.render_html(_etf_data(trades=trades, etf_fills=[{'order_id': 'a',
        'filled_avg_price': '101.5', 'slippage_bp': '49.5'}]))
    assert 'Fill price' in page and 'Slippage (bp)' in page and '49.5' in page
    assert '101.5' in page and 'XYZ' not in page
    assert 'rebalance · 2 orders' in page and '—' in page
    assert 'Jev' not in page.split('Recent orders')[1].split('Real-money readiness')[0]


def test_etf_collect_uses_last_25_strategy_orders(tmp_path, monkeypatch):
    import csv
    import engine
    monkeypatch.setattr(render, 'BOT_DIR', str(tmp_path))
    monkeypatch.setattr(engine, 'ACTIVE_STRATEGY_PATH', str(tmp_path / 'none.json'))
    (tmp_path / 'journal').mkdir()
    with (tmp_path / 'journal' / 'trade_log.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=['timestamp', 'symbol', 'source', 'status'])
        w.writeheader()
        w.writerows([{'timestamp': f'2026-10-01T{i:02d}:00:00', 'symbol': 'SPY',
                     'source': 'etf_trend', 'status': 'submitted'} for i in range(24)])
        w.writerows([{'timestamp': '2026-10-02T00:00:00', 'symbol': 'AAPL',
                     'source': 'stock_signals', 'status': 'submitted'} for _ in range(30)])
    d = render.collect({'strategy': {'mode': 'etf_trend'}}, fetch_benchmark=False)
    assert len(d['trades']) == 24 and all(r['source'] == 'etf_trend' for r in d['trades'])


def test_live_refresh_preserves_scroll_and_details(monkeypatch):
    pytest.importorskip('flask')
    from dashboard.app import create_app
    monkeypatch.setattr(render, 'collect', lambda *a, **k: _fake_data())
    page = create_app(CFG).test_client().get('/').get_data(as_text=True)
    assert 'http-equiv="refresh"' not in page
    assert 'visibilityState' in page and 'sessionStorage' in page


def test_etf_readiness_and_jev_collapsed():
    page = render.render_html(_etf_data(readiness={'mode': 'stock_signals',
        'status': 'READY_FOR_REVIEW', 'checks': {'min_closed_trades': False}}, policy={'status': 'pending'}))
    assert "First ETF readiness check runs after today's end-of-day job" in page
    assert '<details><summary>Jev shadow scoring' in page
    assert 'Latest learning report' in page
    assert 'Real-money readiness criteria met' not in page


def test_etf_readiness_uses_option_b_spy_guard_not_old_sixty_forty():
    d = _etf_data(readiness={"mode": "etf_trend", "status": "NOT_READY",
                            "checks": {"min_days": False, "drawdown_vs_spy": True},
                            "max_drawdown_pct": 12.0, "spy_max_drawdown_pct": 20.0,
                            "targets": {"min_days": 252, "max_drawdown_pct": 40.0}})
    page = render.render_html(d)
    assert "Drawdown no deeper than SPY" in page
    assert "Drawdown no deeper than 60/40" not in page


def test_etf_tracking_target_comes_from_config(tmp_path, monkeypatch):
    import engine
    monkeypatch.setattr(render, "BOT_DIR", str(tmp_path))
    monkeypatch.setattr(engine, "ACTIVE_STRATEGY_PATH", str(tmp_path / "none.json"))
    cfg = {"strategy": {"mode": "etf_trend"},
           "live_readiness": {"etf_trend": {"max_tracking_error_pct": 2.0}}}
    d = render.collect(cfg, fetch_benchmark=False)
    assert d["tracking_error_target"] == 2.0


def test_etf_mobile_tables_have_compact_styles():
    page = render.render_html(_etf_data())
    assert 'class="table table-vcenter card-table etf-table"' in page
    assert 'class="table card-table etf-orders"' in page
    assert '@media (max-width:576px)' in page
    assert '.etf-table th:nth-child(2)' in page
    assert '.etf-orders th:nth-child(5)' in page
