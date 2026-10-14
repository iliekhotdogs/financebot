"""Discord text stays short, human-readable, and free of raw job output."""
import importlib.util
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eod_delivery_silent_on_success_but_explains_failure(tmp_path):
    from scripts.eod_delivery import summary
    log = tmp_path / 'eod.log'
    log.write_text('INFO fetched SPY\n# ETF trend monitor\n|---|---|\n')
    assert summary(log, 0) == ''  # the bot sends its own short daily summary
    log.write_text('INFO fetched SPY\n[ERROR] learn failed: broker timeout\n|---|---|\n')
    msg = summary(log, 0)
    assert msg.startswith('⚠️') and 'daily check' in msg.lower()
    assert 'broker timeout' not in msg and '|---|' not in msg and len(msg) < 140
    assert summary(log, 1).startswith('🚨')


def test_half_hour_status_is_etf_focused_and_short(tmp_path):
    import csv, json
    digest = _load('/opt/data/scripts/finance_bot_digest.py', 'finance_bot_digest_test')
    j = tmp_path / 'journal'
    j.mkdir()
    now = datetime(2026, 10, 7, 10, 0, tzinfo=ZoneInfo('America/Los_Angeles'))
    with (j / 'equity_curve.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['date', 'updated_at', 'equity', 'last_equity', 'exposure'])
        w.writeheader()
        w.writerow({'date': '2026-10-07', 'updated_at': now.isoformat(),
                    'equity': '100123', 'last_equity': '100000', 'exposure': '98000'})
    (j / 'etf_state.json').write_text(json.dumps({'status': 'idle', 'updated_at': now.isoformat()}))
    message = digest.build(now, base=tmp_path)
    assert '💰' in message and '$100,123' in message
    assert 'advisor' not in message.lower() and 'exposure' not in message.lower()
    assert len(message) < 180 and len(message.splitlines()) <= 3
    (j / 'etf_state.json').write_text(json.dumps({'status': 'blocked — incomplete price data (SPY)',
                                                   'updated_at': now.isoformat()}))
    assert 'SPY' in digest.build(now, base=tmp_path)


def test_daily_message_avoids_strategy_codes_and_jargon(monkeypatch):
    import main, notify
    from dashboard import render
    sent = []
    monkeypatch.setattr(notify, 'send', lambda config, text, kind='info', files=None, max_content_len=None: sent.append(text))
    monkeypatch.setattr(render, 'write_dashboard', lambda *a, **k: '/nonexistent')
    class DP:
        trading_client = True
        def get_account(self): return {'equity': '100123', 'last_equity': '100000'}
        def get_positions(self): return [{'symbol': 'SPY'}]
    main._daily_digest({'strategy': {'mode': 'etf_trend'}, 'dashboard': {'enabled': True}}, DP(), '',
                       {'status': 'NOT_READY', 'tracking_error_pct': None})
    assert '📝 Today in brief' in sent[0]
    msg = sent[-1]
    assert '💰' in msg and '$100,123' in msg and 'Paper' in msg
    assert all(s not in msg for s in ('etf_trend_v', 'tracking error', 'Readiness:', 'shadow-paper'))
    assert len(msg) < 250
