"""Discord posts should be scannable, with detail kept in reports."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
import notify


def test_daily_digest_is_short_and_has_no_raw_report_dump(monkeypatch):
    sent = []
    monkeypatch.setattr(notify, "send", lambda cfg, text, **kw: sent.append((text, kw)) or True)
    monkeypatch.setattr(main, "_daily_digest", main._daily_digest)
    import dashboard.render as render
    monkeypatch.setattr(render, "write_dashboard", lambda *a, **kw: "/opt/data/finance-bot/reports/dashboard.html")
    monkeypatch.setattr(render, "EQUITY_PNG_PATH", "/opt/data/finance-bot/reports/equity.png")

    class DP:
        trading_client = True
        def get_account(self): return {"equity": "102500", "last_equity": "100000"}
        def get_positions(self): return [{"symbol": "SPY"}]

    report = "## Proposed\n- " + "excessive parameter explanations " * 70 + "\n\n## Promotion\n- " + "nested strategy evidence " * 70
    main._daily_digest({"strategy": {"family": "sma_cross", "version": "v1.1"}}, DP(), report,
                       {"status": "NOT_READY", "summary": "failing: insufficient paper trades"})
    summary, legacy = sent
    assert all(len(text) <= 2000 for text, _ in sent)
    assert "📝 Today in brief" in summary[0] and "Performance" in summary[0] and "Trades today" in summary[0]
    assert "Paper trading only" in summary[0]
    assert all("excessive parameter explanations" not in text for text, _ in sent)
    assert all(kw["kind"] == "daily" for _, kw in sent)
    # Preserve every assertion the original short-message test made about
    # the existing digest; the new prose is a separate preceding message.
    assert len(legacy[0]) <= 500
    assert "Account" in legacy[0] and "Stocks" in legacy[0] and "Live trading not approved" in legacy[0]
    assert legacy[1]["files"] and not summary[1].get("files")


def test_discord_send_limits_content_to_scannable_size(monkeypatch):
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
    text = "📊 Summary\n" + "\n".join(f"Detail {i}: " + "blah " * 20 for i in range(30))
    assert notify.send({"notifications": {"discord_channel_id": "123"}}, text)
    assert len(sent["content"]) <= 900
    assert "📊 Summary" in sent["content"]
    assert "Details omitted" in sent["content"]
    assert sent["allowed_mentions"] == {"parse": []}
