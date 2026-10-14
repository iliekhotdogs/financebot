# Deploying the fixed finance-bot to Hermes cloud

All Hermes cron jobs were paused at export time. Leave them paused until step 5.

## 1. Update the code (keep runtime data)

Copy this repo over `/opt/data/finance-bot`, **excluding** data the live bot owns:

```bash
rsync -av --exclude '.venv' --exclude '.env' --exclude '.git' \
  --exclude 'journal/*.csv' --exclude 'journal/*.json' --exclude 'journal/.advisor_last_call' \
  --exclude 'learning/*.csv' --exclude 'learning/*.jsonl' --exclude 'learning/active_strategy.json' \
  --exclude 'learning/reports' --exclude 'daily_reviews' \
  ./ /opt/data/finance-bot/
```

## 2. Install dependencies into the project venv

`openai` was missing (the audit log shows `openai package not installed`, so the advisor never ran).

```bash
cd /opt/data/finance-bot && .venv/bin/pip install -r requirements.txt
```

## 3. Install the two cron wrappers, remove the old ones

```bash
cp deploy/hermes/finance_bot_cycle.sh deploy/hermes/finance_bot_eod.sh /opt/data/scripts/
chmod +x /opt/data/scripts/finance_bot_cycle.sh /opt/data/scripts/finance_bot_eod.sh
cd /opt/data/scripts && rm -f finance_bot_monitor.sh finance_bot_monitor.py \
  paper_trade_cycle.sh paper_trade_cycle.py finance_bot_review.sh \
  finance_bot_daily_review.sh finance_bot_daily_review.py daily_review.py monitor_cycle.py
```

## 3b. Discord notifications

The bot posts directly to Discord (it must *confirm* delivery before a
strategy switch is scheduled). It uses the Hermes gateway's existing
`DISCORD_BOT_TOKEN` from `/opt/data/.env` and posts to
`notifications.discord_channel_id` in `config.yaml` (default
`1555334067862376489`, #📈trading-reports). To use another channel, change
that id. Alternatively set `DISCORD_WEBHOOK_URL` in `/opt/data/.env`.

Check it once:

```bash
cd /opt/data/finance-bot && .venv/bin/python -c "import yaml,notify,main; print(notify.send(yaml.safe_load(open('config.yaml')), 'test message from finance-bot'))"
```

It should print `True` and the message should appear in the channel. If it
prints `False`, auto-promotion will never switch strategies (by design).

## 4. Verify by hand before scheduling

```bash
cd /opt/data/finance-bot
.venv/bin/python -m pytest -q                 # expect all passing
.venv/bin/python main.py health               # Alpaca + yfinance OK
.venv/bin/python main.py run                  # monitor cycle, never orders
FINBOT_VERBOSE=1 /opt/data/scripts/finance_bot_cycle.sh   # prints why it skipped/ran
```

`run` should take seconds, not hours. Check that the advisor shows `max_calls=3`.

## 5. Cron jobs

| Job | Action |
|---|---|
| `d576ceb80db2` Finance bot monitoring cycle | **Repoint** to script `finance_bot_cycle.sh`, schedule `*/5 4-22 * * 1-5`, workdir `/opt/data/finance-bot`, no_agent, deliver `local` |
| `f1d09601edec` finance-bot-paper-trade | **Delete** (trading now happens in the cycle job via `execution.auto_trade`) |
| `103e4b39ff68` Finance bot daily review | **Repoint** to script `finance_bot_eod.sh`, schedule `30 13,20,21 * * 1-5` |
| `4f2855b45a0e` finance-bot-daily-review | **Delete** (duplicate) |
| `8770283fabf0`, `62dc9a16197d`, `a91f0bbee8d2` (Gambler) | Separate project, not touched |
| **New:** finance-bot live dashboard | **Create** script job `finance_bot_dashboard.sh`, schedule `*/10 * * * *`, workdir `/opt/data/finance-bot`, no_agent, deliver `local` (see 6c) |

Why these schedules: Hermes fired the same cron expressions in UTC for some jobs and in local
time for others. Both schedules are wide enough to cover 05:45–13:00 / 13:30 Pacific under
**either** interpretation and in both PST and PDT. `scripts/run_scheduled.py` makes the actual
decision on Pacific wall-clock time and prints nothing outside the window, so Hermes delivers
nothing for those runs. The EOD job writes a per-day marker, so it runs exactly once.

Resume both jobs, then **confirm the first real fire** in `cron/output/<job_id>/`. A job
existing in the list doesn't prove it runs.

## 6. Daily operation

- The cycle places **paper** orders when `execution.auto_trade: true`. Set it to `false` to
  run monitor-only. Live (non-paper) orders are refused in code.
- EOD runs the review, then `forecast-evaluate`, then `learn`: the strategy search,
  decision-policy learning, real-money readiness check, and a Discord daily summary.
  Full report: `learning/reports/<run>.md`.
- **Strategy changes are supervised** (`learning.promotion.allow_auto_promotion: false`):
  candidates that pass every gate remain in shadow paper until explicitly approved with
  `.venv/bin/python main.py learning-approve --candidate-id <ID>`.
  - Undo an approved switch: `.venv/bin/python main.py learning-rollback`
  - If automatic promotion is ever re-enabled, Discord announces it first with a 12-hour
    veto window; cancel with `.venv/bin/python main.py learning-veto`.
- **Real money:** `.venv/bin/python main.py readiness` shows the live-readiness checks on the
  actual paper record. When all pass, Discord gets a "review for real money" alert. Going
  live is a manual decision (new keys, `APCA_PAPER`, and removing the paper guard in
  `data.py`); the bot never does it itself.
- **Check the learner itself** at any time (no keys needed):
  `.venv/bin/python scripts/simulate_learning.py --mode trend|null|real`

## 6b. Seeing your data

| Where | What you get | Setup |
|---|---|---|
| **Discord #📈trading-reports** | every BUY/SELL, halts, strategy switches; a daily summary at ~13:45 PT with an equity-vs-SPY chart image and the full dashboard attached | none (after step 3b) |
| **Google Drive** → `hermes-data/finance-bot/reports/dashboard.html` | full dashboard, refreshed after every cycle; download and open in a browser | none: the existing Hermes drive sync carries it |
| **Alpaca web app** → app.alpaca.markets (Paper account) | broker's own view of positions, orders, fills, equity | none |
| **Live dashboard** | same page, live broker data, auto-refresh every 60s | `python main.py dashboard` (localhost). Remote: set `DASHBOARD_TOKEN` in `/opt/data/.env`, run `python main.py dashboard --host 0.0.0.0`, reach it over Tailscale at `http://<hermes-host>:8080/?token=<token>` (needs Tailscale inside the Hermes container) |

Dashboard contents: equity, today's P&L, return since start, account vs SPY and drawdown charts,
open positions with stops, recent orders, closed trades with win rate, real-money readiness
checks, Jev decision-policy evidence, strategy-switch history, learning leaderboard, and the
latest learning report.

## 6c. Live dashboard on your PC (Tailscale)

Your PCs and Mac are already on a Tailscale network. This puts Hermes on it too,
so any of your devices can open the live dashboard — nothing is exposed to the
internet, and it needs no root inside the container (userspace networking).

1. **Auth key (once).** On your PC: <https://login.tailscale.com/admin/settings/keys>
   → *Generate auth key* → leave *Reusable* off, turn *Pre-approved* on → copy it.
   On Hermes add one line to `/opt/data/.env` (never to the bot's `config.yaml`):
   `TS_AUTHKEY=tskey-auth-...`
2. **Start it by hand once** and read the output:
   `bash /opt/data/scripts/finance_bot_dashboard.sh`
   Expected: `installed tailscale_…tgz`, `started tailscaled`, `joined tailnet as
   hermes-finbot`, `started on 127.0.0.1:8080`, `serving http://hermes-finbot/`.
   A second run must print **nothing** (healthy).
3. **On your PC** (Tailscale running): open <http://hermes-finbot/>. The page
   reloads every 60 s and pulls fresh account/positions from Alpaca on each load;
   the trading data underneath updates every 5-min cycle.
4. **Schedule the keepalive** (step 5 table): every 10 min. If Hermes recycles the
   container or kills background processes, the next run restarts everything and
   says so in its output.
5. After the first successful join you can delete `TS_AUTHKEY` from `.env` — the
   node identity lives in `/opt/data/tailscale/state`. To revoke access at any
   time: Tailscale admin → Machines → `hermes-finbot` → Remove.

If step 2 fails at the download (no outbound access to pkgs.tailscale.com) or the
processes never survive between cron runs, use Discord + the Drive copy of
`reports/dashboard.html` (6b) — those need nothing running on Hermes.

## 6d. ETF trend mode (2026-10-06)

`config.yaml` now has `strategy.mode: etf_trend`. The cron jobs, scripts and schedule stay the same; only what the cycle does changes.

**What happens after install:**

- **Before the open:** cycles show and save the plan; you can see it on the dashboard under *Target allocation → Planned orders*.
- **First cycle after the open (≥ 06:30 PT):** the bot executes the plan once:
  1. Sells every stock that is not part of the strategy (e.g. TXN). This also cancels its broker stop order.
  2. Buys the ETF targets, whole shares, keeping 1% in cash.
  3. Puts the rest in BIL (T-bills).
  4. Posts the rebalance to Discord.
- **Afterwards:** it trades only on trading days 1, 6, 11 and 16 of the month (the day after, at the open), and only when a target moved by more than 0.5% of equity.
- **Every other cycle:** checks the account, writes the equity curve and refreshes the dashboard. Prices are downloaded once per day and cached.

**End of day (`learn`):**

- Writes a monitor report: paper vs model tracking error, and drawdown and Sharpe vs SPY and 60/40.
- Updates readiness and scores Jev's shadow reviews.
- There is no strategy search in this mode.

**Readiness for real money (all must hold):**

- 252 trading days on this strategy
- Tracking error vs model ≤ 3% a year
- Max drawdown ≤ 20%
- Drawdown no deeper than 60/40 over the same days

**Safety:**

- Paper-only enforcement is unchanged.
- If equity falls more than 20% below its peak, the bot only sells and alerts Discord.

**Useful commands:**

```bash
cd /opt/data/finance-bot
.venv/bin/python main.py etf-status          # targets, signals, saved plan
.venv/bin/python main.py run                 # one cycle, plan only, never orders
.venv/bin/python main.py readiness           # monitor report + readiness
.venv/bin/python main.py etf-replay --start 2015-01-01   # live code vs model on history (~1-3 min)
```

**Roll back:** set `strategy.mode: stock_signals` in `config.yaml`. Positions are not touched automatically when switching back, so close the ETFs by hand if you want the stock bot to start from cash.

## 7. Update Hermes memory

Replace the cron sentence in `memories/MEMORY.md` with:

> Cron: `finance_bot_cycle.sh` (*/5 4-22 * * 1-5, gated 05:45–13:00 PT in run_scheduled.py, places paper orders when execution.auto_trade=true) and `finance_bot_eod.sh` (30 13,20,21 * * 1-5, once/day ≥13:30 PT: review → forecast-evaluate → learn). Keys load from /opt/data/.env automatically.

Add: "Strategy switches are announced in #📈trading-reports and auto-applied after 12h
unless `main.py learning-veto` is run."
