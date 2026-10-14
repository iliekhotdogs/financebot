# Setup Guide — Combined Trading Bot (Alpaca + yfinance / Liquid)

## Status: Fully Operational (verified)

This bot combines two trading systems into one unified pipeline:

| Project | Original role | Now integrated as |
|---------|--------------|-------------------|
| **finance-bot** | Alpaca + yfinance data layer, SMA Cross strategy | **Paper trading engine** (data + execution) |
| **gambler** | Liquid Trade perps, risk management, journal, daily review | **Risk framework + journal + review + advisory decision flow + Liquid execution path** |

### Current mode: PAPER TRADING
- **yfinance**: Historical/backfill data (free, unofficial scraping) — used for backtesting and gap-filling
- **Alpaca**: Real-time bars/quotes + paper trading execution (paper=true, $100K paper cash, account ACTIVE)
- **Liquid Trade**: Configured but DISABLED — activates only after paper trading is validated + operator approval

## Commands
```bash
cd /opt/data/finance-bot

# Check all data sources (yfinance + Alpaca + Liquid status)
.venv/bin/python main.py health

# Backtest SMA Cross strategy on historical yfinance data
.venv/bin/python main.py backtest

# Run one monitoring cycle (signal + advisor + Jev + risk checks, NO order placement)
.venv/bin/python main.py run

# Run full advisory decision flow + place paper trades (signal → advisor → Jev → risk → execution)
.venv/bin/python main.py trade

# Check Alpaca paper account + positions
.venv/bin/python main.py account

# Generate daily review report (includes advisor + Jev audit trail)
.venv/bin/python main.py review

# Check Liquid integration status (disabled by default)
.venv/bin/python main.py liquid-status

# Run research roles (technical, fundamental, bull/bear analysis)
.venv/bin/python main.py research

# Learning system commands
.venv/bin/python main.py learning-status           # Show candidates and audit events
.venv/bin/python main.py learning-evaluate        # Run walk-forward evaluation on candidates
.venv/bin/python main.py learning-promote         # Assess promotion readiness
.venv/bin/python main.py learning-select          # Run model selection

# Start web dashboard on localhost:8080
.venv/bin/python main.py dashboard

# Run test suite
.venv/bin/python -m pytest tests/ -v
```

## Architecture

```
yfinance (free, backfill)  ↔  Alpaca (real-time, paper trading)
  │                             │
  ├─ Historical bars              ├─ Real-time bars/quotes (IEX feed)
  ├─ Splits/dividends             ├─ Order execution (paper)
  ├─ Fundamentals                 ├─ Account/portfolio
  └─ Backfill gaps                └─ Market clock

Liquid Trade (Phase 2 — live trading)
  └─ Stock-linked perps (NVDA/AAPL/TSLA) via Co-Invest MCP
  └─ Enabled only after paper trading validated + operator approval

┌─────────────────────────────────────────────────────────────┐
│  ADVISORY DECISION FLOW                                     │
│                                                             │
│  1. Snapshot: market data + account + position + strategy  │
│     (single timestamped point-in-time snapshot)              │
│     ↓                                                       │
│  2. Advisor: receives snapshot → returns advice ONLY       │
│     (recommendation, evidence, risks — no order tools)       │
│     ↓                                                       │
│  3. Jev: receives snapshot + valid actions + advisor input │
│     → makes FINAL trade decision (action, qty, stop, rationale)│
│     ↓                                                       │
│  4. RiskManager: ALL Jev decisions must pass mandatory       │
│     risk checks (position size, exposure, loss limits, etc.) │
│     ↓                                                       │
│  5. Execution: only after risk approval → paper order via    │
│     Alpaca                                                   │
│                                                             │
│  Key constraints:                                            │
│  • Jev is the ONLY decision-maker                            │
│  • Advisor provides advice only (no execution tools)         │
│  • Risk checks are mandatory (can reject/resize/override)    │
│  • Both models get the SAME point-in-time data               │
│  • Invalid/malformed/timeout → fail safe (no trade)           │
└─────────────────────────────────────────────────────────────┘
```

`DataProvider.get_history()` uses Alpaca IEX bars as the primary source for
bars and falls back to yfinance to fill any data gaps.

### Risk Management
`RiskManager` enforces pre-trade limits before every order:
- Per-trade max loss: 2% of balance ($2,000)
- Max position size: 5% of balance ($5,000)
- Total open exposure: 10% of balance ($10,000)
- Daily loss limit: 5% of balance ($5,000)
- Drawdown halt: 15% from peak ($15,000)
- Min order: $15 (platform minimum)
- Max leverage: 1.0x (no leverage)

Risk checks remain **mandatory and unchanged** — both the advisor and Jev are
subject to the same pre-trade gate.

### Journal & Audit Trail
- **`journal/trade_log.csv`** — Original trade log (timestamp, symbol, action, qty, price, order_id, reason, source, status). Unchanged schema.
- **`journal/audit_log.csv`** — Extended audit trail capturing the full decision chain per symbol: snapshot_id → advisor recommendation/evidence/risks → Jev action/qty/rationale → risk check result → execution result. This is what powers the dashboard and daily report's advisor-vs-Jev comparison.

## What Was Built

| File | Purpose |
|---|---|
| `data.py` | DataProvider: unified yfinance/Alpaca data layer |
| `risk.py` | RiskManager: pre-trade risk limit enforcement |
| `main.py` | CLI with health, backfill, backtest, run (monitoring), trade (full flow + execution), account, review, liquid-status |
| `strategies/baseline.py` | SMA Cross strategy (v1.0) with backtest + live signals |
| `strategies/v1_1.py` | SMA Cross v1.1 with 65% Rule + Runner Hug exits |
| `schemas.py` | Validated response schemas for Advisor and Jev (strict, fail-safe) |
| `snapshot.py` | Timestamped point-in-time market/account/position/strategy snapshot builder |
| `advisor.py` | Advisor client (configurable provider/model/timeout, NO order tools, advice-only) |
| `jev.py` | Jev agent (sole decision-maker, receives snapshot + advisor input + valid actions) |
| `journal/trade_log.py` | Append-only CSV trade journal (original schema) |
| `journal/audit_log.py` | Extended audit CSV: full decision chain per symbol |
| `liquid/client.py` | Liquid Trade client stub (Phase 2, disabled) |
| `tests/` | 81 tests covering data, strategy, risk, journal, advisory flow, audit logging |
| `scripts/monitor_cycle.py` | Monitoring cycle with Pacific trading-window gate (calls `main.py run`) |
| `scripts/daily_review.py` | Daily review report generator (includes advisor + Jev audit trail) |

## What Was Fixed
1. **RTF `.env` issue**: Credentials were saved as RTF from TextEdit. Extracted to proper `.env`.
2. **Alpaca data client init**: Removed incorrect `sandbox=True` and `url_override=` params.
3. **Alpaca IEX feed**: Added `feed=DataFeed.IEX` to bars requests (free tier ≠ SIP).
4. **Column name normalization**: Alpaca lowercase → yfinance capitalized.
5. **Timezone fix**: `_backfill_yf` handles tz-aware vs naive datetime comparisons.
6. **Trade count bug**: SMA crossover uses `|diff|==1` (not `==2`).
7. **Drawdown halt bug**: Now compares against `peak_balance`, not current `balance`.

## Recursive Learning and Strategy Selection

The learning system evaluates alternative strategies over time and promotes candidates only when there is credible evidence they improve performance. It runs **outside** the order execution path.

### Architecture
```
learning/
  candidates.py      — Candidate definitions, config bounds, persistent storage
  evaluation.py      — Walk-forward evaluation with cost modeling
  promotion.py       — Promotion gates (paper trading → operator approval)
  model_selection.py — Primary metric + hard constraints selection
  audit.py           — Learning audit log (every candidate/rejection/promotion)
```

### Key Principles
- **Decision-time trading is separate from learning** — the live path uses only the approved strategy
- **Candidates never modify the active strategy** — they exist only as evaluated alternatives
- **No automatic promotion** — operator approval is always required
- **No order credentials in learning service** — learning produces configs and recommendations only
- **Baselines always included** — SMA Cross 20/50 and buy-and-hold in every evaluation

### Evaluation Process
1. Candidates are created with version, hypothesis, intended conditions, required data, and abstain rule
2. Walk-forward evaluation on chronologically separated data (training/validation/out-of-sample)
3. Realistic cost modeling: commission, slippage, spread
4. Comparison against baselines across multiple symbols and regimes
5. Gates: data quality, minimum trades, look-ahead check, baseline beat, profit factor > 1.0

### Promotion Gates
- Minimum paper trading duration (default: 30 days)
- Minimum paper trades (default: 10)
- Minimum paper symbols (default: 2)
- Must beat active strategy on Sharpe ratio
- Explicit operator approval required
- Previous version preserved for rollback

### Model Selection
- Primary metric: Sharpe ratio (configurable)
- Hard constraints: max drawdown, max exposure, min trades
- If no candidate clears all gates, the active version is retained
- Selection is deterministic and auditable

## Research Roles

Specialist analysis roles that provide structured findings to Jev. They advise but do NOT place orders.

### Roles
- **Technical** — SMA, RSI, MACD, ATR, support/resistance
- **Fundamental** — P/E, market cap, dividend yield, sector (via yfinance)
- **News/Sentiment** — News headlines (requires API key configuration)
- **Bull/Bear** — Synthesizes all findings into bullish/bearish thesis

### Integration
Research findings are included in Jev's prompt alongside the advisor's recommendation. Jev makes the final decision considering all inputs.

## Web Dashboard

The dashboard runs on `localhost:8080` (container-internal only) and shows:
- Current mode (paper/live)
- Configuration summary
- Audit trail with all decision outcomes
- Trade journal
- Learning system status (candidates, evaluations, promotions)
- Daily review summaries
- Research role status

## Known Limitations
- **Alpaca free tier** uses IEX data (single venue) — upgrade to paid plan for SIP.
- **yfinance** is unofficial scraping — backfill only, never for live signal generation.
- **Liquid integration** is a stub — requires Hermes MCP runtime for full integration.
- **Cron jobs are script-based** (no LLM) — monitoring just logs signals, doesn't trade.
- **Advisor & Jev require API keys** — set `ADVISOR_API_KEY` and `JEV_API_KEY` in `.env`. If keys are missing, both fail safe (no trade).
- **Paper mode is enforced** — `APCA_PAPER=true` must remain set. The `place_order` method explicitly rejects live orders when `paper=False`.
- **Learning evaluation uses yfinance data** — not Alpaca historical bars, due to API limitations.
- **News sentiment not yet integrated** — requires external news API key.
- **Dashboard is container-internal** — not reachable from the internet.

## Configuration

### Advisor & Jev settings (config.yaml)
```yaml
advisor:
  enabled: true
  provider: "openai"      # or "anthropic" or "mock"
  model: "gpt-4o-mini"
  timeout_seconds: 30
  api_key_env: "ADVISOR_API_KEY"

jev:
  enabled: true
  provider: "openai"      # or "anthropic" or "mock"
  model: "gpt-4o-mini"
  timeout_seconds: 30
  api_key_env: "JEV_API_KEY"
```

### Environment variables (.env)
```
ADVISOR_API_KEY=   # API key for the advisor model (advice only)
JEV_API_KEY=       # API key for Jev (final decision-maker)
```

Set `provider: "mock"` in config.yaml to use deterministic built-in responses (for testing without API keys).
