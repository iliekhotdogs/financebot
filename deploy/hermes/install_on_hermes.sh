#!/bin/bash
# Install / update finance-bot on Hermes cloud.
#
#   tar -xzf finance-bot-hermes.tar.gz -C /opt/data
#   bash /opt/data/finance-bot-update/deploy/hermes/install_on_hermes.sh
#
# What it does (safe to re-run):
#   1. Backs up the current /opt/data/finance-bot (minus .venv) to /opt/data/backups/
#   2. Copies the new code over it. The bundle contains NO data files, so the
#      journal, learning history, .env and daily reviews already there are kept.
#   3. Creates/updates the project venv and installs requirements
#   4. Installs the two cron wrappers into /opt/data/scripts and removes the old ones
#   5. Runs the test suite, a health check and one monitor-only cycle
#   6. Optionally sends a Discord test message (pass --discord-test)
#
# It does NOT change Hermes cron jobs — see HERMES_PROMPT.md for that step.
set -uo pipefail

ROOT="${HERMES_ROOT:-/opt/data}"
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/finance-bot"
SCRIPTS="$ROOT/scripts"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DISCORD_TEST=0
[ "${1:-}" = "--discord-test" ] && DISCORD_TEST=1

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mFAILED: %s\033[0m\n' "$*"; exit 1; }

[ -f "$SRC/main.py" ] && [ -f "$SRC/engine.py" ] || fail "run this from the extracted bundle ($SRC has no main.py/engine.py)"
[ "$SRC" = "$DEST" ] && fail "bundle must be extracted somewhere other than $DEST (e.g. $ROOT/finance-bot-update)"

say "1/6 Backup current install"
mkdir -p "$ROOT/backups"
if [ -d "$DEST" ]; then
  tar -czf "$ROOT/backups/finance-bot-$STAMP.tgz" -C "$ROOT" --exclude 'finance-bot/.venv' finance-bot \
    && echo "backup: $ROOT/backups/finance-bot-$STAMP.tgz" || fail "backup failed"
else
  echo "no existing install — fresh setup"
  mkdir -p "$DEST"
fi

say "2/6 Copy code (data files untouched)"
# Bundle ships code/config only; nothing here overwrites journal/learning data or .env
cp -a "$SRC/." "$DEST/" || fail "copy failed"
rm -rf "$DEST/dashboard/templates"          # replaced by dashboard/render.py
find "$DEST" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null
echo "code installed in $DEST"

say "3/6 Python environment"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
      PY="$(command -v "$c")"; break
    fi
  done
fi
if [ ! -x "$DEST/.venv/bin/python" ]; then
  [ -n "$PY" ] || fail "need Python >= 3.10 to create the venv"
  "$PY" -m venv "$DEST/.venv" || fail "venv creation failed"
fi
if command -v uv >/dev/null; then
  uv pip install -q -p "$DEST/.venv/bin/python" -r "$DEST/requirements.txt" || fail "dependency install failed"
else
  "$DEST/.venv/bin/python" -m pip install -q --upgrade pip >/dev/null 2>&1
  "$DEST/.venv/bin/python" -m pip install -q -r "$DEST/requirements.txt" || fail "dependency install failed"
fi
"$DEST/.venv/bin/python" -c "import alpaca, yfinance, pandas, openai, flask, matplotlib; print('deps OK')" || fail "imports failed"

say "4/6 Cron wrappers"
mkdir -p "$SCRIPTS"
cp "$DEST/deploy/hermes/finance_bot_cycle.sh" "$DEST/deploy/hermes/finance_bot_eod.sh" \
   "$DEST/deploy/hermes/finance_bot_dashboard.sh" "$SCRIPTS/"
chmod +x "$SCRIPTS/finance_bot_cycle.sh" "$SCRIPTS/finance_bot_eod.sh" "$SCRIPTS/finance_bot_dashboard.sh"
for old in finance_bot_monitor.sh finance_bot_monitor.py paper_trade_cycle.sh paper_trade_cycle.py \
           finance_bot_review.sh finance_bot_daily_review.sh finance_bot_daily_review.py \
           daily_review.py monitor_cycle.py; do
  [ -f "$SCRIPTS/$old" ] && mv "$SCRIPTS/$old" "$ROOT/backups/$old.$STAMP" && echo "retired $old (moved to backups)"
done
ls -l "$SCRIPTS"/finance_bot_*.sh

say "5/6 Verify"
cd "$DEST" || fail "cd $DEST"
.venv/bin/python -m pytest -q 2>&1 | tail -3
.venv/bin/python -m pytest -q >/dev/null 2>&1 || fail "tests failed — not proceeding (previous version is in $ROOT/backups)"
echo
FINBOT_NOTIFY=off .venv/bin/python main.py health 2>&1 | grep -v "INFO" | tail -15
echo
echo "--- one monitor-only cycle (never places orders) ---"
FINBOT_NOTIFY=off timeout 280 .venv/bin/python main.py run 2>&1 | grep -v "INFO\|yfinance" | tail -15

if [ "$DISCORD_TEST" = 1 ]; then
  say "6/6 Discord test"
  .venv/bin/python -c "import yaml,notify,main; ok=notify.send(yaml.safe_load(open('config.yaml')),'✅ finance-bot installed on Hermes — notifications working'); print('Discord:', 'OK' if ok else 'FAILED (check DISCORD_BOT_TOKEN in /opt/data/.env and the channel id in config.yaml)')"
else
  say "6/6 Discord test skipped (re-run with --discord-test)"
fi

say "Installed. Next: update the Hermes cron jobs (HERMES_PROMPT.md), then confirm the first run."
echo "Rollback: tar -xzf $ROOT/backups/finance-bot-$STAMP.tgz -C $ROOT"
