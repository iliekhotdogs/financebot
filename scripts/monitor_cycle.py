#!/usr/bin/env python3
"""
Cron monitoring script for the combined trading bot.

Runs one monitoring cycle (`./venv/bin/python main.py run`) and exits.
Intended to be called by a cron job every 5 minutes during market hours.

The script checks:
  1. Whether the current time is within the trading window (05:45-13:00 Pacific)
  2. Whether it's a weekday
  3. If not in window, exits silently (cron suppresses empty output)

Usage (cron):
  */5 12-19 * * 1-5 /opt/data/finance-bot/.venv/bin/python /opt/data/finance-bot/scripts/monitor_cycle.py
"""
import os
import sys
import yaml
from datetime import datetime, date

# Add finance-bot to path
BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BOT_DIR, ".env"), override=False)
load_dotenv(os.getenv("HERMES_ENV_FILE", "/opt/data/.env"), override=False)

from data import DataProvider


def is_trading_window():
    """Check if current Pacific time is within trading window on a weekday."""
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("America/Los_Angeles")
    now = datetime.now(tz)

    weekday = now.weekday()  # 0=Mon, 4=Fri
    if weekday > 4:
        return False, "weekend"

    time_str = now.strftime("%H:%M")
    # Trading window: 05:45 - 13:00 Pacific
    if "05:45" <= time_str <= "13:00":
        return True, time_str
    else:
        return False, time_str


def run_monitoring_cycle():
    """Run one monitoring cycle (no orders). Prefer scripts/run_scheduled.py."""
    from engine import run_cycle
    with open(os.path.join(BOT_DIR, "config.yaml")) as f:
        config = yaml.safe_load(f)
    paper = os.getenv("APCA_PAPER", "true").lower() == "true"
    run_cycle(DataProvider(paper=paper), config, execute=False)


if __name__ == "__main__":
    in_window, reason = is_trading_window()
    if not in_window:
        print(f"[SKIPPED] Outside trading window ({reason})")
        sys.exit(0)

    run_monitoring_cycle()
