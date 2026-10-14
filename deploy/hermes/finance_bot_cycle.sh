#!/bin/bash
# Hermes cron entry point — finance-bot trading cycle (no-agent script job).
# Copy to /opt/data/scripts/ and reference by filename only.
# The Python script gates on 05:45–13:00 America/Los_Angeles weekdays and is
# silent outside it; a file lock prevents overlapping cycles; `timeout` keeps
# a stuck cycle from running into the next slot.
cd /opt/data/finance-bot || exit 1
exec timeout 280 .venv/bin/python scripts/run_scheduled.py cycle 2>&1
