#!/bin/bash
# Hermes cron entry point — finance-bot end of day (no-agent script job):
# daily review → forecast-evaluate → recursive learning loop.
# Runs once per weekday at/after 13:30 America/Los_Angeles (per-day marker).
cd /opt/data/finance-bot || exit 1
exec timeout 3000 .venv/bin/python scripts/run_scheduled.py eod 2>&1
