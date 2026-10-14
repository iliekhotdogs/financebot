"""
Web dashboard for the finance-bot.

Shows:
  - Current mode (monitoring/analyzing/paper trading)
  - Market snapshot and data freshness
  - Research findings from all roles
  - Jev's final proposed action and rationale
  - Risk checks and pass/reject/resize results
  - Order status, fills, fees, P&L
  - Recent decisions and daily review summaries
  - Learning system: candidates, evaluations, promotion status

reports/dashboard.html is regenerated after every cycle (dashboard/render.py).
"""

from .app import create_app

__all__ = ["create_app"]
