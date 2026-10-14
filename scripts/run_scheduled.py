#!/usr/bin/env python3
"""
Single scheduled entry point for Hermes cron (DST-safe).

  run_scheduled.py cycle   — trading cycle; runs only weekdays 05:45–13:00 Pacific
  run_scheduled.py eod     — end-of-day: daily review → forecast-evaluate → learn;
                             runs once per weekday at/after 13:30 Pacific

Why gate here instead of in cron: Hermes cron timezone handling was
inconsistent (jobs fired at UTC and at local interpretations of the same
expression), and a fixed UTC window drifts an hour across PST/PDT. Schedule
cron broadly in UTC (both DST offsets) and let this script decide on Pacific
wall-clock time. The `eod` task writes a per-day marker so the PST and PDT
cron slots never both run it.

Outside the window it exits 0 with EMPTY stdout (Hermes suppresses delivery
on empty output). Set FINBOT_VERBOSE=1 to print skip reasons.
"""
import os
import sys
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOT_DIR)
os.chdir(BOT_DIR)

PACIFIC = ZoneInfo("America/Los_Angeles")
MARKER_DIR = os.path.join(BOT_DIR, "journal", "cache")


def now_pacific() -> datetime:
    override = os.getenv("FINBOT_FAKE_NOW")  # testing: ISO timestamp with offset
    if override:
        return datetime.fromisoformat(override).astimezone(PACIFIC)
    return datetime.now(PACIFIC)


def skip(msg: str) -> int:
    if os.getenv("FINBOT_VERBOSE"):
        print(f"[SKIPPED] {msg}")
    return 0


def in_window(now: datetime, start: dtime, end: dtime) -> bool:
    return now.weekday() < 5 and start <= now.time() <= end


def _marker(task: str, now: datetime) -> str:
    return os.path.join(MARKER_DIR, f"{task}_{now.date().isoformat()}.done")


def main(argv) -> int:
    task = argv[1] if len(argv) > 1 else ""
    now = now_pacific()

    if task == "cycle":
        if not in_window(now, dtime(5, 45), dtime(13, 0)):
            return skip(f"cycle outside 05:45–13:00 PT weekday window ({now:%a %H:%M %Z})")
        import main as bot
        sys.argv = ["main.py", "cycle"]
        bot.main()
        return 0

    if task == "eod":
        if not in_window(now, dtime(13, 30), dtime(16, 0)):
            return skip(f"eod outside 13:30–16:00 PT weekday window ({now:%a %H:%M %Z})")
        marker = _marker("eod", now)
        if os.path.exists(marker):
            return skip(f"eod already ran today ({now.date()})")
        os.makedirs(MARKER_DIR, exist_ok=True)
        open(marker, "w").write(now.isoformat())

        from scripts.daily_review import generate_review
        generate_review()
        import main as bot
        for cmd in (["forecast-evaluate"], ["learn"]):
            sys.argv = ["main.py", *cmd]
            try:
                bot.main()
            except Exception as e:  # one step failing must not block the next
                print(f"[ERROR] {' '.join(cmd)} failed: {type(e).__name__}: {e}")
        return 0

    print("usage: run_scheduled.py cycle|eod", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
