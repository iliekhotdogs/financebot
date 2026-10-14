"""Keep scheduled daily-review details in a log, not in Discord."""
from __future__ import annotations

import sys
from pathlib import Path


def summary(path: Path, exit_code: int) -> str:
    """A daily digest is sent separately; only failures need another post."""
    if exit_code:
        return "🚨 Daily check did not finish. Please check the bot dashboard."
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return "⚠️ Daily check log is missing. Please check the bot dashboard."
    if any(line.startswith(("[ERROR]", "Traceback (most recent call last):"))
           for line in text.splitlines()):
        return "⚠️ Part of the daily check failed. Please check the bot dashboard."
    return ""


if __name__ == "__main__":
    result = summary(Path(sys.argv[1]), int(sys.argv[2]))
    if result:
        print(result)
