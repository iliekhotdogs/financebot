"""
Discord notifications.

Posts directly to Discord so the caller KNOWS the message was delivered
before acting on it (auto-promotion announces first, then switches).

Transport (first available):
  1. DISCORD_WEBHOOK_URL                      — channel webhook
  2. DISCORD_BOT_TOKEN + channel id           — the Hermes gateway bot token
     (config notifications.discord_channel_id or DISCORD_CHANNEL_ID)

Disabled when notifications.enabled is false or FINBOT_NOTIFY=off.
Never raises; returns True only on HTTP 2xx.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

MAX_CONTENT_LEN = 900  # concise mobile-friendly posts; below Discord's 2000 limit


def _compact(text: str, budget: int) -> str:
    """Keep complete leading lines; never cut an explanation mid-word."""
    text = text.strip()
    if len(text) <= budget:
        return text
    suffix = "\n… Details omitted; see the dashboard/report."
    remaining = budget - len(suffix)
    kept = []
    for line in text.splitlines():
        candidate = "\n".join([*kept, line])
        if len(candidate) > remaining:
            break
        kept.append(line)
    if kept:
        return "\n".join(kept) + suffix
    return text[:remaining].rsplit(" ", 1)[0].rstrip() + suffix


def _enabled(config: dict) -> bool:
    if os.getenv("FINBOT_NOTIFY", "").lower() in ("off", "0", "false"):
        return False
    return bool(config.get("notifications", {}).get("enabled", True))


def _multipart(payload: dict, files: list) -> tuple:
    """Discord multipart body: payload_json + files[n]."""
    import uuid
    boundary = uuid.uuid4().hex
    parts = [(f'--{boundary}\r\nContent-Disposition: form-data; name="payload_json"\r\n'
              f'Content-Type: application/json\r\n\r\n{json.dumps(payload)}\r\n').encode()]
    for i, path in enumerate(files):
        name = os.path.basename(path)
        ctype = "image/png" if name.endswith(".png") else "text/html" if name.endswith(".html") else "application/octet-stream"
        with open(path, "rb") as f:
            data = f.read()
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="files[{i}]"; filename="{name}"\r\n'
                      f'Content-Type: {ctype}\r\n\r\n').encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def send(config: dict, text: str, kind: str = "info", files: list = None,
         max_content_len: int = MAX_CONTENT_LEN) -> bool:
    """Send `text` (and optional file attachments) to the configured Discord channel.
    kind gates per-type toggles."""
    ncfg = config.get("notifications", {})
    if not _enabled(config):
        return False
    if kind != "info" and not ncfg.get(kind, True):
        return False
    prefix = str(ncfg.get("prefix", "🤖 finance-bot"))[:80]
    header = f"**{prefix}** — "
    payload = {"content": header + (text if max_content_len == 2000 and len(text) <= 2000 - len(header)
                                    else _compact(text, min(max_content_len, 2000) - len(header))),
               "allowed_mentions": {"parse": []}}
    files = [f for f in (files or []) if f and os.path.exists(f) and os.path.getsize(f) < 8_000_000]
    if files:
        body, ctype = _multipart(payload, files)
    else:
        body, ctype = json.dumps(payload).encode(), "application/json"

    webhook = os.getenv("DISCORD_WEBHOOK_URL")
    token = os.getenv("DISCORD_BOT_TOKEN")
    channel = str(ncfg.get("discord_channel_id") or os.getenv("DISCORD_CHANNEL_ID") or "")
    if webhook:
        url, headers = webhook, {}
    elif token and channel:
        url = f"https://discord.com/api/v10/channels/{channel}/messages"
        headers = {"Authorization": f"Bot {token}"}
    else:
        logger.warning("Discord notification skipped: set DISCORD_WEBHOOK_URL or "
                       "DISCORD_BOT_TOKEN + notifications.discord_channel_id")
        return False

    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": ctype,
        "User-Agent": "DiscordBot (finance-bot, 1.0)",
        **headers,
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            ok = 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        logger.warning("Discord post failed: HTTP %s %s", e.code, e.read()[:200])
        return False
    except Exception as e:
        logger.warning("Discord post failed: %s", e)
        return False
    return ok
