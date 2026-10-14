#!/bin/bash
# Hermes cron entry point — keeps the LIVE dashboard reachable on your tailnet.
# Copy to /opt/data/scripts/ and schedule every 10 min (no_agent, deliver local).
#
# Your PC (any device on the same Tailscale account) then opens:
#     http://hermes-finbot/            (auto-refreshes every 60 s, live broker data)
#
# Idempotent and silent when healthy, so Hermes delivers nothing on normal runs:
#   - downloads the static tailscale binaries once into /opt/data/tailscale
#   - runs tailscaled in USERSPACE mode (no root, no /dev/net/tun needed)
#   - joins the tailnet once with TS_AUTHKEY from /opt/data/.env (state is kept
#     in /opt/data/tailscale/state, so the key is not needed after the first join)
#   - runs the read-only Flask dashboard on 127.0.0.1:8080 (never on a public port)
#   - `tailscale serve` publishes it to the tailnet only, on port 80
# Anything it had to (re)start is printed, so a restart shows up in cron output.
set -u
ROOT="${HERMES_ROOT:-/opt/data}"
BOT="$ROOT/finance-bot"
TSDIR="$ROOT/tailscale"
SOCK="$TSDIR/tailscaled.sock"
HOST="${FINBOT_TS_HOSTNAME:-hermes-finbot}"
PORT="${FINBOT_DASHBOARD_PORT:-8080}"

set -a; [ -f "$ROOT/.env" ] && . "$ROOT/.env"; set +a
mkdir -p "$TSDIR/state"
ts() { "$TSDIR/tailscale" --socket="$SOCK" "$@"; }

# 1. Binaries (once)
if [ ! -x "$TSDIR/tailscaled" ]; then
  case "$(uname -m)" in x86_64) arch=amd64 ;; aarch64|arm64) arch=arm64 ;; *) arch="$(uname -m)" ;; esac
  tgz="$(curl -fsSL 'https://pkgs.tailscale.com/stable/?mode=json' \
        | "$BOT/.venv/bin/python" -c "import json,sys; print(json.load(sys.stdin)['Tarballs']['$arch'])")" \
    || { echo "dashboard: cannot resolve tailscale download for $arch"; exit 1; }
  curl -fsSL "https://pkgs.tailscale.com/stable/$tgz" | tar -xz -C "$TSDIR" --strip-components=1 \
    || { echo "dashboard: tailscale download failed"; exit 1; }
  echo "dashboard: installed $tgz"
fi

# 2. tailscaled (userspace networking)
if ! pgrep -f "$TSDIR/tailscaled" >/dev/null; then
  nohup "$TSDIR/tailscaled" --tun=userspace-networking --statedir="$TSDIR/state" \
        --socket="$SOCK" >"$TSDIR/tailscaled.log" 2>&1 &
  for _ in 1 2 3 4 5 6 7 8 9 10; do [ -S "$SOCK" ] && break; sleep 1; done
  echo "dashboard: started tailscaled"
fi

# 3. Join the tailnet (first run only; afterwards the saved state is reused)
if ! ts status --json 2>/dev/null | grep -q '"BackendState": *"Running"'; then
  if [ -n "${TS_AUTHKEY:-}" ]; then
    ts up --authkey="$TS_AUTHKEY" --hostname="$HOST" >/dev/null 2>&1 || ts up --hostname="$HOST" >/dev/null 2>&1
  else
    ts up --hostname="$HOST" >/dev/null 2>&1
  fi
  ts status --json 2>/dev/null | grep -q '"BackendState": *"Running"' \
    && echo "dashboard: joined tailnet as $HOST" \
    || { echo "dashboard: not logged in to Tailscale — set TS_AUTHKEY in $ROOT/.env (see DEPLOY.md 6c)"; exit 1; }
fi

# 4. Dashboard server (localhost only — Tailscale is the access boundary)
if ! curl -fs -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/api/data"; then
  pkill -f "main.py dashboard" 2>/dev/null
  ( cd "$BOT" && nohup .venv/bin/python main.py dashboard --host 127.0.0.1 --port "$PORT" \
        >"$TSDIR/dashboard.log" 2>&1 & )
  sleep 4
  curl -fs -o /dev/null --max-time 10 "http://127.0.0.1:$PORT/api/data" \
    && echo "dashboard: started on 127.0.0.1:$PORT" \
    || { echo "dashboard: server failed to start — see $TSDIR/dashboard.log"; tail -5 "$TSDIR/dashboard.log"; exit 1; }
fi

# 5. Publish to the tailnet (http://$HOST/)
if ! ts serve status 2>/dev/null | grep -q "127.0.0.1:$PORT"; then
  ts serve --bg --http=80 "$PORT" >/dev/null 2>&1 \
    && echo "dashboard: serving http://$HOST/ on your tailnet" \
    || echo "dashboard: tailscale serve failed — see $TSDIR/tailscaled.log"
fi
exit 0
