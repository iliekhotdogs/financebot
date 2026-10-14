#!/bin/bash
# Build finance-bot-hermes.tar.gz: code + config + deploy scripts, NO data or secrets.
# Extracts to ./finance-bot-update/ (see deploy/hermes/HERMES_PROMPT.md).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$HERE/../finance-bot-hermes.tar.gz}"
STAGE="$(mktemp -d)"
mkdir "$STAGE/finance-bot-update"
( cd "$HERE" && git ls-files -z --cached --others --exclude-standard ) | while IFS= read -r -d '' f; do
  case "$f" in
    journal/*.csv|journal/*.json|journal/.advisor_last_call|learning/*.csv|learning/*.jsonl|learning/*.json|daily_reviews/*|.env) continue ;;
  esac
  [ -f "$HERE/$f" ] || continue
  mkdir -p "$STAGE/finance-bot-update/$(dirname "$f")"
  cp -p "$HERE/$f" "$STAGE/finance-bot-update/$f"
done
COPYFILE_DISABLE=1 tar -czf "$OUT" -C "$STAGE" finance-bot-update
rm -rf "$STAGE"
echo "$OUT"
