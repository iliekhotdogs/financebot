# Message to paste into Hermes

Attach `finance-bot-hermes.tar.gz` to your Hermes chat, then paste everything
in the box below.

---

```
I've attached finance-bot-hermes.tar.gz, an update for /opt/data/finance-bot. Please do the following, in order, and show me the output of each step. Stop and tell me if any step fails.

1. Save the attachment to /opt/data/finance-bot-hermes.tar.gz, then run:
   rm -rf /opt/data/finance-bot-update && tar -xzf /opt/data/finance-bot-hermes.tar.gz -C /opt/data
   bash /opt/data/finance-bot-update/deploy/hermes/install_on_hermes.sh --discord-test

2. Update the cron jobs with cronjob_manage (scripts are referenced by filename only):
   - Job d576ceb80db2 "Finance bot monitoring cycle": set script to finance_bot_cycle.sh,
     schedule "*/5 4-22 * * 1-5", workdir /opt/data/finance-bot, no_agent true, deliver local.
   - Job 103e4b39ff68 "Finance bot daily review": set script to finance_bot_eod.sh,
     schedule "30 13,20,21 * * 1-5", workdir /opt/data/finance-bot, no_agent true, deliver local.
   - Delete job f1d09601edec "finance-bot-paper-trade" (trading now happens in the cycle job).
   - Delete job 4f2855b45a0e "finance-bot-daily-review" (duplicate).
   - Do not touch the three Gambler jobs (8770283fabf0, 62dc9a16197d, a91f0bbee8d2).
   - Resume d576ceb80db2 and 103e4b39ff68.
   The schedules are deliberately wide: scripts/run_scheduled.py gates on Pacific time
   (cycle 05:45–13:00, end-of-day once ≥13:30, weekdays) and prints nothing outside
   the window, so it is correct whether cron runs in UTC or local time, in PST or PDT.

3. List all cron jobs and show me their schedule, script, enabled state and next run.

4. Replace the cron sentence in memories/MEMORY.md with:
   "Finance-bot cron: finance_bot_cycle.sh (*/5 4-22 * * 1-5, gated 05:45–13:00 PT, places PAPER orders when execution.auto_trade=true) and finance_bot_eod.sh (30 13,20,21 * * 1-5, once/day ≥13:30 PT: review → forecast-evaluate → learn → Discord summary). Strategy switches are announced in #📈trading-reports and auto-applied after 12h unless `python main.py learning-veto` is run. Keys load from /opt/data/.env. Dashboard: /opt/data/finance-bot/reports/dashboard.html."

5. After the next weekday 05:50 PT, check /opt/data/cron/output/d576ceb80db2/ and confirm the
   first cycle ran without errors. After 13:45 PT, confirm the end-of-day run posted to Discord.

Never run `python main.py trade` or place orders manually, and never change APCA_PAPER.
```

---

## If something goes wrong

- **Roll back:** the installer prints the backup path, e.g.
  `tar -xzf /opt/data/backups/finance-bot-<timestamp>.tgz -C /opt/data`
- **Stop trading but keep monitoring:** set `execution.auto_trade: false` in
  `/opt/data/finance-bot/config.yaml`.
- **Pause everything:** ask Hermes to pause jobs d576ceb80db2 and 103e4b39ff68.
- **Discord test failed:** the bot reads `DISCORD_BOT_TOKEN` from `/opt/data/.env`
  (Hermes's own gateway token) and posts to channel `1555334067862376489`. The bot needs
  permission to post in #📈trading-reports. Or put a channel webhook in `/opt/data/.env`
  as `DISCORD_WEBHOOK_URL=...`.
