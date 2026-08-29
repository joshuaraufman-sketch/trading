# Running this unattended

The system is designed to run once per weekday evening without being
watched. This is how to set that up and what to check.

## Daily rhythm

```
16:20 ET   scheduled task runs run_scheduled.py
           -> gates on trading day and close time
           -> run_daily.py    decides tomorrow's target weight
           -> run_reconciliation.py  confirms yesterday matched intent
           -> records health

evening    you run check_health.py, or glance at state/health.json
```

## Exit codes

Distinguishing "nothing to do" from "something broke" is the point.
If a quiet weekend looked like a failure, the alerts would be muted
within a fortnight and gone when they mattered.

| Code | Meaning | Action |
|---|---|---|
| 0 | Healthy. Acted, or correctly declined. | none |
| 1 | A run executed and something is wrong. | investigate |
| 2 | Refused: not a trading day, before the close, or calendar coverage lapsed. | none |

Alert on **1 only**.

## Windows Task Scheduler

Create a Basic Task:

- **Trigger:** Daily, 16:20, recur every 1 day.
  Weekends and holidays are handled in code, not the trigger — the
  calendar lives in one place.
- **Action:** Start a program
  - Program: `C:\Users\joshu\Trading\.venv\Scripts\python.exe`
  - Arguments: `scripts\run_scheduled.py`
  - Start in: `C:\Users\joshu\Trading`

Leave `--submit` off until several dry runs have looked right.

Under **Settings**, enable "Run task as soon as possible after a
scheduled start is missed" — a laptop asleep at 16:20 should still run
when it wakes, and the stale-bar check will refuse if it has become too
late to be meaningful.

Do NOT enable "Stop the task if it runs longer than" at a short value.
A slow API response should not leave a half-submitted state.

## The failure that will actually catch you out

A run that fails is loud: it exits 1, writes a health record, and prints
the reason.

A scheduled task that **stops running** is silent. No logs, no errors,
no alerts. Everything looks calm because nothing is happening. The only
signal is the age of the last successful run, which is why
`check_health.py` exists and why staleness is measured in trading
sessions rather than calendar days.

Get in the habit of running it in the evening. It takes a second and it
is the only thing standing between you and a system that quietly stopped
working three weeks ago.

## Going live

Currently paper-only, enforced in two places:

- `_get_trading_client` hardcodes `paper=True`
- `preflight` fails on a non-paper account

Both are deliberate. Going live must be a code change someone thought
about, never an environment variable that can be flipped by accident.

Leverage is guarded separately, also in two places:
`live_strategy.max_weight` and `execution_policy.max_gross_exposure`.
Both would have to change. See DECISIONS.md for why leverage is the last
thing added rather than the first.

## Maintenance

**Holiday table.** `market_calendar.py` covers 2026-2028. Outside those
years `run_scheduled.py` refuses with code 2 rather than guessing. Extend
it well before the window lapses; the original table covered 2026 alone,
which would have silently treated every 2027 holiday as a normal
session.

**Account changes.** If credentials change, `preflight` fails on the
pinned account id and the equity baseline resets. Both are intended. Run
`preflight_check.py`, re-pin the id in `config/account.yaml`.

**Parameter changes.** `config/live_strategy.yaml` must match what
research validated. A test asserts this. Changing it means the deployed
strategy is no longer the one that was tested.
