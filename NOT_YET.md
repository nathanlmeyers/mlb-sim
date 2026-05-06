# Not Yet — Plans, Speculation, and Things That Don't Exist

This file holds aspirational plans that were once in the README but should
not look like active capabilities. Everything below is **planned**, not
built. The active plan of record is [`.context/remediation_plan.md`](./.context/remediation_plan.md).

---

## Why this file exists

The original README had a "Roadmap: Live Trading (Week of 2026-04-20)"
section that read like a near-term commitment. Walk-forward validation
on 2026-04-20 showed the model under-performs the always-home baseline
out-of-sample. The roadmap was retired because none of its preconditions
have been met.

Putting the old roadmap here is a deliberate signal: **plans rot, evidence
doesn't**. If you're tempted to build any of this, re-check it against
`scripts/walk_forward_backtest.py` results first.

---

## Retired roadmap (for historical context only)

### Live trading on Kalshi — RETIRED

Bankroll: $500. Bets: $5-15 at eighth-Kelly / 3% cap.

#### "Day 1 — validate or stop (hard gate)"

Replaced by `scripts/check_live_gate.py`, which uses the walk-forward
backtest as the gate. **It is currently failing**, so this is a
no-op until either more data lands (B2 of remediation plan) or
totals validates (C2).

#### "Day 2-3 — Kalshi trade client"

Was: `betting/kalshi_client.py` with RSA-PSS signing, four functions
(`get_balance`, `place_order`, `get_order`, `cancel_order`). Env-var
auth (`KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`).

Status: **not built**. Don't build it until F1 of the remediation plan
is reached.

#### "Day 4 — risk controls + live mode"

Risk controls: **built** at `betting/risk.py` (Phase F1 prep). The
limits there are stricter than the original roadmap and are locked in
code, not in `config.py`.

Live mode in `paper_trade.py`: **not built**. Originally specified env-var
gate + `--yes-i-really-mean-it` flag + ticker confirmation per bet.

#### "Day 5 — nightly calibration"

`scripts/nightly_calibration.py`: **not built**. Functionality partly
covered by `paper_trade.py clv` and `walk_forward_backtest.py`. Should
become a GHA cron job once those are stable.

#### "Day 6-7 — observe"

N/A — no live bets to observe.

#### "Scaling gate"

Original criteria for raising `MAX_BET_OVERRIDE` above $10:
- 50+ live bets with bootstrap 95% lower-CI ROI > -15%, OR
- 100+ paper+live bets under current config with combined ROI > +3%, OR
- Day-1 backtest ROI > +5% AND 20+ live bets with positive ROI

Replaced by stricter rules in `betting/risk.py`. The new rule is: 50 live
bets at $2 cap, then 3% bankroll cap, with auto-halt on rolling CLV < 0
or 50-bet ROI < -10%.

---

## Explicit non-goals (kept verbatim — these are still right)

- No dashboard.
- No Postgres migration.
- No new sim features.
- No spreads or HR props.
- No adaptive Kelly.
- No multi-sportsbook.
- No market-making.

The model is ahead of the evidence. The scarce resource is bets and
closing-line data, not features or surface area.
