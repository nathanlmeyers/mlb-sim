# MLB Simulation Engine

A pitch-by-pitch MLB game simulation engine with betting pipeline, Kalshi market integration, and paper trading.

**Live repo:** https://github.com/nathanlmeyers/mlb-sim

> **⚠️ HONEST STATE (2026-04-20):** Walk-forward validation shows the moneyline
> model **under-performs an "always bet home" baseline** on out-of-sample games
> (51.1% accuracy vs 51.8% baseline, n=311). The earlier 54.5% in-sample claim
> was overfitting. **Live trading is not justified by current evidence.** See
> [`.context/remediation_plan.md`](./.context/remediation_plan.md) and
> [`NOT_YET.md`](./NOT_YET.md) for the planned path to validation. This
> document below describes what the codebase *does*, not what it *proves*.

---

## What This Does

Simulates MLB games using real player stats, runs daily against confirmed
lineups, compares predictions to Kalshi market prices, and identifies *candidate*
betting edges. Predictions are paper-traded and tracked against DraftKings
closing lines (CLV). The current model has **not** yet demonstrated sustained
out-of-sample edge, so paper trading is restricted to totals (the strongest
in-sample signal — also untested OOS as of writing).

---

## Core Strategy

### The thesis

The market (DraftKings / Kalshi) is efficient on average but has structural blind spots:
1. **Pitcher × park interactions** — an elite starter in a pitcher-friendly park is underpriced on unders
2. **Times-through-the-order dynamics** — starter effectiveness degrades sharply in the 3rd TTO; markets don't always fully price late-game bullpen risk
3. **Early-season overreaction** — 2-3 hot/cold starts can shift public opinion more than the data justifies

### The approach

Rather than trying to beat the market on every game, we:

1. **Simulate every game** using our model (both fast box-score and detailed PA-by-PA)
2. **Blend with market prices** as a Bayesian prior (weight = 0.65) — assume the market knows something we don't
3. **Only bet when we *still* disagree after blending** and the disagreement is ≥ 4%
4. **Never bet against strong market conviction** — skip games where Kalshi prices the market outside 15%–85%
5. **Size with fractional Kelly** (eighth-Kelly, max 3% of bankroll per game) to survive variance

The key insight: our *unblended* model already beats DraftKings closing lines (54.5% vs 52.7% accuracy, 0.2467 vs 0.2499 Brier on a 167-game backtest). But finding 7+ edges per day was a sign we were fooling ourselves — edges that small are within model noise. Tightening to 4% edge + market guardrails reduces bets from ~7/day to ~1-3/day, focusing on genuine disagreements.

---

## Backtest Results

### In-sample (legacy, kept for context — DO NOT trust as edge evidence)

This was the original headline. It evaluated on the same data the
hyperparameters were tuned against.

| Model | ML Accuracy | Brier | Run-Line Acc | O/U Acc |
|-------|-------------|-------|--------------|---------|
| Detailed+TTO ("our best") | 54.5% | 0.2467 | 60.5% | 64.1% |
| DraftKings closing line | 52.7% | 0.2499 | — | — |
| Always-home baseline | 55.1% | 0.2475 | — | — |

Notice "always home" already beats the model. That should have been the
flag. The 54.5% number is fragile and does not survive walk-forward.

### Walk-forward out-of-sample (the number that matters)

Run `python scripts/walk_forward_backtest.py` for fresh numbers. As of
2026-04-20, with 4 cutoffs (Apr 5/8/12/14) and 311 OOS games:

| Metric | Value | Interpretation |
|---|---|---|
| OOS accuracy (ML) | **51.1%** | 95% CI [45.3%, 56.9%] |
| Always-home baseline | **51.8%** | model is below baseline |
| OOS Brier | 0.2518 | calibrated, but no edge |
| Live-config bets fired | 5 | n too small for any verdict |

**Bottom line: the moneyline model has not demonstrated edge OOS.** The
totals signal (the supposedly strongest one) is not yet evaluated OOS at
scale — that requires the ESPN-closing-line backfill (Phase B2 of the
remediation plan) to complete.

---

## Paper Trading Results So Far

| Day | Bets | Record | P&L | Notes |
|-----|------|--------|-----|-------|
| Apr 13 | 7 | 3-4 | +$0.15 | Old 0.50 market weight, 2% edge threshold |
| Apr 15 | 6 | 1-5 | -$1.46 | Bet against extreme market conviction (filter now blocks this) |
| Apr 18 | 1 | 0-1 | -$0.41 | First post-tightening ML bet — lost |
| **Total** | **14** | **4-10** | **-$1.72** | Bankroll: **$18.28** / $20 |

**CLV vs DK closing (run `python scripts/paper_trade.py clv`):** mean
+8.9% on 13 matched bets, 85% positive. **However** that average is
dominated by 3 Apr 15 bets at extreme Kalshi prices (2.5%, 14.5%,
17.5%) — exactly the bets the new `MIN_MARKET_PRICE=0.15` filter would
now block. Excluding those, mean CLV ≈ +1.5% on 10 bets, indistinguishable
from noise.

**Status:** Paper trading is now defaulted to **totals only** (use
`--allow-ml` to override). The walk-forward verdict is that ML loses to
"always home" baseline OOS, so we should not log moneyline bets until
the model demonstrates OOS edge.

---

## Model Architecture

### Simulation engines

**Box score model** (`sim/game.py`)
- Fast (5000 sims in ~2 seconds)
- Negative binomial run distribution (r=4.0)
- Team offense/defense + starter quality factor + park factor
- Used for daily predictions and backtest at scale

**Detailed model** (`sim/detailed_game.py`, `sim/plate_appearance.py`, `sim/inning.py`)
- Count-progression Markov chain per plate appearance
- Times Through the Order penalty (K% drops 17% by 3rd TTO)
- Pitcher fatigue with real pitch counts
- TTO-aware bullpen pull decisions
- Baserunner state machine with empirical advancement rates

### Player models

**Batter** (`models/batter_model.py`)
- K%, BB%, BABIP, HR/FB, GB%/FB%/LD%
- Platoon splits (L vs R)
- Speed score for baserunning
- Exponential decay weighting + Bayesian regression to league avg

**Pitcher** (`models/pitcher_model.py`)
- Same rate stats + strike% (key command metric from ESPN/Kalshi)
- 2025 season prior blended at 0.5 weight with current-season stats
- Reg_n=20 (light regression — pitcher stats stabilize faster than team stats)

### Ensemble (`sim/ensemble.py`)

Pipeline:
1. Run box score + detailed engines in parallel
2. Blend engine outputs (60% box / 40% detailed)
3. Record-based anchor (log5 formula with home edge)
4. Logistic probability calibration
5. **Market prior blend (Kalshi at 0.65 weight, confidence-adjusted)**

When our two engines agree strongly, market weight is discounted by 30%. When they disagree, we lean more on the market.

### Data sources

- **MLB Stats API** (`statsapi`) — game schedules, box scores, lineups, player stats
- **Kalshi API** — live market prices for ML, totals, spreads, HR props
- **ESPN Summary API** — DraftKings closing lines (ML, spread, total)
- **FanGraphs** — park factors (30 stadiums)
- **Baseball Reference** — baserunner advancement rates

Historical data cached locally in `.context/`:
- `season_2025.json` — full box scores (regenerate with `python scripts/fetch_season.py 2025`)
- `season_2026.json` — 332 games, updated daily
- `espn_odds_2026.json` — DraftKings closing lines (refresh with `python scripts/fetch_espn_odds.py 2026`)
- `park_factors_2024.json` — 30 stadiums
- `kalshi_history.json` — daily market prices accumulated (currently 2 days; B3 of remediation plan adds nightly closing snapshots)

---

## Daily Pipeline

### Automated (GitHub Actions)

The pipeline runs **automatically** via `.github/workflows/daily_pipeline.yml`:

- **Settle job** (08:00 ET daily): fetches yesterday's completed games, settles any open bets in the paper ledger, commits data back to the repo.
- **Dispatch job** (10:00 / 13:00 / 16:00 / 19:00 ET): lists today's eligible games and spawns one GHA job per game via a matrix strategy.
- **Per-game jobs**: each job sleeps until T-60 min before first pitch, then runs the simulation pipeline for that single game, logs any qualifying edges to the paper ledger, and commits the results with retry-on-conflict. One job per game keeps the daily run narrow and lets predictions use the most-recent lineup info.

No laptop or manual trigger required. Results are version-controlled in `.context/`.

### Manual run

```bash
python scripts/daily_pipeline.py              # today
python scripts/daily_pipeline.py 2026-04-18   # specific date
python scripts/daily_pipeline.py --game-id <MLB_GAME_PK>  # single-game mode used by per-game GHA jobs
```

Runs 5 steps:
1. Fetch yesterday's completed games → update season data
2. Fetch Kalshi market prices (ML, totals, spreads, HR)
3. Fetch confirmed lineups from MLB Stats API
4. Run simulations via `predict_game()` — box score (5000 sims) + detailed (500 sims) blended 60/40, with market prior if Kalshi prices available
5. Compare model vs market, identify edges ≥ 4%, output report

Output saved to `.context/pipeline_YYYY_MM_DD.json`.

### Paper trading

```bash
python scripts/paper_trade.py log 2026-04-18    # log today's bets from pipeline
python scripts/paper_trade.py settle 2026-04-18 # settle after games finish
python scripts/paper_trade.py report            # full P&L history
```

Paper trading uses a $20 bankroll with eighth-Kelly sizing and 3% max per bet. Ledger is stored in `.context/paper_ledger.json`.

---

## Key Constants (`config.py`)

### Bet selection

```python
TRAINED_MARKET_WEIGHT = 0.65      # Blend weight for Kalshi prices
MARKET_CONFIDENCE_DISCOUNT = 0.30 # Lower market weight when engines agree
MIN_EDGE_THRESHOLD = 0.04         # Minimum 4% edge to log a bet
MIN_MARKET_PRICE = 0.15           # Skip bets when market < 15%
MAX_MARKET_PRICE = 0.85           # Skip bets when market > 85%
KELLY_FRACTION = 0.125            # Eighth-Kelly sizing
MAX_BET_PCT = 0.03                # Max 3% of bankroll per game
```

### Simulation

```python
TTO_MULTIPLIERS = {...}           # Pitcher effectiveness by times through order
STARTER_INNINGS_SHARE = 0.30      # Starter pitches ~30% of innings in early season
HOME_FIELD_ADVANTAGE_RUNS = 0.24  # MLB home teams score +0.24 runs/game
NEGATIVE_BINOMIAL_R = 4.0         # Run distribution dispersion parameter
```

---

## Things We Learned

### 1. Simulation mechanics

- **Count progression matters.** Our first version collapsed each PA into a single outcome draw. Adding a proper pitch-by-pitch count state machine (0-0 → 1-0, 2-1, etc.) with count-dependent outcome probabilities improved the detailed model dramatically.
- **TTO penalty is huge and well-documented.** A starter's K% drops 8-17% by the 3rd TTO. Ignoring it systematically underrates late-inning offenses.
- **Strike% > K/9.** A pitcher's % of pitches that are strikes is a better command metric than aggregate K-rate. ESPN's API returns this directly.
- **2025 priors help pitchers, not teams.** Blending prior-season stats at 0.5 weight for pitchers (reg_n=20) improves model stability. For team-level stats, 2025 priors added noise — the roster changes too much year to year.

### 2. Market dynamics

- **The market is usually right.** Running the model without a market prior was finding 7+ "edges" per day — most were fake. Blending at weight=0.65 cuts this to 1-3 real disagreements.
- **Kalshi offers more than moneyline.** They have totals, run lines, and HR props. Our 64% O/U accuracy in backtest could be actionable on Kalshi's totals markets.
- **Extreme prices mean the market knows something.** When Kalshi prices a team at 2% or 98%, there's info we don't have (injury, lineup, pitcher). Never bet against it with an early-season model.

### 3. Bankroll and psychology

- **Kelly criterion works, but only on real edges.** Betting 8th-Kelly against fake 2% edges just slowly bleeds. Real 5%+ edges, even at 8th-Kelly, can compound meaningfully.
- **Paper trade first, always.** 13 bets with real money at day 2's losses would have felt awful. In paper it's just data. The $1.31 loss told us our thresholds were wrong — cheaply.

### 4. Data pipeline

- **The MLB Stats API is excellent and free.** `statsapi` Python package handles lineups, box scores, and game-by-game player logs with no auth.
- **ESPN's hidden `pickcenter` field has DraftKings closing lines** for the current season. Perfect benchmark.
- **Kalshi settled markets don't include prices** in API responses — we cache daily prices to build our own historical dataset.
- **Cache everything to `.context/`** (gitignored). Lets us backtest without hitting APIs.

---

## Directory Structure

```
├── config.py              # All constants (bet selection, simulation, regression)
├── cli.py                 # Click CLI (init_db, fetch, smoke_test, backtest, backtest_json, fetch_season, train)
├── pyproject.toml
├── .github/workflows/
│   └── daily_pipeline.yml # Automated daily pipeline (settle + per-game matrix)
├── sim/                   # Simulation engines
│   ├── game.py           # Box score model (negative binomial, 5000 sims)
│   ├── detailed_game.py  # Pitch-by-pitch with TTO + bullpen management
│   ├── plate_appearance.py  # Count-progression Markov chain per PA
│   ├── inning.py         # Half-inning loop
│   ├── baserunners.py    # 8-state base machine with empirical advancement rates
│   └── ensemble.py       # Blends both engines (60/40) + record anchor + market prior
├── models/                # Player and environment models
│   ├── batter_model.py   # K%, BB%, BABIP, HR/FB, platoon splits, Bayesian regression
│   ├── pitcher_model.py  # Same rate stats + strike%, 2025 prior blending
│   ├── team_model.py     # Team-level offense/defense aggregates
│   ├── weather.py        # Temperature, wind, dome detection
│   ├── schedule_features.py  # Rest days, travel, fatigue
│   └── calibration.py    # Isotonic regression + logistic fallback
├── data/                  # Data fetching (all read-only)
│   ├── fetch.py          # MLB Stats API wrappers
│   ├── kalshi.py         # Kalshi public API client (read-only, no auth)
│   ├── lineup_fetcher.py # Real lineup + per-player model builder from API
│   ├── db.py             # PostgreSQL schema definitions (optional)
│   └── load.py           # PostgreSQL bulk loaders (optional)
├── betting/               # Bet evaluation
│   ├── ev.py             # EV calculation, fractional Kelly, odds conversion
│   ├── predictions.py    # BettingPrediction dataclass (ML, run line, total evaluation)
│   └── confidence.py     # Engine agreement scoring (scales market weight)
├── backtest/              # Evaluation framework
│   ├── json_backtest.py  # JSON-backed backtest (no DB needed)
│   ├── model_eval.py     # Composite scoring across ML/RL/O-U/Brier
│   └── evaluate.py       # PostgreSQL-backed evaluation (optional)
├── scripts/
│   ├── daily_pipeline.py      # Full daily workflow (5 steps, supports --game-id)
│   ├── list_games_for_dispatch.py  # Outputs game matrix JSON for GHA per-game jobs
│   ├── paper_trade.py         # Paper trading: log, settle, report subcommands
│   ├── daily_picks.py         # Quick standalone daily scanner
│   ├── fetch_daily.py         # Pull yesterday's games
│   ├── fetch_season.py        # Pull full season (2025 or 2026)
│   ├── fetch_kalshi_history.py  # Historical Kalshi settled events
│   ├── fetch_espn_odds.py     # ESPN DraftKings closing lines
│   └── train_ensemble.py      # Grid search ensemble weights
└── .context/              # Cached data (gitignored)
    ├── season_2025.json        # 2559 games with full box scores
    ├── season_2026.json        # 317+ games, updated daily by GHA
    ├── espn_odds_2026.json     # 258 games with DraftKings closing lines
    ├── park_factors_2024.json  # 30 stadiums
    ├── kalshi_history.json     # Daily market prices
    ├── paper_ledger.json       # Paper trading bankroll + bet history
    └── pipeline_*.json         # Daily pipeline outputs
```

---

## Roadmap

The earlier "Live Trading: Week of 2026-04-20" plan was retired after
walk-forward showed the moneyline model has no OOS edge. The current
plan is in [`.context/remediation_plan.md`](./.context/remediation_plan.md),
summarized in [`NOT_YET.md`](./NOT_YET.md). Both should be read before
considering any live trading work.

In short: validate totals OOS first (Phase C), then re-tune (Phase D),
then production hardening (Phase E), then a gated live launch (Phase F).
Realistic earliest live date is **late May 2026** — and only if totals
clear the OOS gate, which they may not.

---

## Honest Assessment

**What works:**
- Simulation engine is calibrated (runs ~4.4 per team, ~52% home win rate)
- Pipeline is end-to-end functional: data → simulate → compare → recommend
- Walk-forward backtest exists and produces honest OOS numbers
- CLV instrumentation exists (`paper_trade.py clv`)
- Risk module + live-trading hard gate are pre-built (`betting/risk.py`,
  `scripts/check_live_gate.py`)

**What does not work yet:**
- ML model has no demonstrated OOS edge (51.1% < 51.8% always-home baseline)
- Totals OOS evaluation exists but needs more closing-line data to be conclusive
- Hyperparameters (TRAINED_MARKET_WEIGHT=0.65, MIN_EDGE_THRESHOLD=0.04) were
  tuned in-sample; D1 of the remediation plan replaces this with real
  walk-forward grid search
- Kalshi historical price snapshots are sparse (2 days); B3 adds nightly closing snapshots
- No real-money trade client exists; `data/kalshi.py` is read-only

**What you should not do:**
- Place real bets. The walk-forward gate is failing and the live gate
  in `scripts/check_live_gate.py` will refuse.
