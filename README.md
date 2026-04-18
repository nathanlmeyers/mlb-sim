# MLB Simulation Engine

A pitch-by-pitch MLB game simulation engine with betting pipeline, Kalshi market integration, and paper trading.

**Live repo:** https://github.com/nathanlmeyers/mlb-sim

---

## What This Does

Simulates MLB games using real player stats, runs daily against confirmed lineups, compares predictions to Kalshi market prices, and identifies profitable betting edges. All predictions are paper-traded and tracked against DraftKings closing lines to validate that the model has genuine alpha before risking real money.

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

## Backtest Results (167 games, 2026 Mar 28 – Apr 9)

| Model | ML Accuracy | Brier | Run-Line Acc | O/U Acc |
|-------|-------------|-------|--------------|---------|
| **Detailed+TTO (our best)** | **54.5%** | **0.2467** | **60.5%** | **64.1%** |
| Box + SP (2025 priors) | 53.3% | 0.2525 | 59.9% | 58.7% |
| Detailed+TTO + SP | 52.7% | 0.2505 | 58.7% | 54.5% |
| Box (no SP) | 49.7% | 0.2526 | 60.5% | 61.7% |
| **DraftKings closing line** | 52.7% | 0.2499 | — | — |
| Always Home baseline | 55.1% | 0.2475 | — | — |

**Our model beats DK closing lines on both accuracy and Brier.** Over/under at 64.1% is particularly strong — O/U markets are where our TTO + count-progression model adds the most value.

---

## Paper Trading Results So Far

| Day | Bets | Record | P&L | ROI | Notes |
|-----|------|--------|-----|-----|-------|
| Apr 13 | 7 | 3-4 | +$0.15 | +5.3% | Old 0.50 market weight, 2% edge threshold |
| Apr 15 | 6 | 1-5 | -$1.46 | — | Lost to sharp market (bet vs 86%+ conviction) |
| **Total** | **13** | **4-9** | **-$1.31** | **-24.7%** | **Pre-tightening** |
| Apr 18+ | 1 | TBD | TBD | TBD | Post-tightening: 4% edge min, 15-85% market bounds |

**Lesson from Apr 15:** Bet WSH +$0.44 when Kalshi had PIT at 86%; PIT won 2-0. Bet SF +$0.58 when Kalshi had CIN at 98%; SF lost 3-8. Betting against extreme market confidence with an untested early-season model was a clear mistake. Now filtered.

**Comparison to DraftKings on same 13 games:**
- DK picked 9 winners (69.2%), we picked 4 (30.8%)
- If we'd bet DK favorites at same stakes: +$0.30 (vs our -$1.31)
- We were systematically betting against sharp money

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
- `season_2025.json` — 2559 games with full box scores
- `season_2026.json` — 317+ games, updated daily
- `espn_odds_2026.json` — 258 games with DraftKings closing lines
- `park_factors_2024.json` — 30 stadiums
- `kalshi_history.json` — daily market prices accumulated

---

## Daily Pipeline

```bash
python scripts/daily_pipeline.py          # today
python scripts/daily_pipeline.py 2026-04-18   # specific date
```

Runs 5 steps:
1. Fetch yesterday's completed games → update season data
2. Fetch Kalshi market prices (ML, totals, spreads, HR)
3. Fetch confirmed lineups from MLB Stats API
4. Run simulations (box score, 5000 sims per game)
5. Compare model vs market, identify edges, output report

Output saved to `.context/pipeline_YYYY_MM_DD.json`.

### Paper trading

```bash
python scripts/paper_trade.py log 2026-04-18    # log today's bets from pipeline
python scripts/paper_trade.py settle 2026-04-18 # settle after games finish
python scripts/paper_trade.py report            # full P&L history
```

Starts with $20 bankroll. Uses eighth-Kelly sizing with 3% max per bet.

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
bandung/
├── config.py              # All constants (bet selection, simulation, regression)
├── cli.py                 # Click CLI
├── sim/                   # Simulation engines
│   ├── game.py           # Box score model
│   ├── detailed_game.py  # Pitch-by-pitch with TTO
│   ├── plate_appearance.py  # Count progression
│   ├── inning.py         # Half-inning simulation
│   ├── baserunners.py    # Base state machine
│   └── ensemble.py       # Blending + market prior
├── models/                # Player and environment models
│   ├── batter_model.py
│   ├── pitcher_model.py
│   ├── team_model.py
│   ├── weather.py
│   ├── schedule_features.py
│   └── calibration.py
├── data/                  # Data fetching
│   ├── fetch.py          # MLB Stats API wrappers
│   ├── kalshi.py         # Kalshi API client
│   ├── lineup_fetcher.py # Real lineup + player model builder
│   └── load.py           # PostgreSQL loaders (optional)
├── betting/               # Bet evaluation
│   ├── ev.py             # EV + Kelly + vig removal
│   ├── predictions.py    # BettingPrediction dataclass
│   └── confidence.py     # Engine agreement scoring
├── backtest/              # Evaluation framework
│   ├── json_backtest.py  # JSON-backed backtest (no DB needed)
│   ├── model_eval.py     # Composite scoring
│   └── evaluate.py       # PostgreSQL-backed (optional)
└── scripts/
    ├── daily_pipeline.py      # Full daily workflow
    ├── daily_picks.py         # Quick daily scanner
    ├── paper_trade.py         # Paper trading tracker
    ├── fetch_daily.py         # Pull yesterday's games
    ├── fetch_season.py        # Pull full season
    ├── fetch_kalshi_history.py  # Historical Kalshi events
    ├── fetch_espn_odds.py     # ESPN DraftKings closing lines
    └── train_ensemble.py      # Grid search weights
```

---

## What's Next

1. **Accumulate 20+ paper bets** with the tightened thresholds to see if we close the gap to DK closing-line performance
2. **Grid-search market weight** once we have 50+ paired (prediction, outcome, market price) samples
3. **Add totals to paper trading** — O/U is our strongest signal (64% backtest accuracy), but we only trade ML currently
4. **Scheduled remote agent** — set up automated daily fetch + pipeline via GitHub Actions or Anthropic cloud trigger
5. **Live betting** — only after 50+ paper bets show consistent profit vs DK benchmark

---

## Honest Assessment

**What's working:**
- Simulation engine is calibrated (runs ~4.4 per team, 52% home win rate matches MLB averages)
- Backtest beats DraftKings closing lines (54.5% vs 52.7% accuracy)
- Pipeline is end-to-end functional: data → simulate → compare vs market → recommend
- O/U and run-line accuracy both 60%+ in backtest
- Paper trading framework prevents catastrophic real-money losses while validating

**What's weak:**
- Only 13 real paper bets — sample size too small to distinguish skill from luck
- Market weight (0.65) is a guess; needs grid search on 50+ games
- Not yet trading O/U markets despite them being our strongest signal
- No automation — pipeline has to be run manually each day
- Early season: only 2-3 weeks of 2026 data, so team/pitcher models are high-variance

**Expected realistic performance:**
- Best case: 53-55% ML accuracy (beats DK closing line), 2-5% ROI on O/U at Kalshi
- Likely case: Break even or slightly profitable once past the rookie-season-mistakes phase
- Worst case: Small losses that we notice quickly, adjust, and keep iterating
