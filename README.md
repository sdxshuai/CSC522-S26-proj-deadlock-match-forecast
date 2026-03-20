# Deadlock Match Forecast

CSC 522 (Spring 2026) course project. Predicts match outcomes in Valve's **Deadlock** using pre-match player and hero features collected from the public [deadlock-api.com](https://deadlock-api.com) API.

## Project Structure

```
.
├── data/
│   ├── fetch_matches.py        # Fetch historical match metadata → data/raw/
│   ├── fetch_hero_stats.py     # Fetch global hero win/pick rates → data/raw/
│   ├── fetch_player_stats.py   # Fetch player rank, MMR, hero win rates → data/raw/
│   ├── raw/                    # Raw JSON responses (git-ignored)
│   └── processed/              # Cleaned CSV/Parquet datasets (git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Feature construction & importance
│   └── 03_model_comparison.ipynb     # Multi-model training & ablation study
├── src/
│   ├── preprocess.py           # Data cleaning + pre-match feature vector construction
│   ├── evaluate.py             # Metrics (Accuracy, AUC-ROC, F1, Brier) + ablation
│   └── models/
│       ├── __init__.py
│       ├── base.py             # Shared fit/predict/evaluate interface
│       ├── logistic_regression.py
│       ├── random_forest.py
│       ├── gradient_boost.py   # XGBoost / LightGBM
│       ├── svm.py
│       └── mlp.py              # PyTorch MLP
├── results/
│   ├── plots/                  # ROC curves, feature importance charts, etc.
│   └── metrics/                # Per-model metric tables (CSV)
├── .gitignore
├── requirements.txt
└── README.md
```

## Pre-match Features

| Feature Group | Features | Source |
|---|---|---|
| **Team composition** | One-hot hero picks for each team (6 heroes × 2 teams) | Match detail |
| **Badge / Rank** | `avg_badge_team0`, `avg_badge_team1`, badge difference | Bulk metadata |
| **Player rank at match time** | `player_division`, `player_rank` per player (via `max_match_id` MMR query) | `/v1/players/mmr` |
| **Player historical stats** | Win rate, KDA, `kills_per_min`, `damage_per_min`, `last_hits_per_min` on selected hero | `/v1/players/hero-stats` |
| **Hero meta stats** | Global win rate, pick rate, KDA for each hero | `/v1/analytics/hero-stats` |
| **Hero counter advantage** | Aggregate win rate of team's heroes vs. opposing team's heroes | `/v1/analytics/hero-counter-stats` |
| **Hero synergy score** | Average win rate of hero pairs within each team | `/v1/analytics/hero-synergy-stats` |
| **Match context** | `match_mode` (Ranked/Unranked), `game_mode` | Bulk metadata |

**Target label:** `1` if Team0 wins, `0` if Team1 wins.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
# Create virtual environment and install dependencies
uv sync

# Or install from requirements.txt directly
uv pip install -r requirements.txt

# Run any script within the managed environment
uv run python data/fetch_matches.py --limit 10000
```

## Data Collection

All scripts use the [deadlock-api.com](https://deadlock-api.com) free public API (no API key required).

```bash
# 1. Fetch global hero analytics (one-time, < 1 minute)
uv run python data/fetch_hero_stats.py

# 2. Fetch match data in two phases:
#    Phase 1: bulk metadata list (fast, ~10 req at 4 req/s for 10k matches)
#    Phase 2: individual match detail with player info (10 req/s from cache)
uv run python data/fetch_matches.py --limit 10000

#    Run phases separately if needed:
uv run python data/fetch_matches.py --phase 1 --limit 10000   # list only
uv run python data/fetch_matches.py --phase 2 --rate 10       # detail only

# 3. Fetch per-player stats for all accounts found in match files
uv run python data/fetch_player_stats.py --incremental

# 4. Check data coverage
uv run python data/validate.py
```

Raw responses are saved to `data/raw/`. Processed datasets are saved to `data/processed/`.

See [data/README.md](data/README.md) for full schema documentation and data flow.

## Feature Engineering & Training

Run the notebooks in order:

1. `notebooks/01_eda.ipynb` — Explore raw data distributions
2. `notebooks/02_feature_engineering.ipynb` — Build feature matrix, analyze importance
3. `notebooks/03_model_comparison.ipynb` — Train LR / RF / XGBoost / SVM / MLP, compare results, run ablation study

## Results

Evaluation metrics (Accuracy, AUC-ROC, F1, Brier Score) and plots are saved to `results/`.

## API Reference

| Endpoint | Rate | Notes |
|---|---|---|
| `GET /v1/matches/metadata` | 4 req/s | Bulk, ≤1000 matches, supports `include_player_info` |
| `GET /v1/matches/{id}/metadata` | 100 req/s (cache) | Full match JSON with players |
| `GET /v1/analytics/hero-stats` | 100 req/s | Global hero win rates |
| `GET /v1/analytics/hero-counter-stats` | 100 req/s | Hero vs hero matchup matrix |
| `GET /v1/analytics/hero-synergy-stats` | 100 req/s | Hero pair synergy |
| `GET /v1/players/hero-stats` | 100 req/s | Per-player stats, batch ≤1000 |
| `GET /v1/players/mmr` | 100 req/s | Player MMR, supports `max_match_id` |

Base URL: `https://api.deadlock-api.com` · No authentication required
