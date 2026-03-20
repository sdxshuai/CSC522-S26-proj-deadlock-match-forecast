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

| Feature Group | Description |
|---|---|
| Hero picks | One-hot encoding of each team's hero lineup (6 heroes total in 3v3) |
| Player hero win rate | Each player's historical win rate on their selected hero |
| Hero matchup scores | Aggregate win rates of your heroes vs. enemy heroes |
| Hero synergy scores | Win rates of hero combinations within the same team |
| Rank / Badge | Each player's current rank tier |
| MMR differential | Average MMR difference between the two teams |
| Meta win rate | Global win rate of each hero in the current patch |

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
# 1. Fetch match metadata (adjust --limit as needed)
uv run python data/fetch_matches.py --limit 10000

# 2. Fetch global hero statistics
uv run python data/fetch_hero_stats.py

# 3. Fetch per-player stats for all account IDs found in raw matches
uv run python data/fetch_player_stats.py
```

Raw responses are saved to `data/raw/`. Processed datasets are saved to `data/processed/`.

## Feature Engineering & Training

Run the notebooks in order:

1. `notebooks/01_eda.ipynb` — Explore raw data distributions
2. `notebooks/02_feature_engineering.ipynb` — Build feature matrix, analyze importance
3. `notebooks/03_model_comparison.ipynb` — Train LR / RF / XGBoost / SVM / MLP, compare results, run ablation study

## Results

Evaluation metrics (Accuracy, AUC-ROC, F1, Brier Score) and plots are saved to `results/`.

## API Reference

- Match metadata: `GET https://api.deadlock-api.com/v1/matches/metadata`
- Hero stats: `GET https://api.deadlock-api.com/v1/analytics/hero-stats`
- Heroes: `GET https://assets.deadlock-api.com/v2/heroes`
