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

Huggingface Address of the dataset we created: [https://huggingface.co/datasets/sdxshuai/deadlock-match-forecast]

## Known Data Risks & Limitations

### 1. Player data coverage is not guaranteed (high impact)

`fetch_player_stats.py` collects stats for all `account_id`s found in match files, but **full coverage cannot be guaranteed** due to three structural gaps:

| Gap | Cause | Effect |
|---|---|---|
| `account_id = 0` or `null` | Anonymous / private accounts in match JSON | Player silently excluded from stats |
| API returns no record | Account has no game history on that hero, or is brand new | Hero-stats record absent for that player |
| No MMR history | Player has never ranked up, or account is private | MMR lookup returns empty for that account |

**Mitigation in preprocessing:** treat missing player stats as `NaN` and apply imputation (e.g. global median for numerical features, zero for win-rate-based features). Track per-match missingness rate; drop matches where > N players lack stats if coverage is too sparse.

Use `validate.py` to check coverage before preprocessing:
```bash
uv run python data/validate.py
```
The report prints `hero_stats_ids ∩ match_accounts` and `mmr_ids ∩ match_accounts` coverage percentages.

---

### 2. MMR pre-match snapshot temporal accuracy (medium impact)

The MMR endpoint is called **without** `max_match_id` (fetching full history instead of a per-match snapshot), because that would require one request per `(account_id, match_id)` pair.

In preprocessing, the nearest MMR record before each match's `start_time` is used. If a player has not played ranked matches for an extended period, the "nearest" record may be stale.

**Mitigation:** apply a time-window cap (e.g. 30 days) when looking up historical MMR; mark as missing if no record exists within the window.

---

### 3. Hero global stats aggregated across all patches (medium impact)

`fetch_hero_stats.py` uses `bucket=no_bucket`, which aggregates win/pick rates over the **entire history** of the game. Deadlock patches regularly alter hero balance; mixing stats from different patches introduces noise.

**Mitigation:** explore the `bucket` parameter (monthly buckets) to restrict stats to a recent time window, or at minimum verify in EDA that global win-rate correlates with recent match outcomes.

---

### 4. Unstable match JSON schema (low impact, already handled)

Individual match files (`/v1/matches/{id}/metadata`) may nest the `players` array under different keys (`players`, `match_info.players`, `player_info`, etc.) depending on the API version. Both `fetch_player_stats.py` and `validate.py` already implement a multi-path fallback.

**Mitigation:** ensure `src/preprocess.py` uses the same fallback logic. Log which fallback path was hit per file to detect future schema drift.

---

### 5. Default sample target of 10 000 may be insufficient (high impact)

The one-hot hero encoding alone produces ~66 sparse dimensions (33 heroes × 2 teams). With per-player stats and rank features, the full feature vector is wide relative to 10 000 samples.

**Recommendation:** collect at least **50 000 matches**. Phase 1 completes in ~25 s at 4 req/s; Phase 2 takes ~1.4 hours at 10 req/s.

---

### 6. `average_badge` sparsity (low impact)

Early matches in the dataset return `null` for `average_badge_team0/1`. If match-level badge features are missing for a large fraction of the dataset, the badge-difference feature will need heavy imputation or should be dropped.

**Mitigation:** check badge null-rate in `validate.py` output; fall back to per-player MMR-derived rank if match-level badge is unavailable.

---

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
