# Project Proposal Summary — Deadlock Match Forecast

## Team Members

| Name | Unity ID |
|------|----------|
| Xuesi Zhou | xzhou38 |
| Adam Myers | ajmyers5 |
| Joe Strickland | jpstric2 |
| Zhi Zhang | zzhan224 |

---

## 1. Problem Description

### Abstract

Statistics in gaming have become very prevalent and websites that give in-depth overviews of player information have become very popular. One such game, Deadlock, has an enormous amount of player data and an easily accessible API to retrieve player information. Specifically, our goal is to utilize the mass amount of retrievable player information to determine a prediction on a match's outcome (Win or Lose). This is not only something that players would be interested in seeing when they play a game, but for competitive e-sports, knowing a good prediction on games could give a gambling advantage. To solve this, we would like to build a pipeline that utilizes the information of players in a match (win-rate, rank, average gold, etc.), and can predict the match outcome target attribute.

### Dataset

- **Location**: https://huggingface.co/datasets/sdxshuai/deadlock-match-forecast
- **Construction**: Built by our team via the Deadlock player/match API. No dataset combining required.

### Prediction Task

> Given all 12 players and their attributes, predict whether the match will result in a win or lose for the first 6 players. (Target column: `match_outcome`)

**Problem type**: Binary Classification

### Research Questions

**RQ1**: Can pre-match player statistics reliably predict match outcomes in Deadlock?
- Core problem; evaluated by comparing multiple models on primary metric (ROC-AUC).

**RQ2**: Does preprocessing have a meaningful impact on prediction outcomes?
- Evaluate whether scaling, discretization, log transforms, etc. meaningfully affect performance.

**RQ3**: What player statistics are the most influential predictors of match outcomes?
- Feature importance analysis to determine which statistics (rank, win rate, economic stats, etc.) drive predictions.

### Novel Aspects

1. **Novel Dataset Collection via API Pipeline** — Unlike pre-cleaned guided project datasets, the team directly queries the Deadlock player/match API, building a data pipeline that handles retrieval, inconsistencies, missing entries, and transformation.

2. **Feature Engineering for Teams** — The dataset must be organized for teams, requiring engineering of new Deadlock-specific features, including: team balance score (averaged rank/winrate), experience gap metric (averaged matches played per side), and role-weighted statistics (e.g., kills matter less for support players). This adds complexity beyond any in-class dataset.

---

## 2. Data

### 2.1 Feature Groups

The data is fully tabular (rows and columns). Each instance is **one row per match** (one prediction per row). Feature groups include:

- **Per-player hero performance features (×12 players)**: Stats on the hero being played — wins, matches played, kills/deaths/assists per minute, etc.
- **Per-player MMR / Rank features (×12 players)**: Rank snapshot before the game — MMR rank, division, player rating.
- **Per-player global hero features (×12 players)**: Global hero win rate and total matches (overall hero strength).
- **Team-level aggregations (×2 teams)**: Aggregated win rate, skill levels, kills per minute, etc.

### 2.2 Data Validation

**Inconsistent data types to fix:**
- `Average_badge_team0` / `Average_badge_team1`: nullable integer (null → parsed as float64)
- NaN entries for players with missing stats or private profiles (parsed as float64)
- `start_time`: stored as string `"2024-05-20 00:04:05"` → convert to datetime
- `hero_id` and `assigned_lane`: stored as integers but are **categorical** → convert to one-hot
- `winning_team`: stored as string `"Team0"` → encode as binary (0 = Team0 wins, 1 = Team1 wins); no additional encoding needed after conversion

**Data leakage considerations:**
- All data from the actual match being predicted is removed to prevent leakage
- All player features are derived from historical statistics prior to the match
- Temporal consistency strictly enforced; no post-outcome or future information included
- Global hero statistics aggregated over all matches may implicitly include the target match — this limitation cannot be fully eliminated

**Outliers:**
- Difficult to define; players with high playtime have naturally elevated stats (e.g., matches played)
- Extreme values are kept because they reflect genuine player experience and may be useful for prediction
- Leakage concern: if a model scores above ~72% accuracy or ~0.78 AUROC, check whether player stats were filtered to only include games before each match's `start_time`

### 2.3 Data Groups / Dependencies

Three meaningful groups:
1. **Multiple matches per account** — identified by `account_id`
2. **Hero statistics per account** — joined via `account_id`
3. **Matches by average rank** — grouped by rank within the matches dataset

**Splitting strategy**: Split at the `account_id` level. All matches and hero data for a given `account_id` must be assigned entirely to one split (train/val/test). Ratio: **70/10/20**, stratified by `match_outcome`.

### 2.4 Data Splitting

- **Strategy**: Holdout test set (80% train / 20% test overall; train further split 70/10/20 with validation)
- **Justification**: Thousands of new matches are available per hour via the API; data is not a bottleneck
- **Stratification**: Yes — ensures train/test reflect the same ~50/50 win/loss proportions
- **Test set isolation**: Validation set is used for all design/hyperparameter decisions; the 20% test set is never touched until final evaluation

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Team 0 wins (label=1) | 50,538 | 50.52% |
| Team 1 wins (label=0) | 49,503 | 49.48% |

The dataset is **nearly perfectly balanced** (reflects the true real-world distribution, not a data collection artifact). No minority class exists; no class imbalance mitigation is needed. Both classes are equally important — there is no natural positive class.

### 2.5 Missing Data

- Null values in `Average_badge_team0` / `Average_badge_team1` → rows ignored
- `Not_Scored` (edge case column) → ignored
- **Numeric**: mean/median imputation not planned; null badge fields are handled by float64 parsing
- **Match_outcome**: rows where outcome is not "TeamWin" (player abandoned, draws) are dropped

---

## 3. Features: Transformation, Selection, and Engineering

### Scaling

**Z-score standardization** applied to all numeric player statistics (win rate, matches played, kills/assists per minute, team-level aggregates). Required for Logistic Regression, SVM, and k-NN. Tree-based models less sensitive, but a consistent pipeline is maintained across all models. Extreme values (outliers) will **not** be normalized/capped, as they may carry useful information about very skilled or very poor players (possibly cheaters).

### Functional Transformations

**Log transformation** (`log1p`) applied to right-skewed features:
- `matches_played`, `total_kills`, `total_wins`
- Per-match stats: `kills`, `enemy_kills`, `deaths`, `enemy_deaths`, `assists`, `enemy_assists`, `denies`, `enemy_denies`, `last_hits`, `enemy_last_hits`, `networth`, `enemy_networth`, `obj_damage`, `enemy_obj_damage`, `creeps`, `enemy_creeps`

Applied before z-score standardization.

**Ratio transformation**: KDA = (Kills + Assists) / Deaths — captures player impact more meaningfully than raw kill/death counts.

**Binary transformation**: Flag players as "high performers" if KDA or kills exceed a threshold — may indicate unusually skilled or cheating players.

### Feature Selection

Compare:
- No feature selection (baseline)
- Mutual Information (MI): top-k features by statistical dependence with target; k ∈ {15, 20, 30} selected via validation F1

Feature matrix estimated at 100+ columns (12 players × multiple stat groups). Some redundancy expected; selection may reduce noise for Logistic Regression and k-NN.

### Feature Engineering

| Feature | Source Columns | Logic | Motivation |
|---------|---------------|-------|------------|
| `team_skill_gap` | per-player MMR | mean(Team0 MMR) − mean(Team1 MMR) | Captures team-level skill imbalance |
| `experience_gap` | per-player `matches_played` | mean(Team0) − mean(Team1) | More experienced players perform more consistently |

**Features to drop** (non-predictive):
`match_id`, `game_mode`, `start_time`, `duration_s`, `match_outcome`, `not_scored`

---

## 4. Preprocessing Pipeline

```
Filter rows (keep only TeamWin outcomes)
→ Drop non-predictive columns (match_id, start_time, duration_s, not_scored)
→ Encode categorical via One-Hot:
    hero_id, assigned_lane → One-Hot
    winning_team → One-Hot → [Win_Team0, Win_Team1]
    match_mode → One-Hot → [MatchRanked, MatchUnranked, MatchCoopBot]
    game_mode → One-Hot → [All options]
→ Encode target: winning_team → binary (0 = Team0 wins, 1 = Team1 wins)
→ Drop remaining match-level leakage features
→ Feature Engineering: team_skill_gap, experience_gap, KDA
→ Log Transformation (log1p): skewed numeric features
→ Imputation: float64 for null badge/stat fields; drop abandonment rows
→ Z-score Standardization (all numeric)
→ Feature Selection (MI, k ∈ {15, 20, 30}) — selected via experiment
→ Discretization / Binning (kills, deaths, assists, denies, networth, damage → high/medium/low)
→ Sampling: Not applied — dataset is large (thousands of new matches per hour) and classes are balanced (~50/50); oversampling/undersampling/SMOTE not beneficial
```

### Preprocessing Experiments

1. **Log transformation (include vs. exclude)**: Evaluated on Logistic Regression with validation AUC-ROC
2. **Feature selection MI vs. none**: Compare k ∈ {15, 20, 30} vs. no selection using Logistic Regression and Random Forest on the validation set

---

## 5. Models

### Baselines

| Model | Role |
|-------|------|
| Majority-class predictor (50.52%) | Minimum bar; ROC-AUC = 0.50 |
| Logistic Regression (full features, no tuning) | Strong linear baseline; comparable to Semenov et al. (2016) |
| XGBoost (default hyperparameters) | Strong nonlinear baseline before tuning |

### Main Models

| Model | Justification |
|-------|--------------|
| **Logistic Regression** | Interpretable linear baseline; directly comparable to prior Dota 2 work (Semenov et al.) |
| **Random Forest** | Captures nonlinear interactions between hero stats and MMR; provides feature importance |
| **XGBoost / LightGBM** | Primary model; state-of-the-art on tabular data; run with and without tuning to isolate tuning contribution |

---

## 6. Hyperparameter Tuning

### Hyperparameters to Tune

| Model | Hyperparameter | Justification |
|-------|---------------|---------------|
| Logistic Regression | `C` | Most impactful HP; controls regularization on 100+ feature matrix |
| Random Forest / XGBoost | `max_depth` | Primary lever for bias-variance tradeoff on player stat data |
| XGBoost | `learning_rate` | Must be tuned jointly with `n_estimators`; largest impact on final XGBoost performance |

### Initial Search Grid

| Model | Parameter | Type | Search Space | Values |
|-------|-----------|------|-------------|--------|
| Logistic Regression | `C` | continuous | log-uniform | {0.01, 0.1, 1, 10, 100} |
| Random Forest / XGBoost | `max_depth` | discrete | grid | {3, 5, 10, None} |
| XGBoost | `learning_rate` | continuous | log-uniform | {0.01, 0.1, 0.3} |
| XGBoost | `n_estimators` | — | fixed | 200 |

### Tuning Strategy

- **Method**: Random search (20–30 iterations) — search space too large for grid search; better suited for continuous parameters
- **Evaluation**: 5-fold cross-validation on the training split (not test set); balances stability and computation
- **Complexity estimate**: XGBoost: 4 × 3 = 12 grid combos; with random search (30 iter) × 5-fold = 150 runs; with feature selection (k ∈ 3 options) = up to 450 runs total
- **Reduction**: Fix `n_estimators=200` upfront; coarse sweep first with `learning_rate ∈ {0.01, 0.3}` and `max_depth ∈ {3, 10}`, then refine
- **Fallback**: Average per-team player statistics to reduce feature count if computation is prohibitive

---

## 7. Evaluation Metrics

### Primary Metric: ROC-AUC

Binary classification with no natural positive class; model ranks matches by win probability. Classes are nearly perfectly balanced (50.52% vs 49.48%). Random guess = 0.50, perfect = 1.00.

### Additional Metrics

- **Accuracy** — reliable with balanced classes; naive baseline = 50.52%
- **Macro Precision** — how often a predicted winner is correct
- **Macro Recall** — how often the actual winner is identified
- **Macro F1** — balance of precision and recall; reveals per-class bias

### Performance Thresholds

| Threshold | Meaning |
|-----------|---------|
| Accuracy ≤ 50.52%, ROC-AUC ≤ 0.50 | No better than random guessing |
| Accuracy ≈ 60%, ROC-AUC ≈ 0.65 | Acceptable minimum (based on comparable Dota 2 work) |
| Accuracy > 72%, ROC-AUC > 0.78 | Suspicious — check for data leakage in player stat aggregation |

---

## 8. Algorithmic Bias

Not applicable — this dataset does not include human demographic attributes or protected groups. The subjects are game player accounts, not individuals characterized by protected characteristics.

---

## 9. Related Work

1. **Gu et al., "NeuralAC: Learning Cooperation and Competition Effects for Match Outcome Prediction" (AAAI 2021)** — Uses attention-based neural networks to model intra-team cooperation and inter-team competition for match outcome prediction, outperforming individual player-level baselines. Directly motivates our team-level feature engineering approach.

2. **Yang et al., "Identifying Patterns in Combat that are Predictive of Success in MOBA Games" (FDG 2014)** — Studies League of Legends; explains how team composition, hero synergies, and player skill together determine match outcomes in MOBA games. Provides the domain background for why our features (MMR, hero win rate, team aggregations) are meaningful.

3. **Semenov et al. (2016)** — Used logistic regression on Dota 2 draft features; our Logistic Regression model is directly comparable to this work. Referenced as a performance benchmark.

---

## 10. Execution Plan

### Team Roles

| Person | Role |
|--------|------|
| Person 1 (EDA & Report) | Explore dataset, analyze feature distributions, missing values, and correlations; write data description and results sections |
| Person 2 (Baseline Models) | Implement logistic regression and tree-based baselines |
| Person 3 (Main Model) | Train gradient boosting / random forest with train/val/test split and hyperparameter tuning |
| Person 4 (Feature Engineering) | Design team-level MMR difference, hero synergy/counter scores from raw data files |

### Midterm Milestone

By the midterm, we will have:
- Completed EDA and written the data description section
- Added new team-level features (MMR difference, hero synergy scores)
- Trained and evaluated baseline models (Logistic Regression, tree-based model) on fixed train/val/test split
- Produced a comparison table of AUROC and accuracy across models
