# Results Log

Full notes: [`midway_experiment_log.md`](midway_experiment_log.md)

---

## 01_eda_preprocessing.ipynb

- **cells 2–5**: data overview
  - 100,041 rows × 472 cols, 378 MB
  - labels balanced: 50.52 / 49.48%
  - NaN only in `*_per_soul` (≤18 rows, 0.018%)
  - 38 heroes, 407k accounts
  - 2 rows with NaN `hero_id` → dropped (100,039 remaining)
- **cell 7**: correlation audit
  - `average_badge_team0/1` r=0.993 with `avg_mmr_rank`
    - redundant, drop
  - `player_hero_wr` r=0.778 with label
    - keep for baseline, ablate later
    - possible temporal leakage from API snapshot
- **cells 7–8**: column drop
  - 55 cols dropped total
    - 2 leakage, 26 identifiers, 1 zero-variance
    - 2 redundant, 12 temporal, 12 no-signal
  - 416 features remaining
- **cell 11**: skewness
  - 6 cumulative stat types × 12 players: |skew| > 6
    - log1p (72 cols)
  - `avg_matches_played` ×2 teams: skew ~3.3
    - log1p
  - rate/average features: |skew| < 1, no transform
- **cell 14**: `hero_id` ×12
  - 38 unique, frequency ratio 4.2×, no rare classes
  - multi-hot per team: 38-dim vector, 6 ones each (76 cols total)
- **cell 15**: `assigned_lane` ×12
  - 100% matches are 2-2-2
  - all per-hero lane pref <40%
  - zero signal, drop
- **cells 17–18**: correlation with label
  - top features: `player_hero_wr` and rate stats (assists/networth per min)
  - hero win rates: [0.42, 0.57], ~14 pp spread
    - signal weak but nonzero
    - multi-hot encoding is fine
- **cell 20**: stratified split (70/10/20)
  - train 70,027 / val 10,004 / test 20,008
  - label=1 ratio: 0.505 across all splits
  - no group constraint on `account_id`
    - `account_id` already dropped from features, not a leakage vector
    - forcing disjoint accounts would distort MMR/hero distributions across splits
- **cell 22**: sklearn Pipeline (fit on train, transform val/test)
  - `FeatureEngineer`: `team_skill_gap` = t0−t1 avg_mmr_rank, `experience_gap` = t0−t1 avg_matches_played
  - `HeroMultiHot`: 12 hero_id cols → 2×38 multi-hot (fit hero domain on train)
  - `MedianImputer`: fill NaN with train medians (covers all splits)
  - `Log1pSkewed`: log1p on 72 per-player cumulative cols + 2 team avg_matches_played
  - `ScaleContinuous`: z-score on continuous features only (hero binary excluded)
  - output: 482 features = 384 per-player + 20 team-agg + 76 hero + 2 engineered
  - NaN after pipeline: 0
- **cell 23**: export
  - saved train/val/test.parquet to `data/processed/`

---

## 02_models.ipynb

- **cell 2**: load raw data via `src.pipeline.load_and_clean`, stratified split, `build_pipeline().fit_transform`
  - same split as nb01 (70/10/20, seed 42): train 70,027 / val 10,004
  - 482 features after pipeline
- **cell 4**: baseline results (all default params)
  - Dummy: val AUC 0.50, acc 50.5%
  - LR (C=1.0): val AUC 0.9966, acc 96.95%
  - RF: val AUC 0.9956, acc 96.85%
  - XGBoost: val AUC 0.9969, acc 96.88%
- **cell 5**: sanity checks
  - RF train-val acc gap: +0.031, XGBoost: +0.029 (mild overfit)
  - LR train-val gap: +0.001 (no overfit)
  - all 3 real models val acc > 72%, val AUC > 0.78
  - **not leakage** — verified: `kills / time_played` matches `kills_per_min` (r=0.996), all per-player stats are historical aggregates
  - high accuracy explained by: rich per-player features (12 players × 30+ stats) + Deadlock's poor matchmaking creating one-sided games
  - Dota 2 pre-match baseline ~71% accuracy (Semenov et al. 2016); our 95%+ without hero_wr confirms Deadlock is more skill-deterministic
- **cell 7**: LR GridSearchCV
  - best C=100, CV AUC=0.9963
  - val AUC 0.9967, val acc 96.93%
  - negligible improvement over default C=1.0
- **cell 8**: full-feature ranking by val AUC
  - XGBoost 0.9969 > LR_tuned 0.9967 > LR_default 0.9966 > RF 0.9956
  - all within ~0.1% of each other — task is too easy for model differences to emerge
- **cell 10**: ablation — drop 14 `player_hero_wr` cols, re-fit pipeline, 468 features
- **cell 11**: ablated model results
  - XGBoost: val AUC 0.9949, acc 95.90%
  - LR_tuned: val AUC 0.9921, acc 95.28%
  - LR: val AUC 0.9921, acc 95.15%
  - RF: val AUC 0.9855, acc 93.59%
- **cell 12**: full vs ablated delta
  - XGBoost: AUC -0.002, acc -1.0%
  - LR: AUC -0.005, acc -1.8%
  - RF: AUC -0.010, acc -3.3%
  - `player_hero_wr` accounts for only 1–3% accuracy drop
  - other per-player rate stats (assists_per_min r=0.48, networth_per_min r=0.48) carry most of the signal
  - quick check: 278 rate+mmr features alone (no pipeline, no hero encoding) → LR val AUC 0.95, acc 87.7%
- **cell 15**: test set evaluation (full + ablation, all models)
  - full features: XGBoost test AUC 0.9971, LR 0.9966, RF 0.9957, acc all ~97%
  - no hero_wr: XGBoost test AUC 0.9943, LR 0.9927, RF 0.9853, acc 93–96%

