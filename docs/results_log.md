# Results Log

Full notes: [`midway_experiment_log.md`](midway_experiment_log.md)

---

## 01_eda_preprocessing.ipynb

- **cells 2–5**: data overview
  - 100,041 rows × 472 cols, 378 MB
  - labels balanced: 50.52 / 49.48%
  - NaN only in `*_per_soul` (≤18 rows, 0.018%)
  - 38 heroes, 407k accounts
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
  - one-hot (38×12 = 456 binary cols)
- **cell 15**: `assigned_lane` ×12
  - 100% matches are 2-2-2
  - all per-hero lane pref <40%
  - zero signal, drop
- **cells 17–18**: correlation with label
  - top features: `player_hero_wr` and rate stats (assists/networth per min)
  - hero win rates: [0.42, 0.57], ~14 pp spread
    - signal weak but nonzero
    - one-hot encoding is fine

---

## 02_baseline_models.ipynb

*pending*

- Dummy: val AUC 0.50, acc ~50.5%
- LR: —
- RF: —
- XGBoost: —

---

## Hypotheses

- H1 (AUC ≥ 0.60): pending
- H2 (tree > LR by ≥ 2% AUC): pending
- H3 (team features in MI top-10): pending
- H4 (log1p improves LR by ≥ 1% AUC): pending
