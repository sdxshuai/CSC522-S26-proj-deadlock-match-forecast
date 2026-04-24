# Leakage Analysis

## Procedure
- Compare three feature sets on the same processed dataset: `full`, `feature_analysis`, `minimal`.
- Use two validation splits: `random` (stratified 70/10/20) and `time` (chronological 70/10/20).
- Use Logistic Regression and XGBoost as the reference models.
- On `feature_analysis`, run leave-one-group-out ablation on the transformed matrix with the random split.

## Results

Main ROC-AUC:

| feature set | split | LR | XGBoost |
|---|---:|---:|---:|
| full | random | 0.9966 | 0.9969 |
| full | time | 0.9963 | 0.9968 |
| feature_analysis | random | 0.9774 | 0.9731 |
| feature_analysis | time | 0.9791 | 0.9745 |
| minimal | random | 0.7207 | 0.7100 |
| minimal | time | 0.7192 | 0.7094 |

- `full` is nearly perfect on both splits.
- `feature_analysis` is lower than `full`, but still much higher than `minimal`.
- `minimal` stays around 0.72 AUC on both splits.
- Changing from random split to time split does not remove the inflated scores.

Feature-analysis ablation, delta AUC after dropping one group:

| group | LR | XGBoost |
|---|---:|---:|
| per_minute_rates | -0.2056 | -0.2171 |
| player_hero_wr | -0.0108 | -0.0079 |
| cumulative_counts | -0.0064 | -0.0093 |
| hero_multihot | -0.0037 | -0.0010 |
| mmr | -0.0013 | -0.0018 |
| other groups | about 0 | about 0 |

- The main remaining signal in `feature_analysis` is `per_minute_rates`.
- `player_hero_wr` and `cumulative_counts` still add extra signal.
- The other groups have little effect.

## Summary
- `feature_analysis` is not leakage-safe.
- The strongest remaining leakage-prone groups are `per_minute_rates`, `player_hero_wr`, and `cumulative_counts`.
- `minimal` is the clean pre-match baseline in the current pipeline.
