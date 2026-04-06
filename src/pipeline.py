"""Shared preprocessing pipeline for deadlock-match-forecast."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DROP_LEAKAGE = ['winning_team', 'duration_s']
DROP_ID = (['match_id', 'start_time']
           + [f't{t}_p{p}_account_id' for t in range(2) for p in range(6)]
           + [f't{t}_p{p}_player_slot' for t in range(2) for p in range(6)])
DROP_ZERO_VAR = ['game_mode']
DROP_REDUNDANT = [
    'average_badge_team0', 'average_badge_team1',
] + [f't{t}_p{p}_{f}' for t in range(2) for p in range(6) 
     for f in ['wins', 'kills', 'deaths', 'assists', 'time_played']]
DROP_TEMPORAL = [f't{t}_p{p}_last_played' for t in range(2) for p in range(6)]
DROP_FIXED = [f't{t}_p{p}_assigned_lane' for t in range(2) for p in range(6)]

ALL_DROP = DROP_LEAKAGE + DROP_ID + DROP_ZERO_VAR + DROP_REDUNDANT + DROP_TEMPORAL + DROP_FIXED


def load_and_clean(parquet_path):
    """Load raw parquet → drop invalid rows + non-feature columns → return df_clean, y."""
    df = pd.read_parquet(parquet_path)
    hero_cols = [f't{t}_p{p}_hero_id' for t in range(2) for p in range(6)]
    df = df[~df[hero_cols].isna().any(axis=1)].reset_index(drop=True)
    y = df['label'].copy()
    df_clean = df.drop(columns=ALL_DROP + ['label'], errors='ignore')
    return df_clean, y


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['team_skill_gap'] = X['t0_avg_mmr_rank'] - X['t1_avg_mmr_rank']
        X['experience_gap'] = X['t0_avg_matches_played'] - X['t1_avg_matches_played']
        return X


class HeroMultiHot(BaseEstimator, TransformerMixin):
    """12 hero_id columns → 2×N multi-hot vectors, drop originals."""

    def fit(self, X, y=None):
        hero_cols = [f't{t}_p{p}_hero_id' for t in range(2) for p in range(6)]
        self.hero_domain_ = sorted(X[hero_cols].stack().dropna().unique().astype(int))
        self.hero_to_idx_ = {h: i for i, h in enumerate(self.hero_domain_)}
        return self

    def transform(self, X):
        X = X.copy()
        hero_cols = [f't{t}_p{p}_hero_id' for t in range(2) for p in range(6)]
        for t in range(2):
            arr = np.zeros((len(X), len(self.hero_domain_)), dtype=np.int8)
            for p in range(6):
                for row_i, h in enumerate(X[f't{t}_p{p}_hero_id'].values):
                    if not np.isnan(h) and int(h) in self.hero_to_idx_:
                        arr[row_i, self.hero_to_idx_[int(h)]] = 1
            for i, h in enumerate(self.hero_domain_):
                X[f't{t}_hero_{h}'] = arr[:, i]
        X.drop(columns=hero_cols, inplace=True)
        return X

    def get_hero_feat_cols(self):
        return [f't{t}_hero_{h}' for t in range(2) for h in self.hero_domain_]


class MedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.medians_ = X.median()
        return self

    def transform(self, X):
        return X.fillna(self.medians_)


class Log1pSkewed(BaseEstimator, TransformerMixin):
    cumulative_fields = ['kills', 'deaths', 'assists', 'matches_played', 'wins', 'time_played']

    def fit(self, X, y=None):
        hero_cols = {c for c in X.columns if c.startswith('t0_hero_') or c.startswith('t1_hero_')}
        self.log_cols_ = sorted({
            c for c in X.columns
            if (any(c.endswith(f) or f'_{f}_' in c for f in self.cumulative_fields)
                or 'avg_matches_played' in c)
            and c not in hero_cols
        } & set(X.columns))
        return self

    def transform(self, X):
        X = X.copy()
        X[self.log_cols_] = np.log1p(X[self.log_cols_].clip(lower=0))
        return X


class ScaleContinuous(BaseEstimator, TransformerMixin):
    """Z-score continuous features, skip hero binary."""

    def fit(self, X, y=None):
        hero_cols = {c for c in X.columns if c.startswith('t0_hero_') or c.startswith('t1_hero_')}
        self.num_cols_ = [c for c in X.select_dtypes(include='number').columns if c not in hero_cols]
        self.scaler_ = StandardScaler().fit(X[self.num_cols_])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.num_cols_] = self.scaler_.transform(X[self.num_cols_])
        return X


def build_pipeline():
    return Pipeline([
        ('feat_eng', FeatureEngineer()),
        ('hero_ohe', HeroMultiHot()),
        ('impute', MedianImputer()),
        ('log1p', Log1pSkewed()),
        ('scale', ScaleContinuous()),
    ])
