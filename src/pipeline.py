"""Shared preprocessing pipeline for deadlock-match-forecast."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DROP_LEAKAGE = ['winning_team', 'duration_s']
DROP_ID = (['match_id', 'start_time']
           + [f't{t}_p{p}_account_id' for t in range(2) for p in range(6)]
           + [f't{t}_p{p}_player_slot' for t in range(2) for p in range(6)])
DROP_ZERO_VAR = ['game_mode']
DROP_REDUNDANT = ['average_badge_team0', 'average_badge_team1']
DROP_TEMPORAL = [f't{t}_p{p}_last_played' for t in range(2) for p in range(6)]
DROP_FIXED = [f't{t}_p{p}_assigned_lane' for t in range(2) for p in range(6)]

ALL_DROP = DROP_LEAKAGE + DROP_ID + DROP_ZERO_VAR + DROP_REDUNDANT + DROP_TEMPORAL + DROP_FIXED

META_COLS = ['match_id', 'start_time', 'label'] + [
    f't{t}_p{p}_account_id' for t in range(2) for p in range(6)
]

FA_DROP_FIELDS = ('wins', 'kills', 'deaths', 'assists', 'time_played')
CLEAN_DROP_GROUPS = ('per_minute_rates', 'player_hero_wr', 'cumulative_counts')


def _load_base_frame(parquet_path):
    df = pd.read_parquet(parquet_path)
    hero_cols = [f't{t}_p{p}_hero_id' for t in range(2) for p in range(6)]
    df = df[~df[hero_cols].isna().any(axis=1)].reset_index(drop=True)
    y = df['label'].copy()
    return df, y


def load_and_clean(parquet_path):
    """Load raw parquet → drop invalid rows + non-feature columns → return df_clean, y."""
    df, y = _load_base_frame(parquet_path)
    df_clean = df.drop(columns=ALL_DROP + ['label'], errors='ignore')
    return df_clean, y


def load_with_meta(parquet_path):
    df, y = _load_base_frame(parquet_path)
    df_feat = df.drop(columns=ALL_DROP + ['label'], errors='ignore')
    df_meta = df[[c for c in META_COLS if c in df.columns]].copy()
    return df_feat, y, df_meta


def random_split(n, y, test_size=0.20, val_size=0.125, seed=42):
    idx_tv, idx_test = train_test_split(
        np.arange(n), test_size=test_size, random_state=seed, stratify=y)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=val_size, random_state=seed, stratify=y.iloc[idx_tv])
    return {'train': idx_train, 'val': idx_val, 'test': idx_test}


def time_split(df_meta, train_frac=0.70, val_frac=0.10):
    ts = pd.to_datetime(df_meta['start_time'], unit='s', utc=True)
    order = ts.argsort().values
    n = len(order)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        'train': order[:n_train],
        'val': order[n_train:n_train + n_val],
        'test': order[n_train + n_val:],
    }


def _feature_analysis_frame(df_feat, y):
    df = df_feat.copy()
    for t in (0, 1):
        team_won = (y == (1 - t)).astype(int)
        wr_cols = []
        for i in range(6):
            wins_col = f't{t}_p{i}_wins'
            matches_col = f't{t}_p{i}_matches_played'
            wr_col = f't{t}_p{i}_player_hero_wr'
            df[wr_col] = ((df[wins_col] - team_won) / (df[matches_col] - 1)).replace(
                [np.inf, -np.inf], np.nan
            )
            wr_cols.append(wr_col)
        df[f't{t}_avg_player_hero_wr'] = df[wr_cols].mean(axis=1)

    drop_cols = [
        c for c in df.columns
        if 'ending_level' in c
        or ((c.startswith('t0_p') or c.startswith('t1_p'))
            and any(c.endswith('_' + field) for field in FA_DROP_FIELDS))
    ]
    return df.drop(columns=drop_cols, errors='ignore')


def _minimal_frame(df_feat):
    hero_cols = [f't{t}_p{p}_hero_id' for t in range(2) for p in range(6)]
    mmr_cols = [c for c in df_feat.columns if 'mmr_rank' in c or 'avg_mmr_rank' in c]
    matches_cols = [c for c in df_feat.columns if c.endswith('_matches_played')]
    avg_matches_cols = [c for c in df_feat.columns if 'avg_matches_played' in c]
    keep_cols = sorted(set(hero_cols + mmr_cols + matches_cols + avg_matches_cols))
    return df_feat[keep_cols].copy()


def _clean_frame(df_feat, y):
    df = _feature_analysis_frame(df_feat, y)
    groups = feature_group_registry(df.columns)
    drop_cols = sorted({
        column
        for group_name in CLEAN_DROP_GROUPS
        for column in groups[group_name]
    })
    return df.drop(columns=drop_cols, errors='ignore').copy()


def make_feature_frame(df_feat, y, feature_set='full'):
    if feature_set == 'full':
        return df_feat.copy()
    if feature_set == 'clean':
        return _clean_frame(df_feat, y)
    if feature_set == 'feature_analysis':
        return _feature_analysis_frame(df_feat, y)
    if feature_set == 'minimal':
        return _minimal_frame(df_feat)
    raise ValueError(f'unknown feature_set: {feature_set}')


def load_feature_frame(parquet_path, feature_set='full'):
    df_feat, y, df_meta = load_with_meta(parquet_path)
    return make_feature_frame(df_feat, y, feature_set=feature_set), y, df_meta


def feature_group_registry(columns):
    cols = list(columns)
    return {
        'player_hero_wr': [c for c in cols if 'player_hero_wr' in c],
        'global_hero_wr': [c for c in cols if 'global_hero_wr' in c],
        'global_hero_counts': [c for c in cols if 'global_hero_wins' in c or 'global_hero_matches' in c],
        'cumulative_counts': [
            c for c in cols if any(
                c.endswith('_' + field) or f'_{field}_' in c
                for field in ('matches_played', 'wins', 'kills', 'deaths', 'assists', 'time_played')
            )
        ],
        'per_minute_rates': [c for c in cols if 'per_min' in c or 'per_match' in c or 'per_soul' in c],
        'mmr': [c for c in cols if 'mmr_' in c or 'avg_mmr' in c or 'team_skill_gap' in c],
        'experience_gap': [c for c in cols if 'experience_gap' in c or 'avg_matches_played' in c],
        'hero_multihot': [c for c in cols if c.startswith('t0_hero_') or c.startswith('t1_hero_')],
        'ending_level': [c for c in cols if 'ending_level' in c],
        'accuracy': [c for c in cols if 'accuracy' in c or 'crit_shot' in c],
    }


def transform_splits(df_feat, splits, pipe=None):
    if pipe is None:
        pipe = build_pipeline()
    matrices = {'train': pipe.fit_transform(df_feat.iloc[splits['train']])}
    for split_name in ('val', 'test'):
        if split_name in splits:
            matrices[split_name] = pipe.transform(df_feat.iloc[splits[split_name]])
    return pipe, matrices


def ablate_matrix_group(matrices, drop_cols):
    present = [c for c in drop_cols if c in matrices['train'].columns]
    ablated = {
        split_name: frame.drop(columns=present)
        for split_name, frame in matrices.items()
    }
    return ablated, present


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if {'t0_avg_mmr_rank', 't1_avg_mmr_rank'}.issubset(X.columns):
            X['team_skill_gap'] = X['t0_avg_mmr_rank'] - X['t1_avg_mmr_rank']
        if {'t0_avg_matches_played', 't1_avg_matches_played'}.issubset(X.columns):
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
