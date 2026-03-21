# Data Collection Guide

## Overview

Data is collected in three stages:

```
Stage 1: Hero stats (one-time, global)
   fetch_hero_stats.py  →  raw/hero_stats/

Stage 2: Match data (bulk pagination + per-match detail)
   fetch_matches.py     →  raw/match_list/   (bulk metadata, checkpointed)
                        →  raw/matches/      (individual match JSON, checkpointed)

Stage 3: Player stats (after matches are fetched)
   fetch_player_stats.py →  raw/player_stats/
```

Run `validate.py` at any time to check coverage and data quality.

---

## Directory Structure

```
data/
├── raw/
│   ├── hero_stats/
│   │   ├── hero_stats.json          # Global hero win/loss stats (bucketed)
│   │   ├── hero_counter_stats.json  # Hero vs hero matchup matrix
│   │   └── hero_synergy_stats.json  # Hero pair synergy (same team)
│   ├── match_list/
│   │   ├── batch_0000.json          # Bulk metadata pages (≤1000 matches each)
│   │   ├── batch_0001.json
│   │   └── checkpoint.json          # {"last_match_id": N, "total_listed": N}
│   ├── matches/
│   │   ├── match_12670.json         # Individual match full data (with players)
│   │   └── checkpoint.json          # {"fetched_ids": [...], "total": N}
│   └── player_stats/
│       ├── hero_stats_0000.json     # Per-player hero stats (batches of 1000)
│       ├── mmr_0000.json            # Per-player MMR history (batches of 1000)
│       └── checkpoint.json          # {"processed_ids": [...]}
└── processed/
    ├── matches.parquet              # Feature matrix (output of src/preprocess.py)
    └── matches_meta.json            # Column descriptions and split info
```

---

## API Endpoints

| Endpoint | Rate Limit | Notes |
|---|---|---|
| `GET /v1/matches/metadata` | 4 req/s (IP) | Bulk, ≤1000 per request; `include_player_info=true` for player fields |
| `GET /v1/matches/{id}/metadata` | 100 req/s (from cache) | Full match JSON (CMsgMatchMetaDataContents) |
| `GET /v1/analytics/hero-stats` | 100 req/s | Global hero performance per bucket |
| `GET /v1/analytics/hero-counter-stats` | 100 req/s | Hero vs hero matchup win rates |
| `GET /v1/analytics/hero-synergy-stats` | 100 req/s | Hero pair synergy win rates |
| `GET /v1/players/hero-stats` | 100 req/s | Per-player stats, batch ≤1000 account_ids |
| `GET /v1/players/mmr` | 100 req/s | Player MMR history, batch ≤1000; `max_match_id` for pre-match snapshot |

Base URL: `https://api.deadlock-api.com`  
Authentication: None required (free tier)

---

## Raw Data Schemas

### Bulk match metadata (`raw/match_list/batch_NNNN.json`)

Each batch is a JSON array. Each element:
```json
{
  "match_id": 12670,
  "winning_team": "Team1",          // "Team0" | "Team1"
  "match_mode": "Unranked",         // "Ranked" | "Unranked" | "CoopBot"
  "game_mode": "Normal",
  "duration_s": 2328,
  "start_time": "2024-05-20 00:04:05",
  "average_badge_team0": null,     // int or null
  "average_badge_team1": null,
  "match_outcome": "TeamWin",       // filter: keep only "TeamWin"
  "not_scored": null
}
```

Filter applied during collection:
- `match_outcome == "TeamWin"` (removes abandoned/draw)
- `match_mode in {"Ranked", "Unranked"}` (removes CoopBot)
- `game_mode == "Normal"` (default filter in API)

### Individual match metadata (`raw/matches/match_{id}.json`)

Saved in **slim mode** by default (use `--full` to save raw API response).
Only pre-match fields are retained — all in-game stats are excluded because
they are unavailable at prediction time.

```json
{
  "match_id": 12670,
  "match_outcome": "TeamWin",
  "winning_team": 1,
  "start_time": "2024-05-20 00:04:05",
  "duration_s": 2328,
  "match_mode": "Unranked",
  "game_mode": "normal",
  "average_badge_team0": 90,
  "average_badge_team1": 88,
  "players": [
    {
      "account_id": 123456789,
      "hero_id": 15,
      "team": 0,
      "assigned_lane": 1,
      "player_slot": 2
    }
    // ... 11 more players
  ]
}
```

Excluded (in-game only, unavailable pre-match): `kills`, `deaths`, `assists`,
`net_worth`, `level`, `items`, `stats` (time-series), `death_details`,
`accolades`, `power_up_buffs`, `pings`, etc.

### Player hero stats (`raw/player_stats/hero_stats_NNNN.json`)

Array of `HeroStats` objects per player per hero:
```json
{
  "account_id": 123456789,
  "hero_id": 15,
  "matches_played": 42,
  "last_played": 1718000000,       // Unix timestamp
  "wins": 24,
  "kills": 210,                    // total across all matches
  "deaths": 168,
  "assists": 315,
  "kills_per_min": 0.42,
  "deaths_per_min": 0.33,
  "assists_per_min": 0.63,
  "denies_per_match": 8.5,
  "networth_per_min": 1200.0,
  "last_hits_per_min": 3.2,
  "damage_per_min": 2400.0,
  "ending_level": 18.5             // avg hero level at match end
}
```

### Player MMR (`raw/player_stats/mmr_NNNN.json`)

Array of `MMRHistory` objects (one per match per player):
```json
{
  "account_id": 123456789,
  "match_id": 12670,
  "start_time": 1716163445,
  "player_score": 4.2,             // internal EMA score
  "rank": 52,                      // tier=rank//10, subtier=rank%10
  "division": 5,                   // rank // 10
  "division_tier": 2               // rank % 10
}
```

---

## Processed Dataset Schema

`preprocess.py` joins all raw tables into one flat row per match.
Players are ordered by `(team, player_slot)` and expanded as `t0_p0_*` … `t1_p5_*`.
**No information is dropped** — individual player columns and team-level aggregations coexist.

### Match-level columns (always present)

| Column | Source | Notes |
|---|---|---|
| `match_id` | match JSON | primary key |
| `winning_team` | match JSON | 0 or 1 (integer) |
| `label` | derived | 1 if Team0 wins, 0 if Team1 wins — **prediction target** |
| `duration_s` | match JSON | match length in seconds |
| `start_time` | match JSON | Unix timestamp |
| `average_badge_team0` | match JSON | null for older matches (imputed) |
| `average_badge_team1` | match JSON | null for older matches (imputed) |
| `game_mode` | match JSON | integer enum |

### Per-player columns (×12: `t0_p0_` … `t1_p5_`)

Players are sorted by `(team, player_slot)` within each match. `i` = 0..5 within team.

| Column pattern | Source | Notes |
|---|---|---|
| `t{t}_p{i}_hero_id` | match JSON | hero picked |
| `t{t}_p{i}_assigned_lane` | match JSON | integer lane assignment |
| `t{t}_p{i}_player_slot` | match JSON | 0-11 global slot |
| `t{t}_p{i}_account_id` | match JSON | player identifier |
| `t{t}_p{i}_matches_played` | player hero_stats | total matches on this hero |
| `t{t}_p{i}_wins` | player hero_stats | wins on this hero |
| `t{t}_p{i}_kills_per_min` | player hero_stats | historical KPM on this hero |
| `t{t}_p{i}_deaths_per_min` | player hero_stats | historical DPM on this hero |
| `t{t}_p{i}_assists_per_min` | player hero_stats | historical APM on this hero |
| `t{t}_p{i}_denies_per_match` | player hero_stats | |
| `t{t}_p{i}_networth_per_min` | player hero_stats | |
| `t{t}_p{i}_last_hits_per_min` | player hero_stats | |
| `t{t}_p{i}_damage_per_min` | player hero_stats | |
| `t{t}_p{i}_ending_level` | player hero_stats | avg hero level at match end |
| `t{t}_p{i}_mmr_rank` | player MMR | rank at or before this match |
| `t{t}_p{i}_mmr_division` | player MMR | rank // 10 |
| `t{t}_p{i}_mmr_division_tier` | player MMR | rank % 10 |
| `t{t}_p{i}_mmr_player_score` | player MMR | internal EMA score |
| `t{t}_p{i}_global_hero_wins` | hero_stats | global wins for this hero |
| `t{t}_p{i}_global_hero_matches` | hero_stats | global matches for this hero |

Missing values (private profiles, new players): left as `NaN`.

### Team-aggregation columns (redundant but convenient)

Appended after all per-player columns. Each is a mean over the 6 players of that team.

| Column | Aggregates |
|---|---|
| `t{t}_avg_mmr_rank` | `t{t}_p{i}_mmr_rank` |
| `t{t}_avg_player_hero_wr` | `wins / matches_played` per player |
| `t{t}_avg_player_hero_matches` | `t{t}_p{i}_matches_played` |
| `t{t}_avg_global_hero_wr` | `global_hero_wins / global_hero_matches` per player |
| `t{t}_avg_kills_per_min` | `t{t}_p{i}_kills_per_min` |
| `t{t}_avg_deaths_per_min` | `t{t}_p{i}_deaths_per_min` |
| `t{t}_avg_networth_per_min` | `t{t}_p{i}_networth_per_min` |
| `t{t}_avg_ending_level` | `t{t}_p{i}_ending_level` |

Note: `match_mode` is excluded (current data is 100% Unranked, zero variance). Hero counter/synergy stats are excluded from the flat join (pair-wise matrix, not naturally per-player); may be added later as separate aggregations.

---

## Estimated Data Volumes

| Item | Count | Storage |
|---|---|---|
| Match list (10k matches) | 10 batches | ~3 MB |
| Individual match JSON (slim) | 10,000 files | ~15 MB |
| Individual match JSON (slim) | 50,000 files | ~75 MB |
| Player hero stats | ~50k account_ids | ~200 MB |
| Player MMR | ~50k account_ids | ~100 MB |
| Hero stats | 38 heroes | < 1 MB |
| **Processed feature matrix** | 10,000 rows × ~300 cols | ~20 MB |

---

## Notes

- Match IDs are monotonically increasing; use `min_match_id`/`max_match_id` for pagination.
- `average_badge_team0/1` may be `null` for older matches — imputed with global median in preprocessing.
- Player privacy: some accounts may return no stats (private profiles). These are handled as missing values.
- All timestamps are UTC.
- Checkpoint files prevent re-fetching on re-runs. Delete a checkpoint to re-fetch that stage.
