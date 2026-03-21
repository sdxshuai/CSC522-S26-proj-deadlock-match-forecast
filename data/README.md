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

## Feature Groups

The following pre-match features are engineered during preprocessing:

| Group | Features | Source |
|---|---|---|
| **Team composition** | `t0_hero_{id}`, `t1_hero_{id}` — one-hot over 38 hero IDs (76 cols total) | Match players |
| **Badge/rank** | `avg_badge_team0`, `avg_badge_team1`, `badge_diff` | Match metadata |
| **Hero global stats** | `t0_global_wr`, `t1_global_wr` — team avg of per-hero global win rate | hero_stats |
| **Hero counter** | `t0_counter_score`, `t1_counter_score` — avg counter advantage vs opposing team | hero_counter_stats |
| **Hero synergy** | `t0_synergy_score`, `t1_synergy_score` — avg pairwise synergy within team | hero_synergy_stats |
| **Player skill** | `t0_player_hero_wr`, `t1_player_hero_wr` — team avg player win rate on picked hero | player hero_stats |
| **Player experience** | `t0_player_hero_matches`, `t1_player_hero_matches` — team avg matches played on hero | player hero_stats |
| **Player rank** | `t0_player_rank`, `t1_player_rank` — team avg MMR rank at match time | player mmr |
| **Match context** | `duration_bucket` (derived from `duration_s`) | Match metadata |

Note: `match_mode` is excluded as current data is 100% Unranked (zero variance).

Target: `label` = 1 if Team0 wins, 0 if Team1 wins.

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
| **Processed feature matrix** | 10,000 rows × ~100 cols | ~5 MB |

---

## Notes

- Match IDs are monotonically increasing; use `min_match_id`/`max_match_id` for pagination.
- `average_badge_team0/1` may be `null` for older matches — imputed with global median in preprocessing.
- Player privacy: some accounts may return no stats (private profiles). These are handled as missing values.
- All timestamps are UTC.
- Checkpoint files prevent re-fetching on re-runs. Delete a checkpoint to re-fetch that stage.
