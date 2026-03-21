#!/usr/bin/env python3
"""
Preprocess raw data into a flat feature matrix (Plan B).

Each row = one match.
Players are sorted by (team, player_slot) and expanded as t0_p0_* … t1_p5_*.
Team-level mean aggregations are appended after all per-player columns.

Output: data/processed/matches.parquet
"""

import argparse
import json
import bisect
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# All scalar fields from the player hero_stats records (join keys excluded)
PLAYER_HS_FIELDS = [
    "matches_played", "wins", "time_played", "last_played", "ending_level",
    "kills", "deaths", "assists",
    "kills_per_min", "deaths_per_min", "assists_per_min",
    "denies_per_match", "denies_per_min",
    "networth_per_min", "last_hits_per_min",
    "damage_per_min", "damage_per_soul",
    "damage_mitigated_per_min",
    "damage_taken_per_min", "damage_taken_per_soul",
    "creeps_per_min",
    "obj_damage_per_min", "obj_damage_per_soul",
    "accuracy", "crit_shot_rate",
]

MMR_FIELDS = ["rank", "division", "division_tier", "player_score"]

# Team aggregation specs: output_suffix -> [per-player field or None for derived]
TEAM_AGG_STATS = [
    "kills_per_min", "deaths_per_min", "assists_per_min",
    "networth_per_min", "ending_level", "damage_per_min",
    "matches_played",
]


def load_player_hero_stats():
    """Returns dict: (account_id, hero_id) -> {field: value}"""
    stats = {}
    hs_files = sorted((RAW / "player_stats").glob("hero_stats_*.json"))
    print(f"  Loading {len(hs_files)} player hero_stats batch files...")
    for path in hs_files:
        for rec in json.loads(path.read_text()):
            key = (rec["account_id"], rec["hero_id"])
            stats[key] = {f: rec.get(f) for f in PLAYER_HS_FIELDS}
    return stats


def load_mmr():
    """Returns dict: account_id -> (sorted match_id list, parallel records list)"""
    raw: dict[int, list] = {}
    mmr_files = sorted((RAW / "player_stats").glob("mmr_*.json"))
    print(f"  Loading {len(mmr_files)} MMR batch files...")
    for path in mmr_files:
        for rec in json.loads(path.read_text()):
            aid = rec["account_id"]
            if aid not in raw:
                raw[aid] = []
            raw[aid].append(rec)

    # Sort and split into parallel arrays for fast binary search
    mmr: dict[int, tuple] = {}
    for aid, recs in raw.items():
        recs.sort(key=lambda r: r["match_id"])
        match_ids = [r["match_id"] for r in recs]
        mmr[aid] = (match_ids, recs)
    return mmr


def get_mmr_before(match_ids, recs, match_id):
    """Return most recent MMR record at or before match_id.
    Falls back to the earliest available record if none precedes match_id."""
    pos = bisect.bisect_left(match_ids, match_id)
    if pos == 0:
        return recs[0]  # no prior record; use earliest available as approximation
    return recs[pos - 1]


def load_global_hero_stats():
    """Returns dict: hero_id -> {global_wins, global_matches}
    Aggregated (summed) across all badge buckets."""
    hero_stats: dict[int, dict] = {}
    for rec in json.loads((RAW / "hero_stats" / "hero_stats.json").read_text()):
        hid = rec["hero_id"]
        if hid not in hero_stats:
            hero_stats[hid] = {"global_wins": 0, "global_matches": 0}
        hero_stats[hid]["global_wins"] += rec.get("wins", 0)
        hero_stats[hid]["global_matches"] += rec.get("matches", 0)
    return hero_stats


def process_match(match, player_hs, mmr, global_hs):
    row = {
        "match_id":            match["match_id"],
        "winning_team":        match["winning_team"],
        "label":               1 if match["winning_team"] == 0 else 0,
        "duration_s":          match["duration_s"],
        "start_time":          match["start_time"],
        "average_badge_team0": match.get("average_badge_team0"),
        "average_badge_team1": match.get("average_badge_team1"),
        "game_mode":           match.get("game_mode"),
    }

    # Sort players by (team, player_slot) so indexing within team is stable
    players = sorted(match["players"], key=lambda p: (p["team"], p["player_slot"]))

    team_idx = {0: 0, 1: 0}
    for p in players:
        t = p["team"]
        i = team_idx[t]
        team_idx[t] += 1
        pfx = f"t{t}_p{i}_"

        row[pfx + "hero_id"]       = p["hero_id"]
        row[pfx + "assigned_lane"] = p.get("assigned_lane")
        row[pfx + "player_slot"]   = p["player_slot"]
        row[pfx + "account_id"]    = p["account_id"]

        # Player × hero stats
        hs = player_hs.get((p["account_id"], p["hero_id"]), {})
        for f in PLAYER_HS_FIELDS:
            row[pfx + f] = hs.get(f)  # missing → None → NaN in DataFrame

        # MMR snapshot (latest entry before this match)
        mmr_entry = None
        if p["account_id"] in mmr:
            match_ids, recs = mmr[p["account_id"]]
            mmr_entry = get_mmr_before(match_ids, recs, match["match_id"])
        for f in MMR_FIELDS:
            row[pfx + "mmr_" + f] = mmr_entry[f] if mmr_entry else None

        # Global hero stats
        gh = global_hs.get(p["hero_id"], {})
        row[pfx + "global_hero_wins"]    = gh.get("global_wins")
        row[pfx + "global_hero_matches"] = gh.get("global_matches")

    return row


def add_team_aggregations(df):
    for t in (0, 1):
        pfx = f"t{t}_"

        # Per-player hero win rate (derived column, needed for avg)
        for i in range(6):
            w  = df[f"t{t}_p{i}_wins"]
            mp = df[f"t{t}_p{i}_matches_played"]
            df[f"t{t}_p{i}_player_hero_wr"] = (w / mp).replace([np.inf], np.nan)

        df[pfx + "avg_player_hero_wr"] = df[
            [f"t{t}_p{i}_player_hero_wr" for i in range(6)]
        ].mean(axis=1)

        # Global hero win rate
        for i in range(6):
            gw = df[f"t{t}_p{i}_global_hero_wins"]
            gm = df[f"t{t}_p{i}_global_hero_matches"]
            df[f"t{t}_p{i}_global_hero_wr"] = (gw / gm).replace([np.inf], np.nan)

        df[pfx + "avg_global_hero_wr"] = df[
            [f"t{t}_p{i}_global_hero_wr" for i in range(6)]
        ].mean(axis=1)

        # Simple per-column means
        for stat in TEAM_AGG_STATS:
            cols = [f"t{t}_p{i}_{stat}" for i in range(6)]
            df[pfx + f"avg_{stat}"] = df[cols].mean(axis=1)

        df[pfx + "avg_mmr_rank"] = df[
            [f"t{t}_p{i}_mmr_rank" for i in range(6)]
        ].mean(axis=1)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N match files (for testing)")
    args = parser.parse_args()

    print("=== Deadlock match preprocessing ===")
    if args.limit:
        print(f"  (limit: {args.limit} matches)")

    print("Loading player hero stats...")
    player_hs = load_player_hero_stats()
    print(f"  {len(player_hs):,} (account_id, hero_id) entries")

    print("Loading MMR history...")
    mmr = load_mmr()
    print(f"  {len(mmr):,} accounts")

    print("Loading global hero stats...")
    global_hs = load_global_hero_stats()
    print(f"  {len(global_hs):,} heroes")

    match_files = sorted(
        f for f in (RAW / "matches").iterdir()
        if f.suffix == ".json" and f.name != "checkpoint.json"
    )
    if args.limit:
        match_files = match_files[: args.limit]
    print(f"Processing {len(match_files):,} match files...")

    rows = []
    for i, path in enumerate(match_files):
        match = json.loads(path.read_text())
        rows.append(process_match(match, player_hs, mmr, global_hs))
        if (i + 1) % 10_000 == 0:
            print(f"  {i+1:,} / {len(match_files):,}")

    print("Building DataFrame...")
    df = pd.DataFrame(rows)

    print("Adding team aggregations...")
    df = add_team_aggregations(df)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "matches.parquet"
    df.to_parquet(out_path, index=False)  # always overwrite

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone. Saved to {out_path}")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]:,} cols")
    print(f"  Size:  {size_mb:.1f} MB")
    print(f"  NaN rate: {df.isna().mean().mean():.1%}")


if __name__ == "__main__":
    main()
