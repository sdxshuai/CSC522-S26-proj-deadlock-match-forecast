"""
validate.py
============
Inspect the collected raw data and print a summary report.

Checks:
  - Number of matches in the list vs. number of fully-fetched match files
  - Player data coverage (how many match files have a valid 'players' array)
  - Hero picks completeness (all 12 slots filled per match)
  - Player account_id coverage in player_stats
  - MMR coverage
  - Distribution by match_mode, average_badge bin

Usage:
    uv run python data/validate.py
    uv run python data/validate.py --verbose
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

MATCH_LIST_DIR = Path("data/raw/match_list")
MATCHES_DIR = Path("data/raw/matches")
PLAYER_STATS_DIR = Path("data/raw/player_stats")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | list:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _extract_players(data: dict) -> list[dict]:
    """Return the players list from a match JSON, trying common field names."""
    for key in ("players", "player_info", "player_data", "match_players"):
        v = data.get(key)
        if isinstance(v, list):
            return v

    match_info = data.get("match_info") or {}
    for key in ("players", "player_info"):
        v = match_info.get(key)
        if isinstance(v, list):
            return v

    return []


def _badge_bin(badge: int | None) -> str:
    if badge is None:
        return "unknown"
    tier = badge // 10
    labels = {
        0: "Obscurus",
        1: "Seeker",
        2: "Alchemist",
        3: "Arcanist",
        4: "Ritualist",
        5: "Emissary",
        6: "Archon",
        7: "Oracle",
        8: "Phantom",
        9: "Ascendant",
        10: "Eternus",
        11: "Eternus+",
    }
    return labels.get(tier, f"tier-{tier}")


# ---------------------------------------------------------------------------
# Match list summary
# ---------------------------------------------------------------------------

def summarise_match_list() -> tuple[list[int], Counter]:
    listed_ids: list[int] = []
    mode_counts: Counter = Counter()

    for fpath in sorted(MATCH_LIST_DIR.glob("batch_*.json")):
        batch = _load_json(fpath)
        if not isinstance(batch, list):
            continue
        for m in batch:
            listed_ids.append(m["match_id"])
            mode_counts[m.get("match_mode", "?")] += 1

    return listed_ids, mode_counts


# ---------------------------------------------------------------------------
# Match detail summary
# ---------------------------------------------------------------------------

def summarise_matches(listed_ids: set[int], verbose: bool) -> dict:
    fetched_ids = set()
    with_players = 0
    full_12_players = 0
    badge_counts: Counter = Counter()
    account_ids: set[int] = set()
    missing_players: list[int] = []

    all_files = sorted(MATCHES_DIR.glob("match_*.json"))

    for fpath in all_files:
        mid = int(fpath.stem.split("_")[1])
        fetched_ids.add(mid)

        data = _load_json(fpath)
        players = _extract_players(data)

        if players:
            with_players += 1
            if len(players) == 12:
                full_12_players += 1
            elif verbose:
                print(f"  WARN: match {mid} has {len(players)} players (expected 12)")
            for p in players:
                aid = p.get("account_id")
                if aid:
                    account_ids.add(int(aid))
        else:
            missing_players.append(mid)

        # Badge distribution
        badge = data.get("average_badge_team0") or (
            (data.get("match_info") or {}).get("average_badge_team0")
        )
        badge_counts[_badge_bin(badge)] += 1

    return {
        "fetched_ids": fetched_ids,
        "with_players": with_players,
        "full_12_players": full_12_players,
        "badge_counts": badge_counts,
        "account_ids": account_ids,
        "missing_players": missing_players,
    }


# ---------------------------------------------------------------------------
# Player stats summary
# ---------------------------------------------------------------------------

def summarise_player_stats(match_account_ids: set[int]) -> dict:
    hero_stats_ids: set[int] = set()
    mmr_ids: set[int] = set()

    for fpath in sorted(PLAYER_STATS_DIR.glob("hero_stats_*.json")):
        records = _load_json(fpath)
        if isinstance(records, list):
            for r in records:
                aid = r.get("account_id")
                if aid:
                    hero_stats_ids.add(int(aid))

    for fpath in sorted(PLAYER_STATS_DIR.glob("mmr_*.json")):
        records = _load_json(fpath)
        if isinstance(records, list):
            for r in records:
                aid = r.get("account_id")
                if aid:
                    mmr_ids.add(int(aid))

    return {
        "hero_stats_ids": hero_stats_ids,
        "mmr_ids": mmr_ids,
    }


# ---------------------------------------------------------------------------
# Hero stats summary
# ---------------------------------------------------------------------------

def summarise_hero_stats() -> dict:
    hero_stats_path = Path("data/raw/hero_stats/hero_stats.json")
    counter_path = Path("data/raw/hero_stats/hero_counter_stats.json")
    synergy_path = Path("data/raw/hero_stats/hero_synergy_stats.json")

    n_heroes = len(_load_json(hero_stats_path)) if hero_stats_path.exists() else 0
    n_counters = len(_load_json(counter_path)) if counter_path.exists() else 0
    n_synergies = len(_load_json(synergy_path)) if synergy_path.exists() else 0

    return {
        "heroes": n_heroes,
        "counter_matchups": n_counters,
        "synergies": n_synergies,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate collected raw data")
    parser.add_argument("--verbose", action="store_true", help="Print per-match warnings")
    args = parser.parse_args()

    W = 60  # report width

    print("=" * W)
    print(" DEADLOCK DATA VALIDATION REPORT")
    print("=" * W)

    # --- Hero stats ---
    hs = summarise_hero_stats()
    print("\n[Hero Stats]")
    print(f"  Hero win-rate records:   {hs['heroes']}")
    print(f"  Counter matchup pairs:   {hs['counter_matchups']}")
    print(f"  Synergy pairs:           {hs['synergies']}")

    # --- Match list ---
    listed_ids, mode_counts = summarise_match_list()
    print(f"\n[Match List]  ({MATCH_LIST_DIR})")
    print(f"  Total listed matches:    {len(listed_ids)}")
    if mode_counts:
        for mode, cnt in sorted(mode_counts.items(), key=lambda x: -x[1]):
            print(f"    {mode:<20} {cnt:>6}")
    else:
        print("  No batch files found. Run: uv run python data/fetch_matches.py --phase 1")

    # --- Match detail ---
    print(f"\n[Match Files] ({MATCHES_DIR})")
    if not any(MATCHES_DIR.glob("match_*.json")):
        print("  No match files found. Run: uv run python data/fetch_matches.py --phase 2")
        match_result = {
            "fetched_ids": set(),
            "account_ids": set(),
            "with_players": 0,
            "full_12_players": 0,
            "badge_counts": Counter(),
            "missing_players": [],
        }
    else:
        match_result = summarise_matches(set(listed_ids), verbose=args.verbose)
        fetched = match_result["fetched_ids"]
        account_ids = match_result["account_ids"]

        pct_players = (
            100 * match_result["with_players"] / len(fetched) if fetched else 0
        )
        pct_full = (
            100 * match_result["full_12_players"] / len(fetched) if fetched else 0
        )
        pct_listed = 100 * len(fetched) / len(listed_ids) if listed_ids else 0

        print(f"  Fetched match files:     {len(fetched)}/{len(listed_ids)} ({pct_listed:.1f}% of listed)")
        print(f"  Files with player data:  {match_result['with_players']} ({pct_players:.1f}%)")
        print(f"  Files with 12 players:   {match_result['full_12_players']} ({pct_full:.1f}%)")
        print(f"  Unique account_ids:      {len(account_ids)}")

        if match_result["missing_players"] and args.verbose:
            print(f"  Matches missing players: {len(match_result['missing_players'])}")
            if len(match_result["missing_players"]) <= 10:
                print(f"    IDs: {match_result['missing_players']}")

        print("\n  Badge distribution (team0):")
        for badge, cnt in sorted(
            match_result["badge_counts"].items(), key=lambda x: -x[1]
        )[:8]:
            bar = "█" * min(30, cnt // max(1, len(fetched) // 30))
            print(f"    {badge:<14} {cnt:>5}  {bar}")

    # --- Player stats ---
    print(f"\n[Player Stats] ({PLAYER_STATS_DIR})")
    if not PLAYER_STATS_DIR.exists():
        print("  No player stats found. Run: uv run python data/fetch_player_stats.py")
    else:
        ps = summarise_player_stats(match_result.get("account_ids", set()))
        match_accounts = match_result.get("account_ids", set())
        hero_cov = (
            100 * len(ps["hero_stats_ids"] & match_accounts) / len(match_accounts)
            if match_accounts
            else 0
        )
        mmr_cov = (
            100 * len(ps["mmr_ids"] & match_accounts) / len(match_accounts)
            if match_accounts
            else 0
        )
        print(f"  Hero-stats coverage:     {len(ps['hero_stats_ids'])} accounts ({hero_cov:.1f}% of match players)")
        print(f"  MMR coverage:            {len(ps['mmr_ids'])} accounts ({mmr_cov:.1f}% of match players)")

    # --- Readiness summary ---
    print("\n" + "=" * W)
    print(" READINESS")
    print("=" * W)
    n_full = match_result.get("full_12_players", 0)
    n_hero = len(ps["hero_stats_ids"]) if (PLAYER_STATS_DIR.exists() and "ps" in dir()) else 0
    n_mmr = len(ps["mmr_ids"]) if (PLAYER_STATS_DIR.exists() and "ps" in dir()) else 0
    hero_done = hs["heroes"] > 0
    print(f"  Hero stats ready:        {'YES' if hero_done else 'NO — run fetch_hero_stats.py'}")
    print(f"  Usable match records:    {n_full}")
    print(f"  Player hero stats:       {n_hero} accounts")
    print(f"  Player MMR:              {n_mmr} accounts")
    if n_full >= 1000:
        print("\n  Ready for preprocessing. Run: uv run python src/preprocess.py")
    else:
        print(f"\n  Need more data. Target: 10 000 matches with full player data.")
        print(f"  Run: uv run python data/fetch_matches.py --limit 10000")
    print("=" * W)


if __name__ == "__main__":
    main()
