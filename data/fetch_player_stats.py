"""
fetch_player_stats.py
======================
Fetch per-player hero stats and MMR for all players found in the collected
match files.

Two datasets are fetched:
  hero_stats — /v1/players/hero-stats?account_ids=...
  mmr        — /v1/players/mmr?account_ids=...

For the MMR endpoint, we pass max_match_id=<match_id> so the returned rank
reflects the player's rank *before* that match (pre-match feature).  Because
this requires one request per unique (account_id, match_id) pair, this option
is skipped by default — instead we fetch the player's full MMR history and
look up the nearest rank in preprocessing.

Usage:
    # Fetch stats for all players found in data/raw/matches/
    uv run python data/fetch_player_stats.py

    # Incremental: skip already-queried account_ids
    uv run python data/fetch_player_stats.py --incremental

    # Override matches dir
    uv run python data/fetch_player_stats.py --matches-dir data/raw/matches

Both endpoints accept up to 1 000 account_ids per request at 100 req/s.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
from log_utils import get_logger  # noqa: E402

log = logging.getLogger(__name__)

BASE_URL = "https://api.deadlock-api.com/v1"
MATCHES_DIR = Path("data/raw/matches")
PLAYER_STATS_DIR = Path("data/raw/player_stats")
BATCH_SIZE = 1000   # max account_ids per request

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | list:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _get(url: str, params: dict | None = None, retries: int = 5) -> requests.Response:
    delay = 1.0
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = delay * (2 ** attempt)
                log.warning(f"Rate limited (429). Waiting {wait:.1f}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = delay * (2 ** attempt)
            log.warning(f"Request error ({exc}). Retrying in {wait:.1f}s …")
            time.sleep(wait)
    raise RuntimeError(f"All retries failed for {url}")


# ---------------------------------------------------------------------------
# Extract account_ids from match files
# ---------------------------------------------------------------------------

def extract_account_ids(matches_dir: Path) -> set[int]:
    """
    Scan all match_{id}.json files and extract unique account_ids.

    The CMsgMatchMetaDataContents JSON structure may vary slightly.
    We look for 'players' at the top level or nested under 'match_info'.
    Each player element should have an 'account_id' field.
    """
    account_ids: set[int] = set()
    files = sorted(matches_dir.glob("match_*.json"))

    if not files:
        log.warning(f"No match files found in {matches_dir}")
        return account_ids

    for fpath in files:
        try:
            data = json.loads(fpath.read_text())
        except Exception:
            continue

        # Try top-level 'players' first, then 'match_info.players'
        players = data.get("players")
        if players is None:
            players = (data.get("match_info") or {}).get("players")
        if players is None:
            # Some protos use 'player_info' or 'player_data'
            for key in ("player_info", "player_data", "match_players"):
                players = data.get(key)
                if players:
                    break

        if not isinstance(players, list):
            continue

        for p in players:
            aid = p.get("account_id")
            if aid:
                account_ids.add(int(aid))

    return account_ids


# ---------------------------------------------------------------------------
# Fetch hero stats
# ---------------------------------------------------------------------------

def fetch_hero_stats(
    account_ids: list[int],
    out_dir: Path,
    already_done: set[int],
) -> None:
    pending = [aid for aid in account_ids if aid not in already_done]
    batches = [pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]

    log.info(
        f"[Hero Stats] {len(pending)} accounts to fetch in "
        f"{len(batches)} batches …"
    )

    # Determine starting batch index from existing files
    existing = sorted(out_dir.glob("hero_stats_*.json"))
    batch_idx = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 0

    hs_start = time.time()
    total_records = 0
    errors = 0

    for i, batch in enumerate(batches, 1):
        params = {"account_ids": ",".join(str(a) for a in batch)}
        try:
            resp = _get(f"{BASE_URL}/players/hero-stats", params=params)
            records = resp.json()
        except Exception as exc:
            log.warning(f"Hero-stats batch failed ({exc}), skipping {len(batch)} accounts")
            errors += 1
            batch_idx += 1
            time.sleep(0.5)
            continue

        out_path = out_dir / f"hero_stats_{batch_idx:04d}.json"
        _save_json(out_path, records)
        total_records += len(records)
        elapsed = time.time() - hs_start
        pct = 100 * i / len(batches)
        log.info(
            f"  Batch {batch_idx:04d} ({i}/{len(batches)}, {pct:.0f}%): "
            f"{len(batch)} accounts → {len(records)} records "
            f"| elapsed={elapsed:.1f}s"
        )
        batch_idx += 1
        time.sleep(0.05)  # ~20 req/s (well under 100/s limit)

    elapsed_hs = time.time() - hs_start
    log.info(f"[Hero Stats] Done. {total_records} records in {elapsed_hs:.1f}s | errors={errors}")


# ---------------------------------------------------------------------------
# Fetch MMR history
# ---------------------------------------------------------------------------

def fetch_mmr(
    account_ids: list[int],
    out_dir: Path,
    already_done: set[int],
) -> None:
    pending = [aid for aid in account_ids if aid not in already_done]
    batches = [pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]

    log.info(
        f"[MMR] {len(pending)} accounts to fetch in "
        f"{len(batches)} batches …"
    )

    existing = sorted(out_dir.glob("mmr_*.json"))
    batch_idx = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 0

    mmr_start = time.time()
    total_records = 0
    errors = 0

    for i, batch in enumerate(batches, 1):
        params = {"account_ids": ",".join(str(a) for a in batch)}
        try:
            resp = _get(f"{BASE_URL}/players/mmr", params=params)
            records = resp.json()
        except Exception as exc:
            log.warning(f"MMR batch failed ({exc}), skipping {len(batch)} accounts")
            errors += 1
            batch_idx += 1
            time.sleep(0.5)
            continue

        out_path = out_dir / f"mmr_{batch_idx:04d}.json"
        _save_json(out_path, records)
        total_records += len(records)
        elapsed = time.time() - mmr_start
        pct = 100 * i / len(batches)
        log.info(
            f"  Batch {batch_idx:04d} ({i}/{len(batches)}, {pct:.0f}%): "
            f"{len(batch)} accounts → {len(records)} MMR records "
            f"| elapsed={elapsed:.1f}s"
        )
        batch_idx += 1
        time.sleep(0.05)

    elapsed_mmr = time.time() - mmr_start
    log.info(f"[MMR] Done. {total_records} records in {elapsed_mmr:.1f}s | errors={errors}")


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _load_processed_ids(out_dir: Path) -> set[int]:
    """Return account_ids already stored in existing output files."""
    processed: set[int] = set()
    for fpath in out_dir.glob("hero_stats_*.json"):
        try:
            records = json.loads(fpath.read_text())
            for r in records:
                aid = r.get("account_id")
                if aid:
                    processed.add(int(aid))
        except Exception:
            pass
    return processed


def _load_mmr_processed_ids(out_dir: Path) -> set[int]:
    processed: set[int] = set()
    for fpath in out_dir.glob("mmr_*.json"):
        try:
            records = json.loads(fpath.read_text())
            for r in records:
                aid = r.get("account_id")
                if aid:
                    processed.add(int(aid))
        except Exception:
            pass
    return processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global log
    log = get_logger("fetch_player_stats")
    parser = argparse.ArgumentParser(
        description="Fetch per-player hero stats and MMR history",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--matches-dir",
        default=str(MATCHES_DIR),
        help="Directory containing match_{id}.json files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PLAYER_STATS_DIR),
        help="Output directory for player stats",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip account_ids already present in output files",
    )
    parser.add_argument(
        "--hero-stats-only",
        action="store_true",
        help="Fetch only hero stats (skip MMR)",
    )
    parser.add_argument(
        "--mmr-only",
        action="store_true",
        help="Fetch only MMR (skip hero stats)",
    )
    args = parser.parse_args()

    matches_dir = Path(args.matches_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Scanning {matches_dir} for account_ids …")
    all_ids = sorted(extract_account_ids(matches_dir))
    log.info(f"  Found {len(all_ids)} unique account_ids")

    if not all_ids:
        log.warning("No account_ids found. Run fetch_matches.py (phase 2) first.")
        return

    if not args.mmr_only:
        already_hero = _load_processed_ids(out_dir) if args.incremental else set()
        if args.incremental and already_hero:
            log.info(f"  Skipping {len(already_hero)} already-processed accounts (hero stats)")
        fetch_hero_stats(all_ids, out_dir, already_hero)

    if not args.hero_stats_only:
        already_mmr = _load_mmr_processed_ids(out_dir) if args.incremental else set()
        if args.incremental and already_mmr:
            log.info(f"  Skipping {len(already_mmr)} already-processed accounts (MMR)")
        fetch_mmr(all_ids, out_dir, already_mmr)

    log.info("All player stats fetched.")


if __name__ == "__main__":
    main()
