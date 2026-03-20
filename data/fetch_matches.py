"""
fetch_matches.py
=================
Collect match data from deadlock-api.com in two phases.

Phase 1 — match list
    Paginate /v1/matches/metadata (bulk endpoint) to collect match IDs and
    basic metadata.  Results are saved to data/raw/match_list/batch_NNNN.json.
    A checkpoint (data/raw/match_list/checkpoint.json) tracks the highest
    match_id seen so re-runs add only new matches.

Phase 2 — per-match detail
    For each match_id from Phase 1, fetch /v1/matches/{id}/metadata which
    returns the full CMsgMatchMetaDataContents JSON, including the players
    array (account_id, hero_id, team, etc.).
    Results are saved to data/raw/matches/match_{id}.json.
    A checkpoint (data/raw/matches/checkpoint.json) tracks already-fetched IDs.

Usage:
    # Run both phases, target 10 000 matches
    uv run python data/fetch_matches.py --limit 10000

    # Phase 1 only (populate match list)
    uv run python data/fetch_matches.py --phase 1 --limit 20000

    # Phase 2 only (fetch match detail for already-listed matches)
    uv run python data/fetch_matches.py --phase 2 --limit 10000 --rate 10

    # Continue a previous run (checkpoint is respected automatically)
    uv run python data/fetch_matches.py --limit 10000

Phase 1 bulk endpoint:  4 req/s IP limit (1 000 matches per request → very fast)
Phase 2 individual endpoint: 100 req/s from cache (use --rate to control)

Filters applied in Phase 1:
  - match_outcome == "TeamWin"  (removes abandoned/draw games)
  - match_mode in {"Ranked", "Unranked"}  (removes CoopBot)
  - game_mode == "normal"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
MATCH_LIST_DIR = Path("data/raw/match_list")
MATCHES_DIR = Path("data/raw/matches")

VALID_MODES = {"Ranked", "Unranked"}   # match_mode values to keep
BULK_LIMIT = 1000                       # max results per bulk metadata request


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, retries: int = 5) -> requests.Response:
    """GET with exponential backoff on 429/5xx."""
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
    raise RuntimeError(f"All {retries} retries failed for {url}")


# ---------------------------------------------------------------------------
# Phase 1: Bulk match list
# ---------------------------------------------------------------------------

def _filter_match(m: dict) -> bool:
    """Return True if this match should be kept for the dataset."""
    if m.get("match_outcome") != "TeamWin":
        return False
    if m.get("match_mode") not in VALID_MODES:
        return False
    return True


def phase1(limit: int) -> list[int]:
    """
    Paginate /v1/matches/metadata from newest to oldest and collect up to
    `limit` match IDs that pass the filter.  Returns all collected match_ids
    (including previously saved ones).

    Uses order_direction=desc with a max_match_id cursor so each run begins
    at the most recent available match and pages backwards in time.
    """
    MATCH_LIST_DIR.mkdir(parents=True, exist_ok=True)
    cp_path = MATCH_LIST_DIR / "checkpoint.json"
    cp = _load_json(cp_path)

    # Cursor: fetch matches below this ID on each page.
    # None on the first call → API returns the absolute latest matches.
    max_match_id = cp.get("max_match_id_next")  # None means "start from latest"
    total_listed = cp.get("total_listed", 0)
    batch_idx = cp.get("batch_idx", 0)

    log.info(f"[Phase 1] Starting match list collection (already have {total_listed} matches)")
    log.info(f"          Resuming from max_match_id={'latest' if max_match_id is None else max_match_id}")

    phase1_start = time.time()

    while total_listed < limit:
        params: dict = {
            "limit": BULK_LIMIT,
            "game_mode": "normal",
            "order_by": "match_id",
            "order_direction": "desc",
        }
        if max_match_id is not None:
            params["max_match_id"] = max_match_id

        resp = _get(f"{BASE_URL}/matches/metadata", params=params)

        # The API returns either JSON or octet-stream; handle both
        content_type = resp.headers.get("Content-Type", "")
        if "json" in content_type:
            batch = resp.json()
        else:
            # Try JSON anyway (spec may mis-report content type)
            try:
                batch = resp.json()
            except Exception:
                log.error(
                    f"Unexpected content-type '{content_type}'. "
                    "Cannot decode response. Stopping Phase 1."
                )
                break

        if not batch:
            log.info("No more matches returned. Phase 1 complete.")
            break

        # Filter matches
        kept = [m for m in batch if _filter_match(m)]

        if kept:
            out_path = MATCH_LIST_DIR / f"batch_{batch_idx:04d}.json"
            _save_json(out_path, kept)
            total_listed += len(kept)
            batch_idx += 1
            elapsed = time.time() - phase1_start
            log.info(
                f"Batch {batch_idx-1:04d}: fetched {len(batch)}, kept {len(kept)}"
                f" | total={total_listed} | elapsed={elapsed:.1f}s"
            )

        # Advance cursor: next page starts below the lowest match_id in this page
        min_id_in_batch = min(m["match_id"] for m in batch)
        max_match_id = min_id_in_batch - 1

        # Update checkpoint
        cp = {
            "max_match_id_next": max_match_id,
            "total_listed": total_listed,
            "batch_idx": batch_idx,
        }
        _save_json(cp_path, cp)

        if len(batch) < BULK_LIMIT:
            log.debug(f"Short batch ({len(batch)} < {BULK_LIMIT}), continuing anyway.")

        time.sleep(0.26)  # ~4 req/s max for bulk endpoint

    elapsed_p1 = time.time() - phase1_start
    log.info(f"[Phase 1] Done. Total listed: {total_listed} | wall-time: {elapsed_p1:.1f}s")

    # Return all collected match IDs
    return _load_all_match_ids()


def _load_all_match_ids() -> list[int]:
    """Read all batch files and return sorted list of match_ids."""
    ids: list[int] = []
    for f in sorted(MATCH_LIST_DIR.glob("batch_*.json")):
        batch = json.loads(f.read_text())
        ids.extend(m["match_id"] for m in batch)
    return sorted(set(ids))


# ---------------------------------------------------------------------------
# Phase 2: Per-match detail
# ---------------------------------------------------------------------------

def _load_fetched_ids() -> set[int]:
    cp = _load_json(MATCHES_DIR / "checkpoint.json")
    return set(cp.get("fetched_ids", []))


def _save_fetched_ids(fetched_ids: set[int]) -> None:
    cp_path = MATCHES_DIR / "checkpoint.json"
    _save_json(cp_path, {"fetched_ids": sorted(fetched_ids), "total": len(fetched_ids)})


def phase2(match_ids: list[int], limit: int, rate: float) -> None:
    """
    Fetch /v1/matches/{id}/metadata for each match_id not yet downloaded.
    Saves to data/raw/matches/match_{id}.json.

    rate: target requests per second (default 10 for safety; cache allows 100/s)
    """
    MATCHES_DIR.mkdir(parents=True, exist_ok=True)
    fetched_ids = _load_fetched_ids()

    pending = [mid for mid in match_ids if mid not in fetched_ids]
    pending = pending[:limit]

    log.info(
        f"[Phase 2] Need to fetch {len(pending)} individual match files "
        f"(already have {len(fetched_ids)}, target {limit})"
    )
    log.info(f"          Rate: {rate} req/s | ETA ≈ {len(pending)/rate/60:.1f} min")

    sleep_s = 1.0 / rate
    done = 0
    errors = 0
    phase2_start = time.time()
    window_start = phase2_start
    window_done = 0  # requests in current reporting window

    for match_id in pending:
        url = f"{BASE_URL}/matches/{match_id}/metadata"
        try:
            resp = _get(url)
        except Exception as exc:
            log.warning(f"Failed to fetch match {match_id}: {exc}")
            errors += 1
            continue

        content_type = resp.headers.get("Content-Type", "")
        if "json" in content_type:
            data = resp.json()
        else:
            try:
                data = resp.json()
            except Exception:
                log.warning(f"Non-JSON response for match {match_id} (skipping)")
                continue

        out_path = MATCHES_DIR / f"match_{match_id}.json"
        out_path.write_text(json.dumps(data))
        fetched_ids.add(match_id)
        done += 1
        window_done += 1

        if done % 100 == 0:
            _save_fetched_ids(fetched_ids)
            now = time.time()
            elapsed_total = now - phase2_start
            window_elapsed = now - window_start
            actual_rate = window_done / window_elapsed if window_elapsed > 0 else 0
            remaining = len(pending) - done
            eta_s = remaining / actual_rate if actual_rate > 0 else float('inf')
            pct = 100 * done / len(pending)
            log.info(
                f"Progress: {done}/{len(pending)} ({pct:.1f}%) "
                f"| rate={actual_rate:.1f} req/s "
                f"| elapsed={elapsed_total:.0f}s "
                f"| ETA≈{eta_s/60:.1f}min "
                f"| errors={errors}"
            )
            window_start = now
            window_done = 0

        time.sleep(sleep_s)

    _save_fetched_ids(fetched_ids)
    elapsed_p2 = time.time() - phase2_start
    log.info(
        f"[Phase 2] Done. Fetched {done} new matches (total: {len(fetched_ids)}) "
        f"| errors={errors} | wall-time={elapsed_p2:.0f}s "
        f"| avg-rate={(done / elapsed_p2):.1f} req/s"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global log
    log = get_logger("fetch_matches")
    parser = argparse.ArgumentParser(
        description="Collect Deadlock match data from deadlock-api.com",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
        help="Target number of matches to collect",
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default="all",
        help="Which phase to run: 1=list only, 2=detail only, all=both",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Phase-2 fetch rate in req/s (max ~100 from cache)",
    )
    args = parser.parse_args()

    if args.phase in ("1", "all"):
        match_ids = phase1(args.limit)
    else:
        match_ids = _load_all_match_ids()
        log.info(f"[Phase 2] Loaded {len(match_ids)} match IDs from existing batch files")

    if args.phase in ("2", "all"):
        phase2(match_ids, limit=args.limit, rate=args.rate)


if __name__ == "__main__":
    main()
