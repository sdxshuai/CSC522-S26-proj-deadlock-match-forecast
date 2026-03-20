"""
fetch_hero_stats.py
====================
One-time script to fetch global hero analytics from deadlock-api.com.

Fetches three datasets:
  1. hero_stats.json        - per-hero win/loss rates, KDA, etc.
  2. hero_counter_stats.json - hero vs hero matchup matrix
  3. hero_synergy_stats.json - hero pair synergy (same team)

Usage:
    uv run python data/fetch_hero_stats.py
    uv run python data/fetch_hero_stats.py --output-dir data/raw/hero_stats

These files are used downstream by src/preprocess.py to build feature columns
for hero win_rate, counter advantage, and synergy advantage.
"""

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
DEFAULT_OUT = Path("data/raw/hero_stats")


def fetch_json(url: str, params: dict | None = None) -> list | dict:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_hero_stats(out_dir: Path) -> None:
    log.info("Fetching /v1/analytics/hero-stats ...")
    # bucket=no_bucket aggregates across all time (no time window filter applied)
    data = fetch_json(f"{BASE_URL}/analytics/hero-stats", params={"bucket": "no_bucket"})
    out_path = out_dir / "hero_stats.json"
    out_path.write_text(json.dumps(data, indent=2))
    log.info(f"  Saved {len(data)} hero records → {out_path}")


def fetch_counter_stats(out_dir: Path) -> None:
    log.info("Fetching /v1/analytics/hero-counter-stats ...")
    # same_lane_filter=false gives broader matchup data across all lanes
    data = fetch_json(
        f"{BASE_URL}/analytics/hero-counter-stats",
        params={"same_lane_filter": "false", "min_matches": 50},
    )
    out_path = out_dir / "hero_counter_stats.json"
    out_path.write_text(json.dumps(data, indent=2))
    log.info(f"  Saved {len(data)} matchup records → {out_path}")


def fetch_synergy_stats(out_dir: Path) -> None:
    log.info("Fetching /v1/analytics/hero-synergy-stats ...")
    data = fetch_json(
        f"{BASE_URL}/analytics/hero-synergy-stats",
        params={"same_lane_filter": "false", "min_matches": 50},
    )
    out_path = out_dir / "hero_synergy_stats.json"
    out_path.write_text(json.dumps(data, indent=2))
    log.info(f"  Saved {len(data)} synergy records → {out_path}")


def main() -> None:
    global log
    log = get_logger("fetch_hero_stats")
    parser = argparse.ArgumentParser(description="Fetch global hero analytics")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUT),
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fetch_hero_stats(out_dir)
    time.sleep(0.5)  # polite spacing between analytics calls
    fetch_counter_stats(out_dir)
    time.sleep(0.5)
    fetch_synergy_stats(out_dir)

    log.info(f"Done. All hero stats written to {out_dir}")


if __name__ == "__main__":
    main()
