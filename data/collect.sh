#!/usr/bin/env bash
# collect.sh — Full data collection pipeline for Deadlock match forecast
#
# Usage:
#   bash data/collect.sh                  # full run (default 50 000 matches)
#   bash data/collect.sh --limit 100      # smoke test with 100 matches
#   bash data/collect.sh --limit 10000 --rate 20
#
# Options:
#   --limit N        Target number of matches to collect (default: 50000)
#   --rate  R        Phase-2 requests-per-second (default: 10)
#   --phase 1|2|all  Run only a specific phase of fetch_matches.py (default: all)
#   --skip-hero      Skip fetching hero stats (already fetched)
#   --skip-player    Skip fetching player stats
#   --dry-run        Print commands without executing them
#   -h / --help      Show this help message

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
LIMIT=50000
RATE=10
PHASE="all"
SKIP_HERO=0
SKIP_PLAYER=0
DRY_RUN=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)    LIMIT="$2";  shift 2 ;;
    --rate)     RATE="$2";   shift 2 ;;
    --phase)    PHASE="$2";  shift 2 ;;
    --skip-hero)   SKIP_HERO=1;   shift ;;
    --skip-player) SKIP_PLAYER=1; shift ;;
    --dry-run)     DRY_RUN=1;     shift ;;
    -h|--help)
      sed -n '2,18p' "$0"   # print the usage comment block
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RUN() {
  echo ""
  echo ">>> $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$@"
  fi
}

# Wrapper: run a Python script via uv
UV_PYTHON() {
  RUN uv run python3 "$@"
}

HEADER() {
  echo ""
  echo "============================================================"
  echo "  $*"
  echo "============================================================"
}

# Make sure we are at the project root (the directory that contains data/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Add common uv install locations to PATH (including workspace-local bin/)
export PATH="$HOME/bin:$HOME/.local/bin:$HOME/.cargo/bin:$(dirname "$PROJECT_ROOT")/bin:$PATH"

if ! command -v uv &>/dev/null; then
  echo "ERROR: 'uv' not found. Install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Ensure dependencies are installed
# ---------------------------------------------------------------------------
HEADER "Syncing dependencies (uv sync)"
RUN uv sync

START_TIME=$(date +%s)
echo ""
echo "Deadlock data collection pipeline"
echo "  project root : $PROJECT_ROOT"
echo "  limit        : $LIMIT matches"
echo "  phase-2 rate : $RATE req/s"
echo "  phase        : $PHASE"
echo "  skip-hero    : $SKIP_HERO"
echo "  skip-player  : $SKIP_PLAYER"
echo "  dry-run      : $DRY_RUN"

# ---------------------------------------------------------------------------
# Step 1 — Hero global stats (one-time, fast)
# ---------------------------------------------------------------------------
if [[ $SKIP_HERO -eq 0 ]]; then
  HEADER "Step 1/3 — Hero global stats"
  UV_PYTHON data/fetch_hero_stats.py
else
  echo ""
  echo "[Step 1/3] Skipped (--skip-hero)"
fi

# ---------------------------------------------------------------------------
# Step 2 — Match list + per-match detail
# ---------------------------------------------------------------------------
HEADER "Step 2/3 — Match data (phase: $PHASE, limit: $LIMIT)"
UV_PYTHON data/fetch_matches.py \
  --phase "$PHASE" \
  --limit "$LIMIT" \
  --rate  "$RATE"

# ---------------------------------------------------------------------------
# Step 3 — Per-player stats (hero stats + MMR)
# ---------------------------------------------------------------------------
if [[ $SKIP_PLAYER -eq 0 ]]; then
  HEADER "Step 3/3 — Player stats (hero stats + MMR)"
  UV_PYTHON data/fetch_player_stats.py --incremental
else
  echo ""
  echo "[Step 3/3] Skipped (--skip-player)"
fi

# ---------------------------------------------------------------------------
# Step 4 — Validate collected data
# ---------------------------------------------------------------------------
HEADER "Validation report"
UV_PYTHON data/validate.py

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "============================================================"
echo "  Pipeline complete in ${ELAPSED}s"
echo "  Raw data : $PROJECT_ROOT/data/raw/"
echo "============================================================"
