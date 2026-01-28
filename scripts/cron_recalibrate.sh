#!/bin/bash
#
# Cron-friendly wrapper: runs fetch_history.py to refresh historical data, then
# runs recalibrate_models.py. Run from project root, or set PROJECT_ROOT.
# Logs to logs/recalibrate.log (or RECALIBRATE_LOG); exits with the last
# script's exit code (fetch or recalibrate).
#
# -------- Setting the period (inside this script) --------
# Set RECALIBRATE_PERIOD_HOURS so calibration runs at most every N hours.
# Cron should run this script at or finer than that interval (e.g. hourly if
# period is 6).
#
#   RECALIBRATE_PERIOD_HOURS=24   # default: at most once per day
#   RECALIBRATE_PERIOD_HOURS=6    # at most every 6 hours
#   RECALIBRATE_PERIOD_HOURS=168 # at most once per week (24*7)
#
# Optional: set in the script below, or in crontab:
#   0 * * * * RECALIBRATE_PERIOD_HOURS=6 /path/to/scripts/cron_recalibrate.sh
#
# -------- Crontab examples --------
# Use absolute path; when run without a TTY the script appends to logs/recalibrate.log.
#
#   # Every hour (script will run calibration only when RECALIBRATE_PERIOD_HOURS have passed)
#   0 * * * * /path/to/synth-subnet-garch/scripts/cron_recalibrate.sh
#
#   # Daily at 04:00 UTC (period in script controls actual frequency)
#   0 4 * * * /path/to/synth-subnet-garch/scripts/cron_recalibrate.sh
#
# Optional: RECALIBRATE_LOG=..., CRON_RECALIBRATE_ARGS="--hf-window-days 60"
#           CRON_FETCH_ARGS="--days 365 --end now" (args for fetch_history.py).
#
# Before each recalibration, the script runs fetch_history.py to refresh
# historical data; then it runs recalibrate_models.py. Both must succeed for
# the run to count toward the period.
#
# -------- Real-time / daemon mode (no cron) --------
# Run in the foreground; the script wakes every RECALIBRATE_CHECK_INTERVAL_MINUTES,
# sees if RECALIBRATE_PERIOD_HOURS have passed since last success, and runs
# calibration when due. No cron needed.
#
#   ./scripts/cron_recalibrate.sh --daemon
#   RECALIBRATE_DAEMON=1 ./scripts/cron_recalibrate.sh
#
#   RECALIBRATE_PERIOD_HOURS=6 RECALIBRATE_CHECK_INTERVAL_MINUTES=5 ./scripts/cron_recalibrate.sh --daemon
#
# RECALIBRATE_CHECK_INTERVAL_MINUTES: wake and check this often (default 5).
# RECALIBRATE_PERIOD_HOURS: run calibration at most this often (default 24).
#

# Parse --daemon / -d for real-time mode
RECALIBRATE_DAEMON="${RECALIBRATE_DAEMON:-0}"
for arg in "$@"; do
    case "$arg" in
        --daemon|-d) RECALIBRATE_DAEMON=1; shift; break ;;
    esac
done

# Period: run calibration at most every N hours (default 24)
RECALIBRATE_PERIOD_HOURS="${RECALIBRATE_PERIOD_HOURS:-24}"
# How often to wake and check when running as daemon (minutes)
RECALIBRATE_CHECK_INTERVAL_MINUTES="${RECALIBRATE_CHECK_INTERVAL_MINUTES:-5}"

# Project root: directory containing scripts/ and data/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${RECALIBRATE_LOG:-$LOG_DIR/recalibrate.log}"
LAST_RUN_FILE="${RECALIBRATE_LAST_RUN:-$LOG_DIR/recalibrate.lastrun}"
EXTRA_ARGS="${CRON_RECALIBRATE_ARGS:-}"
FETCH_SCRIPT="$PROJECT_ROOT/scripts/fetch_history.py"
FETCH_ARGS="${CRON_FETCH_ARGS:-}"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"
SCRIPT="$PROJECT_ROOT/scripts/recalibrate_models.py"

# Use venv if present (e.g. bt_venv from Makefile)
if [ -x "$PROJECT_ROOT/bt_venv/bin/python3" ]; then
    PYTHON="$PROJECT_ROOT/bt_venv/bin/python3"
fi

# Returns 0 if we should run calibration, 1 if we should skip (within period)
should_run_calibration() {
    local now_ts last_ts elapsed period_sec
    now_ts=$(date +%s)
    if [ ! -f "$LAST_RUN_FILE" ]; then
        return 0
    fi
    last_ts=$(cat "$LAST_RUN_FILE" 2>/dev/null)
    [ -z "$last_ts" ] && return 0
    elapsed=$((now_ts - last_ts)) 2>/dev/null || elapsed=0
    period_sec=$((RECALIBRATE_PERIOD_HOURS * 3600))
    if [ "$elapsed" -ge 0 ] 2>/dev/null && [ "$elapsed" -lt "$period_sec" ]; then
        return 1
    fi
    return 0
}

run_calibration() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Starting fetch_history.py (refresh historical data)"
    "$PYTHON" "$FETCH_SCRIPT" $FETCH_ARGS
    local fetch_ex=$?
    if [ "$fetch_ex" -ne 0 ]; then
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] fetch_history.py failed with exit code $fetch_ex; skipping recalibration"
        return $fetch_ex
    fi
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Starting recalibrate_models.py"
    "$PYTHON" "$SCRIPT" $EXTRA_ARGS
    local ex=$?
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Finished recalibrate with exit code $ex"
    if [ "$ex" -eq 0 ]; then
        date +%s > "$LAST_RUN_FILE"
    fi
    return $ex
}

do_one_run() {
    if ! should_run_calibration; then
        return 0
    fi
    if [ ! -t 1 ] && [ -n "$LOG_FILE" ]; then
        run_calibration >> "$LOG_FILE" 2>&1
    else
        run_calibration
    fi
}

if [ "$RECALIBRATE_DAEMON" = "1" ]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Daemon started (period=${RECALIBRATE_PERIOD_HOURS}h, check every ${RECALIBRATE_CHECK_INTERVAL_MINUTES}min)"
    while true; do
        do_one_run
        sleep $((RECALIBRATE_CHECK_INTERVAL_MINUTES * 60))
    done
fi

# One-shot (cron or manual): skip if within period, else run
now_ts=$(date +%s)
if [ -f "$LAST_RUN_FILE" ]; then
    last_ts=$(cat "$LAST_RUN_FILE" 2>/dev/null)
    if [ -n "$last_ts" ]; then
        elapsed=$((now_ts - last_ts)) 2>/dev/null || elapsed=0
        period_sec=$((RECALIBRATE_PERIOD_HOURS * 3600))
        if [ "$elapsed" -ge 0 ] 2>/dev/null && [ "$elapsed" -lt "$period_sec" ]; then
            exit 0
        fi
    fi
fi

if [ ! -t 1 ] && [ -n "$LOG_FILE" ]; then
    run_calibration >> "$LOG_FILE" 2>&1
    EXIT=$?
else
    run_calibration
    EXIT=$?
fi
exit "$EXIT"
