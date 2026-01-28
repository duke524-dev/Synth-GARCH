#!/usr/bin/env python
"""
Pure-offline test for the Synth miner: simulate paths and optionally compute CRPS.

Uses prompt config (HF/LF) and optional historical anchor/real prices from
data/raw/pyth/{asset}/{freq}.parquet. Generates the same number of prompts per
day as the real validator: 120 per HF asset per day, 24 per LF asset per day.

Usage examples:
  # All assets and prompt types, one date (default)
  python scripts/test_miner_offline.py

  # CRPS for 7 days starting 2026-01-21 (120 HF or 24 LF prompts per asset per day)
  python scripts/test_miner_offline.py --prompt-type LF --asset BTC --start-time 2026-01-21 --test-days 7 --crps

  # HF only, BTC only, 5 days from start, with CRPS
  python scripts/test_miner_offline.py --prompt-type HF --asset BTC --start-time 2026-01-21 --test-days 5 --crps

  # LF, two assets, custom [start,end] and 3 evenly spaced dates
  python scripts/test_miner_offline.py --prompt-type LF --asset SPYX AAPLX --start-time 2026-01-01 --end 2026-01-15 --test-days 3 --crps
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd

from synth.miner.simulations import generate_simulations
from synth.validator.crps_calculation import calculate_crps_for_miner
from synth.validator.prompt_config import HIGH_FREQUENCY, LOW_FREQUENCY

# Paths relative to project root (run from project root)
DATA_RAW_ROOT = Path("data/raw/pyth")

# Use validator prompt config so prompt counts and assets match the validator.
HF_ASSETS = HIGH_FREQUENCY.asset_list
LF_ASSETS = LOW_FREQUENCY.asset_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline test for Synth miner: simulate and optionally CRPS."
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["HF", "LF", "all"],
        default="all",
        help="Prompt type to test (default: all).",
    )
    parser.add_argument(
        "--asset",
        type=str,
        nargs="*",
        default=None,
        help="Assets to test (default: all for selected prompt type(s)).",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=None,
        metavar="N",
        help=(
            "With --start-time: window [start-time, start-time+N], one test per validator "
            "prompt (120 HF or 24 LF per asset per day). If only --start-time: that one day."
        ),
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help=(
            "Start date (ISO 8601 UTC). If only this is given: test that one day only. "
            "With --test-days N: test N days from this start."
        ),
    )
    parser.add_argument(
        "--end",
        type=str,
        default="now",
        help="End of date range when not using --start-time + --test-days (default: now).",
    )
    parser.add_argument(
        "--crps",
        action="store_true",
        help="Compute CRPS using historical real prices.",
    )
    return parser.parse_args()


def parse_time(s: str) -> datetime:
    if s == "now":
        return datetime.now(timezone.utc).replace(microsecond=0)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Date-only (e.g. 2026-01-21) -> UTC midnight
    if "T" not in s and " " not in s:
        s = s + "T00:00:00+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def load_anchor_price(asset: str, freq: str, start_dt: datetime) -> float | None:
    """Return close price at or just before start_dt from stored raw prices."""
    path = DATA_RAW_ROOT / asset / f"{freq}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty or "close" not in df.columns:
        return None
    df = df.sort_index()
    if df.index.tzinfo is None:
        df = df.tz_localize(timezone.utc)
    else:
        df = df.tz_convert(timezone.utc)
    mask = df.index <= start_dt
    if not mask.any():
        return None
    return float(df.loc[mask, "close"].iloc[-1])


def load_real_prices_for_horizon(
    asset: str,
    freq: str,
    start_dt: datetime,
    time_length_sec: int,
    time_increment_sec: int,
) -> np.ndarray | None:
    """Return 1D array of closes on grid [start_dt, start_dt + time_length_sec], step time_increment_sec."""
    path = DATA_RAW_ROOT / asset / f"{freq}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty or "close" not in df.columns:
        return None
    df = df.sort_index()
    if df.index.tzinfo is None:
        df = df.tz_localize(timezone.utc)
    else:
        df = df.tz_convert(timezone.utc)
    end_dt = start_dt + timedelta(seconds=time_length_sec)
    grid = pd.date_range(
        start=start_dt,
        end=end_dt,
        freq=f"{time_increment_sec}s",
        tz=timezone.utc,
    )
    values = []
    for t in grid:
        if t in df.index:
            values.append(float(df.loc[t, "close"]))
        else:
            # reindex to get last available; if none, use nan
            before = df[df.index <= t]
            if before.empty:
                values.append(np.nan)
            else:
                values.append(float(before["close"].iloc[-1]))
    return np.array(values, dtype=float)


def validator_prompt_start_times(
    prompt_config: object,
    asset: str,
    day_start_dt: datetime,
) -> List[datetime]:
    """
    Return the list of prompt start datetimes the validator would use for this
    (prompt_config, asset) on the given calendar day (UTC).

    - HF: total_cycle_minutes=12, 4 assets → 120 prompts/asset/day at
      day_start + asset_offset + k*12 min, k=0..119.
    - LF: total_cycle_minutes=60, initial_delay=60 → 24 prompts/asset/day at
      day_start + 60 + k*60 min, k=0..23.
    """
    tz = timezone.utc
    day_start = day_start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if day_start.tzinfo is None:
        day_start = day_start.replace(tzinfo=tz)
    else:
        day_start = day_start.astimezone(tz)

    asset_list = getattr(prompt_config, "asset_list", [])
    total = int(getattr(prompt_config, "total_cycle_minutes", 60))
    initial_delay = int(getattr(prompt_config, "initial_delay", 0))

    if asset not in asset_list:
        return []

    n_assets = len(asset_list)
    step_minutes = total / n_assets  # minutes between prompts in a cycle
    # Same-asset interval = one full cycle
    interval_minutes = total

    # HF: 120 per asset per day (24*60/12)
    # LF: 24 per asset per day (24*60/60)
    prompts_per_asset_per_day = int(24 * 60 / interval_minutes)

    asset_index = asset_list.index(asset)
    # First prompt of day for this asset: day_start + initial_delay + asset_offset
    asset_offset_minutes = asset_index * step_minutes
    first = day_start + timedelta(minutes=initial_delay + asset_offset_minutes)

    def round_to_minute(dt: datetime) -> datetime:
        return dt.replace(second=0, microsecond=0)

    return [
        round_to_minute(first + timedelta(minutes=k * interval_minutes))
        for k in range(prompts_per_asset_per_day)
    ]


def build_test_cases(
    prompt_type: str,
    assets: List[str] | None,
) -> List[Tuple[object, str]]:
    """Return list of (prompt_config, asset)."""
    cases: List[Tuple[object, str]] = []
    if prompt_type == "all":
        for a in HF_ASSETS:
            if assets is None or a in assets:
                cases.append((HIGH_FREQUENCY, a))
        for a in LF_ASSETS:
            if assets is None or a in assets:
                cases.append((LOW_FREQUENCY, a))
        return cases
    prompt = HIGH_FREQUENCY if prompt_type == "HF" else LOW_FREQUENCY
    pool = HF_ASSETS if prompt_type == "HF" else LF_ASSETS
    for a in pool:
        if assets is None or a in assets:
            cases.append((prompt, a))
    return cases


def run_one_test(
    prompt: object,
    asset: str,
    start_time_iso: str,
    run_crps: bool,
) -> Tuple[bool, str, float | None]:
    """
    Run one (prompt, asset, date) test. Returns (ok, message, crps_sum or None).
    """
    time_increment = prompt.time_increment
    time_length = prompt.time_length
    num_simulations = prompt.num_simulations
    freq = "1min" if time_increment == 60 else "5min"
    start_dt = parse_time(start_time_iso)

    anchor = load_anchor_price(asset, freq, start_dt)
    if anchor is None:
        return (
            False,
            f"No anchor price for {asset} {freq} at {start_time_iso}",
            None,
        )

    # Mock get_asset_price so we use historical anchor without network or current_price API.
    try:
        with patch("synth.miner.simulations.get_asset_price", return_value=anchor):
            sim_output = generate_simulations(
                asset=asset,
                start_time=start_time_iso,
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=num_simulations,
            )
    except Exception as e:
        return False, f"generate_simulations failed: {e}", None

    if not isinstance(sim_output, (list, tuple)) or len(sim_output) < 3:
        return False, f"Bad sim output type/len: {type(sim_output)} len={len(sim_output) if hasattr(sim_output, '__len__') else '?'}", None

    start_unix = sim_output[0]
    dt = sim_output[1]
    paths = sim_output[2:]
    expected_len = int(time_length / time_increment) + 1
    if len(paths) != num_simulations:
        return False, f"Expected {num_simulations} paths, got {len(paths)}", None
    for i, p in enumerate(paths):
        if len(p) != expected_len:
            return False, f"Path {i} length {len(p)} != {expected_len}", None
    arr = np.array(paths, dtype=float)
    if not np.isfinite(arr).all():
        return False, "Non-finite values in paths", None

    crps_sum = None
    if run_crps:
        real = load_real_prices_for_horizon(
            asset, freq, start_dt, time_length, time_increment
        )
        if real is None or len(real) != len(paths[0]):
            return False, "CRPS: could not load real prices or length mismatch", None
        try:
            crps_sum, _ = calculate_crps_for_miner(
                arr,
                real,
                time_increment,
                prompt.scoring_intervals,
            )
        except Exception as e:
            return False, f"CRPS failed: {e}", None
        if crps_sum == -1.0:
            return False, "CRPS returned -1 (e.g. zero price in sims)", None

    return True, "ok", crps_sum


def main() -> None:
    args = parse_args()

    # Only --start-time: test that one day only.
    if args.start_time is not None and args.test_days is None:
        start_dt = parse_time(args.start_time)
        end_dt = start_dt
        test_dates = [start_dt]
    # --start-time and --test-days: window [start_time, start_time + test_days], one test per day.
    elif args.start_time is not None and args.test_days is not None:
        if args.test_days <= 0:
            print("--test-days must be >= 1")
            return
        start_dt = parse_time(args.start_time)
        end_dt = start_dt + timedelta(days=args.test_days)
        test_dates = [
            start_dt + timedelta(days=i)
            for i in range(args.test_days)
        ]
    else:
        # No --start-time: range from --end (default now), start = end - 1 day, one date by default.
        end_dt = parse_time(args.end)
        start_dt = end_dt - timedelta(days=1)
        n = 1 if args.test_days is None else max(1, args.test_days)
        if n == 1:
            test_dates = [start_dt]
        else:
            step = (end_dt - start_dt) / (n - 1) if n > 1 else timedelta(0)
            test_dates = [start_dt + step * i for i in range(n)]

    cases = build_test_cases(args.prompt_type, args.asset)
    if not cases:
        print("No (prompt, asset) cases to run.")
        return

    n_dates = len(test_dates)
    print(
        f"Offline miner test: prompt_type={args.prompt_type}, "
        f"assets={args.asset or 'all'}, test_dates={n_dates}, crps={args.crps}"
    )
    if n_dates == 1 and args.start_time and args.test_days is None:
        print(f"Single day: {start_dt}")
    else:
        print(f"Date range: {start_dt} .. {end_dt}")
    print(
        f"Cases: {len(cases)} (same prompt/asset list as validator); "
        "prompts per day = same count as validator (HF 120, LF 24 per asset)."
    )
    print()

    fails = 0
    # Collect CRPS per (asset, prompt_label) for summary when --crps
    crps_by_key: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    total_tests = 0

    for prompt, asset in cases:
        for d in test_dates:
            day_start = d.replace(hour=0, minute=0, second=0, microsecond=0)
            if day_start.tzinfo is None:
                day_start = day_start.replace(tzinfo=timezone.utc)
            starts = validator_prompt_start_times(prompt, asset, day_start)
            for start_dt in starts:
                # Use full ISO with seconds (never HH:MM-only)
                start_iso = start_dt.isoformat(timespec="seconds")
                ok, msg, crps_sum = run_one_test(
                    prompt, asset, start_iso, args.crps
                )
                total_tests += 1
                label = f"{asset} {prompt.label} {start_iso}"
                if ok:
                    if crps_sum is not None:
                        crps_by_key[(asset, prompt.label)].append(crps_sum)
                    extra = f" CRPS={crps_sum:.4f}" if crps_sum is not None else ""
                    print(f"  OK   {label}{extra}")
                else:
                    print(f"  FAIL {label}: {msg}")
                    fails += 1

    print(f"\nRan {total_tests} tests (same prompt count per day as validator).")

    if fails:
        print(f"\n{fails} test(s) failed.")
    else:
        print("\nAll tests passed.")

    if args.crps and crps_by_key:
        print("\n--- CRPS summary (avg / min / max) per asset and prompt type ---")
        for (asset, prompt_label) in sorted(crps_by_key.keys()):
            vals = crps_by_key[(asset, prompt_label)]
            n = len(vals)
            avg = sum(vals) / n
            mn = min(vals)
            mx = max(vals)
            print(f"  {asset} {prompt_label}: n={n}  avg={avg:.4f}  min={mn:.4f}  max={mx:.4f}")
        print("---")


if __name__ == "__main__":
    main()
