#!/usr/bin/env python
"""
Fetch historical price data from the Pyth Benchmarks TradingView shim
for use in miner model training.

This script downloads last N days of history (default: 365) ending at a
given date (default: now) for all assets and frequencies needed by the
recalibration script and stores both raw prices and computed returns.

Outputs:
    data/raw/pyth/{asset}/{freq}.parquet
    data/processed/{asset}/{freq}_returns.parquet

Usage examples:
    # Fetch last 1 year of data up to now (UTC) for all assets
    python scripts/fetch_history.py

    # Fetch last 180 days ending at a specific date
    python scripts/fetch_history.py --days 180 --end 2026-01-27T00:00:00Z
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from dotenv import load_dotenv

from synth.validator.price_data_provider import PriceDataProvider


DATA_RAW_ROOT = Path("data/raw/pyth")
DATA_PROC_ROOT = Path("data/processed")


HF_ASSETS: List[str] = ["BTC", "ETH", "XAU", "SOL"]
LF_ASSETS_CRYPTO: List[str] = ["BTC", "ETH", "XAU", "SOL"]
LF_ASSETS_EQUITY: List[str] = ["SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]

# Mapping from Pyth synthetic equity tickers to Yahoo Finance/Alpha Vantage
# equity tickers.
EQUITY_YF_TICKER_MAP: Dict[str, str] = {
    "SPYX": "SPY",
    "NVDAX": "NVDA",
    "TSLAX": "TSLA",
    "AAPLX": "AAPL",
    "GOOGLX": "GOOGL",
}

# Mapping from crypto assets to Alpha Vantage crypto symbol/market pairs.
# These are used with the CRYPTO_INTRADAY endpoint.
CRYPTO_AV_SYMBOL_MAP: Dict[str, Tuple[str, str]] = {
    "BTC": ("BTC", "USD"),
    "ETH": ("ETH", "USD"),
    "XAU": ("XAU", "USD"),
    "SOL": ("SOL", "USD"),
}


load_dotenv()
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch historical price data from Pyth Benchmarks TradingView"
            " shim for miner model training."
        )
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of history to fetch (default: 365).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="now",
        help=(
            "End timestamp in ISO 8601 (UTC). "
            "Use 'now' for current time (default)."
        ),
    )
    return parser.parse_args()


def parse_end_time(end_str: str) -> datetime:
    if end_str == "now":
        return datetime.now(timezone.utc).replace(microsecond=0)

    # Accept both ...Z and offset formats
    if end_str.endswith("Z"):
        end_str = end_str[:-1] + "+00:00"
    return datetime.fromisoformat(end_str).astimezone(timezone.utc)


def build_time_range(end: datetime, days: int) -> Tuple[int, int]:
    start = end - timedelta(days=days)
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    return start_ts, end_ts


def fetch_pyth_history(
    asset: str, resolution_minutes: int, start_ts: int, end_ts: int
) -> pd.DataFrame:
    """
    Fetch historical prices for a single asset and resolution from Pyth.

    The TradingView shim endpoint limits the maximum number of points that
    can be returned in a single request. To reliably obtain up to a year of
    data, we fetch in smaller sequential chunks and concatenate the results.

    Returns a DataFrame with columns:
        - time: datetime64[ns, UTC]
        - close: float
    """
    symbol = PriceDataProvider._get_token_mapping(asset)  # noqa: SLF001

    # Conservative limit to avoid server-side range/size errors.
    # TradingView-like APIs often cap responses at ~5000 bars.
    max_bars = 4000
    seconds_per_bar = resolution_minutes * 60
    max_span = max_bars * seconds_per_bar

    all_frames: List[pd.DataFrame] = []
    total_rows = 0
    current_start = start_ts

    while current_start < end_ts:
        current_end = min(current_start + max_span, end_ts)

        print(
            f"  - Pyth chunk {asset} {resolution_minutes}min:"
            f" from {datetime.fromtimestamp(current_start, tz=timezone.utc)}"
            f" to {datetime.fromtimestamp(current_end, tz=timezone.utc)}"
        )

        params = {
            "symbol": symbol,
            "resolution": resolution_minutes,
            "from": current_start,
            "to": current_end,
        }

        response = requests.get(
            PriceDataProvider.BASE_URL, params=params, timeout=30
        )
        response.raise_for_status()
        data = response.json()

        if not data or "t" not in data or "c" not in data:
            # Move to next chunk; this may happen near the edges.
            current_start = current_end
            continue

        times = data["t"]
        closes = data["c"]

        if not times:
            current_start = current_end
            continue

        df = pd.DataFrame({"time": times, "close": closes})
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.drop_duplicates(subset="time").set_index("time").sort_index()
        all_frames.append(df)
        total_rows += len(df)
        print(f"    -> fetched {len(df)} rows (total so far: {total_rows})")

        # Step forward; +1 to avoid re-requesting the last bar.
        current_start = current_end + 1

    if not all_frames:
        return pd.DataFrame(columns=["time", "close"])

    out = pd.concat(all_frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def save_raw_prices(asset: str, freq_label: str, df: pd.DataFrame) -> None:
    """
    Save raw prices to data/raw/pyth/{asset}/{freq}.parquet.
    """
    if df.empty:
        print(f"No data for {asset} at {freq_label}, skipping raw save.")
        return

    out_dir = DATA_RAW_ROOT / asset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{freq_label}.parquet"
    df.to_parquet(out_path)
    print(f"Saved raw prices: {out_path} ({len(df)} rows)")


def prices_to_returns(df: pd.DataFrame) -> pd.Series:
    """
    Convert a prices DataFrame with 'close' column to percent log-returns.
    """
    if df.empty:
        return pd.Series(dtype=float)

    prices = df["close"].astype(float).sort_index()
    log_prices = np.log(prices)
    rets = 100.0 * (log_prices.diff())
    return rets.dropna()


def save_returns(asset: str, freq_label: str, returns: pd.Series) -> None:
    """
    Save returns to data/processed/{asset}/{freq}_returns.parquet
    with a single column 'ret'.
    """
    if returns.empty:
        print(f"No returns for {asset} at {freq_label}, skipping save.")
        return

    out_dir = DATA_PROC_ROOT / asset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{freq_label}_returns.parquet"
    df = pd.DataFrame({"ret": returns})
    df.to_parquet(out_path)
    print(f"Saved returns: {out_path} ({len(df)} rows)")


def fetch_yahoo_history(
    asset: str, resolution_minutes: int, start_ts: int, end_ts: int
) -> pd.DataFrame:
    """
    Fetch historical prices for an equity-like asset from Yahoo Finance.

    This is used as a fallback when Pyth does not provide data, primarily
    for the synthetic equity assets.
    """
    ticker = EQUITY_YF_TICKER_MAP.get(asset)
    if ticker is None:
        print(f"  - No Yahoo ticker mapping for {asset}, skipping Yahoo.")
        return pd.DataFrame(columns=["time", "close"])

    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    # Map resolution to Yahoo Finance interval string.
    if resolution_minutes == 1:
        interval = "1m"
    elif resolution_minutes == 5:
        interval = "5m"
    else:
        interval = f"{resolution_minutes}m"

    print(
        f"  - Fetching Yahoo history for {asset} ({ticker})"
        f" at interval {interval} from {start_dt} to {end_dt}"
    )

    try:
        data = yf.download(
            ticker,
            start=start_dt,
            end=end_dt,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  - Yahoo fetch failed for {asset}: {exc}")
        return pd.DataFrame(columns=["time", "close"])

    if data.empty or "Close" not in data.columns:
        print(f"  - Yahoo returned no data for {asset}.")
        return pd.DataFrame(columns=["time", "close"])

    df = data[["Close"]].rename(columns={"Close": "close"})
    df = df.tz_convert(timezone.utc) if df.index.tzinfo else df.tz_localize(
        timezone.utc
    )
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.reset_index().rename(columns={"index": "time"}).set_index("time")


def fetch_alpha_vantage_history(
    asset: str, resolution_minutes: int, start_ts: int, end_ts: int
) -> pd.DataFrame:
    """
    Fetch historical prices for an equity-like asset from Alpha Vantage.

    This is used as a primary fallback when Pyth does not provide data for
    synthetic equity assets.
    """
    if not ALPHAVANTAGE_API_KEY:
        print(
            "  - ALPHAVANTAGE_API_KEY not set; "
            "skipping Alpha Vantage fallback."
        )
        return pd.DataFrame(columns=["time", "close"])

    ticker = EQUITY_YF_TICKER_MAP.get(asset, asset)

    # Alpha Vantage supports only specific intraday intervals.
    supported_intervals = {1, 5, 15, 30, 60}
    if resolution_minutes not in supported_intervals:
        print(
            f"  - Alpha Vantage does not support {resolution_minutes}min "
            f"interval directly; skipping for {asset}."
        )
        return pd.DataFrame(columns=["time", "close"])

    interval = f"{resolution_minutes}min"

    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    print(
        f"  - Fetching Alpha Vantage history for {asset} ({ticker})"
        f" at interval {interval} from {start_dt} to {end_dt}"
    )

    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": interval,
        "outputsize": "full",
        "datatype": "json",
        "apikey": ALPHAVANTAGE_API_KEY,
    }

    try:
        response = requests.get(
            "https://www.alphavantage.co/query", params=params, timeout=30
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        print(f"  - Alpha Vantage fetch failed for {asset}: {exc}")
        return pd.DataFrame(columns=["time", "close"])

    # Handle common error/limit messages.
    if "Error Message" in data:
        print(
            f"  - Alpha Vantage returned an error for {asset}: "
            f"{data['Error Message']}"
        )
        return pd.DataFrame(columns=["time", "close"])
    if "Note" in data:
        print(
            f"  - Alpha Vantage notice for {asset} (likely rate limit): "
            f"{data['Note']}"
        )
        return pd.DataFrame(columns=["time", "close"])

    ts_key = f"Time Series ({interval})"
    series = data.get(ts_key)
    if not series:
        # Log actual response shape to debug API changes or free-tier limits.
        top_level = list(data.keys())
        print(
            f"  - Alpha Vantage returned no time series data for {asset} "
            f"with key '{ts_key}'. Top-level keys: {top_level}"
        )
        # Fallback: use any key that looks like "Time Series (<interval>)"
        for k in data:
            if k.startswith("Time Series (") and k.endswith(")"):
                series = data[k]
                if isinstance(series, dict) and series:
                    print(f"  - Using alternative key '{k}' for {asset}.")
                    break
        else:
            series = None
        if not series:
            return pd.DataFrame(columns=["time", "close"])

    records: List[Tuple[datetime, float]] = []
    for ts_str, values in series.items():
        close_str = values.get("4. close")
        if close_str is None:
            continue
        try:
            close = float(close_str)
        except ValueError:
            continue
        records.append((ts_str, close))

    if not records:
        print(f"  - Alpha Vantage time series empty for {asset}.")
        return pd.DataFrame(columns=["time", "close"])

    df = pd.DataFrame(records, columns=["time", "close"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    # Alpha Vantage intraday timestamps are in US/Eastern by default.
    try:
        df = df.tz_localize("America/New_York").tz_convert(timezone.utc)
    except TypeError:
        # Already tz-aware; just ensure UTC.
        df = df.tz_convert(timezone.utc)

    # Restrict to requested time window.
    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_alpha_vantage_crypto_history(
    asset: str, resolution_minutes: int, start_ts: int, end_ts: int
) -> pd.DataFrame:
    """
    Fetch historical prices for a crypto asset from Alpha Vantage.

    This is used as a fallback when Pyth does not provide data for the
    high-frequency and low-frequency crypto assets (BTC, ETH, XAU, SOL).
    """
    if not ALPHAVANTAGE_API_KEY:
        print(
            "  - ALPHAVANTAGE_API_KEY not set; "
            "skipping Alpha Vantage crypto fallback."
        )
        return pd.DataFrame(columns=["time", "close"])

    symbol_market = CRYPTO_AV_SYMBOL_MAP.get(asset)
    if symbol_market is None:
        print(f"  - No Alpha Vantage crypto mapping for {asset}, skipping.")
        return pd.DataFrame(columns=["time", "close"])

    symbol, market = symbol_market

    # Alpha Vantage crypto intraday supports these intervals.
    supported_intervals = {1, 5, 15, 30, 60}
    if resolution_minutes not in supported_intervals:
        print(
            f"  - Alpha Vantage CRYPTO_INTRADAY does not support"
            f" {resolution_minutes}min interval directly; skipping for {asset}."
        )
        return pd.DataFrame(columns=["time", "close"])

    interval = f"{resolution_minutes}min"

    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    print(
        f"  - Fetching Alpha Vantage CRYPTO_INTRADAY for {asset} ({symbol}/{market})"
        f" at interval {interval} from {start_dt} to {end_dt}"
    )

    params = {
        "function": "CRYPTO_INTRADAY",
        "symbol": symbol,
        "market": market,
        "interval": interval,
        "outputsize": "full",
        "datatype": "json",
        "apikey": ALPHAVANTAGE_API_KEY,
    }

    try:
        response = requests.get(
            "https://www.alphavantage.co/query", params=params, timeout=30
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        print(f"  - Alpha Vantage CRYPTO_INTRADAY fetch failed for {asset}: {exc}")
        return pd.DataFrame(columns=["time", "close"])

    # Handle common error/limit messages.
    if "Error Message" in data:
        print(
            f"  - Alpha Vantage CRYPTO_INTRADAY error for {asset}: "
            f"{data['Error Message']}"
        )
        return pd.DataFrame(columns=["time", "close"])
    if "Note" in data:
        print(
            f"  - Alpha Vantage CRYPTO_INTRADAY notice for {asset}"
            f" (likely rate limit): {data['Note']}"
        )
        return pd.DataFrame(columns=["time", "close"])

    ts_key = f"Time Series Crypto ({interval})"
    series = data.get(ts_key)
    if not series:
        # Log actual response shape; free tier may return different structure or empty.
        top_level = list(data.keys())
        print(
            f"  - Alpha Vantage CRYPTO_INTRADAY returned no time series data for"
            f" {asset} with key '{ts_key}'. Top-level keys: {top_level}"
        )
        # Fallback: use any key matching "Time Series Crypto (...)"
        for k in data:
            if "Time Series Crypto (" in k and isinstance(data[k], dict):
                cand = data[k]
                if cand:
                    series = cand
                    print(f"  - Using alternative key '{k}' for {asset}.")
                    break
        else:
            series = None
        if not series:
            return pd.DataFrame(columns=["time", "close"])

    records: List[Tuple[datetime, float]] = []
    for ts_str, values in series.items():
        close_str = values.get("4. close")
        if close_str is None:
            continue
        try:
            close = float(close_str)
        except ValueError:
            continue
        records.append((ts_str, close))

    if not records:
        print(f"  - Alpha Vantage CRYPTO_INTRADAY time series empty for {asset}.")
        return pd.DataFrame(columns=["time", "close"])

    df = pd.DataFrame(records, columns=["time", "close"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    # Alpha Vantage crypto timestamps are in UTC.
    if df.index.tzinfo is None:
        df = df.tz_localize(timezone.utc)
    else:
        df = df.tz_convert(timezone.utc)

    # Restrict to requested time window.
    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_and_store_for_asset(
    asset: str,
    resolutions: Dict[str, int],
    start_ts: int,
    end_ts: int,
) -> None:
    for freq_label, minutes in resolutions.items():
        print(
            f"Fetching {asset} at resolution {minutes} min"
            f" for ts range [{start_ts}, {end_ts}]"
        )

        # Try to load existing raw data to only backfill missing timestamps
        raw_path = DATA_RAW_ROOT / asset / f"{freq_label}.parquet"
        existing_df: pd.DataFrame | None = None
        if raw_path.exists():
            try:
                existing_df = pd.read_parquet(raw_path).sort_index()
                print(
                    f"  - Found existing raw data for {asset} {freq_label}"
                    f" with {len(existing_df)} rows"
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  - Failed to load existing raw data for"
                    f" {asset} {freq_label}: {exc}"
                )

        freq_str = f"{minutes}min"

        # Build expected index over requested range to detect gaps.
        # Align start and end times to the frequency grid to avoid
        # off-minute seconds causing everything to look "missing".
        start_dt_raw = datetime.fromtimestamp(start_ts, tz=timezone.utc)
        end_dt_raw = datetime.fromtimestamp(end_ts, tz=timezone.utc)
        start_dt = pd.Timestamp(start_dt_raw).floor(freq_str).to_pydatetime()
        end_dt = pd.Timestamp(end_dt_raw).floor(freq_str).to_pydatetime()

        full_index = pd.date_range(
            start=start_dt,
            end=end_dt,
            freq=freq_str,
            tz=timezone.utc,
        )

        if existing_df is None or existing_df.empty:
            # No existing data: fetch the entire requested range.
            try:
                aligned_start_ts = int(start_dt.timestamp())
                aligned_end_ts = int(end_dt.timestamp())
                df = fetch_pyth_history(
                    asset, minutes, aligned_start_ts, aligned_end_ts
                )
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED fetch for {asset} {freq_label}: {exc}")
                continue
        else:
            # Restrict existing data to requested range.
            existing_df = existing_df.loc[
                (existing_df.index >= start_dt)
                & (existing_df.index <= end_dt)
            ]
            missing_index = full_index.difference(existing_df.index)

            if missing_index.empty:
                print(
                    f"  - No missing data detected for {asset} {freq_label};"
                    " reusing existing data."
                )
                df = existing_df
            else:
                print(
                    f"  - Detected {len(missing_index)} missing timestamps for"
                    f" {asset} {freq_label}; backfilling."
                )
                # Group missing timestamps into contiguous ranges
                missing_index = missing_index.sort_values()
                ranges: List[tuple[datetime, datetime]] = []
                range_start = missing_index[0]
                prev = missing_index[0]
                step = pd.Timedelta(minutes=minutes)

                for ts in missing_index[1:]:
                    if ts - prev != step:
                        ranges.append((range_start, prev))
                        range_start = ts
                    prev = ts
                ranges.append((range_start, prev))

                fetched_frames: List[pd.DataFrame] = []
                for r_start, r_end in ranges:
                    r_start_ts = int(r_start.timestamp())
                    r_end_ts = int(r_end.timestamp())
                    print(
                        f"    - Backfill range {r_start} to {r_end}"
                        f" ({freq_label})"
                    )
                    try:
                        part = fetch_pyth_history(
                            asset, minutes, r_start_ts, r_end_ts
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"      FAILED backfill for {asset}"
                            f" {freq_label}: {exc}"
                        )
                        continue
                    if not part.empty:
                        fetched_frames.append(part)

                if fetched_frames:
                    df = pd.concat([existing_df] + fetched_frames)
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                else:
                    print(
                        f"  - No additional data fetched for {asset}"
                        f" {freq_label}; using existing only."
                    )
                    df = existing_df

        # If we still have no data for an equity asset, try external providers.
        # For crypto assets, attempt to backfill any remaining gaps using
        # Alpha Vantage CRYPTO_INTRADAY and merge results.
        if asset in HF_ASSETS or asset in LF_ASSETS_CRYPTO:
            if df is None or df.empty:
                current_index = pd.DatetimeIndex([], tz=timezone.utc)
            else:
                df = df.sort_index()
                current_index = df.loc[
                    (df.index >= start_dt) & (df.index <= end_dt)
                ].index

            remaining_missing = full_index.difference(current_index)

            if not remaining_missing.empty:
                print(
                    f"  - Still {len(remaining_missing)} missing timestamps for"
                    f" {asset} {freq_label} after Pyth; attempting"
                    " Alpha Vantage crypto backfill."
                )

                ext_df = fetch_alpha_vantage_crypto_history(
                    asset,
                    minutes,
                    int(start_dt.timestamp()),
                    int(end_dt.timestamp()),
                )

                if not ext_df.empty:
                    ext_df = ext_df.sort_index()
                    ext_index = ext_df.index
                    to_fill = remaining_missing.intersection(ext_index)

                    if not to_fill.empty:
                        print(
                            f"  - Filling {len(to_fill)} timestamps for {asset}"
                            f" {freq_label} from Alpha Vantage crypto."
                        )
                        ext_slice = ext_df.loc[to_fill]
                        if df is None or df.empty:
                            df = ext_slice
                        else:
                            df = pd.concat([df, ext_slice])
                            df = df[~df.index.duplicated(keep="last")].sort_index()

        # For equity assets, attempt to backfill any remaining gaps using
        # external providers (Alpha Vantage, then Yahoo) and merge results.
        if asset in LF_ASSETS_EQUITY:
            # Recompute missing timestamps against the final df we have so far.
            if df is None or df.empty:
                current_index = pd.DatetimeIndex([], tz=timezone.utc)
            else:
                # Ensure we only look at the requested range.
                df = df.sort_index()
                current_index = df.loc[
                    (df.index >= start_dt) & (df.index <= end_dt)
                ].index

            remaining_missing = full_index.difference(current_index)

            if not remaining_missing.empty:
                print(
                    f"  - Still {len(remaining_missing)} missing timestamps for"
                    f" {asset} {freq_label} after Pyth; attempting"
                    " Alpha Vantage / Yahoo backfill."
                )

                # Fetch external history over the full requested range.
                ext_df = fetch_alpha_vantage_history(
                    asset, minutes, int(start_dt.timestamp()), int(end_dt.timestamp())
                )

                if ext_df.empty:
                    print(
                        f"  - Alpha Vantage returned no usable data for {asset}"
                        f" {freq_label}; attempting Yahoo Finance fallback."
                    )
                    ext_df = fetch_yahoo_history(
                        asset,
                        minutes,
                        int(start_dt.timestamp()),
                        int(end_dt.timestamp()),
                    )

                if not ext_df.empty:
                    # Restrict external data to just the missing timestamps,
                    # then merge into df.
                    ext_df = ext_df.sort_index()
                    ext_index = ext_df.index
                    to_fill = remaining_missing.intersection(ext_index)

                    if not to_fill.empty:
                        print(
                            f"  - Filling {len(to_fill)} timestamps for {asset}"
                            f" {freq_label} from external provider(s)."
                        )
                        ext_slice = ext_df.loc[to_fill]
                        if df is None or df.empty:
                            df = ext_slice
                        else:
                            df = pd.concat([df, ext_slice])
                            df = df[~df.index.duplicated(keep="last")].sort_index()

        save_raw_prices(asset, freq_label, df)
        rets = prices_to_returns(df)
        save_returns(asset, freq_label, rets)


def main() -> None:
    args = parse_args()
    end_dt = parse_end_time(args.end)
    start_ts, end_ts = build_time_range(end_dt, args.days)

    print(
        f"Fetching history from Pyth for last {args.days} days,"
        f" from {datetime.fromtimestamp(start_ts, tz=timezone.utc)}"
        f" to {datetime.fromtimestamp(end_ts, tz=timezone.utc)}"
    )

    # HF assets: 1min data (for HF models)
    hf_resolutions = {"1min": 1}
    for asset in HF_ASSETS:
        fetch_and_store_for_asset(asset, hf_resolutions, start_ts, end_ts)

    # LF crypto: 5min data
    lf_crypto_resolutions = {"5min": 5}
    for asset in LF_ASSETS_CRYPTO:
        fetch_and_store_for_asset(
            asset, lf_crypto_resolutions, start_ts, end_ts
        )

    # LF equities: 5min data
    lf_equity_resolutions = {"5min": 5}
    for asset in LF_ASSETS_EQUITY:
        fetch_and_store_for_asset(
            asset, lf_equity_resolutions, start_ts, end_ts
        )

    print("Done fetching and storing historical data.")


if __name__ == "__main__":
    main()

