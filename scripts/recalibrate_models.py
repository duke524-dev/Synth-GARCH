#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from arch.univariate import ConstantMean, GARCH, StudentsT


DATA_ROOT = Path("data/processed")
MODEL_ROOT = Path("models")


HF_ASSETS = ["BTC", "ETH", "XAU", "SOL"]
LF_ASSETS_CRYPTO = ["BTC", "ETH", "XAU", "SOL"]
LF_ASSETS_EQUITY = ["SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]


@dataclass
class ModelSpec:
    asset: str
    label: str  # "HF" or "LF"
    freq: str
    window_days: int


def load_returns(asset: str, freq: str) -> pd.Series:
    """
    Load precomputed returns for a given asset and frequency.

    Expects a file at data/processed/{asset}/{freq}_returns.parquet with a
    column named 'ret' and a DateTimeIndex in UTC. Drops NaN and inf so
    training uses only valid observations (e.g. when fetch left gaps).
    """
    path = DATA_ROOT / asset / f"{freq}_returns.parquet"
    if not path.exists():
        msg = f"Missing returns file: {path}"
        raise FileNotFoundError(msg)

    df = pd.read_parquet(path)
    if "ret" not in df.columns:
        msg = f"'ret' column not found in {path}"
        raise ValueError(msg)

    ret = df["ret"].sort_index()
    n_before = len(ret)
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    n_dropped = n_before - len(ret)
    if n_dropped > 0:
        print(f"  [{asset} {freq}] Dropped {n_dropped} invalid/NaN returns; using {len(ret)}.")
    return ret


def slice_last_window(
    returns: pd.Series, window_days: int, end_dt: datetime | None = None
) -> pd.Series:
    """
    Slice the last `window_days` of observations from a returns series.

    If end_dt is provided, the window will end at min(end_dt, last_timestamp)
    instead of at the latest observation. Drops any remaining NaN/inf in the
    window so the training path sees only valid observations.
    """
    if returns.empty:
        msg = "Empty returns series"
        raise ValueError(msg)

    last_ts = returns.index.max()
    if end_dt is not None:
        # Ensure we do not go beyond the available data.
        end = min(last_ts, end_dt)
    else:
        end = last_ts

    start = end - pd.Timedelta(days=window_days)
    window = returns[(returns.index > start) & (returns.index <= end)]
    # Process missing: drop NaN/inf so GARCH training uses only valid data.
    window = window.replace([np.inf, -np.inf], np.nan).dropna()
    if len(window) < 1000:
        msg = (
            "Not enough data in window: "
            f"got {len(window)} points for {window_days} days"
        )
        raise ValueError(msg)
    return window


# Scale factor applied to returns before fitting so optimizer sees values
# in ~1–1000 (arch recommends this for stability). last_sigma is converted
# back to original units when saving.
FIT_SCALE = 100.0

# Optimizer options for arch fit(): more iterations and looser tolerance
# to improve convergence (XAU, some LF crypto often hit SLSQP code 8).
# Passed to scipy.optimize.minimize (SLSQP).
FIT_OPTIONS = {
    "maxiter": 1000,
    "ftol": 1e-4,
}


def fit_gjr_garch_t(
    returns: pd.Series,
):
    """
    Fit ConstantMean + GJR-GARCH(1,1) with Student-t innovations.

    Returns are scaled by FIT_SCALE before fitting to satisfy arch's
    scale recommendation; sigma is in scaled units, use scale=FIT_SCALE
    when saving to get last_sigma in original units.
    """
    from arch.univariate import GARCH as GJR_GARCH

    y = returns * FIT_SCALE
    am = ConstantMean(y)
    am.volatility = GJR_GARCH(p=1, o=1, q=1)
    am.distribution = StudentsT()

    res = am.fit(
        update_freq=100,
        disp="off",
        tol=1e-4,
        options=FIT_OPTIONS,
    )
    sigma = res.conditional_volatility
    return res, sigma


def fit_garch_t(
    returns: pd.Series,
):
    """
    Fit ConstantMean + symmetric GARCH(1,1) with Student-t innovations.

    Returns are scaled by FIT_SCALE before fitting; use scale=FIT_SCALE
    when saving to get last_sigma in original units.
    """
    y = returns * FIT_SCALE
    am = ConstantMean(y)
    am.volatility = GARCH(p=1, q=1)
    am.distribution = StudentsT()

    res = am.fit(
        update_freq=100,
        disp="off",
        tol=1e-4,
        options=FIT_OPTIONS,
    )
    sigma = res.conditional_volatility
    return res, sigma


def save_model_result(
    spec: ModelSpec,
    res,
    sigma: pd.Series,
    scale: float = 1.0,
) -> None:
    """
    Save model parameters and last volatility state to a JSON file.

    If returns were scaled by `scale` before fitting (e.g. 100 for percent
    returns), pass it so last_sigma is converted back to original units.
    """
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    last_sigma = float(sigma.iloc[-1]) / scale
    last_time = sigma.index[-1].isoformat()

    pc = getattr(res, "param_cov", None)
    if pc is None:
        param_cov_list: List[Any] = []
    elif hasattr(pc, "values"):
        param_cov_list = pc.values.tolist()
    else:
        param_cov_list = pc.tolist()

    out: Dict[str, Any] = {
        "asset": spec.asset,
        "label": spec.label,
        "freq": spec.freq,
        "window_days": spec.window_days,
        "params": res.params.to_dict(),
        "param_covariance": param_cov_list,
        "distribution": "StudentsT",
        "vol_model": res.model.volatility.__class__.__name__,
        "last_sigma": last_sigma,
        "last_time": last_time,
        "n_obs": int(res.nobs),
    }

    out_path = MODEL_ROOT / f"{spec.asset}_{spec.label}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved model: {out_path}")


def recalibrate_one(spec: ModelSpec, end_dt: datetime | None) -> None:
    """
    Recalibrate a single (asset, label) model.
    """
    print(
        f"Recalibrating {spec.asset} [{spec.label}] using"
        f" {spec.freq}, window={spec.window_days} days"
    )

    returns = load_returns(spec.asset, spec.freq)
    window_ret = slice_last_window(returns, spec.window_days, end_dt=end_dt)

    if spec.label == "HF":
        res, sigma = fit_gjr_garch_t(window_ret)
    else:
        res, sigma = fit_garch_t(window_ret)

    save_model_result(spec, res, sigma, scale=FIT_SCALE)


def build_specs(
    hf_window_days: int,
    lf_crypto_window_days: int,
    lf_equity_window_days: int,
) -> List[ModelSpec]:
    """
    Construct the list of model specifications to recalibrate.
    """
    specs: List[ModelSpec] = []

    for asset in HF_ASSETS:
        specs.append(
            ModelSpec(
                asset=asset,
                label="HF",
                freq="1min",
                window_days=hf_window_days,
            )
        )

    for asset in LF_ASSETS_CRYPTO:
        specs.append(
            ModelSpec(
                asset=asset,
                label="LF",
                freq="5min",
                window_days=lf_crypto_window_days,
            )
        )

    for asset in LF_ASSETS_EQUITY:
        specs.append(
            ModelSpec(
                asset=asset,
                label="LF",
                freq="5min",
                window_days=lf_equity_window_days,
            )
        )

    return specs


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for recalibration configuration.
    """
    parser = argparse.ArgumentParser(
        description="Recalibrate GARCH-based models for the Synth miner."
    )
    parser.add_argument(
        "--hf-window-days",
        type=int,
        default=60,
        help="Window length in days for HF (1min) models. Default: 60.",
    )
    parser.add_argument(
        "--lf-crypto-window-days",
        type=int,
        default=90,
        help=(
            "Window length in days for LF crypto (5min) models."
            " Default: 90."
        ),
    )
    parser.add_argument(
        "--lf-equity-window-days",
        type=int,
        default=180,
        help=(
            "Window length in days for LF equity/XAU (5min) models."
            " Default: 180."
        ),
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
    """
    Parse an end time string into an aware UTC datetime.
    """
    if end_str == "now":
        return datetime.now(timezone.utc).replace(microsecond=0)

    # Accept both ...Z and offset formats
    if end_str.endswith("Z"):
        end_str = end_str[:-1] + "+00:00"
    return datetime.fromisoformat(end_str).astimezone(timezone.utc)


def main() -> None:
    """
    Entry point for the recalibration script.
    """
    args = parse_args()

    end_dt = parse_end_time(args.end)

    specs = build_specs(
        hf_window_days=args.hf_window_days,
        lf_crypto_window_days=args.lf_crypto_window_days,
        lf_equity_window_days=args.lf_equity_window_days,
    )

    failures: List[tuple[ModelSpec, str]] = []
    for spec in specs:
        try:
            recalibrate_one(spec, end_dt=end_dt)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {spec.asset} [{spec.label}]: {exc}")
            failures.append((spec, str(exc)))

    if failures:
        print("\nSome recalibrations failed:")
        for spec, msg in failures:
            print(f"- {spec.asset} [{spec.label}]: {msg}")
    else:
        print("\nAll recalibrations succeeded.")


if __name__ == "__main__":
    main()

