import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import bittensor as bt
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Hermes Pyth API documentation: https://hermes.pyth.network/docs/

TOKEN_MAP = {
    "BTC": "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH": "ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "XAU": "765d2ba906dbc32ca17cc11f5310a89e9ee1f6420508c63861f2f8ba4ee34bb2",
    "SOL": "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
    "SPYX": "2817b78438c769357182c04346fddaad1178c82f4048828fe0997c3c64624e14",
    "NVDAX": "4244d07890e4610f46bbde67de8f43a4bf8b569eebe904f136b469f148503b7f",
    "TSLAX": "47a156470288850a440df3a6ce85a55917b813a19bb5b31128a33a986566a362",
    "AAPLX": "978e6cc68a119ce066aa830017318563a9ed04ec3a0a6439010fc11296a58675",
    "GOOGLX": "b911b0329028cd0283e4259c33809d62942bd2716a58084e5f31d64c00b5424e",
}

pyth_base_url = "https://hermes.pyth.network/v2/updates/price/latest"


@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=2),
    reraise=True,
)
def get_asset_price(asset: str = "BTC") -> float | None:
    """
    Fetch the latest price for an asset from the Pyth Hermes API.
    """
    pyth_params = {"ids[]": [TOKEN_MAP[asset]]}
    response = requests.get(pyth_base_url, params=pyth_params, timeout=10)
    if response.status_code != 200:
        print("Error in response of Pyth API")
        return None

    data = response.json()
    parsed_data = data.get("parsed", [])
    if not parsed_data:
        print("Empty parsed data from Pyth API")
        return None

    asset_data = parsed_data[0]
    price = int(asset_data["price"]["price"])
    expo = int(asset_data["price"]["expo"])

    live_price: float = price * (10**expo)
    return live_price


def _get_model_dir() -> Path:
    """
    Return the directory where trained GARCH models are stored.

    The default is a 'models' directory at the project root, but this can
    be overridden via the SYNTH_MODEL_DIR environment variable.
    """
    env_path = os.getenv("SYNTH_MODEL_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()

    # Default: project_root / "models"
    # This file lives at synth/miner/price_simulation.py
    return Path(__file__).resolve().parents[2] / "models"


def _infer_prompt_label(
    time_increment: int, time_length: int
) -> str:
    """
    Infer whether a request is high- or low-frequency based on the validator
    prompt configuration.

    HIGH_FREQUENCY:
        time_increment = 60, time_length = 3600
    LOW_FREQUENCY:
        time_increment = 300, time_length = 86400

    Returns "HF" or "LF".
    """
    if time_increment == 60 and time_length == 3600:
        return "HF"
    return "LF"


def _load_model_state(asset: str, label: str) -> Dict[str, Any]:
    """
    Load a previously trained model state for a given asset and label.

    The file is expected at: {model_dir}/{asset}_{label}.json
    """
    model_dir = _get_model_dir()
    model_path = model_dir / f"{asset}_{label}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with model_path.open("r") as f:
        return json.load(f)


def _extract_gjr_garch_params(
    params: Dict[str, Any]
) -> Tuple[float, float, float, float, float]:
    """
    Extract (mu, omega, alpha, gamma, beta) from an arch ConstantMean+GJR-GARCH
    parameter dictionary.
    """
    mu = float(params.get("mu", 0.0))
    omega = float(params.get("omega", 0.0))

    # arch names parameters alpha[1], gamma[1], beta[1] by default
    alpha = float(
        params.get("alpha[1]", params.get("alpha1", params.get("alpha", 0.0)))
    )
    gamma = float(
        params.get("gamma[1]", params.get("gamma1", params.get("gamma", 0.0)))
    )
    beta = float(
        params.get("beta[1]", params.get("beta1", params.get("beta", 0.0)))
    )

    return mu, omega, alpha, gamma, beta


def _extract_t_df(params: Dict[str, Any]) -> float:
    """
    Extract the degrees of freedom parameter nu for a Student-t distribution
    from an arch parameter dictionary.
    """
    if "nu" in params:
        return float(params["nu"])
    if "shape" in params:
        return float(params["shape"])
    return 8.0


def simulate_single_price_path_gjr_garch_t(
    current_price: float,
    time_increment: int,
    time_length: int,
    asset: str,
    model_state: Dict[str, Any],
) -> np.ndarray:
    """
    Simulate a single price path using a GJR-GARCH(1,1) model with
    Student-t innovations, using parameters estimated offline with the
    `arch` package.

    Returns an array of length (num_steps + 1) with prices, starting
    from current_price.
    """
    num_steps = int(time_length / time_increment)
    if num_steps <= 0:
        return np.array([current_price], dtype=float)

    params: Dict[str, Any] = model_state.get("params", {})
    bt.logging.info(f"params: {params}")
    mu, omega, alpha, gamma, beta = _extract_gjr_garch_params(params)
    nu = _extract_t_df(params)

    bt.logging.info(f"nu: {nu}")

    last_sigma = float(model_state.get("last_sigma", 0.0))
    if last_sigma <= 0.0:
        # Fallback to a small positive volatility to avoid numerical issues.
        last_sigma = 1e-3

    prices = np.empty(num_steps + 1, dtype=float)
    prices[0] = current_price

    sigma_t = last_sigma
    r_prev = 0.0
    logged_overflow_warning = False

    for t in range(1, num_steps + 1):
        # Draw standardized Student-t shock.
        eps = np.random.standard_t(nu)

        # Returns were trained in percent space; keep the same convention.
        r_t = mu + sigma_t * eps

        # Log large or non-finite return components once per path, each value individually.
        if (
            not logged_overflow_warning
            and (not np.isfinite(r_t) or abs(r_t) > 1_000)
        ):
            bt.logging.warning("GJR-GARCH step overflow risk: asset=%s, t=%d", asset, t)
            bt.logging.warning("  mu=%s", mu)
            bt.logging.warning("  sigma_t=%s", sigma_t)
            bt.logging.warning("  eps=%s", eps)
            bt.logging.warning("  r_t=%s", r_t)
            logged_overflow_warning = True

        # Update price using log-returns (percent to fraction).
        prices[t] = prices[t - 1] * np.exp(r_t / 100.0)

        # Update variance for next step via GJR-GARCH(1,1) recursion.
        indicator = 1.0 if r_prev < 0.0 else 0.0
        var_next = (
            omega
            + alpha * (r_prev**2)
            + gamma * (r_prev**2) * indicator
            + beta * (sigma_t**2)
        )
        var_next = max(var_next, 1e-12)
        sigma_t = float(np.sqrt(var_next))
        r_prev = r_t

    return prices


def simulate_crypto_price_paths_gjr_garch(
    current_price: float,
    time_increment: int,
    time_length: int,
    num_simulations: int,
    asset: str,
) -> np.ndarray:
    """
    Simulate price paths using a trained GJR-GARCH(1,1)-t model when available.

    If the model file is missing or invalid, this function raises, allowing
    the caller to decide on a fallback strategy.
    """
    label = _infer_prompt_label(time_increment, time_length)
    model_state = _load_model_state(asset, label)

    paths = []
    for _ in range(num_simulations):
        price_path = simulate_single_price_path_gjr_garch_t(
            current_price=current_price,
            time_increment=time_increment,
            time_length=time_length,
            asset=asset,
            model_state=model_state,
        )
        paths.append(price_path)

    return np.array(paths, dtype=float)


def simulate_single_price_path_gbm(
    current_price: float, time_increment: int, time_length: int, sigma: float
) -> np.ndarray:
    """
    Simulate a single asset price path using a simple GBM-like random walk
    with constant volatility. This serves as a fallback when no trained
    GARCH model is available.
    """
    one_hour = 3600
    dt = time_increment / one_hour
    num_steps = int(time_length / time_increment)
    if num_steps <= 0:
        return np.array([current_price], dtype=float)

    std_dev = sigma * np.sqrt(dt)
    price_change_pcts = np.random.normal(0, std_dev, size=num_steps)
    cumulative_returns = np.cumprod(1 + price_change_pcts)
    cumulative_returns = np.insert(cumulative_returns, 0, 1.0)
    price_path = current_price * cumulative_returns
    return price_path


def simulate_crypto_price_paths_gbm(
    current_price: float,
    time_increment: int,
    time_length: int,
    num_simulations: int,
    sigma: float,
) -> np.ndarray:
    """
    Simulate multiple asset price paths using the simple GBM-like fallback
    model with constant volatility.
    """
    price_paths = []
    for _ in range(num_simulations):
        price_path = simulate_single_price_path_gbm(
            current_price, time_increment, time_length, sigma
        )
        price_paths.append(price_path)

    return np.array(price_paths, dtype=float)

