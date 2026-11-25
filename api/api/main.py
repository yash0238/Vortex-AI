"""FastAPI backend for VORTEX-AI CG-NSDE interactive dashboard.

Serves data artifacts, model inference, scenario generation, and evaluation
metrics via REST API endpoints.
"""

from __future__ import annotations

import io
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torchsde
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from data.node_features import compute_node_features_compact

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "AXISBANK.NS",
    "SBIN.NS", "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "WIPRO.NS", "POWERGRID.NS", "NTPC.NS", "M&M.NS", "TECHM.NS",
    "TMPV.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "BAJAJ-AUTO.NS", "CIPLA.NS",
    "DRREDDY.NS", "DIVISLAB.NS", "HEROMOTOCO.NS", "ONGC.NS", "COALINDIA.NS",
    "BPCL.NS", "GRASIM.NS", "ADANIPORTS.NS", "EICHERMOT.NS", "APOLLOHOSP.NS",
    "HINDALCO.NS", "TATACONSUM.NS", "BRITANNIA.NS", "SHREECEM.NS", "UPL.NS",
    "BAJAJFINSV.NS", "INDUSINDBK.NS",
]

MODEL_STATE: dict[str, Any] = {}
CHECKPOINT_DIR = BASE_DIR / "models" / "checkpoints"


def require_loaded_generator() -> None:
    if MODEL_STATE.get("generator") is None:
        raise HTTPException(
            status_code=503,
            detail="No trained CG-NSDE generator checkpoint is loaded. Train and load a compatible VORTEXModel checkpoint before running inference.",
        )


class ModelConfig(BaseModel):
    n_stocks: int = 50
    T: int = 60
    latent_dim: int = 64
    proj_dim: int = 32
    in_feats: int = 6
    gat_heads: int = 4
    gat_dropout: float = 0.1
    tau: float = 0.07
    lr: float = 0.001
    sde_hidden_dim: int = 128
    batch_size: int = 8


def load_data() -> dict[str, np.ndarray]:
    """Load all data artifacts, handling length mismatches."""
    windows = np.load(RAW_DIR / "windows.npy", allow_pickle=True)
    regimes_daily = np.load(RAW_DIR / "regimes.npy", allow_pickle=True)
    regimes_path = RAW_DIR / "window_regimes_v2.npy"
    if regimes_path.exists():
        regimes = np.load(regimes_path, allow_pickle=True)
    else:
        first_window_end = windows.shape[1] - 1
        regimes = regimes_daily[first_window_end:first_window_end + len(windows)]
    adj = np.load(RAW_DIR / "adj_matrices.npy", allow_pickle=True)
    node_features_path = RAW_DIR / "window_node_features.npy"
    returns = np.load(RAW_DIR / "returns.npy", allow_pickle=True)
    close_path = RAW_DIR / "nifty50_close.csv"
    tickers = pd.read_csv(close_path, index_col=0, nrows=0).columns.tolist()

    if node_features_path.exists():
        node_feats = np.load(node_features_path, allow_pickle=True)
        if node_feats.ndim == 4:
            node_feats = node_feats[:, -1]
    else:
        volume_path = RAW_DIR / "nifty50_volume.csv"
        volume = None
        if volume_path.exists():
            volume = pd.read_csv(volume_path, index_col=0).to_numpy(dtype=np.float32)
            if volume.shape != returns.shape:
                volume = None
        daily_features = compute_node_features_compact(returns, volume, tickers)
        first_window_end = windows.shape[1] - 1
        node_feats = daily_features[first_window_end:first_window_end + len(windows)]

    if len(tickers) != windows.shape[2]:
        raise ValueError(
            f"Ticker count ({len(tickers)}) does not match data width ({windows.shape[2]})."
        )

    n = min(len(windows), len(regimes), len(adj), len(node_feats))
    return {
        "windows": windows[:n],
        "window_regimes": regimes[:n],
        "adj_matrices": adj[:n],
        "node_features": node_feats[:n],
        "returns": returns,
        "daily_regimes": regimes_daily,
        "tickers": tickers,
    }


def build_forecast_features(returns: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Build only information available through each day for a causal risk baseline."""
    market = np.nan_to_num(returns, nan=0.0).mean(axis=1)
    cross_sectional_vol = np.nan_to_num(returns, nan=0.0).std(axis=1)
    features: list[list[float]] = []
    names = ["market_return_5d", "market_return_20d", "volatility_20d", "volatility_60d", "negative_breadth_5d", "drawdown_60d"]
    for index in range(60, len(market)):
        window_5 = market[index - 5:index]
        window_20 = market[index - 20:index]
        window_60 = market[index - 60:index]
        equity_curve = np.exp(np.cumsum(window_60))
        drawdown = float(np.min(equity_curve / np.maximum.accumulate(equity_curve) - 1.0))
        features.append([
            float(window_5.sum()),
            float(window_20.sum()),
            float(cross_sectional_vol[index - 20:index].mean()),
            float(cross_sectional_vol[index - 60:index].mean()),
            float((returns[index - 5:index] < 0).mean()),
            drawdown,
        ])
    return np.asarray(features, dtype=np.float64), names


def build_forecast_state(data: dict[str, np.ndarray], horizon: int = 5) -> dict[str, Any]:
    """Fit a chronological logistic baseline for next-horizon crisis probability."""
    returns = np.asarray(data["returns"], dtype=np.float64)
    daily_regimes = np.asarray(data["daily_regimes"], dtype=np.int64)
    features, feature_names = build_forecast_features(returns)
    start = 60
    targets = np.asarray([
        int(daily_regimes[index + 1:index + horizon + 1].max())
        for index in range(start, len(daily_regimes) - horizon)
    ], dtype=np.int64)
    usable_features = features[:len(targets)]
    split = max(int(len(targets) * 0.7), 1)
    scaler = StandardScaler().fit(usable_features[:split])
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    classifier.fit(scaler.transform(usable_features[:split]), targets[:split])
    holdout_probability = classifier.predict_proba(scaler.transform(usable_features[split:]))[:, 1]
    holdout_prediction = (holdout_probability >= 0.5).astype(int)
    metrics = {
        "holdout_start_index": int(start + split),
        "holdout_samples": int(len(targets) - split),
        "holdout_positive_rate": float(targets[split:].mean()),
        "accuracy": float(accuracy_score(targets[split:], holdout_prediction)),
        "brier_score": float(brier_score_loss(targets[split:], holdout_probability)),
        "roc_auc": float(roc_auc_score(targets[split:], holdout_probability)) if len(np.unique(targets[split:])) > 1 else None,
        "split": "chronological 70/30 holdout",
    }
    full_scaler = StandardScaler().fit(usable_features)
    full_classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    full_classifier.fit(full_scaler.transform(usable_features), targets)
    latest_features = features[-1:]
    latest_probability = float(full_classifier.predict_proba(full_scaler.transform(latest_features))[0, 1])
    return {
        "scaler": full_scaler,
        "classifier": full_classifier,
        "features": features,
        "feature_names": feature_names,
        "latest_probability": latest_probability,
        "metrics": metrics,
        "horizon_days": horizon,
        "target_definition": f"at least one crisis-labelled day in the next {horizon} trading days",
    }


def build_directional_screen(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    returns = np.nan_to_num(np.asarray(data["returns"], dtype=np.float64), nan=0.0)
    tickers = data["tickers"]
    rows = []
    for index, ticker in enumerate(tickers):
        recent = returns[-20:, index]
        return_20 = float(np.expm1(recent.sum()))
        volatility = float(recent.std() * np.sqrt(252))
        score = float(return_20 / (volatility + 1e-8))
        rows.append({
            "symbol": ticker,
            "return_20d": return_20,
            "volatility_annualized": volatility,
            "direction_score": score,
            "signal": "positive trend" if score > 0.25 else "negative trend" if score < -0.25 else "mixed",
        })
    return sorted(rows, key=lambda row: row["direction_score"], reverse=True)


def nan_to_float(arr: np.ndarray) -> list:
    """Convert numpy array to JSON-safe list, replacing NaN/Inf."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.tolist()


def chunk_array(arr: np.ndarray, chunk_size: int = 50) -> list:
    """Break large array into chunks for JSON serialization."""
    flat = arr.flatten()
    chunks = []
    for i in range(0, len(flat), chunk_size):
        chunks.append(float(flat[i]))
    return chunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and build model on startup."""
    global MODEL_STATE
    MODEL_STATE["data"] = load_data()
    MODEL_STATE["device"] = torch.device("cpu")
    MODEL_STATE["forecast"] = build_forecast_state(MODEL_STATE["data"])
    MODEL_STATE["directional_screen"] = build_directional_screen(MODEL_STATE["data"])
    yield


app = FastAPI(
    title="VORTEX-AI CG-NSDE API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_STATE.get("generator") is not None,
        "generator_checkpoint_available": any(CHECKPOINT_DIR.glob("*.ckpt")),
    }


@app.get("/api/stats")
async def get_stats():
    """Get overall data and model statistics."""
    data = MODEL_STATE["data"]
    return {
        "n_windows": int(len(data["windows"])),
        "n_stocks": int(data["windows"].shape[2]),
        "window_length": int(data["windows"].shape[1]),
        "crisis_ratio": float(data["window_regimes"].mean()),
        "n_crisis": int(data["window_regimes"].sum()),
        "n_normal": int(len(data["window_regimes"]) - data["window_regimes"].sum()),
        "returns_shape": list(data["returns"].shape),
        "daily_crisis_days": int(data["daily_regimes"].sum()),
        "daily_total_days": int(len(data["daily_regimes"])),
    }


@app.get("/api/forecast/outlook")
async def get_forecast_outlook(horizon_days: int = Query(5, ge=1, le=20)):
    """Return a calibrated baseline next-horizon crisis outlook and stock screen."""
    data = MODEL_STATE["data"]
    forecast = MODEL_STATE["forecast"]
    if horizon_days != forecast["horizon_days"]:
        forecast = build_forecast_state(data, horizon_days)
    probability = forecast["latest_probability"]
    regime = "elevated risk" if probability >= 0.6 else "watch" if probability >= 0.35 else "lower risk"
    directional = forecast.get("directional_screen") or MODEL_STATE["directional_screen"]
    return {
        "horizon_days": horizon_days,
        "crisis_probability": probability,
        "risk_regime": regime,
        "confidence": float(abs(probability - 0.5) * 2),
        "target_definition": forecast["target_definition"],
        "as_of": str(data["returns"].shape[0] - 1),
        "features": {
            name: float(value)
            for name, value in zip(forecast["feature_names"], forecast["features"][-1])
        },
        "drivers": [
            {"name": name, "value": float(value)}
            for name, value in zip(forecast["feature_names"], forecast["features"][-1])
        ],
        "validation": forecast["metrics"],
        "directional_screen": {
            "strongest_positive": directional[:5],
            "strongest_negative": list(reversed(directional[-5:])),
        },
        "method": "Causal logistic-regression baseline over rolling market return, volatility, breadth, and drawdown features. This is a research forecast, not a guaranteed price prediction or investment advice.",
    }


@app.get("/api/returns")
async def get_returns(
    stock_idx: int = Query(0, ge=0, le=49),
    window_idx: int = Query(0, ge=0, le=3639),
):
    """Get returns data for a specific stock and window."""
    data = MODEL_STATE["data"]
    returns = data["returns"]
    if stock_idx >= returns.shape[1]:
        raise HTTPException(status_code=400, detail="Invalid stock index")

    cum_returns = np.cumsum(returns[:, stock_idx])
    regimes_daily = data["daily_regimes"]

    return {
        "ticker": data["tickers"][stock_idx],
        "cumulative_returns": nan_to_float(cum_returns),
        "daily_returns": nan_to_float(returns[:, stock_idx]),
        "regimes": regimes_daily.tolist(),
        "dates": list(range(len(returns))),
    }


def normalize_market_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9^.-]{1,24}", normalized):
        raise HTTPException(status_code=400, detail="Invalid market symbol.")
    return normalized


def market_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


@app.get("/api/market/search")
def search_market_instruments(
    q: str = Query(..., min_length=2, max_length=80),
    limit: int = Query(10, ge=1, le=20),
):
    """Search NSE/BSE equities through Yahoo Finance's public search API."""
    try:
        matches = yf.Search(q.strip(), max_results=limit * 3, news_count=0).quotes
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Market search is temporarily unavailable: {exc}") from exc

    instruments = []
    seen: set[str] = set()
    for match in matches:
        symbol = str(match.get("symbol", "")).upper()
        if symbol in seen or not symbol.endswith((".NS", ".BO")):
            continue
        seen.add(symbol)
        instruments.append({
            "symbol": symbol,
            "name": match.get("shortname") or match.get("longname") or symbol,
            "exchange": "NSE" if symbol.endswith(".NS") else "BSE",
            "currency": match.get("currency") or "INR",
            "quote_type": match.get("quoteType") or "EQUITY",
        })
        if len(instruments) == limit:
            break
    return {"query": q.strip(), "results": instruments}


@app.get("/api/market/quote")
def get_market_quote(symbol: str = Query(..., min_length=1, max_length=24)):
    """Return the latest available Yahoo Finance quote and its session date."""
    normalized = normalize_market_symbol(symbol)
    try:
        ticker = yf.Ticker(normalized)
        history = ticker.history(period="5d", interval="1d", auto_adjust=False)
        if history.empty:
            raise HTTPException(status_code=404, detail=f"No recent market data for {normalized}.")
        fast = dict(ticker.fast_info)
        latest = history.iloc[-1]
        previous = history.iloc[-2] if len(history) > 1 else None
        previous_close = market_float(fast.get("previousClose"))
        if previous_close is None and previous is not None:
            previous_close = market_float(previous["Close"])
        price = market_float(fast.get("lastPrice")) or market_float(latest["Close"])
        volume = market_float(fast.get("lastVolume"))
        if volume is None or volume <= 0:
            volume = market_float(latest["Volume"])
        change = price - previous_close if price is not None and previous_close is not None else None
        return {
            "symbol": normalized,
            "price": price,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change / previous_close * 100 if change is not None and previous_close else None,
            "open": market_float(fast.get("open")) or market_float(latest["Open"]),
            "day_high": market_float(fast.get("dayHigh")) or market_float(latest["High"]),
            "day_low": market_float(fast.get("dayLow")) or market_float(latest["Low"]),
            "volume": int(volume) if volume is not None and volume > 0 else None,
            "market_cap": market_float(fast.get("marketCap")),
            "currency": fast.get("currency") or "INR",
            "exchange": "NSE" if normalized.endswith(".NS") or normalized == "^NSEI" else "BSE",
            "as_of": history.index[-1].isoformat(),
            "source": "Yahoo Finance via yfinance",
            "freshness_note": "Latest available quote; may be delayed and is not licensed exchange streaming data.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Quote is temporarily unavailable for {normalized}: {exc}") from exc


@app.get("/api/market/history")
def get_market_history(
    symbol: str = Query(..., min_length=1, max_length=24),
    period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"] = "1y",
    interval: Literal["1m", "5m", "15m", "30m", "60m", "1d", "1wk", "1mo"] = "1d",
):
    """Return OHLCV bars for a selected NSE/BSE symbol and chart range."""
    normalized = normalize_market_symbol(symbol)
    try:
        bars = yf.Ticker(normalized).history(period=period, interval=interval, auto_adjust=False)
        if bars.empty:
            raise HTTPException(status_code=404, detail=f"No history returned for {normalized}.")
        rows = []
        for timestamp, bar in bars.iterrows():
            rows.append({
                "date": timestamp.isoformat(),
                "open": market_float(bar["Open"]),
                "high": market_float(bar["High"]),
                "low": market_float(bar["Low"]),
                "close": market_float(bar["Close"]),
                "volume": int(market_float(bar["Volume"]) or 0),
            })
        return {
            "symbol": normalized,
            "period": period,
            "interval": interval,
            "source": "Yahoo Finance via yfinance",
            "freshness_note": "Historical bars; availability and delay depend on Yahoo Finance.",
            "bars": rows,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"History is temporarily unavailable for {normalized}: {exc}") from exc


@app.get("/api/window")
async def get_window(window_idx: int = Query(0, ge=0, le=3639)):
    """Get a single window's data for detailed inspection."""
    data = MODEL_STATE["data"]
    return {
        "window_idx": window_idx,
        "regime": int(data["window_regimes"][window_idx]),
        "returns": nan_to_float(data["windows"][window_idx]),
        "adjacency": nan_to_float(data["adj_matrices"][window_idx]),
        "regime_label": "Crisis" if data["window_regimes"][window_idx] else "Normal",
    }


@app.get("/api/adjacency")
async def get_adjacency(window_idx: int = Query(0, ge=0, le=3639)):
    """Get adjacency matrix for heatmap visualization."""
    data = MODEL_STATE["data"]
    adj = data["adj_matrices"][window_idx]
    regimes = data["window_regimes"]

    non_zero = int((adj > 0).sum())
    density = float(non_zero / (adj.shape[0] * adj.shape[1]))

    # NetworkX graph for centrality
    G = nx.from_numpy_array(adj, create_using=nx.Graph())
    try:
        centrality = nx.degree_centrality(G)
        top_nodes = sorted(centrality.items(), key=lambda x: -x[1])[:10]
        top_tickers = [data["tickers"][i] for i, _ in top_nodes]
    except Exception:
        top_tickers = data["tickers"][:10]

    return {
        "z": nan_to_float(adj),
        "x": data["tickers"],
        "y": data["tickers"],
        "regime": "Crisis" if regimes[window_idx] else "Normal",
        "regime_idx": int(regimes[window_idx]),
        "edge_density": density,
        "non_zero_edges": non_zero,
        "top_connected": top_tickers,
    }


@app.get("/api/node-features")
async def get_node_features(window_idx: int = Query(0, ge=0, le=3639)):
    """Get node features for a specific window."""
    data = MODEL_STATE["data"]
    feats = data["node_features"][window_idx]
    feat_names = ["Return", "Abs Return", "Volume", "Sector ID", "Band Limit", "Dist to Band"]

    return {
        "tickers": data["tickers"],
        "feature_names": feat_names,
        "data": nan_to_float(feats),
    }


@app.get("/api/regime-distribution")
async def get_regime_distribution():
    """Get regime distribution statistics."""
    data = MODEL_STATE["data"]
    regimes = data["window_regimes"]
    crisis_windows = np.where(regimes == 1)[0]

    return {
        "total_windows": len(regimes),
        "crisis_count": int(regimes.sum()),
        "normal_count": int(len(regimes) - regimes.sum()),
        "crisis_ratio": float(regimes.mean()),
        "crisis_indices": crisis_windows.tolist()[:100],
    }


@app.post("/api/model/gat-forward")
async def gat_forward(config: ModelConfig = None):
    """Run GAT encoder forward pass on a batch of data."""
    require_loaded_generator()
    if config is None:
        config = ModelConfig()
    data = MODEL_STATE["data"]
    device = MODEL_STATE["device"]

    from models.gat_encoder import DynamicGATEncoder, pool_graph_embedding

    gat = DynamicGATEncoder(
        in_feats=config.in_feats,
        hidden=config.latent_dim,
        out_feats=config.latent_dim,
        heads=config.gat_heads,
        dropout=config.gat_dropout,
    ).to(device)
    gat.eval()

    n = min(config.batch_size, len(data["windows"]))
    windows = torch.from_numpy(data["windows"][:n]).float().to(device)
    adj = torch.nan_to_num(torch.from_numpy(data["adj_matrices"][:n]).float(),
                           nan=0.0, posinf=1.0, neginf=0.0).to(device)
    node_feats = torch.nan_to_num(
        torch.from_numpy(data["node_features"][:n]).float(),
        nan=0.0, posinf=0.0, neginf=0.0
    ).to(device)

    with torch.no_grad():
        node_embs, learned_adj = gat(node_feats, adj)
        graph_emb = pool_graph_embedding(node_embs)

    return {
        "node_embs_shape": list(node_embs.shape),
        "learned_adj_shape": list(learned_adj.shape),
        "graph_emb_shape": list(graph_emb.shape),
        "learned_adj_sample": nan_to_float(learned_adj[0].cpu().numpy()),
        "node_embs_pca": nan_to_float(
            PCA(n_components=2).fit_transform(node_embs[0].cpu().numpy())
        ),
        "attention_stats": {
            "mean": float(learned_adj[0].mean().item()),
            "std": float(learned_adj[0].std().item()),
            "min": float(learned_adj[0].min().item()),
            "max": float(learned_adj[0].max().item()),
        },
    }


@app.post("/api/model/sde-forward")
async def sde_forward(config: ModelConfig = None):
    """Run Neural SDE forward pass on a batch of data."""
    require_loaded_generator()
    if config is None:
        config = ModelConfig()
    data = MODEL_STATE["data"]
    device = MODEL_STATE["device"]

    from models.gat_encoder import DynamicGATEncoder, pool_graph_embedding
    from models.neural_sde import LatentSDEModel

    gat = DynamicGATEncoder(
        in_feats=config.in_feats,
        hidden=config.latent_dim,
        out_feats=config.latent_dim,
        heads=config.gat_heads,
        dropout=config.gat_dropout,
    ).to(device)
    sde_model = LatentSDEModel(
        n_stocks=config.n_stocks,
        latent_dim=config.latent_dim,
        T=config.T,
        sde_hidden_dim=config.sde_hidden_dim,
    ).to(device)
    gat.eval()
    sde_model.eval()

    n = min(config.batch_size, len(data["windows"]))
    x = torch.from_numpy(data["windows"][:n]).float().to(device)
    adj = torch.nan_to_num(torch.from_numpy(data["adj_matrices"][:n]).float(),
                           nan=0.0, posinf=1.0, neginf=0.0).to(device)
    node_feats = torch.nan_to_num(
        torch.from_numpy(data["node_features"][:n]).float(),
        nan=0.0, posinf=0.0, neginf=0.0
    ).to(device)

    with torch.no_grad():
        node_embs, learned_adj = gat(node_feats, adj)
        graph_emb = pool_graph_embedding(node_embs)
        r_hat, zs = sde_model(x, graph_emb)

    ts = list(range(config.T))

    return {
        "z_path_sample": nan_to_float(zs[0, :, :4].cpu().numpy()),
        "r_hat_sample": nan_to_float(r_hat[0, :, :5].cpu().numpy()),
        "x_sample": nan_to_float(x[0, :, :5].cpu().numpy()),
        "ts": ts,
        "latent_stats": {
            "mean": float(zs.mean().item()),
            "std": float(zs.std().item()),
            "max": float(zs.abs().max().item()),
        },
        "returns_stats": {
            "mean": float(r_hat.mean().item()),
            "std": float(r_hat.std().item()),
        },
    }


@app.post("/api/scenario/generate")
async def generate_scenario(config: ModelConfig = None, noise_scale: float = 1.0, n_scenarios: int = 5, stock_idx: int = 0):
    """Generate synthetic scenarios from the CG-NSDE model."""
    require_loaded_generator()
    if config is None:
        config = ModelConfig()
    data = MODEL_STATE["data"]
    device = MODEL_STATE["device"]

    from models.gat_encoder import DynamicGATEncoder, pool_graph_embedding
    from models.neural_sde import LatentSDEModel

    gat = DynamicGATEncoder(
        in_feats=config.in_feats,
        hidden=config.latent_dim,
        out_feats=config.latent_dim,
        heads=config.gat_heads,
        dropout=config.gat_dropout,
    ).to(device)
    sde = LatentSDEModel(
        n_stocks=config.n_stocks,
        latent_dim=config.latent_dim,
        T=config.T,
        sde_hidden_dim=config.sde_hidden_dim,
    ).to(device)
    gat.eval()
    sde.eval()

    n = min(n_scenarios, len(data["windows"]))
    x = torch.from_numpy(data["windows"][:n]).float().to(device)
    adj = torch.nan_to_num(torch.from_numpy(data["adj_matrices"][:n]).float(),
                           nan=0.0, posinf=1.0, neginf=0.0).to(device)
    node_feats = torch.nan_to_num(
        torch.from_numpy(data["node_features"][:n]).float(),
        nan=0.0, posinf=0.0, neginf=0.0
    ).to(device)

    with torch.no_grad():
        node_embs, _ = gat(node_feats, adj)
        graph_emb = pool_graph_embedding(node_embs)

        original_g = sde.sde.g

        def scaled_g(t, y):
            return original_g(t, y) * noise_scale

        sde.sde.g = scaled_g
        r_hat, zs = sde(x, graph_emb)
        sde.sde.g = original_g

    real_returns = x[:, :, stock_idx].cpu().numpy()
    synth_returns = r_hat[:, :, stock_idx].cpu().numpy()

    real_flat = real_returns.flatten()
    synth_flat = synth_returns.flatten()

    from scipy.stats import kurtosis as scipy_kurtosis
    real_kurt = float(scipy_kurtosis(real_flat, fisher=True))
    synth_kurt = float(scipy_kurtosis(synth_flat, fisher=True))

    def mean_acf_sq(arr):
        sq = arr ** 2
        acf_vals = []
        for i in range(min(5, arr.shape[-1])):
            col = sq[0, :, i] if arr.ndim == 3 else sq[0, :, i]
            if len(col) > 1:
                c = np.corrcoef(col[1:], col[:-1])
                if c.size > 1 and not np.isnan(c[0, 1]):
                    acf_vals.append(c[0, 1])
        return float(np.mean(acf_vals)) if acf_vals else 0.0

    real_acf = mean_acf_sq(real_returns)
    synth_acf = mean_acf_sq(synth_returns)

    return {
        "scenarios": nan_to_float(synth_returns),
        "real_avg": nan_to_float(real_returns.mean(axis=0)),
        "tickers": [data["tickers"][stock_idx]],
        "stock_idx": stock_idx,
        "stats": {
            "real_kurtosis": real_kurt,
            "synth_kurtosis": synth_kurt,
            "kurtosis_pass": synth_kurt > 3.0,
            "real_acf_sq": real_acf,
            "synth_acf_sq": synth_acf,
            "real_mean": float(real_flat.mean()),
            "synth_mean": float(synth_flat.mean()),
            "real_std": float(real_flat.std()),
            "synth_std": float(synth_flat.std()),
        },
    }


@app.get("/api/evaluation/metrics")
async def get_evaluation_metrics(batch_size: int = 8):
    """Run full evaluation metrics on model output."""
    require_loaded_generator()
    data = MODEL_STATE["data"]
    device = MODEL_STATE["device"]
    cfg = ModelConfig()

    from models.gat_encoder import DynamicGATEncoder, pool_graph_embedding
    from models.neural_sde import LatentSDEModel
    from models.contrastive import SupConLoss, ProjectionHead
    from training.losses import total_loss

    gat = DynamicGATEncoder(
        in_feats=cfg.in_feats, hidden=cfg.latent_dim,
        out_feats=cfg.latent_dim, heads=cfg.gat_heads,
        dropout=cfg.gat_dropout,
    ).to(device)
    sde = LatentSDEModel(
        n_stocks=cfg.n_stocks, latent_dim=cfg.latent_dim,
        T=cfg.T, sde_hidden_dim=cfg.sde_hidden_dim,
    ).to(device)
    proj = ProjectionHead(cfg.latent_dim, cfg.proj_dim).to(device)
    supcon = SupConLoss(temperature=cfg.tau).to(device)

    n = min(batch_size, len(data["windows"]))
    x = torch.from_numpy(data["windows"][:n]).float().to(device)
    adj = torch.nan_to_num(torch.from_numpy(data["adj_matrices"][:n]).float(),
                           nan=0.0, posinf=1.0, neginf=0.0).to(device)
    node_feats = torch.nan_to_num(
        torch.from_numpy(data["node_features"][:n]).float(),
        nan=0.0, posinf=0.0, neginf=0.0
    ).to(device)
    regimes = torch.from_numpy(data["window_regimes"][:n]).float().to(device)

    with torch.no_grad():
        node_embs, learned_adj = gat(node_feats, adj)
        graph_emb = pool_graph_embedding(node_embs)
        r_hat, zs = sde(x, graph_emb)
        z_proj = proj(zs)

    from scipy.stats import kurtosis as scipy_kurtosis

    real_flat = x.detach().cpu().numpy().flatten()
    synth_flat = r_hat.detach().cpu().numpy().flatten()

    real_kurt = float(scipy_kurtosis(real_flat, fisher=True))
    synth_kurt = float(scipy_kurtosis(synth_flat, fisher=True))

    def mean_acf_sq_batch(arr):
        sq = arr ** 2
        acf_vals = []
        for i in range(min(5, arr.shape[-1])):
            col = sq[0, :, i] if arr.ndim == 3 else sq[:, i]
            if len(col) > 1:
                c = np.corrcoef(col[1:], col[:-1])
                if c.size > 1 and not np.isnan(c[0, 1]):
                    acf_vals.append(c[0, 1])
        return float(np.mean(acf_vals)) if acf_vals else 0.0

    real_acf = mean_acf_sq_batch(x.detach().cpu().numpy())
    synth_acf = mean_acf_sq_batch(r_hat.detach().cpu().numpy())

    real_returns = x.detach().cpu().numpy().reshape(-1, x.shape[-1])
    synth_returns = r_hat.detach().cpu().numpy().reshape(-1, r_hat.shape[-1])
    corr_real = np.corrcoef(real_returns, rowvar=False)
    corr_synth = np.corrcoef(synth_returns, rowvar=False)
    corr_error = float(np.linalg.norm(corr_real - corr_synth, "fro")) if corr_real.shape == corr_synth.shape else float("NaN")

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    n_samples = min(50, len(x))
    X_disc = np.concatenate([
        x[:n_samples].reshape(n_samples, -1),
        r_hat[:n_samples].detach().cpu().numpy().reshape(n_samples, -1),
    ])
    y_disc = np.array([1] * n_samples + [0] * n_samples)

    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    disc_scores = cross_val_score(clf, X_disc, y_disc, cv=3, scoring="accuracy")
    disc_score = float(disc_scores.mean())

    normal_idx = np.where(data["window_regimes"][:n] == 0)[0]
    crisis_idx = np.where(data["window_regimes"][:n] == 1)[0]

    if len(normal_idx) > 1 and len(crisis_idx) > 1:
        corr_n = np.corrcoef(r_hat[normal_idx].mean(0).T.cpu().numpy())
        corr_c = np.corrcoef(r_hat[crisis_idx].mean(0).T.cpu().numpy())
        def avg_abs_offdiag(c):
            nn = c.shape[0]
            mask = ~np.eye(nn, dtype=bool)
            return float(np.abs(c[mask]).mean())
        avg_n = avg_abs_offdiag(corr_n)
        avg_c = avg_abs_offdiag(corr_c)
        boost = float((avg_c - avg_n) / (avg_n + 1e-8))
    else:
        avg_n = 0.0
        avg_c = 0.0
        boost = 0.0

    loss_components = {}
    with torch.no_grad():
        l_rec = float(reconstruction_loss(x, r_hat, lambda_styl=0.1).item())
        l_graph = float(graph_consistency_loss(learned_adj, adj).item())
        l_con = float(supcon(z_proj, regimes).item())

    return {
        "statistical": {
            "kurtosis_real": real_kurt,
            "kurtosis_synth": synth_kurt,
            "kurtosis_pass": synth_kurt > 3.0,
            "acf_sq_real": real_acf,
            "acf_sq_synth": synth_acf,
            "corr_error": corr_error,
            "real_mean": float(real_flat.mean()),
            "synth_mean": float(synth_flat.mean()),
            "real_std": float(real_flat.std()),
            "synth_std": float(synth_flat.std()),
        },
        "discriminative": {
            "score": disc_score,
            "pass": disc_score < 0.60,
        },
        "contagion": {
            "normal_avg_corr": avg_n,
            "crisis_avg_corr": avg_c,
            "boost": boost,
            "pass": boost > 0.5,
        },
        "loss_components": {
            "L_reconstruction": l_rec,
            "L_graph": l_graph,
            "L_contrastive": l_con,
            "L_circuit_filter": 0.0,
        },
    }


@app.get("/api/training/simulate")
async def training_simulation(epochs: int = 10, batch_size: int = 8, lr: float = 0.001):
    """Run a quick training simulation and return loss curve."""
    data = MODEL_STATE["data"]
    device = MODEL_STATE["device"]
    cfg = ModelConfig(batch_size=batch_size)

    from models.gat_encoder import DynamicGATEncoder, pool_graph_embedding
    from models.neural_sde import LatentSDEModel
    from models.contrastive import SupConLoss, ProjectionHead
    from training.losses import total_loss

    torch.manual_seed(42)

    gat = DynamicGATEncoder(
        in_feats=cfg.in_feats, hidden=cfg.latent_dim,
        out_feats=cfg.latent_dim, heads=cfg.gat_heads,
        dropout=cfg.gat_dropout,
    ).to(device)
    sde = LatentSDEModel(
        n_stocks=cfg.n_stocks, latent_dim=cfg.latent_dim,
        T=cfg.T, sde_hidden_dim=cfg.sde_hidden_dim,
    ).to(device)
    proj = ProjectionHead(cfg.latent_dim, cfg.proj_dim).to(device)
    supcon = SupConLoss(temperature=cfg.tau).to(device)

    optimizer = torch.optim.Adam(
        list(gat.parameters()) + list(sde.parameters()) + list(proj.parameters()),
        lr=lr,
    )

    losses = []
    n = min(batch_size, len(data["windows"]))

    for epoch in range(epochs):
        idx = torch.randperm(len(data["windows"]))[:n]
        x = torch.from_numpy(data["windows"][idx]).float().to(device)
        adj = torch.nan_to_num(torch.from_numpy(data["adj_matrices"][idx]).float(),
                               nan=0.0, posinf=1.0, neginf=0.0).to(device)
        node_feats = torch.nan_to_num(
            torch.from_numpy(data["node_features"][idx]).float(),
            nan=0.0, posinf=0.0, neginf=0.0
        ).to(device)
        reg = torch.from_numpy(data["window_regimes"][idx]).float().to(device)

        optimizer.zero_grad()
        node_embs, learned_adj = gat(node_feats, adj)
        graph_emb = pool_graph_embedding(node_embs)
        r_hat, zs = sde(x, graph_emb)
        z_proj = proj(zs)

        loss, _ = total_loss(
            r_real=x, r_hat=r_hat,
            a_learned=learned_adj, a_empirical=adj,
            z_proj=z_proj, regime_labels=reg,
            supcon_loss_fn=supcon,
            lam_rec=1.0, lam_graph=0.1, lam_con=0.1, lam_na=0.0,
            lambda_styl=0.1,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(gat.parameters()) + list(sde.parameters()) + list(proj.parameters()),
            1.0,
        )
        optimizer.step()
        losses.append(float(loss.item()))

    del optimizer

    return {
        "epochs": epochs,
        "losses": losses,
        "final_loss": float(losses[-1]),
        "initial_loss": float(losses[0]),
        "trend": "decreasing" if losses[-1] < losses[0] else "flat",
    }


import networkx as nx
from sklearn.decomposition import PCA
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from training.losses import graph_consistency_loss, reconstruction_loss
