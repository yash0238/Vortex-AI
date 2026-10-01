"""Interactive VORTEX-AI Dashboard.

A production-ready Streamlit application demonstrating the full CG-NSDE
pipeline: data exploration, GAT encoder, Neural SDE, scenario generation,
and evaluation metrics.

Run: streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
import torch
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx

from models.gat_encoder import DynamicGATEncoder, pool_graph_embedding
from models.neural_sde import LatentSDEModel, GraphConditionedSDE
from models.contrastive import SupConLoss, ProjectionHead
from training.losses import total_loss, stylized_facts_loss, graph_consistency_loss, reconstruction_loss

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
CKPT_DIR = BASE_DIR / "models" / "checkpoints"

NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "AXISBANK.NS",
    "SBIN.NS", "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "WIPRO.NS", "POWERGRID.NS", "NTPC.NS", "M&M.NS", "TECHM.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "BAJAJ-AUTO.NS", "CIPLA.NS",
    "DRREDDY.NS", "DIVISLAB.NS", "HEROMOTOCO.NS", "ONGC.NS", "COALINDIA.NS",
    "BPCL.NS", "GRASIM.NS", "ADANIPORTS.NS", "EICHERMOT.NS", "APOLLOHOSP.NS",
    "HINDALCO.NS", "TATACONSUM.NS", "BRITANNIA.NS", "SHREECEM.NS", "UPL.NS",
    "BAJAJFINSV.NS", "SBILIFE.NS", "HDFCLIFE.NS", "INDUSINDBK.NS", "LTI.NS",
]

st.set_page_config(
    page_title="VORTEX-AI — CG-NSDE Dashboard",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Cached data and model loading
# ============================================================================

@st.cache_data(show_time=False)
def load_data():
    """Load all data artifacts from data/raw/."""
    windows = np.load(RAW_DIR / "windows.npy", allow_pickle=True)
    adj_matrices = np.load(RAW_DIR / "adj_matrices.npy", allow_pickle=True)
    node_feats = np.load(RAW_DIR / "window_node_features.npy", allow_pickle=True)
    regimes_v2 = np.load(RAW_DIR / "window_regimes_v2.npy", allow_pickle=True)
    returns = np.load(RAW_DIR / "returns.npy", allow_pickle=True)
    regimes = np.load(RAW_DIR / "regimes.npy", allow_pickle=True)

    n = min(len(windows), len(regimes_v2), len(adj_matrices), len(node_feats))
    return {
        "windows": windows[:n],
        "adj_matrices": adj_matrices[:n],
        "node_features": node_feats[:n],
        "window_regimes": regimes_v2[:n],
        "returns": returns[:n],
        "regimes": regimes,
    }


@st.cache_resource(show_time=False)
def get_model_config():
    return {
        "n_stocks": 50,
        "T": 60,
        "latent_dim": 64,
        "proj_dim": 32,
        "in_feats": 6,
        "gat_heads": 4,
        "gat_dropout": 0.1,
        "tau": 0.07,
        "sde_hidden_dim": 128,
    }


@st.cache_resource(show_time=False)
def build_model(cfg, device_str="cpu"):
    """Build a fresh (untrained) VORTEX-AI model."""
    torch.manual_seed(42)
    device = torch.device(device_str)

    gat = DynamicGATEncoder(
        in_feats=cfg["in_feats"],
        hidden=cfg["latent_dim"],
        out_feats=cfg["latent_dim"],
        heads=cfg["gat_heads"],
        dropout=cfg["gat_dropout"],
    ).to(device)

    sde = LatentSDEModel(
        n_stocks=cfg["n_stocks"],
        latent_dim=cfg["latent_dim"],
        T=cfg["T"],
        sde_hidden_dim=cfg["sde_hidden_dim"],
    ).to(device)

    proj = ProjectionHead(cfg["latent_dim"], cfg["proj_dim"]).to(device)
    supcon = SupConLoss(temperature=cfg["tau"]).to(device)

    return gat, sde, proj, supcon, device


# ============================================================================
# Sidebar navigation
# ============================================================================

st.sidebar.title("🌀 VORTEX-AI")
st.sidebar.caption("CG-NSDE: Contrastive Graph-Neural SDE for NSE Stress Testing")

page = st.sidebar.radio(
    "Navigation",
    options=["Overview", "Data Pipeline", "GAT Encoder", "Neural SDE",
             "Scenario Generation", "Evaluation", "Training"],
    index=0,
    format_func=lambda x: {
        "Overview": "📊 Overview",
        "Data Pipeline": "📈 Data Pipeline",
        "GAT Encoder": "🔗 GAT Encoder",
        "Neural SDE": "🧠 Neural SDE",
        "Scenario Generation": "🎲 Scenario Generation",
        "Evaluation": "📏 Evaluation",
        "Training": "🏋️ Training",
    }.get(x, x),
)

st.sidebar.divider()
st.sidebar.markdown("### ⚙️ Model Config")
cfg = get_model_config()
cfg["lr"] = 0.001
cfg["latent_dim"] = st.sidebar.slider("Latent dim", 16, 256, cfg["latent_dim"], step=16)
cfg["gat_heads"] = st.sidebar.slider("GAT heads", 1, 8, cfg["gat_heads"])
cfg["sde_hidden_dim"] = st.sidebar.slider("SDE hidden dim", 32, 256, cfg["sde_hidden_dim"], step=32)
cfg["tau"] = st.sidebar.slider("SupCon τ", 0.01, 0.5, cfg["tau"], step=0.01)
batch_size = st.sidebar.slider("Batch size", 1, 32, 8)

show_data_stats = st.sidebar.checkbox("Show data stats", value=True)


# ============================================================================
# Page: Overview
# ============================================================================

def page_overview(data):
    st.title("🌀 VORTEX-AI: CG-NSDE Dashboard")
    st.markdown("""
    **Contrastive Graph-Neural Stochastic Differential Equations for NSE Stress Testing**

    This dashboard demonstrates the full VORTEX-AI pipeline for generating synthetic
    NIFTY-50 market scenarios, including realistic financial crashes, by combining:
    - Dynamic Graph Attention Networks for cross-asset structure
    - Neural SDEs for temporal path generation
    - Supervised contrastive learning for regime separation
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Windows", f"{len(data['windows']):,}")
    with col2:
        crisis_ratio = data["window_regimes"].mean()
        st.metric("Crisis Windows", f"{crisis_ratio:.1%}")
    with col3:
        st.metric("Assets", str(data["windows"].shape[2]))
    with col4:
        st.metric("Window Length", str(data["windows"].shape[1]) + " days")

    st.markdown("---")
    st.subheader("System Architecture")

    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, fillcolor=lightblue];
        A [label="RAW DATA\\nNIFTY-50 (2010-2024)"];
        B [label="BLOCK 0\\nPreprocessing\\nlog-returns, regimes, windows"];
        C [label="BLOCK 1\\nDynamic GAT Encoder\\nGATConv(4 heads) → attention"];
        D [label="BLOCK 2\\nLatent Encoder\\nGRU → z0"];
        E [label="BLOCK 3\\nNeural SDE\\ntorchsde Euler-Maruyama"];
        F [label="BLOCK 4\\nSupCon Regime Head\\nProjection + InfoNCE"];
        G [label="BLOCK 5\\nDecoder + Circuit Filter\\nMLP decode + NSE bands"];
        H [label="OUTPUT\\nSynthetic Scenarios"];
        A -> B -> C -> D -> E -> F -> G -> H;
        C -> E [label="G_t (graph emb)", style=dashed, color=red];
    }
    """, use_container_width=True)

    st.markdown("---")

    # Cumulative returns chart
    cum_returns = np.cumsum(data["returns"], axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(cum_returns))),
        y=cum_returns[:, 0],
        mode="lines",
        name=NIFTY50_TICKERS[0],
        line=dict(color="blue"),
    ))
    crisis_mask = data["regimes"] == 1
    crisis_idx = np.where(crisis_mask)[0]
    for idx in crisis_idx:
        fig.add_vrect(
            x0=idx, x1=idx + 1,
            fillcolor="red", opacity=0.15, layer="below", line_width=0,
        )
    fig.update_layout(
        title="Cumulative Returns — NIFTY-50 (first stock shown)",
        xaxis_title="Trading Days",
        yaxis_title="Cumulative Return",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Page: Data Pipeline
# ============================================================================

def page_data_pipeline(data):
    st.title("📈 Data Pipeline")
    st.markdown("### Data Artifacts Overview")

    windows = data["windows"]
    regimes = data["window_regimes"]
    adj = data["adj_matrices"]
    node_feats = data["node_features"]

    tab1, tab2, tab3, tab4 = st.tabs(["Returns", "Regimes", "Adjacency Graphs", "Node Features"])

    with tab1:
        st.subheader("Log Returns Distribution")
        fig = px.histogram(
            windows.flatten(),
            nbins=100,
            title="Window Returns Distribution (all windows, all stocks)",
            labels={"value": "Log Return"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Box(y=windows.flatten(), name="All returns"))
        fig2.update_layout(title="Return Box Plot", yaxis_title="Log Return")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Crisis Regime Labeling")
        crisis_count = int(regimes.sum())
        normal_count = len(regimes) - crisis_count
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Crisis Windows", crisis_count)
            st.metric("Normal Windows", normal_count)
        with col2:
            st.metric("Crisis Ratio", f"{regimes.mean():.1%}")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=regimes,
            nbinsx=2,
            name="Windows",
            marker_color=["green", "red"],
            text=["Normal", "Crisis"],
            textposition="outside",
        ))
        fig.update_layout(
            title="Window Regime Distribution",
            xaxis=dict(tickvals=[0, 1], ticktext=["Normal (0)", "Crisis (1)"]),
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Adjacency Matrix (Empirical Graph)")
        win_idx = st.slider("Window index", 0, len(adj) - 1, len(adj) // 2, key="adj_slider")
        adj_win = adj[win_idx]

        fig = go.Figure(data=go.Heatmap(
            z=adj_win,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            x=NIFTY50_TICKERS,
            y=NIFTY50_TICKERS,
        ))
        fig.update_layout(
            title=f"Empirical Adjacency — Window {win_idx} (Regime: {'Crisis' if regimes[win_idx] else 'Normal'})",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        density = (adj_win > 0).sum() / (50 * 50)
        st.metric("Edge Density", f"{density:.1%}")

    with tab4:
        st.subheader("Node Features")
        feats_win = node_feats[win_idx]
        last_feats = feats_win[-1]

        feat_names = ["Return", "|Return|", "Volume", "Sector ID", "Band Limit", "Dist to Band"]
        fig = go.Figure(data=go.Heatmap(
            z=last_feats.T,
            colorscale="Viridis",
            x=NIFTY50_TICKERS,
            y=feat_names,
        ))
        fig.update_layout(
            title=f"Node Features — Window {win_idx} (last timestep)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Feature statistics:**")
        feat_df = pd.DataFrame(last_feats, columns=feat_names, index=NIFTY50_TICKERS[:len(last_feats)])
        st.dataframe(feat_df.describe().T, use_container_width=True)


# ============================================================================
# Page: GAT Encoder
# ============================================================================

def page_gat_encoder(data, cfg, batch_size):
    st.title("🔗 GAT Encoder — Dynamic Graph Attention")

    st.markdown(f"""
    The DynamicGATEncoder uses two GATConv layers with {cfg["gat_heads"]} attention heads
    to encode node features over the empirical adjacency graph. The attention
    weights from the second layer are extracted as the learned adjacency matrix.
    """)

    gat, sde, proj, supcon, device = build_model(cfg)

    windows = torch.from_numpy(data["windows"]).float().to(device)
    adj = torch.nan_to_num(torch.from_numpy(data["adj_matrices"]).float(), nan=0.0, posinf=1.0, neginf=0.0).to(device)
    node_feats = torch.nan_to_num(
        torch.from_numpy(data["node_features"][:, -1, :, :]).float(),
        nan=0.0, posinf=0.0, neginf=0.0
    ).to(device)
    regimes = data["window_regimes"]

    with st.spinner("Running GAT forward pass..."):
        node_embs, learned_adj = gat(node_feats[:batch_size], adj[:batch_size])

    st.success("✅ Forward pass complete")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Input shape", f"{tuple(node_feats[:batch_size].shape)}")
        st.metric("Node embeddings", f"{tuple(node_embs.shape)}")
    with col2:
        st.metric("Learned adjacency", f"{tuple(learned_adj.shape)}")
        st.metric("Graph embedding", f"{pool_graph_embedding(node_embs).shape}")

    # Visualize attention weights
    st.subheader("Attention Weight Visualization")
    adj_idx = st.slider("Batch sample", 0, batch_size - 1, 0, key="gat_slider")
    sample_learned = learned_adj[adj_idx].detach().cpu().numpy()
    sample_empirical = adj[adj_idx].detach().cpu().numpy()

    tab1, tab2 = st.tabs(["Learned Adjacency (Attention)", "Empirical Adjacency"])
    with tab1:
        fig = go.Figure(data=go.Heatmap(
            z=sample_learned,
            colorscale="Viridis",
            x=NIFTY50_TICKERS,
            y=NIFTY50_TICKERS,
        ))
        fig.update_layout(title=f"Learned Adjacency (Attention Weights) — Sample {adj_idx}")
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig2 = go.Figure(data=go.Heatmap(
            z=sample_empirical,
            colorscale="RdBu_r",
            x=NIFTY50_TICKERS,
            y=NIFTY50_TICKERS,
        ))
        fig2.update_layout(title=f"Empirical Adjacency — Sample {adj_idx}")
        st.plotly_chart(fig2, use_container_width=True)

    # Attention non-uniformity check
    st.subheader("Attention Analysis")
    attn_flat = sample_learned.flatten()
    mask = attn_flat > 0
    non_zero = attn_flat[mask] if mask.any() else np.array([0])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Attention std", f"{non_zero.std():.4f}")
        st.metric("Attention range", f"[{non_zero.min():.4f}, {non_zero.max():.4f}]")
    with col2:
        is_uniform = non_zero.std() < 0.01
        st.metric(
            "Non-uniform?", 
            "✅ Yes" if not is_uniform else "❌ No (uniform)",
            help="Attention weights should be non-uniform for meaningful graph learning",
        )

    # Graph consistency loss
    gc_loss = graph_consistency_loss(learned_adj, adj[:batch_size])
    st.metric("Graph Consistency Loss", f"{gc_loss.item():.6f}")


# ============================================================================
# Page: Neural SDE
# ============================================================================

def page_neural_sde(data, cfg, batch_size):
    st.title("🧠 Neural SDE — Graph-Conditioned Latent Paths")

    st.markdown(f"""
    The LatentSDEModel encodes return windows via a GRU, then integrates
    a graph-conditioned Itô SDE using Euler-Maruyama:

    ```
    dX_t = μ_θ(X_t, G_t, t) dt + σ_θ(X_t, G_t, t) dW_t
    ```

    where G_t is the pooled GAT graph embedding. The latent path is then
    decoded back to per-stock return space.
    """)

    gat, sde, proj, supcon, device = build_model(cfg)

    windows = torch.from_numpy(data["windows"]).float().to(device)
    adj = torch.nan_to_num(torch.from_numpy(data["adj_matrices"]).float(), nan=0.0, posinf=1.0, neginf=0.0).to(device)
    node_feats = torch.nan_to_num(
        torch.from_numpy(data["node_features"][:, -1, :, :]).float(),
        nan=0.0, posinf=0.0, neginf=0.0
    ).to(device)

    x = windows[:batch_size]
    nf = node_feats[:batch_size]
    adj_b = adj[:batch_size]

    with st.spinner("Running full forward pass (GAT → SDE → Decoder)..."):
        node_embs, learned_adj = gat(nf, adj_b)
        graph_emb = pool_graph_embedding(node_embs)

        r_hat, zs = sde(x, graph_emb)

    st.success("✅ Full pipeline forward pass complete")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Input x", f"{tuple(x.shape)}")
        st.metric("Graph embedding", f"{tuple(graph_emb.shape)}")
    with col2:
        st.metric("Latent paths zs", f"{tuple(zs.shape)}")
        st.metric("Decoded r_hat", f"{tuple(r_hat.shape)}")
    with col3:
        st.metric("zs finite", f"{torch.isfinite(zs).all().item()}")
        st.metric("r_hat finite", f"{torch.isfinite(r_hat).all().item()}")

    st.subheader("Latent Path Trajectories")
    sample_idx = st.slider("Batch sample", 0, batch_size - 1, 0, key="sde_slider")
    zs_sample = zs[sample_idx, :, :4].detach().cpu().numpy()
    ts = np.linspace(0, 1, cfg["T"])
    fig = go.Figure()
    for i in range(min(4, zs_sample.shape[1])):
        fig.add_trace(go.Scatter(
            x=ts, y=zs_sample[:, i],
            mode="lines", name=f"Latent dim {i}",
            line=dict(width=2),
        ))
    fig.update_layout(
        title=f"Latent SDE Paths — Sample {sample_idx}",
        xaxis_title="Time (normalized)",
        yaxis_title="Latent State",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Reconstructed vs Input Returns")
    col1, col2 = st.columns(2)
    with col1:
        r_sample = x[sample_idx, :, :5].detach().cpu().numpy()
        fig2 = go.Figure()
        for i in range(min(5, r_sample.shape[1])):
            fig2.add_trace(go.Scatter(
                x=list(range(cfg["T"])), y=r_sample[:, i],
                mode="lines", name=NIFTY50_TICKERS[i],
            ))
        fig2.update_layout(title="Input Returns (first 5 stocks)", xaxis_title="Day", height=300)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        rhat_sample = r_hat[sample_idx, :, :5].detach().cpu().numpy()
        fig3 = go.Figure()
        for i in range(min(5, rhat_sample.shape[1])):
            fig3.add_trace(go.Scatter(
                x=list(range(cfg["T"])), y=rhat_sample[:, i],
                mode="lines", name=NIFTY50_TICKERS[i],
            ))
        fig3.update_layout(title="Reconstructed Returns (decoder output)", xaxis_title="Day", height=300)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Latent Space PCA")
    zs_all = zs.detach().cpu().numpy().reshape(-1, zs.shape[-1])
    regimes_batch = data["window_regimes"]
    if len(regimes_batch) >= len(x):
        regimes_batch = regimes_batch[:batch_size]
    else:
        regimes_batch = np.zeros(batch_size)

    pca = PCA(n_components=2)
    zs_2d = pca.fit_transform(zs_all)
    colors = ["blue" if r == 0 else "red" for r in np.repeat(regimes_batch, cfg["T"] * zs.shape[-1] // batch_size)]
    fig4 = go.Figure(data=go.Scatter(
        x=zs_2d[:, 0], y=zs_2d[:, 1],
        mode="markers",
        marker=dict(
            size=5,
            color=colors[:len(zs_2d)],
            opacity=0.5,
        ),
    ))
    fig4.update_layout(
        title=f"Latent Space PCA (explained var: {pca.explained_variance_ratio_.sum():.1%})",
        xaxis_title="PC1", yaxis_title="PC2",
        height=400,
    )
    st.plotly_chart(fig4, use_container_width=True)


# ============================================================================
# Page: Scenario Generation
# ============================================================================

def page_scenario_generation(data, cfg, batch_size):
    st.title("🎲 Scenario Generation")

    st.markdown("""
    Generate synthetic market scenarios from the CG-NSDE model. Adjust the
    noise scale to control the volatility of generated paths.
    """)

    noise_scale = st.slider("Noise scale (SDE diffusion multiplier)", 0.1, 5.0, 1.0, step=0.1)
    n_scenarios = st.slider("Number of scenarios", 1, 10, 5)
    selected_stock = st.selectbox("Stock to visualize", NIFTY50_TICKERS, index=0)
    stock_idx = NIFTY50_TICKERS.index(selected_stock)

    gat, sde, proj, supcon, device = build_model(cfg)

    windows = torch.from_numpy(data["windows"]).float().to(device)
    adj = torch.nan_to_num(torch.from_numpy(data["adj_matrices"]).float(), nan=0.0, posinf=1.0, neginf=0.0).to(device)
    node_feats = torch.nan_to_num(
        torch.from_numpy(data["node_features"][:, -1, :, :]).float(),
        nan=0.0, posinf=0.0, neginf=0.0
    ).to(device)

    x = windows[:n_scenarios]
    nf = node_feats[:n_scenarios]
    adj_b = adj[:n_scenarios]

    with st.spinner("Generating scenarios..."):
        node_embs, _ = gat(nf, adj_b)
        graph_emb = pool_graph_embedding(node_embs)

        # Scale the diffusion network's output by noise_scale
        original_g = sde.sde.g

        def scaled_g(t, y):
            return original_g(t, y) * noise_scale

        sde.sde.g = scaled_g

        r_hat, zs = sde(x, graph_emb)
        sde.sde.g = original_g

    st.success(f"✅ Generated {n_scenarios} scenarios")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Real returns range", f"[{x[:, :, stock_idx].min():.4f}, {x[:, :, stock_idx].max():.4f}]")
    with col2:
        st.metric("Synthetic range", f"[{r_hat[:, :, stock_idx].min():.4f}, {r_hat[:, :, stock_idx].max():.4f}]")

    st.subheader(f"Synthetic Scenarios — {selected_stock}")
    fig = go.Figure()
    for i in range(n_scenarios):
        fig.add_trace(go.Scatter(
            x=list(range(cfg["T"])),
            y=r_hat[i, :, stock_idx].detach().cpu().numpy(),
            mode="lines",
            name=f"Scenario {i+1}",
            opacity=0.8,
        ))
    fig.add_trace(go.Scatter(
        x=list(range(cfg["T"])),
        y=x[:n_scenarios, :, stock_idx].mean(dim=0).detach().cpu().numpy(),
        mode="lines",
        name="Real (avg)",
        line=dict(color="black", width=3, dash="dash"),
    ))
    fig.update_layout(
        title=f"Synthetic {selected_stock} Returns (noise scale = {noise_scale})",
        xaxis_title="Day",
        yaxis_title="Return",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scenario Statistics Comparison")
    real_flat = x.detach().cpu().numpy().flatten()
    synth_flat = r_hat.detach().cpu().numpy().flatten()

    stats_df = pd.DataFrame({
        "Metric": ["Mean", "Std Dev", "Min", "Max", "Skewness", "Kurtosis (excess)"],
        "Real": [
            real_flat.mean(), real_flat.std(), real_flat.min(), real_flat.max(),
            pd.Series(real_flat).skew(), pd.Series(real_flat).kurtosis(),
        ],
        "Synthetic": [
            synth_flat.mean(), synth_flat.std(), synth_flat.min(), synth_flat.max(),
            pd.Series(synth_flat).skew(), pd.Series(synth_flat).kurtosis(),
        ],
    })
    st.dataframe(stats_df, use_container_width=True)

    kurtosis_pass = pd.Series(synth_flat).kurtosis() > 3.0
    st.info(f"**Fat tail check**: Synthetic kurtosis > 3.0 (excess) = {'✅ Pass' if kurtosis_pass else '❌ Fail'}", icon="📊")


# ============================================================================
# Page: Evaluation
# ============================================================================

def page_evaluation(data, cfg, batch_size):
    st.title("📏 Evaluation Metrics")

    st.markdown("""
    Evaluate synthetic scenarios against real data using statistical fidelity,
    discriminative score, and contagion tests.
    """)

    gat, sde, proj, supcon, device = build_model(cfg)

    windows = torch.from_numpy(data["windows"]).float().to(device)
    adj = torch.nan_to_num(torch.from_numpy(data["adj_matrices"]).float(), nan=0.0, posinf=1.0, neginf=0.0).to(device)
    node_feats = torch.nan_to_num(
        torch.from_numpy(data["node_features"][:, -1, :, :]).float(),
        nan=0.0, posinf=0.0, neginf=0.0
    ).to(device)

    x = windows[:batch_size]
    nf = node_feats[:batch_size]
    adj_b = adj[:batch_size]

    with st.spinner("Running model for evaluation..."):
        node_embs, learned_adj = gat(nf, adj_b)
        graph_emb = pool_graph_embedding(node_embs)
        r_hat, zs = sde(x, graph_emb)

    real_returns = x.detach().cpu().numpy()
    synth_returns = r_hat.detach().cpu().numpy()

    tab1, tab2, tab3, tab4 = st.tabs(["Statistical", "Discriminative", "Contagion", "Full Loss"])

    with tab1:
        st.subheader("Statistical Fidelity (Stylized Facts)")

        from scipy.stats import kurtosis as scipy_kurtosis

        real_kurt = scipy_kurtosis(real_returns.reshape(-1), fisher=True)
        synth_kurt = scipy_kurtosis(synth_returns.reshape(-1), fisher=True)

        def mean_acf_sq(arr):
            sq = arr ** 2
            acf_vals = []
            for i in range(min(5, arr.shape[-1])):
                col = sq[0, :, i] if arr.ndim == 3 else sq[0, :, i]
                if len(col) > 1:
                    acf_vals.append(np.corrcoef(col[1:], col[:-1])[0, 1])
            return np.nanmean(acf_vals) if acf_vals else 0

        real_acf = mean_acf_sq(real_returns)
        synth_acf = mean_acf_sq(synth_returns)

        corr_real = np.corrcoef(real_returns[0].T)
        corr_synth = np.corrcoef(synth_returns[0].T)
        corr_error = np.linalg.norm(corr_real - corr_synth, "fro")

        metrics = pd.DataFrame({
            "Metric": ["Kurtosis (excess)", "ACF(squared, lag-1)", "Correlation Error (Frobenius)", "Mean", "Std Dev"],
            "Real": [f"{real_kurt:.4f}", f"{real_acf:.4f}", f"{corr_error:.4f}",
                     f"{real_returns.mean():.4f}", f"{real_returns.std():.4f}"],
            "Synthetic": [f"{synth_kurt:.4f}", f"{synth_acf:.4f}", "N/A",
                         f"{synth_returns.mean():.4f}", f"{synth_returns.std():.4f}"],
            "Target": ["> 3.0", "> 0.12", "< 2.5", "~0", "~real"],
        })
        st.dataframe(metrics, use_container_width=True)

        kurtosis_pass = synth_kurt > 3.0
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Kurtosis > 3.0", "✅" if kurtosis_pass else "❌", f"{synth_kurt:.2f}")
        with col2:
            st.metric("ACF(sq) > 0.12", "✅" if synth_acf > 0.12 else "❌", f"{synth_acf:.4f}")
        with col3:
            st.metric("Corr Error < 2.5", "✅" if corr_error < 2.5 else "❌", f"{corr_error:.2f}")

    with tab2:
        st.subheader("Discriminative Score")

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score

        n_samples = min(100, len(real_returns), len(synth_returns))
        X = np.concatenate([
            real_returns[:n_samples].reshape(n_samples, -1),
            synth_returns[:n_samples].reshape(n_samples, -1),
        ])
        y = np.array([1] * n_samples + [0] * n_samples)

        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        disc_score = scores.mean()

        st.metric("Discriminative Score", f"{disc_score:.3f}",
                  help="Lower is better. Target < 0.60 (ideal ~0.50 = indistinguishable)")
        st.progress(disc_score / 2, text=f"Score: {disc_score:.3f}")

        status = "✅ Pass" if disc_score < 0.60 else "❌ Fail"
        st.info(f"**Status**: {status} (score {disc_score:.3f}, target < 0.60)")

    with tab3:
        st.subheader("Contagion Test")

        regimes = data["window_regimes"]
        normal_idx = np.where(regimes[:batch_size] == 0)[0]
        crisis_idx = np.where(regimes[:batch_size] == 1)[0]

        if len(normal_idx) > 0 and len(crisis_idx) > 0:
            corr_n = np.corrcoef(synth_returns[normal_idx].mean(0).T)
            corr_c = np.corrcoef(synth_returns[crisis_idx].mean(0).T)

            def avg_abs_offdiag(corr):
                n = corr.shape[0]
                mask = ~np.eye(n, dtype=bool)
                return np.abs(corr[mask]).mean() if mask.sum() > 0 else 0

            avg_n = avg_abs_offdiag(corr_n)
            avg_c = avg_abs_offdiag(corr_c)
            boost = (avg_c - avg_n) / (avg_n + 1e-8)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Normal avg |corr|", f"{avg_n:.4f}")
            with col2:
                st.metric("Crisis avg |corr|", f"{avg_c:.4f}")
            with col3:
                st.metric("Correlation Boost", f"{boost:.1%}")

            contagion_pass = boost > 0.5
            st.info(f"**Contagion test**: {'✅ Pass' if contagion_pass else '❌ Fail'} (boost {boost:.1%}, target > 50%)", icon="🔗")
        else:
            st.warning("Not enough normal/crisis samples in batch for contagion test.")

    with tab4:
        st.subheader("Total Loss Components")
        z_proj = proj(zs)
        regimes_batch = torch.from_numpy(data["window_regimes"][:batch_size]).float().to(device)

        try:
            loss, _ = total_loss(
                r_real=x, r_hat=r_hat,
                a_learned=learned_adj, a_empirical=adj_b,
                z_proj=z_proj, regime_labels=regimes_batch,
                supcon_loss_fn=supcon,
                lam_rec=1.0, lam_graph=0.1, lam_con=0.1, lam_na=0.0,
                lambda_styl=0.1,
            )
            l_rec = reconstruction_loss(x, r_hat, lambda_styl=0.1)
            l_graph = graph_consistency_loss(learned_adj, adj_b)
            l_con = supcon(z_proj, regimes_batch)

            loss_df = pd.DataFrame({
                "Component": ["L_reconstruction", "L_graph", "L_contrastive", "L_circuit_filter", "TOTAL"],
                "Value": [
                    f"{l_rec.item():.6f}",
                    f"{l_graph.item():.6f}",
                    f"{l_con.item():.6f}",
                    "0.0 (lam_na=0)",
                    f"{loss.item():.6f}",
                ],
            })
            st.dataframe(loss_df, use_container_width=True)
            st.success(f"Total loss: {loss.item():.6f} (all components finite: {torch.isfinite(loss).item()})")
        except Exception as e:
            st.error(f"Loss computation error: {e}")


# ============================================================================
# Page: Training
# ============================================================================

def page_training(data, cfg):
    st.title("🏋️ Training Dashboard")

    st.markdown("""
    Monitor the CG-NSDE training process. The model combines four loss
    components into a single objective (masterplan Part D.6):

    ```
    L_total = λ_rec · L_rec + λ_graph · L_graph
             + λ_con · L_con + λ_NA · L_circuit
    ```
    """)

    st.subheader("Loss Configuration")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        lam_rec = st.number_input("λ_rec", 0.0, 10.0, 1.0, 0.1, key="lam_rec")
    with col2:
        lam_graph = st.number_input("λ_graph", 0.0, 10.0, 0.1, 0.05, key="lam_graph")
    with col3:
        lam_con = st.number_input("λ_con", 0.0, 10.0, 0.1, 0.05, key="lam_con")
    with col4:
        lam_na = st.number_input("λ_NA (circuit)", 0.0, 10.0, 0.0, 0.1, key="lam_na")

    st.info(f"""
    Current weights: rec={lam_rec}, graph={lam_graph}, con={lam_con}, circuit={lam_na}

    **Training roadmap (masterplan Part C):**
    - Weeks 3-4: GAT Encoder + Static Graph ✅
    - Weeks 5-6: Neural SDE Integration ✅
    - Week 7: SupCon Loss + Regime Separation ✅
    - Week 8: NSE Circuit Filter + Full Forward Pass ✅
    - Week 9: End-to-End Training (in progress)
    """)

    st.subheader("Model Architecture Summary")
    gat, sde, proj, supcon, device = build_model(cfg)

    layers = []
    for name, module in [("GAT Encoder", gat), ("Latent SDE", sde), ("Projection Head", proj), ("SupCon Loss", supcon)]:
        n_params = sum(p.numel() for p in module.parameters()) if hasattr(module, "parameters") else 0
        layers.append({"Component": name, "Parameters": f"{n_params:,}"})

    arch_df = pd.DataFrame(layers)
    st.dataframe(arch_df, use_container_width=True)
    total_params = sum(l["Parameters"].replace(",", "") for l in layers)
    st.metric("Total Parameters", f"{int(total_params):,}")

    st.subheader("Quick Training Simulation")
    n_epochs = st.slider("Simulation epochs", 1, 20, 5)

    if st.button("Run Simulation", type="primary"):
        windows = torch.from_numpy(data["windows"]).float().to(device)
        adj = torch.nan_to_num(torch.from_numpy(data["adj_matrices"]).float(), nan=0.0, posinf=1.0, neginf=0.0).to(device)
        node_feats = torch.nan_to_num(
            torch.from_numpy(data["node_features"][:, -1, :, :]).float(),
            nan=0.0, posinf=0.0, neginf=0.0
        ).to(device)
        regimes = data["window_regimes"]

        progress = st.progress(0)
        loss_chart = st.empty()

        optimizer = torch.optim.Adam(
            list(gat.parameters()) + list(sde.parameters()) +
            list(proj.parameters()),
            lr=cfg["lr"],
        )

        losses = []
        for epoch in range(n_epochs):
            progress.progress((epoch + 1) / n_epochs)

            idx = np.random.choice(len(windows), cfg["batch_size"], replace=False)
            x = windows[idx]
            nf = node_feats[idx]
            adj_b = adj[idx]
            reg = torch.from_numpy(regimes[idx]).float().to(device)

            optimizer.zero_grad()

            node_embs, learned_adj = gat(nf, adj_b)
            graph_emb = pool_graph_embedding(node_embs)
            r_hat, zs = sde(x, graph_emb)
            z_proj = proj(zs)

            loss, _ = total_loss(
                r_real=x, r_hat=r_hat,
                a_learned=learned_adj, a_empirical=adj_b,
                z_proj=z_proj, regime_labels=reg,
                supcon_loss_fn=supcon,
                lam_rec=lam_rec, lam_graph=lam_graph, lam_con=lam_con, lam_na=lam_na,
                lambda_styl=0.1,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(gat.parameters()) + list(sde.parameters()) + list(proj.parameters()),
                1.0,
            )
            optimizer.step()

            losses.append(loss.item())
            loss_chart.line_chart(losses)

        st.success(f"Training simulation complete. Final loss: {losses[-1]:.6f}")
        st.metric("Loss trend", "↓ Decreasing" if losses[-1] < losses[0] else "→ Flat")

        # Show loss curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(losses))), y=losses,
            mode="lines+markers", name="Training Loss",
        ))
        fig.update_layout(title="Training Loss Curve", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)

        del optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ============================================================================
# Main
# ============================================================================

def main():
    try:
        data = load_data()
    except FileNotFoundError as e:
        st.error(f"Data not found: {e}")
        st.caption("Run the data pipeline first: `python data/download.py && python data/preprocess.py`")
        st.stop()

    if page == "Overview":
        page_overview(data)
    elif page == "Data Pipeline":
        page_data_pipeline(data)
    elif page == "GAT Encoder":
        page_gat_encoder(data, cfg, batch_size)
    elif page == "Neural SDE":
        page_neural_sde(data, cfg, batch_size)
    elif page == "Scenario Generation":
        page_scenario_generation(data, cfg, batch_size)
    elif page == "Evaluation":
        page_evaluation(data, cfg, batch_size)
    elif page == "Training":
        page_training(data, cfg)


if __name__ == "__main__":
    main()
