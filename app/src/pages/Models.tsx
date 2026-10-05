import { useEffect, useState } from "react";
import { GitBranch, Brain, BarChart3, Activity } from "lucide-react";
import { apiClient } from "../lib/api";
import type { GatForwardResult, SdeForwardResult, ModelConfig } from "../types";
import StatsGrid from "../components/StatsGrid";
import Heatmap from "../components/Heatmap";
import Chart from "../components/Chart";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";

const NIFTY50_TICKERS = [
  "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
  "ICICIBANK", "KOTAKBANK", "BHARTIARTL", "ITC", "AXISBANK",
  "SBIN", "LT", "BAJFINANCE", "HCLTECH", "ASIANPAINT",
  "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
  "WIPRO", "POWERGRID", "NTPC", "M&M", "TECHM",
  "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "BAJAJ-AUTO", "CIPLA",
  "DRREDDY", "DIVISLAB", "HEROMOTOCO", "ONGC", "COALINDIA",
  "BPCL", "GRASIM", "ADANIPORTS", "EICHERMOT", "APOLLOHOSP",
  "HINDALCO", "TATACONSUM", "BRITANNIA", "SHREECEM", "UPL",
  "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "INDUSINDBK", "LTI",
];



interface ModelsProps {
  health: "loading" | "connected" | "disconnected";
  tab?: "gat" | "sde";
}

const defaultConfig: ModelConfig = {
  n_stocks: 50,
  T: 60,
  latent_dim: 64,
  proj_dim: 32,
  in_feats: 6,
  gat_heads: 4,
  gat_dropout: 0.1,
  tau: 0.07,
  lr: 0.001,
  sde_hidden_dim: 128,
  batch_size: 8,
};

export default function Models({ health, tab = "gat" }: ModelsProps) {
  const [activeTab, setActiveTab] = useState(tab);
  const [gatResult, setGatResult] = useState<GatForwardResult | null>(null);
  const [sdeResult, setSdeResult] = useState<SdeForwardResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<ModelConfig>(defaultConfig);

  useEffect(() => {
    setActiveTab(tab);
  }, [tab]);

  const runGatForward = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiClient.gatForward(config);
      setGatResult(result);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const runSdeForward = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiClient.sdeForward(config);
      setSdeResult(result);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  if (health === "disconnected") {
    return <ErrorState title="Backend Disconnected" message="Start the FastAPI server to use this page." />;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold mb-2">Model Architecture</h1>
        <p className="text-sm text-gray-500">
          Dynamic GAT Encoder + Neural SDE + SupCon Regime Head
        </p>
      </div>

      {/* Config panel */}
      <div className="chart-container">
        <h3 className="text-lg font-medium mb-3">Model Configuration</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div>
            <label className="param-label">Latent dim</label>
            <input
              type="range" min="16" max="256" value={config.latent_dim || 64}
              onChange={(e) => setConfig({ ...config, latent_dim: parseInt(e.target.value) })}
              className="w-full"
            />
            <span className="text-xs text-gray-500">{config.latent_dim}</span>
          </div>
          <div>
            <label className="param-label">GAT heads</label>
            <input
              type="range" min="1" max="8" value={config.gat_heads || 4}
              onChange={(e) => setConfig({ ...config, gat_heads: parseInt(e.target.value) })}
              className="w-full"
            />
            <span className="text-xs text-gray-500">{config.gat_heads}</span>
          </div>
          <div>
            <label className="param-label">SDE hidden</label>
            <input
              type="range" min="32" max="256" step="32" value={config.sde_hidden_dim || 128}
              onChange={(e) => setConfig({ ...config, sde_hidden_dim: parseInt(e.target.value) })}
              className="w-full"
            />
            <span className="text-xs text-gray-500">{config.sde_hidden_dim}</span>
          </div>
          <div>
            <label className="param-label">SupCon τ</label>
            <input
              type="range" min="0.01" max="0.5" step="0.01" value={config.tau || 0.07}
              onChange={(e) => setConfig({ ...config, tau: parseFloat(e.target.value) })}
              className="w-full"
            />
            <span className="text-xs text-gray-500">{config.tau}</span>
          </div>
          <div>
            <label className="param-label">Batch size</label>
            <input
              type="range" min="1" max="32" value={config.batch_size || 8}
              onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
              className="w-full"
            />
            <span className="text-xs text-gray-500">{config.batch_size}</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-border">
        <div className="flex gap-4">
          <button
            onClick={() => setActiveTab("gat")}
            className={`flex items-center gap-2 px-4 py-2 border-b-2 ${
              activeTab === "gat" ? "border-primary text-primary" : "border-transparent text-gray-500"
            }`}
          >
            <GitBranch size={16} /> GAT Encoder
          </button>
          <button
            onClick={() => setActiveTab("sde")}
            className={`flex items-center gap-2 px-4 py-2 border-b-2 ${
              activeTab === "sde" ? "border-primary text-primary" : "border-transparent text-gray-500"
            }`}
          >
            <Brain size={16} /> Neural SDE
          </button>
        </div>
      </div>

      {activeTab === "gat" && (
        <div className="space-y-4">
          {!gatResult && (
            <button
              onClick={runGatForward}
              disabled={loading}
              className="btn-primary"
            >
              {loading ? "Running..." : "▶ Run GAT Forward Pass"}
            </button>
          )}

          {loading && <LoadingState text="Running GAT encoder forward pass..." />}

          {error && <ErrorState title="Error" message={error} />}

          {gatResult && (
            <>
              <StatsGrid cards={[
                { label: "Node Embeddings", value: gatResult.node_embs_shape?.join("×") || "—", icon: <GitBranch size={20} /> },
                { label: "Learned Adjacency", value: gatResult.learned_adj_shape?.join("×") || "—", icon: <BarChart3 size={20} /> },
                { label: "Graph Embedding", value: gatResult.graph_emb_shape?.join("×") || "—", icon: <Brain size={20} /> },
                { label: "Attention Std", value: gatResult.attention_stats?.std.toFixed(4) || "—", icon: <Activity size={20} />, description: "Non-uniform check" },
              ]} />

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Heatmap
                  data={gatResult.learned_adj_sample}
                  xLabels={NIFTY50_TICKERS.slice(0, gatResult.learned_adj_sample.length)}
                  yLabels={NIFTY50_TICKERS.slice(0, gatResult.learned_adj_sample.length)}
                  title="Learned Adjacency (Attention Weights)"
                  colorscale="viridis"
                />
                <div className="chart-container">
                  <h3 className="text-lg font-medium mb-3">Attention Weight Distribution</h3>
                  <Chart
                    data={{
                      datasets: [{
                        label: "Attention weights",
                        data: gatResult.learned_adj_sample.flat(),
                        backgroundColor: "rgba(59, 130, 246, 0.6)",
                      }],
                    }}
                    options={{
                      responsive: true,
                      plugins: { legend: { display: false } },
                      scales: { x: { title: { display: true, text: "Edge index" } } },
                    }}
                    height={300}
                    type="bar"
                  />
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {activeTab === "sde" && (
        <div className="space-y-4">
          {!sdeResult && (
            <button
              onClick={runSdeForward}
              disabled={loading}
              className="btn-primary"
            >
              {loading ? "Running..." : "▶ Run Neural SDE Forward Pass"}
            </button>
          )}

          {loading && <LoadingState text="Running SDE forward pass..." />}
          {error && <ErrorState title="Error" message={error} />}

          {sdeResult && (
            <>
              <StatsGrid cards={[
                { label: "Latent dim", value: sdeResult.z_path_sample[0].length, icon: <Brain size={20} /> },
                { label: "Time steps", value: sdeResult.ts.length, icon: <Activity size={20} /> },
                { label: "Latent std", value: sdeResult.latent_stats.std.toFixed(4), icon: <BarChart3 size={20} /> },
                { label: "Returns std", value: sdeResult.returns_stats.std.toFixed(4), icon: <GitBranch size={20} /> },
              ]} />

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="chart-container">
                  <h3 className="text-lg font-medium mb-3">Latent Path Trajectories</h3>
                  <Chart
                    data={{
                      datasets: sdeResult.z_path_sample[0].map((_: any, i: number) => ({
                        label: `Dim ${i}`,
                        data: sdeResult.z_path_sample.map((row: number[]) => row[i]),
                        borderColor: `hsl(${i * 60}, 70%, 50%)`,
                        borderWidth: 2,
                        pointRadius: 0,
                      })),
                    }}
                    options={{
                      responsive: true,
                      plugins: { legend: { position: "top" } },
                      scales: {
                        x: { title: { display: true, text: "Time step" } },
                        y: { title: { display: true, text: "Latent state" } },
                      },
                    }}
                    height={300}
                  />
                </div>
                <div className="chart-container">
                  <h3 className="text-lg font-medium mb-3">Returns: Input vs Reconstructed</h3>
                  <Chart
                    data={{
                      datasets: [
                        ...sdeResult.r_hat_sample[0].map((_: any, i: number) => ({
                          label: `Synth ${NIFTY50_TICKERS[i]}`,
                          data: sdeResult.r_hat_sample.map((row: number[]) => row[i]),
                          borderColor: `hsl(${i * 30}, 70%, 60%)`,
                          borderWidth: 1,
                          pointRadius: 0,
                          opacity: 0.7,
                        })),
                        {
                          label: "Real avg",
                          data: sdeResult.x_sample.map((row: number[]) => row.reduce((a: number, b: number) => a + b, 0) / row.length),
                          borderColor: "#ef4444",
                          borderWidth: 2,
                          borderDash: [5, 5],
                          pointRadius: 0,
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      plugins: { legend: { position: "bottom", labels: { font: { size: 10 } } } },
                      scales: { x: { title: { display: true, text: "Day" } } },
                    }}
                    height={300}
                  />
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
