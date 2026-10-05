import { useEffect, useState } from "react";
import { Database, BarChart3, Activity, Layers } from "lucide-react";
import { apiClient } from "../lib/api";
import type { DataStats,  AdjacencyData, NodeFeaturesData, RegimeDistribution } from "../types";
import StatsGrid from "../components/StatsGrid";
import Heatmap from "../components/Heatmap";
import Chart from "../components/Chart";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";

interface DataPipelineProps {
  stats: DataStats | null;
  health: "loading" | "connected" | "disconnected";
}

export default function DataPipeline({ stats, health }: DataPipelineProps) {
  const [activeTab, setActiveTab] = useState<"regimes" | "adjacency" | "features">("regimes");
  const [regimeDist, setRegimeDist] = useState<RegimeDistribution | null>(null);
  const [windowIdx, setWindowIdx] = useState(0);
  const [adjacency, setAdjacency] = useState<AdjacencyData | null>(null);
  const [nodeFeatures, setNodeFeatures] = useState<NodeFeaturesData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (health === "connected") {
      setLoading(true);
      apiClient.getRegimeDistribution()
        .then(setRegimeDist)
        .catch(setError)
        .finally(() => setLoading(false));
    }
  }, [health]);

  useEffect(() => {
    if (health === "connected") {
      apiClient.getAdjacency(windowIdx)
        .then(setAdjacency)
        .catch(setError);
    }
  }, [health, windowIdx]);

  useEffect(() => {
    if (health === "connected") {
      apiClient.getNodeFeatures(windowIdx)
        .then(setNodeFeatures)
        .catch(setError);
    }
  }, [health, windowIdx]);

  if (health === "disconnected" || error) {
    return <ErrorState title="Connection Error" message={error || "Backend disconnected"} />;
  }

  if (!stats || loading) {
    return <LoadingState text="Loading data pipeline..." />;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold mb-2">Data Pipeline</h1>
        <p className="text-sm text-gray-500">
          NIFTY-50 data preprocessing, regime labeling, and graph construction.
        </p>
      </div>

      {/* Data Stats */}
      <StatsGrid cards={[
        { label: "Windows", value: stats.n_windows.toLocaleString(), icon: <Layers size={20} /> },
        { label: "Window Size", value: `${stats.window_length} days`, icon: <Activity size={20} /> },
        { label: "Stocks", value: stats.n_stocks, icon: <Database size={20} /> },
        { label: "Crisis %", value: `${(stats.crisis_ratio * 100).toFixed(1)}%`, icon: <BarChart3 size={20} /> },
      ]} />

      {/* Tabs */}
      <div className="border-b border-border">
        <div className="flex gap-4">
          {[
            { id: "regimes", label: "Regime Distribution", icon: <Activity size={16} /> },
            { id: "adjacency", label: "Adjacency Graph", icon: <BarChart3 size={16} /> },
            { id: "features", label: "Node Features", icon: <Layers size={16} /> },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-2 border-b-2 transition-colors ${
                activeTab === tab.id
                  ? "border-primary text-primary"
                  : "border-transparent text-gray-500 hover:text-gray-300"
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Window selector */}
      <div className="flex items-center gap-4">
        <label className="text-sm text-gray-400">Window:</label>
        <input
          type="range"
          min="0"
          max={stats.n_windows - 1}
          value={windowIdx}
          onChange={(e) => setWindowIdx(parseInt(e.target.value))}
          className="w-64"
        />
        <span className="text-sm text-gray-300">#{windowIdx}</span>
        {regimeDist && (
          <span className={`text-sm font-medium ${
            regimeDist.crisis_indices.includes(windowIdx) ? "text-red-400" : "text-green-400"
          }`}>
            {regimeDist.crisis_indices.includes(windowIdx) ? "⚠️ Crisis" : "✓ Normal"}
          </span>
        )}
      </div>

      {/* Tab content */}
      {activeTab === "regimes" && regimeDist && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="chart-container">
            <h3 className="text-lg font-medium mb-3">Regime Distribution</h3>
            <Chart
              data={{
                datasets: [
                  {
                    label: "Normal",
                    data: [regimeDist.normal_count],
                    backgroundColor: "#22c55e",
                  },
                  {
                    label: "Crisis",
                    data: [regimeDist.crisis_count],
                    backgroundColor: "#ef4444",
                  },
                ],
              }}
              options={{
                indexAxis: "y",
                responsive: true,
                plugins: { legend: { position: "top" } },
                scales: { x: { stacked: true }, y: { stacked: true } },
              }}
              height={200}
              type="bar"
            />
          </div>
          <div className="chart-container">
            <h3 className="text-lg font-medium mb-3">Regime Statistics</h3>
            <table className="w-full text-sm border-collapse">
              <tbody>
                <tr className="border-b border-border">
                  <td className="py-2 text-gray-400">Total windows</td>
                  <td className="py-2 text-right font-medium">{regimeDist.total_windows.toLocaleString()}</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="py-2 text-gray-400">Crisis windows</td>
                  <td className="py-2 text-right font-medium text-red-400">{regimeDist.crisis_count}</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="py-2 text-gray-400">Normal windows</td>
                  <td className="py-2 text-right font-medium text-green-400">{regimeDist.normal_count}</td>
                </tr>
                <tr>
                  <td className="py-2 text-gray-400">Crisis ratio</td>
                  <td className="py-2 text-right font-medium">{regimeDist.crisis_ratio.toFixed(1)}%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === "adjacency" && adjacency && (
        <Heatmap
          data={adjacency.z}
          xLabels={adjacency.x}
          yLabels={adjacency.y}
          title={`Empirical Adjacency — Window ${windowIdx} (${adjacency.regime})`}
          colorscale="rdylbu"
          maxValue={1}
          minValue={-1}
          height={400}
        />
      )}

      {activeTab === "features" && nodeFeatures && (
        <div className="space-y-4">
          <h3 className="text-lg font-medium">Node Features — Window {windowIdx}</h3>
          <Heatmap
            data={nodeFeatures.data.map((row: number[]) =>
              row.map((v: number) => Number(v.toFixed(3)))
            )}
            xLabels={nodeFeatures.feature_names}
            yLabels={nodeFeatures.tickers.slice(0, 10)}
            title="6-Dimensional Node Features"
            colorscale="viridis"
            height={300}
          />
          <div className="grid grid-cols-6 gap-2 mt-4">
            {nodeFeatures.feature_names.map((name, i) => (
              <div key={i} className="metric-card text-center">
                <span className="stat-label">{name}</span>
                <span className="stat-value block mt-1">{i}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
