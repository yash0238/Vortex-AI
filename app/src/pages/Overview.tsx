import { useEffect, useState } from "react";
import { Globe, Activity, TrendingUp, Calendar } from "lucide-react";
import { apiClient } from "../lib/api";
import type { DataStats } from "../types";
import StatsGrid from "../components/StatsGrid";
import Chart from "../components/Chart";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";

interface OverviewProps {
  stats: DataStats | null;
  health: "loading" | "connected" | "disconnected";
}

export default function Overview({ stats, health }: OverviewProps) {
  const [returnsData, setReturnsData] = useState<any>(null);

  useEffect(() => {
    if (stats) {
      apiClient.getReturns(0, 0)
        .then(setReturnsData)
        .catch(() => {});
    }
  }, [stats]);

  const statusColor = health === "connected" ? "text-green-400" : health === "loading" ? "text-yellow-400" : "text-red-400";

  if (health === "disconnected") {
    return (
      <div className="space-y-4">
        <ErrorState
          title="Backend Disconnected"
          message="The FastAPI backend is not running. Start it with: `python api/main.py`"
        />
      </div>
    );
  }

  if (!stats) {
    return <LoadingState text="Loading project statistics..." />;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">VORTEX-AI: CG-NSDE Dashboard</h1>
          <p className="text-sm text-gray-500 mt-1">
            Contrastive Graph-Neural Stochastic Differential Equations for NSE Stress Testing
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${health === "connected" ? "bg-green-400" : "bg-yellow-400 animate-pulse"}`}></div>
          <span className={`text-sm ${statusColor}`}>Backend: {health}</span>
        </div>
      </div>

      {/* Key Metrics */}
      <StatsGrid cards={[
        { label: "Total Windows", value: stats.n_windows.toLocaleString(), icon: <Calendar size={20} />, description: "60-day sliding windows" },
        { label: "Assets", value: stats.n_stocks, icon: <Globe size={20} />, description: "NIFTY-50 stocks" },
        { label: "Crisis Windows", value: `${stats.n_crisis} (${stats.crisis_ratio.toFixed(1)}%)`, icon: <TrendingUp size={20} />, description: "Regime-labeled crisis periods", change: `${stats.crisis_ratio.toFixed(1)}%`, changeType: "neutral" },
        { label: "Total Trading Days", value: stats.daily_total_days.toLocaleString(), icon: <Activity size={20} />, description: "2010-2024 data range" },
      ]} />

      {/* Architecture Diagram */}
      <div className="chart-container mt-6">
        <h3 className="text-lg font-medium mb-3">System Architecture (CG-NSDE)</h3>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                <th className="border border-border px-3 py-2 text-left bg-card-bg/50">Block</th>
                <th className="border border-border px-3 py-2 text-left bg-card-bg/50">Component</th>
                <th className="border border-border px-3 py-2 text-left bg-card-bg/50">Description</th>
                <th className="border border-border px-3 py-2 text-left bg-card-bg/50">Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border border-border px-3 py-2">0</td>
                <td className="border border-border px-3 py-2">Preprocessing</td>
                <td className="border border-border px-3 py-2">log-returns, regime labeling, 60-day windows, Pearson+Granger adjacency</td>
                <td className="border border-border px-3 py-2"><span className="text-green-400">Complete</span></td>
              </tr>
              <tr>
                <td className="border border-border px-3 py-2">1</td>
                <td className="border border-border px-3 py-2">Dynamic GAT Encoder</td>
                <td className="border border-border px-3 py-2">2-layer GATConv (4 heads), attention-based learned adjacency</td>
                <td className="border border-border px-3 py-2"><span className="text-green-400">Complete</span></td>
              </tr>
              <tr>
                <td className="border border-border px-3 py-2">2</td>
                <td className="border border-border px-3 py-2">Latent Encoder</td>
                <td className="border border-border px-3 py-2">GRU encoder z0 initial condition for SDE</td>
                <td className="border border-border px-3 py-2"><span className="text-green-400">Implemented</span></td>
              </tr>
              <tr>
                <td className="border border-border px-3 py-2">3</td>
                <td className="border border-border px-3 py-2">Neural SDE</td>
                <td className="border border-border px-3 py-2">torchsde SDEIto (Euler-Maruyama), graph-conditioned drift/diffusion</td>
                <td className="border border-border px-3 py-2"><span className="text-green-400">Implemented</span></td>
              </tr>
              <tr>
                <td className="border border-border px-3 py-2">4</td>
                <td className="border border-border px-3 py-2">SupCon Regime Head</td>
                <td className="border border-border px-3 py-2">ProjectionHead + SupConLoss (Khosla et al. 2020)</td>
                <td className="border border-border px-3 py-2"><span className="text-green-400">Implemented</span></td>
              </tr>
              <tr>
                <td className="border border-border px-3 py-2">5</td>
                <td className="border border-border px-3 py-2">Circuit Filter + Decoder</td>
                <td className="border border-border px-3 py-2">MLP decoder to returns, NSE band constraints (+-5/10/20%)</td>
                <td className="border border-border px-3 py-2"><span className="text-green-400">Implemented</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Cumulative Returns Chart */}
      {returnsData && (
        <div className="chart-container">
          <h3 className="text-lg font-medium mb-3">
            Cumulative Returns - {returnsData.ticker}
          </h3>
          <Chart
            data={{
              datasets: [
                {
                  label: returnsData.ticker,
                  data: returnsData.cumulative_returns,
                  borderColor: "#3b82f6",
                  backgroundColor: "rgba(59, 130, 246, 0.1)",
                  borderWidth: 2,
                  fill: true,
                  pointRadius: 0,
                },
              ],
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {},
              scales: {
                x: { title: { display: true, text: "Trading Days" } },
                y: { title: { display: true, text: "Cumulative Return" } },
              },
            }}
            height={320}
          />
        </div>
      )}

      {/* Crisis days overlay */}
      {returnsData && returnsData.regimes && (
        <div className="text-xs text-gray-500 mt-2">
          Crisis days marked in red. Total crisis days:{" "}
          {returnsData.regimes.filter((r: number) => r === 1).length} of {returnsData.regimes.length}
        </div>
      )}
    </div>
  );
}