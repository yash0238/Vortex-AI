import { useState } from "react";
import { Zap, TrendingUp, BarChart3, Activity, } from "lucide-react";
import { apiClient } from "../lib/api";
import type { DataStats, ScenarioResult } from "../types";
import StatsGrid from "../components/StatsGrid";
import Chart from "../components/Chart";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";

interface ScenarioGenerationProps {
  stats: DataStats | null;
  health: "loading" | "connected" | "disconnected";
  modelLoaded: boolean;
}

const NIFTY50_TICKERS = [
  "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
  "ICICIBANK", "KOTAKBANK", "BHARTIARTL", "ITC", "AXISBANK",
  "SBIN", "LT", "BAJFINANCE", "HCLTECH", "ASIANPAINT",
  "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
  "WIPRO", "POWERGRID", "NTPC", "M&M", "TECHM",
  "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "BAJAJ-AUTO", "CIPLA",
  "DRREDDY", "DIVISLAB", "HEROMOTOCO", "ONGC", "COALINDIA",
  "BPCL", "GRASIM", "ADANIPORTS", "EICHERMOT", "APOLLOHISP",
  "HINDALCO", "TATACONSUM", "BRITANNIA", "SHREECEM", "UPL",
  "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "INDUSINDBK", "LTI",
];

export default function ScenarioGeneration({ stats, health, modelLoaded }: ScenarioGenerationProps) {
  const [result, setResult] = useState<ScenarioResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [noiseScale, setNoiseScale] = useState(1.0);
  const [nScenarios, setNScenarios] = useState(5);
  const [stockIdx, setStockIdx] = useState(0);

  const handleGenerate = async () => {
    if (health !== "connected") return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.generateScenario(
        { batch_size: nScenarios },
        noiseScale,
        nScenarios,
        stockIdx,
      );
      setResult(res);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  if (health === "disconnected") {
    return <ErrorState title="Backend Disconnected" message="Start the API server to use this page." />;
  }
  if (!modelLoaded) {
    return <ErrorState title="Trained generator not loaded" message="A compatible CG-NSDE checkpoint is required before synthetic scenarios can be generated." />;
  }

  if (!stats) {
    return <LoadingState text="Loading statistics..." />;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold mb-2">Scenario Generation</h1>
        <p className="text-sm text-gray-500">
          Generate synthetic NIFTY-50 market scenarios using the graph-conditioned Neural SDE.
        </p>
      </div>

      {/* Controls */}
      <div className="chart-container">
        <h3 className="text-lg font-medium mb-3">Generation Parameters</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="param-label">Noise Scale</label>
            <input
              type="range" min="0.1" max="5.0" step="0.1" value={noiseScale}
              onChange={(e) => setNoiseScale(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>Low</span>
              <span className="font-medium text-gray-300">{noiseScale.toFixed(1)}</span>
              <span>High</span>
            </div>
          </div>
          <div>
            <label className="param-label">Number of Scenarios</label>
            <input
              type="range" min="1" max="10" value={nScenarios}
              onChange={(e) => setNScenarios(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="text-center text-xs text-gray-500">{nScenarios} scenario{nScenarios > 1 ? "s" : ""}</div>
          </div>
          <div>
            <label className="param-label">Stock</label>
            <select
              value={stockIdx}
              onChange={(e) => setStockIdx(parseInt(e.target.value))}
              className="w-full bg-card-bg border border-border rounded-lg px-2 py-1 text-sm"
            >
              {NIFTY50_TICKERS.map((ticker, i) => (
                <option key={ticker} value={i}>{i + 1}. {ticker}</option>
              ))}
            </select>
          </div>
        </div>

        <button
          onClick={handleGenerate}
          disabled={loading || health !== "connected"}
          className="btn-primary mt-4 flex items-center gap-2"
        >
          <Zap size={16} />
          {loading ? "Generating..." : "Generate Scenarios"}
        </button>
      </div>

      {loading && <LoadingState text="Running Neural SDE inference..." subtext="Integrating graph-conditioned SDE paths" />}

      {error && <ErrorState message={error} onRetry={handleGenerate} />}

      {result && (
        <div className="space-y-6">
          {/* Stats */}
          <StatsGrid cards={[
            { label: "Generated Scenarios", value: result.scenarios.length, icon: <Zap size={20} /> },
            { label: "Real Kurtosis", value: result.stats.real_kurtosis.toFixed(2), icon: <BarChart3 size={20} />, description: "Fat-tailed distribution check" },
            { label: "Synth Kurtosis", value: result.stats.synth_kurtosis.toFixed(2), icon: <TrendingUp size={20} />, description: result.stats.kurtosis_pass ? "✅ Pass (>3.0)" : "❌ Fail (<3.0)" },
            { label: "Vol Clustering", value: result.stats.synth_acf_sq.toFixed(4), icon: <Activity size={20} />, description: "ACF of squared returns" },
          ]} />

          {/* Scenario chart */}
          <div className="chart-container">
            <h3 className="text-lg font-medium mb-3">
              Synthetic Scenarios — {NIFTY50_TICKERS[result.stock_idx]}
              {" "} (noise scale: {noiseScale.toFixed(1)})
            </h3>
            <Chart
              type="line"
              height={320}
              data={{
                datasets: [
                  ...result.scenarios.map((_, i: number) => ({
                    label: `Scenario ${i + 1}`,
                    data: result.scenarios[i],
                    borderColor: `hsl(${(i * 60) % 360}, 70%, 55%)`,
                    borderWidth: 1.5,
                    opacity: 0.8,
                    pointRadius: 0,
                  })),
                  {
                    label: "Real (avg)",
                    data: result.real_avg,
                    borderColor: "#ef4444",
                    borderWidth: 2.5,
                    borderDash: [5, 5],
                    pointRadius: 0,
                  },
                ],
              }}
              options={{
                plugins: {
                  legend: { position: "bottom" },
                },
                scales: {
                  x: { title: { display: true, text: "Day" } },
                  y: { title: { display: true, text: "Return" } },
                },
              }}
            />
          </div>

          {/* Statistics comparison */}
          <div className="chart-container">
            <h3 className="text-lg font-medium mb-3">Real vs Synthetic Statistics</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {[
                { label: "Mean", real: result.stats.real_mean, synth: result.stats.synth_mean },
                { label: "Std Dev", real: result.stats.real_std, synth: result.stats.synth_std },
                { label: "Kurtosis", real: result.stats.real_kurtosis, synth: result.stats.synth_kurtosis },
                { label: "ACF(sq)", real: result.stats.real_acf_sq, synth: result.stats.synth_acf_sq },
                { label: "Kurtosis Pass", real: "—", synth: result.stats.kurtosis_pass ? "✅" : "❌" },
              ].map((stat) => (
                <div key={stat.label} className="text-center">
                  <p className="text-xs text-gray-500">{stat.label}</p>
                  <div className="grid grid-cols-2 gap-1 mt-1">
                    <div>
                      <p className="text-xs text-gray-600">Real</p>
                      <p className="text-sm font-medium">
                        {typeof stat.real === "number" ? stat.real.toFixed(4) : stat.real}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-600">Synth</p>
                      <p className="text-sm font-medium text-primary">
                        {typeof stat.synth === "number" ? stat.synth.toFixed(4) : stat.synth}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
