import { useState } from "react";
import { Activity,    } from "lucide-react";
import { apiClient } from "../lib/api";
import type { TrainingResult } from "../types";
import Chart from "../components/Chart";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";

interface TrainingProps {
  health: "loading" | "connected" | "disconnected";
}

export default function Training({ health }: TrainingProps) {
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(8);
  const [lr, setLr] = useState(0.001);

  const handleTrain = async () => {
    if (health !== "connected") return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.trainingSimulation(epochs, batchSize, lr);
      setResult(res);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const lossWeights = [
    { label: "L_reconstruction", value: 1.0, desc: "MSE + stylized facts" },
    { label: "L_graph", value: 0.1, desc: "GAT adjacency MSE" },
    { label: "L_contrastive", value: 0.1, desc: "SupCon regime separation" },
    { label: "L_circuit_filter", value: 0.0, desc: "NSE band soft penalty" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold mb-2">Training Dashboard</h1>
        <p className="text-sm text-gray-500">
          Monitor the CG-NSDE training process with the full 4-component loss.
        </p>
      </div>

      {/* Loss equation */}
      <div className="chart-container">
        <h3 className="text-lg font-medium mb-3">Loss Function</h3>
        <div className="text-center my-4">
          <p className="text-lg font-mono">
            L = λ_rec · L_rec + λ_graph · L_graph + λ_con · L_con + λ_NA · L_circuit
          </p>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {lossWeights.map((w) => (
            <div key={w.label} className="metric-card">
              <p className="stat-label">{w.label}</p>
              <p className="stat-value text-primary">λ = {w.value}</p>
              <p className="text-xs text-gray-600 mt-1">{w.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Parameters */}
      <div className="chart-container">
        <h3 className="text-lg font-medium mb-3">Training Parameters</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="param-label">Epochs ({epochs})</label>
            <input
              type="range" min="1" max="50" value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="param-label">Batch size ({batchSize})</label>
            <input
              type="range" min="1" max="32" value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
          <div>
            <label className="param-label">Learning rate ({lr})</label>
            <input
              type="range" min="0.0001" max="0.01" step="0.0001" value={lr}
              onChange={(e) => setLr(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>

        <button
          onClick={handleTrain}
          disabled={loading || health !== "connected"}
          className="btn-primary mt-4 flex items-center gap-2"
        >
          <Activity size={16} />
          {loading ? "Training..." : "Start Training Simulation"}
        </button>
      </div>

      {/* Results */}
      {loading && (
        <LoadingState text="Training in progress..." subtext={`Epoch ${result?.epochs || 0}/${epochs} — gradient clipping active (max_norm=1.0)`} />
      )}

      {error && <ErrorState message={error} onRetry={handleTrain} />}

      {result && !loading && (
        <div className="space-y-6">
          {/* Summary */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="metric-card">
              <p className="stat-label">Initial Loss</p>
              <p className="stat-value">{result.initial_loss.toFixed(4)}</p>
            </div>
            <div className="metric-card">
              <p className="stat-label">Final Loss</p>
              <p className="stat-value">{result.final_loss.toFixed(4)}</p>
            </div>
            <div className="metric-card">
              <p className="stat-label">Trend</p>
              <p className={`stat-value ${result.trend === "decreasing" ? "text-green-400" : "text-yellow-400"}`}>
                {result.trend === "decreasing" ? "↓ Decreasing" : "→ Flat"}
              </p>
            </div>
            <div className="metric-card">
              <p className="stat-label">Epochs Run</p>
              <p className="stat-value">{result.epochs}</p>
            </div>
          </div>

          {/* Loss curve */}
          <div className="chart-container">
            <h3 className="text-lg font-medium mb-3">Training Loss Curve</h3>
            <Chart
              height={320}
              data={{
                labels: result.losses.map((_, i) => i + 1),
                datasets: [{
                  label: "Training Loss",
                  data: result.losses,
                  borderColor: "#3b82f6",
                  backgroundColor: "rgba(59, 130, 246, 0.1)",
                  fill: true,
                  borderWidth: 2,
                  pointRadius: 3,
                  pointBackgroundColor: "#3b82f6",
                }],
              }}
              options={{
                
                scales: { y: { type: "logarithmic" } },
              }}
            />
          </div>

          {/* Training roadmap */}
          <div className="chart-container">
            <h3 className="text-lg font-medium mb-3">Implementation Roadmap</h3>
            <div className="space-y-3">
              {[
                { week: "1-2", task: "Data Pipeline + Baselines", status: "complete" },
                { week: "3-4", task: "GAT Encoder + Static Graph", status: "complete" },
                { week: "5-6", task: "Neural SDE Integration (torchsde)", status: "complete" },
                { week: "7", task: "SupCon Loss + Regime Separation", status: "complete" },
                { week: "8", task: "Circuit Filter + Full Forward Pass", status: "complete" },
                { week: "9", task: "End-to-End Training + Debugging", status: "active" },
                { week: "10", task: "Evaluation (statistical + discriminative)", status: "pending" },
                { week: "11", task: "Ablations + Baseline Comparison", status: "pending" },
                { week: "12", task: "Paper Writing + Submission", status: "pending" },
              ].map((phase) => (
                <div key={phase.week} className="flex items-center gap-4">
                  <span className="w-16 text-sm font-medium text-gray-500">Wk {phase.week}</span>
                  <div className={`flex-1 h-2 rounded-full ${phase.status === "complete" ? "bg-green-500" : phase.status === "active" ? "bg-primary" : "bg-gray-600"}`}></div>
                  <span className={`w-64 text-sm ${phase.status === "complete" ? "text-green-400" : phase.status === "active" ? "text-primary" : "text-gray-500"}`}>
                    {phase.task}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded ${phase.status === "complete" ? "bg-green-500/10 text-green-400" : phase.status === "active" ? "bg-primary/10 text-primary" : "bg-gray-600/10 text-gray-500"}`}>
                    {phase.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
