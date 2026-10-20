import { useEffect, useState } from "react";
import { BarChart3, Shield, Target, Brain, TrendingUp, Activity, } from "lucide-react";
import { apiClient } from "../lib/api";
import type { DataStats, EvaluationResult } from "../types";
import StatsGrid from "../components/StatsGrid";
import Chart from "../components/Chart";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";

interface EvaluationProps {
  stats: DataStats | null;
  health: "loading" | "connected" | "disconnected";
  modelLoaded: boolean;
}

export default function Evaluation({ stats, health, modelLoaded }: EvaluationProps) {
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [batchSize] = useState(8);

  const runEvaluation = async () => {
    if (health !== "connected") return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.getEvaluation(batchSize);
      setResult(res);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (health === "connected" && modelLoaded) {
      runEvaluation();
    }
  }, [health, modelLoaded]);

  if (health === "disconnected") {
    return <ErrorState title="Backend Disconnected" message="Start the API server to use this page." />;
  }
  if (!modelLoaded) {
    return <ErrorState title="Trained model not loaded" message="Live model evaluation is disabled because this workspace has no compatible CG-NSDE generator checkpoint. See the saved three-seed evaluation summary for validated results." />;
  }

  if (!stats || (loading && !result)) {
    return <LoadingState text="Running evaluation metrics..." subtext="Statistical fidelity, discriminative score, and contagion tests" />;
  }

  if (error) {
    return <ErrorState message={error} onRetry={runEvaluation} />;
  }

  if (!result) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Evaluation Metrics</h1>
        <button onClick={runEvaluation} className="btn-primary">
          Run Evaluation
        </button>
      </div>
    );
  }

  const s = result.statistical;
  const d = result.discriminative;
  const c = result.contagion;
  const l = result.loss_components;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold mb-2">Evaluation Metrics</h1>
          <p className="text-sm text-gray-500">
            Statistical fidelity, discriminative score, contagion analysis, and loss components.
          </p>
        </div>
        <button onClick={runEvaluation} disabled={loading} className="btn-secondary flex items-center gap-2">
          <Activity size={16} />
          {loading ? "Running..." : "Re-run"}
        </button>
      </div>

      {/* Tier 1: Statistical Fidelity */}
      <div className="chart-container">
        <h3 className="text-lg font-medium mb-3 flex items-center gap-2">
          <BarChart3 size={20} /> Tier 1 — Statistical Fidelity
        </h3>
        <StatsGrid cards={[
          {
            label: "Kurtosis (real)", value: s.kurtosis_real.toFixed(2),
            icon: <BarChart3 size={20} />,
            subtext: "Excess kurtosis of real returns",
          },
          {
            label: "Kurtosis (synth)",
            value: s.kurtosis_synth.toFixed(2),
            icon: <TrendingUp size={20} />,
            subtext: s.kurtosis_pass ? "✅ Fat-tailed (>3.0)" : "❌ Gaussian (<3.0)",
          },
          {
            label: "ACF² (real)", value: s.acf_sq_real.toFixed(4),
            icon: <Activity size={20} />,
            subtext: "Volatility clustering",
          },
          {
            label: "ACF² (synth)", value: s.acf_sq_synth.toFixed(4),
            icon: <Activity size={20} />,
            subtext: "Synthetic volatility clustering",
          },
        ]} />
        <StatsGrid cards={[
          { label: "Correlation Error", value: s.corr_error.toFixed(2), icon: <Target size={20} />, subtext: "Frobenius norm (lower is better)" },
          { label: "Real Mean", value: s.real_mean.toFixed(4), icon: <BarChart3 size={20} /> },
          { label: "Syn Mean", value: s.synth_mean.toFixed(4), icon: <TrendingUp size={20} /> },
          { label: "Real Std", value: s.real_std.toFixed(4), icon: <Activity size={20} /> },
        ]} />
      </div>

      {/* Tier 2: Cross-Sectional Structure */}
      <div className="chart-container">
        <h3 className="text-lg font-medium mb-3 flex items-center gap-2">
          <Brain size={20} /> Tier 2 — Cross-Sectional Structure
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="metric-card">
            <p className="stat-label">Crisis Correlation Boost</p>
            <p className="stat-value text-2xl">{(c.boost * 100).toFixed(1)}%</p>
            <p className="text-xs mt-1 text-gray-600">
              Crisis vs normal avg correlation
            </p>
            <div className="mt-2">
              <p className={`text-xs ${c.pass ? "status-pass" : "status-fail"}`}>
                {c.pass ? "✅ Pass (>50%)" : "❌ Fail (≤50%)"}
              </p>
            </div>
          </div>
          <div className="metric-card">
            <p className="stat-label">Normal avg |corr|</p>
            <p className="stat-value">{c.normal_avg_corr.toFixed(4)}</p>
          </div>
          <div className="metric-card">
            <p className="stat-label">Crisis avg |corr|</p>
            <p className="stat-value">{c.crisis_avg_corr.toFixed(4)}</p>
          </div>
        </div>
      </div>

      {/* Tier 3: Discriminative Score */}
      <div className="chart-container">
        <h3 className="text-lg font-medium mb-3 flex items-center gap-2">
          <Shield size={20} /> Tier 3 — Discriminative Score
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="metric-card">
            <p className="stat-label">Discriminative Score</p>
            <p className="stat-value text-2xl">{d.score.toFixed(3)}</p>
            <p className="text-xs mt-1 text-gray-600">
              GBC classifier accuracy (real vs synthetic)
            </p>
            <div className="mt-2">
              <p className={`text-xs ${d.pass ? "status-pass" : "status-fail"}`}>
                {d.pass ? "✅ Pass (<0.60)" : "❌ Fail (≥0.60)"}
              </p>
            </div>
          </div>
          <div className="metric-card md:col-span-2">
            <Chart
              type="bar"
              height={150}
              data={{
                labels: ["Real", "Synthetic"],
                datasets: [{
                  label: "Accuracy",
                  data: [d.score, 1 - d.score],
                  backgroundColor: ["rgba(59, 130, 246, 0.6)", "rgba(239, 68, 68, 0.6)"],
                }],
              }}
              options={{
                indexAxis: "y",
                plugins: { legend: { display: false } },
                scales: { x: { min: 0, max: 1 } },
              }}
            />
          </div>
        </div>
      </div>

      {/* Loss Components */}
      <div className="chart-container">
        <h3 className="text-lg font-medium mb-3 flex items-center gap-2">
          <Target size={20} /> Loss Components
        </h3>
        <Chart
          height={200}
          data={{
            labels: ["L_rec", "L_graph", "L_con", "L_circuit"],
            datasets: [{
              label: "Loss value",
              data: [l.L_reconstruction, l.L_graph, l.L_contrastive, l.L_circuit_filter],
              backgroundColor: ["#3b82f6", "#a855f7", "#f59e0b", "#10b981"],
            }],
          }}
          options={{
            plugins: { legend: { display: false } },
            scales: { y: { type: "logarithmic" } },
          }}
        />
        <div className="grid grid-cols-4 gap-2 mt-4 text-center">
          <div><p className="text-xs text-gray-500">L_rec</p><p className="font-medium">{l.L_reconstruction.toFixed(4)}</p></div>
          <div><p className="text-xs text-gray-500">L_graph</p><p className="font-medium">{l.L_graph.toFixed(4)}</p></div>
          <div><p className="text-xs text-gray-500">L_con</p><p className="font-medium">{l.L_contrastive.toFixed(4)}</p></div>
          <div><p className="text-xs text-gray-500">L_circuit</p><p className="font-medium">{l.L_circuit_filter.toFixed(4)}</p></div>
        </div>
      </div>

      {/* Summary */}
      <div className="chart-container">
        <h3 className="text-lg font-medium mb-3">Evaluation Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { label: "Kurtosis > 3.0", pass: s.kurtosis_pass, detail: s.kurtosis_synth.toFixed(2) },
            { label: "ACF² > 0.12", pass: s.acf_sq_synth > 0.12, detail: s.acf_sq_synth.toFixed(4) },
            { label: "Disc. Score < 0.60", pass: d.pass, detail: `${d.score.toFixed(3)}` },
            { label: "Contagion Boost > 50%", pass: c.pass, detail: `${(c.boost * 100).toFixed(1)}%` },
          ].map((item) => (
            <div key={item.label} className={`metric-card text-center ${item.pass ? "border-green-500/30" : "border-red-500/30"}`}>
              <p className="stat-label">{item.label}</p>
              <p className={`stat-value mt-1 ${item.pass ? "text-green-400" : "text-red-400"}`}>
                {item.pass ? "✅ PASS" : "❌ FAIL"}
              </p>
              <p className="text-xs text-gray-600 mt-1">{item.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
