import { useEffect, useState } from "react";
import { AlertTriangle, ArrowDownRight, ArrowUpRight, BrainCircuit, RefreshCw, ShieldCheck } from "lucide-react";
import { apiClient } from "../lib/api";
import type { ForecastOutlook } from "../types";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function featureLabel(name: string): string {
  return name.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default function ForwardOutlook() {
  const [horizon, setHorizon] = useState(5);
  const [outlook, setOutlook] = useState<ForecastOutlook | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    setError(null);
    apiClient.getForecastOutlook(horizon)
      .then(setOutlook)
      .catch((requestError) => setError(requestError instanceof Error ? requestError.message : "Forecast unavailable."))
      .finally(() => setLoading(false));
  };

  useEffect(load, [horizon]);

  if (loading && !outlook) return <LoadingState text="Calculating forward market outlook..." subtext="Causal historical baseline with chronological holdout validation" />;
  if (error && !outlook) return <ErrorState title="Forward outlook unavailable" message={error} onRetry={load} />;
  if (!outlook) return null;

  const riskTone = outlook.crisis_probability >= 0.6 ? "high" : outlook.crisis_probability >= 0.35 ? "watch" : "low";
  const riskColor = riskTone === "high" ? "text-rose-300" : riskTone === "watch" ? "text-amber-300" : "text-lime-300";

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-widest text-gray-500">Research forecast / decision support</p>
          <h1 className="mt-2 text-2xl font-bold">Forward Market Outlook</h1>
          <p className="mt-1 max-w-3xl text-sm leading-6 text-gray-400">A probabilistic early-warning baseline for future crisis conditions, plus a transparent technical screen for stocks with strong recent directional movement.</p>
        </div>
        <button type="button" className="btn-secondary gap-2" onClick={load} disabled={loading} aria-label="Refresh forecast" title="Refresh forecast">
          <RefreshCw size={16} className={loading ? "animate-spin" : ""} /> Refresh
        </button>
        <label className="forecast-horizon-control">Horizon
          <select value={horizon} onChange={(event) => setHorizon(Number(event.target.value))}>
            {[3, 5, 10, 20].map((days) => <option key={days} value={days}>{days} sessions</option>)}
          </select>
        </label>
      </header>

      {error && <p role="status" className="text-sm text-amber-200">{error}</p>}

      <section className={`forecast-hero forecast-${riskTone}`} aria-label="Forward crisis risk">
        <div>
          <p className="text-xs uppercase tracking-widest text-gray-500">Next {outlook.horizon_days} trading days</p>
          <p className={`mt-2 text-5xl font-semibold tabular-nums ${riskColor}`}>{percent(outlook.crisis_probability)}</p>
          <h2 className={`mt-2 text-xl font-semibold ${riskColor}`}>{outlook.risk_regime}</h2>
          <p className="mt-2 max-w-xl text-sm text-gray-300">{outlook.target_definition}.</p>
        </div>
        <div className="forecast-confidence">
          <BrainCircuit size={22} className={riskColor} />
          <span>Model confidence</span>
          <strong>{percent(outlook.confidence)}</strong>
          <small>Confidence means distance from 50/50, not a guarantee.</small>
        </div>
      </section>

      <section className="forecast-disclaimer" role="note">
        <AlertTriangle size={18} />
        <p>{outlook.method} It does not predict an exact price, and it is not investment advice.</p>
      </section>

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="chart-container">
          <div className="mb-4 flex items-center justify-between gap-3">
            <div><h2 className="text-lg font-semibold">Market-condition drivers</h2><p className="mt-1 text-xs text-gray-500">Features available before the forecast window.</p></div>
            <ShieldCheck size={19} className="text-sky-300" />
          </div>
          <div className="forecast-driver-list">
            {outlook.drivers.map((driver) => (
              <div className="forecast-driver" key={driver.name}>
                <span>{featureLabel(driver.name)}</span><strong>{driver.value.toFixed(4)}</strong>
              </div>
            ))}
          </div>
        </div>
        <div className="chart-container">
          <div className="mb-4"><h2 className="text-lg font-semibold">Does the baseline generalize?</h2><p className="mt-1 text-xs text-gray-500">Chronological 70/30 holdout, never random-shuffled.</p></div>
          <div className="forecast-validation-grid">
            <div><span>ROC-AUC</span><strong>{outlook.validation.roc_auc === null ? "N/A" : outlook.validation.roc_auc.toFixed(3)}</strong></div>
            <div><span>Brier score</span><strong>{outlook.validation.brier_score.toFixed(3)}</strong></div>
            <div><span>Accuracy</span><strong>{percent(outlook.validation.accuracy)}</strong></div>
            <div><span>Holdout samples</span><strong>{outlook.validation.holdout_samples.toLocaleString()}</strong></div>
          </div>
          <p className="mt-4 text-xs text-gray-500">These are baseline validation statistics, not proof that the full CG-NSDE generator is trained or superior.</p>
        </div>
      </section>

      <section className="chart-container" aria-labelledby="direction-title">
        <div className="mb-4"><h2 id="direction-title" className="text-lg font-semibold">Directional stock screen</h2><p className="mt-1 text-xs text-gray-500">20-day cumulative return divided by annualized recent volatility. This is a technical ranking, not a price prediction.</p></div>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div>
            <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-lime-300"><ArrowUpRight size={16} /> Strongest positive momentum</h3>
            <div className="forecast-stock-list">{outlook.directional_screen.strongest_positive.map((stock) => <div className="forecast-stock-row" key={stock.symbol}><span>{stock.symbol}</span><strong className="text-lime-300">{percent(stock.return_20d)}</strong><small>{(stock.volatility_annualized * 100).toFixed(1)}% vol</small></div>)}</div>
          </div>
          <div>
            <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-rose-300"><ArrowDownRight size={16} /> Strongest negative momentum</h3>
            <div className="forecast-stock-list">{outlook.directional_screen.strongest_negative.map((stock) => <div className="forecast-stock-row" key={stock.symbol}><span>{stock.symbol}</span><strong className="text-rose-300">{percent(stock.return_20d)}</strong><small>{(stock.volatility_annualized * 100).toFixed(1)}% vol</small></div>)}</div>
          </div>
        </div>
      </section>

      <section className="chart-container">
        <h2 className="text-lg font-semibold">How to interpret this for funds</h2>
        <p className="mt-2 text-sm leading-6 text-gray-400">The stock model estimates broad market-regime risk. It does not claim to forecast an individual mutual fund because a fund's result depends on its current holdings, cash, derivatives, expenses, and NAV process. Use the Mutual Funds page for live historical NAV behavior; future fund-risk attribution requires current portfolio holdings and a separate validated model.</p>
      </section>
    </div>
  );
}