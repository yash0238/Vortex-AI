// TypeScript type definitions for VORTEX-AI CG-NSDE API.

export type Regime = 0 | 1;

export interface DataStats {
  n_windows: number;
  n_stocks: number;
  window_length: number;
  crisis_ratio: number;
  n_crisis: number;
  n_normal: number;
  returns_shape: number[];
  daily_crisis_days: number;
  daily_total_days: number;
}

export interface ReturnsData {
  ticker: string;
  cumulative_returns: number[];
  daily_returns: number[];
  regimes: number[];
  dates: number[];
}

export interface WindowData {
  window_idx: number;
  regime: number;
  returns: number[][];
  adjacency: number[][];
  regime_label: string;
}

export interface AdjacencyData {
  z: number[][];
  x: string[];
  y: string[];
  regime: string;
  regime_idx: number;
  edge_density: number;
  non_zero_edges: number;
  top_connected: string[];
}

export interface NodeFeaturesData {
  tickers: string[];
  feature_names: string[];
  data: number[][];
}

export interface RegimeDistribution {
  total_windows: number;
  crisis_count: number;
  normal_count: number;
  crisis_ratio: number;
  crisis_indices: number[];
}

export interface ModelConfig {
  n_stocks?: number;
  T?: number;
  latent_dim?: number;
  proj_dim?: number;
  in_feats?: number;
  gat_heads?: number;
  gat_dropout?: number;
  tau?: number;
  lr?: number;
  sde_hidden_dim?: number;
  batch_size?: number;
}

export interface AttentionStats {
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface GatForwardResult {
  node_embs_shape: number[];
  learned_adj_shape: number[];
  graph_emb_shape: number[];
  learned_adj_sample: number[][];
  node_embs_pca: number[][];
  attention_stats: AttentionStats;
}

export interface SdeForwardResult {
  z_path_sample: number[][];
  r_hat_sample: number[][];
  x_sample: number[][];
  ts: number[];
  latent_stats: { mean: number; std: number; max: number };
  returns_stats: { mean: number; std: number };
}

export interface ScenarioStats {
  real_kurtosis: number;
  synth_kurtosis: number;
  kurtosis_pass: boolean;
  real_acf_sq: number;
  synth_acf_sq: number;
  real_mean: number;
  synth_mean: number;
  real_std: number;
  synth_std: number;
}

export interface ScenarioResult {
  scenarios: number[][];
  real_avg: number[];
  tickers: string[];
  stock_idx: number;
  stats: ScenarioStats;
}

export interface LossComponents {
  L_reconstruction: number;
  L_graph: number;
  L_contrastive: number;
  L_circuit_filter: number;
}

export interface StatisticalMetrics {
  kurtosis_real: number;
  kurtosis_synth: number;
  kurtosis_pass: boolean;
  acf_sq_real: number;
  acf_sq_synth: number;
  corr_error: number;
  real_mean: number;
  synth_mean: number;
  real_std: number;
  synth_std: number;
}

export interface DiscriminativeMetrics {
  score: number;
  pass: boolean;
}

export interface ContagionMetrics {
  normal_avg_corr: number;
  crisis_avg_corr: number;
  boost: number;
  pass: boolean;
}

export interface EvaluationResult {
  statistical: StatisticalMetrics;
  discriminative: DiscriminativeMetrics;
  contagion: ContagionMetrics;
  loss_components: LossComponents;
}

export interface TrainingResult {
  epochs: number;
  losses: number[];
  final_loss: number;
  initial_loss: number;
  trend: string;
}

export interface MarketInstrument {
  symbol: string;
  name: string;
  exchange: string;
  currency: string;
  quote_type: string;
}

export interface MarketQuote {
  symbol: string;
  price: number | null;
  previous_close: number | null;
  change: number | null;
  change_percent: number | null;
  open: number | null;
  day_high: number | null;
  day_low: number | null;
  volume: number | null;
  market_cap: number | null;
  currency: string;
  exchange: string;
  as_of: string;
  source: string;
  freshness_note: string;
}

export interface MarketBar {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number;
}

export type MarketPeriod = "1d" | "5d" | "1mo" | "3mo" | "6mo" | "1y" | "2y" | "5y" | "max";
export type MarketInterval = "1m" | "5m" | "15m" | "30m" | "60m" | "1d" | "1wk" | "1mo";

export interface MarketHistory {
  symbol: string;
  period: MarketPeriod;
  interval: MarketInterval;
  source: string;
  freshness_note: string;
  bars: MarketBar[];
}
