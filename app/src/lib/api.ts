import axios from "axios";
import type {
  AdjacencyData,
  DataStats,
  EvaluationResult,
  GatForwardResult,
  MarketHistory,
  MarketInstrument,
  MarketInterval,
  MarketPeriod,
  MarketQuote,
  ModelConfig,
  NodeFeaturesData,
  RegimeDistribution,
  ReturnsData,
  ScenarioResult,
  SdeForwardResult,
  TrainingResult,
} from "../types";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000",
  timeout: 120_000,
});

export const apiClient = {
  health: async () => (await client.get<{
    status: string;
    model_loaded: boolean;
    generator_checkpoint_available: boolean;
  }>("/api/health")).data,
  getStats: async () => (await client.get<DataStats>("/api/stats")).data,
  getReturns: async (stockIdx: number, windowIdx: number) =>
    (await client.get<ReturnsData>("/api/returns", {
      params: { stock_idx: stockIdx, window_idx: windowIdx },
    })).data,
  getRegimeDistribution: async () =>
    (await client.get<RegimeDistribution>("/api/regime-distribution")).data,
  getAdjacency: async (windowIdx: number) =>
    (await client.get<AdjacencyData>("/api/adjacency", {
      params: { window_idx: windowIdx },
    })).data,
  getNodeFeatures: async (windowIdx: number) =>
    (await client.get<NodeFeaturesData>("/api/node-features", {
      params: { window_idx: windowIdx },
    })).data,
  gatForward: async (config: ModelConfig) =>
    (await client.post<GatForwardResult>("/api/model/gat-forward", config)).data,
  sdeForward: async (config: ModelConfig) =>
    (await client.post<SdeForwardResult>("/api/model/sde-forward", config)).data,
  generateScenario: async (
    config: ModelConfig,
    noiseScale: number,
    nScenarios: number,
    stockIdx: number,
  ) =>
    (await client.post<ScenarioResult>("/api/scenario/generate", config, {
      params: {
        noise_scale: noiseScale,
        n_scenarios: nScenarios,
        stock_idx: stockIdx,
      },
    })).data,
  getEvaluation: async (batchSize: number) =>
    (await client.get<EvaluationResult>("/api/evaluation/metrics", {
      params: { batch_size: batchSize },
    })).data,
  trainingSimulation: async (epochs: number, batchSize: number, lr: number) =>
    (await client.get<TrainingResult>("/api/training/simulate", {
      params: { epochs, batch_size: batchSize, lr },
    })).data,
  searchMarket: async (query: string, limit = 10) =>
    (await client.get<{ query: string; results: MarketInstrument[] }>("/api/market/search", {
      params: { q: query, limit },
    })).data,
  getMarketQuote: async (symbol: string) =>
    (await client.get<MarketQuote>("/api/market/quote", { params: { symbol } })).data,
  getMarketHistory: async (symbol: string, period: MarketPeriod, interval: MarketInterval) =>
    (await client.get<MarketHistory>("/api/market/history", {
      params: { symbol, period, interval },
    })).data,
};