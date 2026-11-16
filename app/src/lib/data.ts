import { useEffect, useState } from "react";

/* The dashboard reads only static JSON exported from results/ by `python -m crisisbench.export_web`. */
/* eslint-disable @typescript-eslint/no-explicit-any */
export type Any = any;
export type RefMetric = { kind: "stat" | "dist" | "disc"; real: number; lo?: number; hi?: number; half?: number; ceiling?: number };
export type BenchRow = Record<string, Any> & { generator: string; name: string; family_tag: string; seeds: number };
export type Bench = { rows: BenchRow[]; reference: { n_windows: number; n_crisis: number; metrics: Record<string, RefMetric>; mac_crisis: number; mac_normal: number };
  metric_tau: { metrics: string[]; matrix: number[][] } | null; alt_rule: BenchRow[] | null; ablation: BenchRow[] | null };
export type Core = { built: string; protocol: Any; pretty: Record<string, string>; summary: Record<string, Any>; bench: Record<string, Bench>;
  gates: Record<string, string>; heldout: Any[] | null; risk: Any[] | null; risk_series: Any | null; metric_validity: Any[] | null;
  contamination: Any[] | null; dgp: Any[] | null; cgnsde: Any | null; hypotheses: Record<string, Any> | null };
export type Market = Any;
export type Samples = Record<string, Record<string, Any>>;

const cache = new Map<string, Promise<Any>>();
function load<T>(name: string): Promise<T> {
  if (!cache.has(name)) cache.set(name, fetch(`${import.meta.env.BASE_URL}data/${name}.json`).then((r) => { if (!r.ok) throw new Error(`${name}.json: ${r.status}`); return r.json(); }));
  return cache.get(name) as Promise<T>;
}
export function useData<T>(name: "core" | "markets" | "samples"): { data: T | null; error: string | null } {
  const [s, set] = useState<{ data: T | null; error: string | null }>({ data: null, error: null });
  useEffect(() => { let on = true; load<T>(name).then((d) => on && set({ data: d, error: null })).catch((e) => on && set({ data: null, error: String(e) })); return () => { on = false; }; }, [name]);
  return s;
}

export const FAMILY: Record<string, { label: string; color: string; blurb: string }> = {
  "neural-ours": { label: "Our model", color: "var(--s-ours)", blurb: "CG-NSDE, the model built in this project" },
  neural: { label: "Neural baselines", color: "var(--s-neural)", blurb: "TimeGAN, QuantGAN, TimeVAE" },
  regime: { label: "Regime models", color: "var(--s-regime)", blurb: "simple statistical models that switch between calm and crisis" },
  "dynamic-classical": { label: "GARCH family", color: "var(--s-dyn)", blurb: "classical volatility models used by risk desks" },
  resampling: { label: "Resampling", color: "var(--s-resample)", blurb: "replays blocks of real history" },
  static: { label: "Static models", color: "var(--s-static)", blurb: "one fixed distribution for every day" },
};
export const famColor = (tag: string) => FAMILY[tag]?.color ?? "var(--s-static)";

export const API = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";
