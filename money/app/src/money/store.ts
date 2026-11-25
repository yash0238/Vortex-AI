import { useEffect, useState } from "react";
import type { Engine, Holding } from "./engine";

export type GoalKind = "house" | "education" | "retirement" | "car" | "wealth" | "travel";

export type Profile = {
  name: string;
  goalKind: GoalKind;
  goal: number; // today's rupees
  years: number;
  start: number;
  sip: number;
  stepUp: number;
  inflation: number;
  tolerance: number; // largest fall (fraction) the user says they can sit through
  emergency: boolean;
  crashPlan: boolean;
  holdings: Holding[];
};

export type Progress = {
  xp: number; seen: string[]; badges: string[]; tipChecks: number; quiz: Record<string, boolean>;
  streak?: { last: string; count: number; best: number };
  history?: { d: string; h: number }[];         // one Money Health snapshot per day
  week?: string; missions?: string[];           // missions completed in the current ISO week
};

export const MISSIONS = [
  { id: "crash", label: "Run a crash test on your portfolio", xp: 25 },
  { id: "xray", label: "X-ray a real mutual fund", xp: 25 },
  { id: "retire", label: "Check if your retirement money lasts", xp: 25 },
  { id: "quiz", label: "Answer the daily quiz", xp: 15 },
  { id: "tip", label: "Reality-check a stock tip", xp: 15 },
  { id: "goal", label: "Try a different SIP in the goal planner", xp: 15 },
  { id: "divers", label: "See how your holdings behave in a crash", xp: 15 },
  { id: "gap", label: "See what panic-selling in 2020 would have cost", xp: 20 },
];

export const today = () => new Date().toISOString().slice(0, 10);
export function isoWeek(d = new Date()): string {
  const t = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
  const day = t.getUTCDay() || 7;
  t.setUTCDate(t.getUTCDate() + 4 - day);
  const y0 = new Date(Date.UTC(t.getUTCFullYear(), 0, 1));
  return `${t.getUTCFullYear()}-W${Math.ceil(((t.getTime() - y0.getTime()) / 86400000 + 1) / 7)}`;
}
// three missions per week, rotating
export function weekMissions(week: string) {
  const h = [...week].reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 7);
  const idx = [h % MISSIONS.length, (h >>> 3) % MISSIONS.length, (h >>> 7) % MISSIONS.length];
  const uniq = [...new Set(idx)];
  for (let i = 0; uniq.length < 3; i++) if (!uniq.includes(i)) uniq.push(i);
  return uniq.slice(0, 3).map((i) => MISSIONS[i]);
}

export const LEVEL_TITLES = ["Saver", "Planner", "Investor", "Steady hand", "Storm-proof", "Vortex master", "Legend"];

export const GOALS: { k: GoalKind; label: string; emoji: string; amount: number; years: number }[] = [
  { k: "house", label: "Dream home", emoji: "🏡", amount: 3000000, years: 10 },
  { k: "education", label: "Child's education", emoji: "🎓", amount: 2500000, years: 12 },
  { k: "retirement", label: "Retire comfortably", emoji: "🌴", amount: 30000000, years: 25 },
  { k: "car", label: "New car", emoji: "🚗", amount: 1200000, years: 4 },
  { k: "travel", label: "Big trip", emoji: "✈️", amount: 500000, years: 3 },
  { k: "wealth", label: "Grow wealth", emoji: "📈", amount: 10000000, years: 15 },
];

const KEY = "vortexMoney.v1";

function read<T>(k: string): T | null {
  try { const s = localStorage.getItem(k); return s ? (JSON.parse(s) as T) : null; } catch { return null; }
}
function write(k: string, v: unknown) {
  try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode: keep in memory only */ }
}

export function usePersisted<T>(k: string, init: T): [T, (v: T | ((p: T) => T)) => void] {
  const [v, setV] = useState<T>(() => read<T>(`${KEY}.${k}`) ?? init);
  useEffect(() => { write(`${KEY}.${k}`, v); }, [k, v]);
  return [v, setV];
}

// ------------------------------------------------------------ building blocks a layman understands
export function basket(eng: Engine, id: string, label: string, sub: string, filter: (sector: string) => boolean, amount: number): Holding {
  const idx = eng.assets.map((a, i) => (filter(a.sector) ? i : -1)).filter((i) => i >= 0);
  const ex = new Array(eng.N).fill(0);
  idx.forEach((i) => (ex[i] = 1 / idx.length));
  return { id, kind: "basket", label, sub, amount, exposure: ex };
}

export function stock(eng: Engine, i: number, amount: number): Holding {
  const ex = new Array(eng.N).fill(0);
  ex[i] = 1;
  const a = eng.assets[i];
  return { id: `stk-${a.ticker}`, kind: "stock", label: a.name, sub: `${a.sector} · single stock`, amount, exposure: ex };
}

export const safe = (amount: number): Holding =>
  ({ id: "safe", kind: "safe", label: "Safe money", sub: "FD / debt fund · assumed 6.5% a year", amount, rateY: 0.065 });

export function catalogue(eng: Engine): Holding[] {
  const s = (x: string) => (sec: string) => sec === x;
  return [
    basket(eng, "bsk-large", "Large-cap index style", "All 47 big NSE companies, equal weight", () => true, 0),
    basket(eng, "bsk-banks", "Banking & finance", "Banks + NBFCs basket", (x) => x === "Banks" || x === "Finance", 0),
    basket(eng, "bsk-it", "IT services", "TCS, Infosys, HCL, Wipro, TechM", s("IT"), 0),
    basket(eng, "bsk-pharma", "Pharma & health", "Sun, Cipla, Dr Reddy's, Divi's, Apollo", (x) => x === "Pharma" || x === "Healthcare", 0),
    basket(eng, "bsk-fmcg", "Everyday brands (FMCG)", "HUL, ITC, Nestle, Britannia, Tata Consumer", s("FMCG"), 0),
    basket(eng, "bsk-auto", "Auto", "Maruti, M&M, Tata Motors, Bajaj, Hero, Eicher", s("Auto"), 0),
    basket(eng, "bsk-energy", "Energy & PSU", "Reliance, ONGC, Coal India, BPCL, NTPC, Power Grid",
      (x) => x === "Energy" || x === "Utilities", 0),
    basket(eng, "bsk-metals", "Metals & materials", "Tata Steel, JSW, Hindalco, cement", (x) => x === "Metals" || x === "Materials", 0),
    safe(0),
  ];
}

export function starter(eng: Engine, kind: "careful" | "balanced" | "bold", total: number): Holding[] {
  const c = Object.fromEntries(catalogue(eng).map((h) => [h.id, h]));
  const pick = (id: string, f: number) => ({ ...c[id], amount: Math.round(total * f) });
  if (kind === "careful") return [pick("bsk-large", 0.35), pick("bsk-fmcg", 0.05), pick("safe", 0.6)];
  if (kind === "balanced") return [pick("bsk-large", 0.5), pick("bsk-fmcg", 0.1), pick("bsk-pharma", 0.05), pick("safe", 0.35)];
  return [pick("bsk-large", 0.55), pick("bsk-it", 0.15), pick("bsk-banks", 0.15), pick("bsk-auto", 0.1), pick("safe", 0.05)];
}

export const LEVELS = [0, 60, 150, 280, 450, 700, 1000];
export const levelOf = (xp: number) => {
  let l = 0;
  LEVELS.forEach((t, i) => { if (xp >= t) l = i; });
  const next = LEVELS[Math.min(l + 1, LEVELS.length - 1)], cur = LEVELS[l];
  return { level: l + 1, frac: next > cur ? (xp - cur) / (next - cur) : 1, next };
};
