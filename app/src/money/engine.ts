import { fundJson } from "./api";

// VORTEX Money scenario engine. Everything runs in the browser on scenarios exported by
// crisisbench/export_money.py (CG-NSDE output) plus a little real history.

export type Asset = { ticker: string; name: string; sector: string };

export type Engine = {
  model: string;
  N: number;
  assets: Asset[];
  nMonth: number;
  genMonth: Float32Array; // (nMonth, N) monthly log returns, rows come in blocks of 3 from one window
  nCrash: number;
  crashWeek: Float32Array; // (nCrash, 12, N) weekly log returns, each starts a stress episode
  corrCalm: number[][];
  corrCrash: number[][];
  realMonths: string[];
  realMonth: Float32Array; // (months, N)
  realEwDates: string[];
  realEw: Float32Array;
  tip: Record<string, number[]>;
  tipQ: number[];
  sdM: number[]; // per-stock monthly log-return sd in the scenarios
};

export type HoldingKind = "stock" | "basket" | "fund" | "safe";

export type Holding = {
  id: string;
  kind: HoldingKind;
  label: string;
  sub: string;
  amount: number; // rupees currently invested (or share of the plan)
  exposure?: number[]; // weights on the N stocks (log-return exposure)
  beta?: number; // funds: sensitivity to the equal-weight large-cap basket
  residM?: number; // funds: monthly idiosyncratic log-return sd
  rateY?: number; // safe money: yearly rate
};

const dec = (b64: string): Float32Array => {
  const bin = atob(b64);
  const n = bin.length / 2;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    if (v >= 32768) v -= 65536;
    out[i] = v / 1e4;
  }
  return out;
};

export async function loadEngine(): Promise<Engine> {
  const r = await fetch(`${import.meta.env.BASE_URL}money/engine.json`);
  if (!r.ok) throw new Error(`engine.json: HTTP ${r.status}`);
  const j = await r.json();
  const e: Engine = {
    model: j.model, N: j.n_assets, assets: j.assets, nMonth: j.n_month, genMonth: dec(j.gen_month),
    nCrash: j.n_crash, crashWeek: dec(j.crash_week), corrCalm: j.corr_calm, corrCrash: j.corr_crash,
    realMonths: j.real_months, realMonth: dec(j.real_month), realEwDates: j.real_ew_dates, realEw: dec(j.real_ew),
    tip: j.tip_quantiles, tipQ: j.tip_q, sdM: [],
  };
  for (let i = 0; i < e.N; i++) {
    let m = 0, v = 0;
    for (let t = 0; t < e.nMonth; t++) m += e.genMonth[t * e.N + i];
    m /= e.nMonth;
    for (let t = 0; t < e.nMonth; t++) v += (e.genMonth[t * e.N + i] - m) ** 2;
    e.sdM.push(Math.sqrt(v / e.nMonth));
  }
  return e;
}

// ------------------------------------------------------------------ randomness (seeded, reproducible)
export function rng(seed: number) {
  let s = seed >>> 0 || 1;
  const u = () => {
    s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
  const n = () => {
    const a = Math.max(u(), 1e-12), b = u();
    return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
  };
  return { u, n };
}

// ------------------------------------------------------------------ holdings -> exposures
export function exposureOf(h: Holding, N: number): number[] {
  if (h.kind === "safe") return new Array(N).fill(0);
  if (h.kind === "fund") return new Array(N).fill((h.beta ?? 1) / N);
  return h.exposure ?? new Array(N).fill(1 / N);
}

const total = (hs: Holding[]) => hs.reduce((a, h) => a + Math.max(h.amount, 0), 0);
export const weightsOf = (hs: Holding[]) => {
  const t = total(hs) || 1;
  return hs.map((h) => Math.max(h.amount, 0) / t);
};

export type Prepared = { E: Float32Array; w: number[]; resid: number[]; safeM: number[]; H: number; N: number; cache: Map<Float32Array, Float32Array> };

export function prepare(hs: Holding[], N: number): Prepared {
  const H = hs.length;
  const E = new Float32Array(H * N);
  hs.forEach((h, k) => exposureOf(h, N).forEach((v, i) => (E[k * N + i] = v)));
  return {
    E, w: weightsOf(hs), H,
    resid: hs.map((h) => (h.kind === "fund" ? h.residM ?? 0 : 0)),
    // safe money earns its rate; a fund earns the safe rate on the part not tied to the market (CAPM-style:
    // r = rf + beta (market - rf) + own noise), so a debt fund with beta ~ 0 grows like safe money
    safeM: hs.map((h) => (h.kind === "safe" ? Math.log(1 + (h.rateY ?? 0.065)) / 12
      : h.kind === "fund" ? (1 - Math.min(h.beta ?? 1, 1)) * Math.log(1.065) / 12 : 0)),
    N, cache: new Map(),
  };
}

// linear (stock-exposure) part of every holding's log return for every row of a scenario matrix, computed once
function linear(p: Prepared, src: Float32Array): Float32Array {
  let L = p.cache.get(src);
  if (L) return L;
  const N = p.N, rows = src.length / N;
  L = new Float32Array(p.H * rows);
  for (let k = 0; k < p.H; k++) {
    const base = k * N;
    const nz: number[] = [];
    for (let i = 0; i < N; i++) if (p.E[base + i] !== 0) nz.push(i);
    for (let r = 0; r < rows; r++) {
      let s = 0;
      const o = r * N;
      for (const i of nz) s += p.E[base + i] * src[o + i];
      L[k * rows + r] = s;
    }
  }
  p.cache.set(src, L);
  return L;
}

// simple return of each holding for scenario row `row` of src; residual noise scaled by sqrt(frac)
export function holdingReturns(p: Prepared, src: Float32Array, row: number, frac: number, z: () => number, out: number[]) {
  const L = linear(p, src), rows = src.length / p.N;
  for (let k = 0; k < p.H; k++) {
    let s = L[k * rows + row];
    if (p.resid[k] > 0) s += p.resid[k] * Math.sqrt(frac) * z() - 0.5 * p.resid[k] * p.resid[k] * frac;
    s += p.safeM[k] * frac;
    out[k] = Math.expm1(s);
  }
}

// ------------------------------------------------------------------ goal planner (Monte Carlo)
export type GoalInput = { start: number; sip: number; years: number; goal: number; inflation: number; stepUp: number };
export type GoalResult = {
  prob: number; months: number; fan: { p10: number[]; p50: number[]; p90: number[] }; naive12: number[];
  finals: Float64Array; goalNominal: number; median: number; bad: number; invested: number;
};

export function simulateGoal(eng: Engine, hs: Holding[], g: GoalInput, paths = 2000, seed = 7): GoalResult {
  const N = eng.N, M = Math.max(1, Math.round(g.years * 12));
  const p = prepare(hs, N);
  const r = rng(seed);
  const nWin = Math.floor(eng.nMonth / 3);
  const step = Math.max(1, Math.floor(M / 60));
  const cols = Math.floor(M / step) + 1;
  const grid = Array.from({ length: cols }, () => new Float64Array(paths));
  const finals = new Float64Array(paths);
  const hr = new Array(p.H).fill(0);
  for (let s = 0; s < paths; s++) {
    let V = g.start, win = 0, pos = 3, sip = g.sip;
    grid[0][s] = V;
    for (let m = 1; m <= M; m++) {
      if (pos >= 3) { win = Math.floor(r.u() * nWin); pos = 0; }
      holdingReturns(p, eng.genMonth, win * 3 + pos, 1, r.n, hr);
      pos++;
      let R = 0;
      for (let k = 0; k < p.H; k++) R += p.w[k] * hr[k];
      V = V * (1 + R) + sip;
      if (m % 12 === 0) sip *= 1 + g.stepUp;
      if (m % step === 0) grid[m / step][s] = V;
    }
    finals[s] = V;
  }
  const q = (a: Float64Array, f: number) => { const b = Float64Array.from(a).sort(); return b[Math.min(b.length - 1, Math.floor(f * b.length))]; };
  const goalNominal = g.goal * Math.pow(1 + g.inflation, g.years);
  let hit = 0;
  finals.forEach((v) => { if (v >= goalNominal) hit++; });
  const naive12: number[] = [];
  let Vn = g.start, sipn = g.sip, invested = g.start;
  naive12.push(Vn);
  for (let m = 1; m <= M; m++) {
    Vn = Vn * Math.pow(1.12, 1 / 12) + sipn; invested += sipn;
    if (m % 12 === 0) sipn *= 1 + g.stepUp;
    if (m % step === 0) naive12.push(Vn);
  }
  return {
    prob: hit / paths, months: M, goalNominal, finals, invested, naive12,
    fan: { p10: grid.map((c) => q(c, 0.1)), p50: grid.map((c) => q(c, 0.5)), p90: grid.map((c) => q(c, 0.9)) },
    median: q(finals, 0.5), bad: q(finals, 0.1),
  };
}

export function sipFor(eng: Engine, hs: Holding[], g: GoalInput, target = 0.85): number {
  let lo = 0, hi = Math.max(g.sip * 4, g.goal / Math.max(g.years * 12, 1) * 1.5, 1000);
  for (let i = 0; i < 4 && simulateGoal(eng, hs, { ...g, sip: hi }, 700, 11).prob < target; i++) hi *= 2;
  for (let i = 0; i < 12; i++) {
    const mid = (lo + hi) / 2;
    if (simulateGoal(eng, hs, { ...g, sip: mid }, 700, 11).prob >= target) hi = mid; else lo = mid;
  }
  return Math.ceil(hi / 100) * 100;
}

// ------------------------------------------------------------------ crash rehearsal
export type CrashResult = {
  weeks: number; p10: number[]; p50: number[]; p90: number[]; troughMedian: number; troughBad: number;
  worstPath: number[]; lossBadRs: number; lossMedRs: number;
};

export function simulateCrash(eng: Engine, hs: Holding[], seed = 3): CrashResult {
  const N = eng.N, W = 12, p = prepare(hs, N), r = rng(seed);
  const V0 = total(hs) || 1;
  const paths: number[][] = [];
  const troughs: number[] = [];
  const hr = new Array(p.H).fill(0);
  for (let s = 0; s < eng.nCrash; s++) {
    let v = 1;
    const path = [1];
    let lo = 1;
    for (let t = 0; t < W; t++) {
      holdingReturns(p, eng.crashWeek, s * W + t, 12 / 52, r.n, hr);
      let R = 0;
      for (let k = 0; k < p.H; k++) R += p.w[k] * hr[k];
      v *= 1 + R;
      path.push(v);
      lo = Math.min(lo, v);
    }
    paths.push(path);
    troughs.push(lo - 1);
  }
  const col = (t: number, f: number) => { const c = paths.map((x) => x[t]).sort((a, b) => a - b); return c[Math.floor(f * (c.length - 1))]; };
  const sortedT = [...troughs].sort((a, b) => a - b);
  const bad = sortedT[Math.floor(0.1 * sortedT.length)], med = sortedT[Math.floor(0.5 * sortedT.length)];
  const worstIdx = troughs.indexOf(sortedT[Math.floor(0.05 * sortedT.length)]);
  return {
    weeks: W, p10: paths[0].map((_, t) => col(t, 0.1)), p50: paths[0].map((_, t) => col(t, 0.5)),
    p90: paths[0].map((_, t) => col(t, 0.9)), troughMedian: med, troughBad: bad, worstPath: paths[worstIdx],
    lossBadRs: -bad * V0, lossMedRs: -med * V0,
  };
}

// ------------------------------------------------------------------ real history replay
export type HistoryResult = {
  labels: string[]; value: number[]; worstDrop: number; worstFrom: string; worstTo: string; recoverMonths: number | null;
  covidDrop: number | null; covidRecover: number | null; upYears: number;
};

export function historyReplay(eng: Engine, hs: Holding[]): HistoryResult {
  const N = eng.N, p = prepare(hs, N), T = eng.realMonths.length;
  const hr = new Array(p.H).fill(0);
  const value = [1];
  const zero = () => 0;
  for (let m = 0; m < T; m++) {
    holdingReturns(p, eng.realMonth, m, 1, zero, hr);
    let R = 0;
    for (let k = 0; k < p.H; k++) R += p.w[k] * hr[k];
    value.push(value[value.length - 1] * (1 + R));
  }
  const labels = ["start", ...eng.realMonths];
  let peak = value[0], peakI = 0, worst = 0, wFrom = 0, wTo = 0;
  for (let i = 1; i < value.length; i++) {
    if (value[i] > peak) { peak = value[i]; peakI = i; }
    const dd = value[i] / peak - 1;
    if (dd < worst) { worst = dd; wFrom = peakI; wTo = i; }
  }
  let rec: number | null = null;
  for (let i = wTo; i < value.length; i++) if (value[i] >= value[wFrom]) { rec = i - wTo; break; }
  const iJan = labels.indexOf("2020-01");
  let covidDrop: number | null = null, covidRec: number | null = null;
  if (iJan > 0) {
    const pk = value[iJan];
    let lo = pk, loI = iJan;
    for (let i = iJan; i < Math.min(iJan + 6, value.length); i++) if (value[i] < lo) { lo = value[i]; loI = i; }
    covidDrop = lo / pk - 1;
    for (let i = loI; i < value.length; i++) if (value[i] >= pk) { covidRec = i - loI; break; }
  }
  let up = 0, n = 0;
  for (let i = 12; i < value.length; i++) { n++; if (value[i] > value[i - 12]) up++; }
  return { labels, value, worstDrop: worst, worstFrom: labels[wFrom], worstTo: labels[wTo], recoverMonths: rec,
    covidDrop, covidRecover: covidRec, upYears: n ? up / n : 0 };
}

// ------------------------------------------------------------------ SIP through a crash
export type SipCrashResult = { months: number; keep: number[]; stop: number[]; invested: number[]; keepFinal: number; stopFinal: number; extraPct: number };

export function sipThroughCrash(eng: Engine, hs: Holding[], sip: number, years = 5, pauseMonths = 12, seed = 5): SipCrashResult {
  const N = eng.N, p = prepare(hs, N), r = rng(seed), M = years * 12, nWin = Math.floor(eng.nMonth / 3);
  const P = 600, hr = new Array(p.H).fill(0);
  const keepA = Array.from({ length: M + 1 }, () => 0), stopA = keepA.slice(), inv = keepA.slice();
  for (let s = 0; s < P; s++) {
    const c = Math.floor(r.u() * eng.nCrash);
    let K = 0, S = 0, I = 0, win = 0, pos = 3;
    for (let m = 1; m <= M; m++) {
      let R = 0;
      if (m <= 3) {                                    // the crash quarter: 12 crisis weeks -> 3 months
        let g = 1;
        for (let t = (m - 1) * 4; t < m * 4; t++) {
          holdingReturns(p, eng.crashWeek, c * 12 + t, 12 / 52, r.n, hr);
          let x = 0;
          for (let k = 0; k < p.H; k++) x += p.w[k] * hr[k];
          g *= 1 + x;
        }
        R = g - 1;
      } else {
        if (pos >= 3) { win = Math.floor(r.u() * nWin); pos = 0; }
        holdingReturns(p, eng.genMonth, win * 3 + pos, 1, r.n, hr);
        pos++;
        for (let k = 0; k < p.H; k++) R += p.w[k] * hr[k];
      }
      K = K * (1 + R) + sip;
      S = S * (1 + R) + (m <= pauseMonths ? 0 : sip);
      I += sip;
      keepA[m] += K / P; stopA[m] += S / P; inv[m] += I / P;
    }
  }
  const kf = keepA[M], sf = stopA[M];
  return { months: M, keep: keepA, stop: stopA, invested: inv, keepFinal: kf, stopFinal: sf, extraPct: sf > 0 ? kf / sf - 1 : 0 };
}

// ------------------------------------------------------------------ diversification that disappears
export type DiversResult = { nHold: number; calm: number; crash: number; pairs: { a: number; b: number; calm: number; crash: number }[] };

function holdingCorr(eng: Engine, hs: Holding[], C: number[][]): number[][] {
  // covariance in monthly log-return units: stocks sd_i sd_j C_ij; a fund adds its own residual variance
  const N = eng.N, sd = eng.sdM;
  const ex = hs.map((h) => exposureOf(h, N));
  const cov = ex.map((a) => ex.map((b) => {
    let s = 0;
    for (let i = 0; i < N; i++) if (a[i]) for (let j = 0; j < N; j++) if (b[j]) s += a[i] * b[j] * sd[i] * sd[j] * C[i][j];
    return s;
  }));
  hs.forEach((h, k) => { if (h.kind === "fund" && h.residM) cov[k][k] += h.residM * h.residM; });
  return cov.map((row, i) => row.map((v, j) => v / Math.sqrt(Math.max(cov[i][i] * cov[j][j], 1e-12))));
}

export function diversification(eng: Engine, hs: Holding[]): DiversResult {
  const risky = hs.filter((h) => h.kind !== "safe" && h.amount > 0);
  if (risky.length === 0) return { nHold: 0, calm: 0, crash: 0, pairs: [] };
  const w = weightsOf(risky);
  const eff = (R: number[][]) => {
    let s = 0;
    for (let i = 0; i < w.length; i++) for (let j = 0; j < w.length; j++) s += w[i] * w[j] * R[i][j];
    return 1 / Math.max(s, 1e-9);
  };
  const Rc = holdingCorr(eng, risky, eng.corrCalm), Rx = holdingCorr(eng, risky, eng.corrCrash);
  const pairs: DiversResult["pairs"] = [];
  for (let a = 0; a < risky.length; a++) for (let b = a + 1; b < risky.length; b++)
    pairs.push({ a: hs.indexOf(risky[a]), b: hs.indexOf(risky[b]), calm: Rc[a][b], crash: Rx[a][b] });
  return { nHold: risky.length, calm: eff(Rc), crash: eff(Rx), pairs };
}

// ------------------------------------------------------------------ tip reality check
export function tipCheck(eng: Engine, pct: number, weeks: 1 | 4 | 13 | 52) {
  const qs = eng.tip[String(weeks)], lv = eng.tipQ;
  const x = pct / 100;
  if (x > qs[qs.length - 1]) return { share: 0, never: true, maxSeen: qs[qs.length - 1] };
  for (let i = qs.length - 1; i > 0; i--) {
    if (x >= qs[i - 1]) {
      const f = (x - qs[i - 1]) / Math.max(qs[i] - qs[i - 1], 1e-9);
      const level = lv[i - 1] + f * (lv[i] - lv[i - 1]);
      return { share: 1 - level, never: false, maxSeen: qs[qs.length - 1] };
    }
  }
  return { share: 0.5, never: false, maxSeen: qs[qs.length - 1] };
}

// ------------------------------------------------------------------ mutual funds (mfapi.in, public AMFI NAVs)
export type FundMatch = { schemeCode: number; schemeName: string };

export async function searchFunds(q: string): Promise<FundMatch[]> {
  return (await fundJson(`/mf/search?q=${encodeURIComponent(q)}`)) as FundMatch[];
}

export type NavPoint = { date: string; nav: number }; // ISO date, ascending
export async function fundNav(code: number): Promise<{ name: string; category: string; house: string; navs: NavPoint[] }> {
  const j = (await fundJson(`/mf/${code}`)) as { meta?: Record<string, string>; data: { date: string; nav: string }[] };
  const navs = j.data.map((d) => { const [dd, mm, yy] = d.date.split("-"); return { date: `${yy}-${mm}-${dd}`, nav: parseFloat(d.nav) }; })
    .filter((x) => isFinite(x.nav) && x.nav > 0).sort((a, b) => (a.date < b.date ? -1 : 1));
  return { name: j.meta?.scheme_name ?? `Fund ${code}`, category: j.meta?.scheme_category ?? "", house: j.meta?.fund_house ?? "", navs };
}

export async function fundProfile(eng: Engine, code: number): Promise<{ name: string; beta: number; residM: number; days: number; category: string }> {
  const j = (await fundJson(`/mf/${code}`)) as { meta?: Record<string, string>; data: { date: string; nav: string }[] };
  const nav = new Map<string, number>();
  for (const d of j.data) {
    const [dd, mm, yy] = d.date.split("-");
    nav.set(`${yy}-${mm}-${dd}`, parseFloat(d.nav));
  }
  const xs: number[] = [], ys: number[] = [];
  let prev: number | null = null;
  for (let i = 0; i < eng.realEwDates.length; i++) {
    const v = nav.get(eng.realEwDates[i]);
    if (v === undefined || !isFinite(v) || v <= 0) { prev = null; continue; }
    if (prev !== null) { xs.push(eng.realEw[i]); ys.push(Math.log(v / prev)); }
    prev = v;
  }
  if (xs.length < 250) throw new Error("not enough overlapping history to measure this fund (need about a year)");
  const mx = xs.reduce((a, b) => a + b) / xs.length, my = ys.reduce((a, b) => a + b) / ys.length;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
  const beta = sxy / sxx;
  let ss = 0;
  for (let i = 0; i < xs.length; i++) ss += (ys[i] - my - beta * (xs[i] - mx)) ** 2;
  const residD = Math.sqrt(ss / (xs.length - 2));
  return { name: j.meta?.scheme_name ?? `Fund ${code}`, beta: Math.max(0, Math.min(beta, 1.8)), residM: residD * Math.sqrt(21),
    days: xs.length, category: j.meta?.scheme_category ?? "" };
}

// ------------------------------------------------------------------ formatting
export const rs = (v: number, digits = 1) => {
  const a = Math.abs(v), s = v < 0 ? "−" : "";
  if (a >= 1e7) return `${s}₹${(a / 1e7).toFixed(digits)} Cr`;
  if (a >= 1e5) return `${s}₹${(a / 1e5).toFixed(digits)} L`;
  if (a >= 1e3) return `${s}₹${(a / 1e3).toFixed(a >= 1e4 ? 0 : 1)}k`;
  return `${s}₹${Math.round(a)}`;
};
export const pct = (v: number, d = 0) => `${v >= 0 ? "" : "−"}${Math.abs(v * 100).toFixed(d)}%`;
