// Extra analytics for VORTEX Money: Fund X-ray, retirement withdrawals, behaviour gap, plan sensitivity.
import { holdingReturns, prepare, rng, simulateGoal, type Engine, type GoalInput, type Holding, type NavPoint } from "./engine";

// ------------------------------------------------------------------ Fund X-ray (real NAV history)
export type FundXray = {
  cagr: { y1: number | null; y3: number | null; y5: number | null; all: number | null };
  vol: number; maxDD: number; ddPeak: string; ddTrough: string; recoveryDays: number | null;
  worstYear: { year: string; ret: number } | null; bestYear: { year: string; ret: number } | null;
  worst3m: number; upYears: number; years: number;
  series: { date: string; nav: number; dd: number }[];
};

const days = (a: string, b: string) => (Date.parse(b) - Date.parse(a)) / 86400000;

export function fundXray(navs: NavPoint[]): FundXray {
  const n = navs.length, last = navs[n - 1];
  const at = (yrs: number) => {
    const target = Date.parse(last.date) - yrs * 365.25 * 86400000;
    if (Date.parse(navs[0].date) > target + 20 * 86400000) return null;
    let i = 0;
    while (i < n - 1 && Date.parse(navs[i + 1].date) <= target) i++;
    return Math.pow(last.nav / navs[i].nav, 1 / yrs) - 1;
  };
  const span = days(navs[0].date, last.date) / 365.25;
  let peak = navs[0].nav, peakD = navs[0].date, mdd = 0, mPeak = peakD, mTrough = peakD;
  const series: FundXray["series"] = [];
  const step = Math.max(1, Math.floor(n / 400));
  navs.forEach((p, i) => {
    if (p.nav > peak) { peak = p.nav; peakD = p.date; }
    const dd = p.nav / peak - 1;
    if (dd < mdd) { mdd = dd; mPeak = peakD; mTrough = p.date; }
    if (i % step === 0 || i === n - 1) series.push({ date: p.date, nav: p.nav, dd });
  });
  let rec: number | null = null;
  const pk = navs.find((p) => p.date === mPeak)!.nav;
  for (const p of navs) if (p.date > mTrough && p.nav >= pk) { rec = Math.round(days(mTrough, p.date)); break; }
  const lr: number[] = [];
  for (let i = 1; i < n; i++) lr.push(Math.log(navs[i].nav / navs[i - 1].nav));
  const mu = lr.reduce((a, b) => a + b, 0) / Math.max(lr.length, 1);
  const vol = Math.sqrt(lr.reduce((a, b) => a + (b - mu) ** 2, 0) / Math.max(lr.length - 1, 1)) * Math.sqrt(250);
  const byYear = new Map<string, { first: number; last: number }>();
  navs.forEach((p) => { const y = p.date.slice(0, 4); const e = byYear.get(y); if (!e) byYear.set(y, { first: p.nav, last: p.nav }); else e.last = p.nav; });
  const yrs = [...byYear.entries()];
  const yr: { year: string; ret: number }[] = [];
  for (let i = 1; i < yrs.length; i++) yr.push({ year: yrs[i][0], ret: yrs[i][1].last / yrs[i - 1][1].last - 1 });
  const full = yr.filter((y) => y.year !== last.date.slice(0, 4));
  let w3 = 0;
  for (let i = 0, j = 0; i < n; i++) {
    while (j < n && days(navs[i].date, navs[j].date) < 91) j++;
    if (j < n) w3 = Math.min(w3, navs[j].nav / navs[i].nav - 1);
  }
  return {
    cagr: { y1: at(1), y3: at(3), y5: at(5), all: span > 1 ? Math.pow(last.nav / navs[0].nav, 1 / span) - 1 : null },
    vol, maxDD: mdd, ddPeak: mPeak, ddTrough: mTrough, recoveryDays: rec,
    worstYear: full.length ? full.reduce((a, b) => (b.ret < a.ret ? b : a)) : null,
    bestYear: full.length ? full.reduce((a, b) => (b.ret > a.ret ? b : a)) : null,
    worst3m: w3, upYears: full.length ? full.filter((y) => y.ret > 0).length / full.length : 0, years: span, series,
  };
}

// ------------------------------------------------------------------ "Will my money last?" (retirement withdrawals)
export type SwpResult = { prob: number; years: number; p10: number[]; p50: number[]; p90: number[]; medianEnd: number; runOutMedianYear: number | null; safeMonthly: number };

function swpRun(eng: Engine, hs: Holding[], corpus: number, monthly: number, years: number, inflation: number, paths: number, seed: number) {
  const N = eng.N, p = prepare(hs, N), r = rng(seed), M = years * 12, nWin = Math.floor(eng.nMonth / 3);
  const hr = new Array(p.H).fill(0);
  const yearly = Array.from({ length: years + 1 }, () => new Float64Array(paths));
  let ok = 0;
  const runOut: number[] = [];
  for (let s = 0; s < paths; s++) {
    let V = corpus, w = monthly, win = 0, pos = 3, dead = -1;
    yearly[0][s] = V;
    for (let m = 1; m <= M; m++) {
      if (pos >= 3) { win = Math.floor(r.u() * nWin); pos = 0; }
      holdingReturns(p, eng.genMonth, win * 3 + pos, 1, r.n, hr);
      pos++;
      let R = 0;
      for (let k = 0; k < p.H; k++) R += p.w[k] * hr[k];
      V = Math.max(0, (V - w) * (1 + R));
      if (V <= 0 && dead < 0) dead = m;
      if (m % 12 === 0) { w *= 1 + inflation; yearly[m / 12][s] = V; }
    }
    if (dead < 0) ok++; else runOut.push(dead / 12);
  }
  return { ok: ok / paths, yearly, runOut };
}

export function simulateSwp(eng: Engine, hs: Holding[], corpus: number, monthly: number, years: number, inflation = 0.06): SwpResult {
  const P = 1500;
  const { ok, yearly, runOut } = swpRun(eng, hs, corpus, monthly, years, inflation, P, 21);
  const q = (a: Float64Array, f: number) => { const b = Float64Array.from(a).sort(); return b[Math.floor(f * (b.length - 1))]; };
  runOut.sort((a, b) => a - b);
  // the largest starting withdrawal that lasts the full period in 90% of futures
  let lo = 0, hi = Math.max(monthly * 3, corpus / (years * 12));
  for (let i = 0; i < 14; i++) {
    const mid = (lo + hi) / 2;
    if (swpRun(eng, hs, corpus, mid, years, inflation, 500, 9).ok >= 0.9) lo = mid; else hi = mid;
  }
  return {
    prob: ok, years, p10: yearly.map((c) => q(c, 0.1)), p50: yearly.map((c) => q(c, 0.5)), p90: yearly.map((c) => q(c, 0.9)),
    medianEnd: q(yearly[years], 0.5), runOutMedianYear: runOut.length > P * 0.5 ? runOut[Math.floor(runOut.length / 2)] : null,
    safeMonthly: Math.floor(lo / 500) * 500,
  };
}

// ------------------------------------------------------------------ behaviour gap: panic-selling in 2020, replayed on real data
export type GapResult = { months: string[]; stayed: number[]; panicked: number[]; stayedEnd: number; panickedEnd: number; cost: number; sellAt: string; backAt: string };

export function behaviourGap(eng: Engine, hs: Holding[], sip: number): GapResult | null {
  const N = eng.N, p = prepare(hs, N), T = eng.realMonths.length, hr = new Array(p.H).fill(0);
  const i0 = eng.realMonths.indexOf("2019-01");
  const iSell = eng.realMonths.indexOf("2020-03"), iBack = eng.realMonths.indexOf("2020-12");
  if (i0 < 0 || iSell < 0 || iBack < 0) return null;
  const zero = () => 0;
  let A = 0, B = 0, cashB = 0;
  const stayed: number[] = [], panicked: number[] = [], months: string[] = [];
  for (let m = i0; m < T; m++) {
    holdingReturns(p, eng.realMonth, m, 1, zero, hr);
    let R = 0;
    for (let k = 0; k < p.H; k++) R += p.w[k] * hr[k];
    A = A * (1 + R) + sip;
    if (m < iSell) B = B * (1 + R) + sip;
    else if (m === iSell) { B = B * (1 + R); cashB += B + sip; B = 0; }        // sells at the March 2020 low, stops the SIP
    else if (m < iBack) { cashB += sip; }                                         // sits in cash, SIP paused
    else if (m === iBack) { B = cashB * (1 + R) + sip; cashB = 0; }              // comes back after the rally
    else B = B * (1 + R) + sip;
    stayed.push(A); panicked.push(B + cashB); months.push(eng.realMonths[m]);
  }
  return { months, stayed, panicked, stayedEnd: A, panickedEnd: B + cashB, cost: A - (B + cashB), sellAt: "Mar 2020", backAt: "Dec 2020" };
}

// ------------------------------------------------------------------ plan sensitivity and outcome spread
export function chanceCurve(eng: Engine, hs: Holding[], g: GoalInput): { sip: number; prob: number }[] {
  const base = Math.max(g.sip, 1000);
  return [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3].map((f) => ({ sip: Math.round(base * f / 500) * 500, prob: simulateGoal(eng, hs, { ...g, sip: base * f }, 600, 13).prob }));
}

export function histogram(values: Float64Array, bins = 24): { lo: number; hi: number; counts: number[] } {
  const v = Float64Array.from(values).sort();
  const lo = v[Math.floor(v.length * 0.01)], hi = v[Math.floor(v.length * 0.99)];
  const counts = new Array(bins).fill(0);
  for (const x of v) { const b = Math.floor(((x - lo) / Math.max(hi - lo, 1e-9)) * bins); if (b >= 0 && b < bins) counts[b]++; }
  return { lo, hi, counts };
}
