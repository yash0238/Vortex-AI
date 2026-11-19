import { useEffect, useMemo, useState } from "react";
import { Loader2, Plus, Search, ScanSearch } from "lucide-react";
import { fundNav, fundProfile, pct, rs, searchFunds, simulateCrash, type Engine, type FundMatch, type Holding } from "../engine";
import { fundXray, type FundXray } from "../analytics";
import type { Profile } from "../store";
import { AreaChart, Count } from "../ui";
import type { Track } from "../App";

const POPULAR = ["Nifty 50 index", "Parag Parikh Flexi", "HDFC Flexi Cap", "SBI Small Cap", "ICICI Prudential Bluechip", "Liquid fund"];

export function FundXrayScreen({ eng, profile, update, track }: { eng: Engine; profile: Profile; update: (p: Partial<Profile>) => void; track: Track }) {
  const [q, setQ] = useState("");
  const [matches, setMatches] = useState<FundMatch[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [fund, setFund] = useState<{ code: number; name: string; category: string; house: string; x: FundXray; beta: number; residM: number } | null>(null);
  const [range, setRange] = useState<"3y" | "5y" | "all">("5y");

  useEffect(() => {
    if (q.trim().length < 3) { setMatches([]); return; }
    const t = setTimeout(() => {
      setBusy("search"); setErr(null);
      searchFunds(q.trim()).then((m) => setMatches(m.slice(0, 7))).catch(() => setErr("Fund search needs an internet connection.")).finally(() => setBusy(null));
    }, 300);
    return () => clearTimeout(t);
  }, [q]);

  const open = async (m: FundMatch) => {
    setBusy(String(m.schemeCode)); setErr(null);
    try {
      const [nav, prof] = await Promise.all([fundNav(m.schemeCode), fundProfile(eng, m.schemeCode).catch(() => null)]);
      if (nav.navs.length < 60) throw new Error("This fund has too little history to X-ray.");
      setFund({ code: m.schemeCode, name: nav.name, category: nav.category, house: nav.house, x: fundXray(nav.navs), beta: prof?.beta ?? 1, residM: prof?.residM ?? 0.02 });
      setMatches([]); setQ("");
      track("xray");
    } catch (e) { setErr((e as Error).message); }
    setBusy(null);
  };

  const crash = useMemo(() => {
    if (!fund) return null;
    const h: Holding = { id: "x", kind: "fund", label: fund.name, sub: "", amount: 100000, beta: fund.beta, residM: fund.residM };
    return simulateCrash(eng, [h]);
  }, [fund, eng]);
  const held = fund ? profile.holdings.some((h) => h.id === `mf-${fund.code}`) : false;
  const add = () => {
    if (!fund || held) return;
    update({ holdings: [...profile.holdings, { id: `mf-${fund.code}`, kind: "fund", label: fund.name.length > 46 ? fund.name.slice(0, 45) + "…" : fund.name,
      sub: `${fund.category || "Mutual fund"} · moves ${fund.beta.toFixed(2)}× the market`, amount: 50000, beta: fund.beta, residM: fund.residM }] });
  };
  const series = useMemo(() => {
    if (!fund) return [];
    const cut = range === "all" ? "" : new Date(Date.now() - (range === "3y" ? 3 : 5) * 365.25 * 86400000).toISOString().slice(0, 10);
    return fund.x.series.filter((s) => s.date >= cut);
  }, [fund, range]);
  const c = (v: number | null) => (v === null ? "–" : pct(v, 1));
  const verdict = fund ? (fund.x.maxDD > -0.12 ? ["Calm", "good"] : fund.x.maxDD > -0.3 ? ["Moderate swings", "warn"] : ["Big crash risk", "bad"]) : null;

  return (
    <>
      <div className="pagehead"><div><div className="eyebrow">Look inside any Indian mutual fund before you invest</div><h1 className="title">Fund X-ray</h1></div></div>
      <div className="grid">
        <div className="card c12 hero">
          <div className="row" style={{ flexWrap: "nowrap" }}>
            <Search size={20} color="var(--muted)" />
            <input className="input" style={{ fontSize: 17, padding: "15px 16px" }} placeholder="Search any fund: e.g. Parag Parikh, Nifty 50 index, SBI small cap"
              value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
          {matches.length === 0 && !fund && (
            <div className="row" style={{ marginTop: 14 }}>{POPULAR.map((p) => <button key={p} className="chip" onClick={() => setQ(p)}>{p}</button>)}</div>
          )}
          {busy === "search" && <p className="muted" style={{ marginTop: 10 }}><Loader2 size={13} className="spin" /> searching AMFI data…</p>}
          <div className="stack" style={{ marginTop: matches.length ? 12 : 0 }}>
            {matches.map((m) => (
              <button key={m.schemeCode} className="list-row" style={{ width: "100%", textAlign: "left" }} disabled={!!busy} onClick={() => open(m)}>
                <span style={{ fontSize: 14 }}>{m.schemeName}</span>
                {busy === String(m.schemeCode) ? <Loader2 size={16} className="spin" /> : <ScanSearch size={16} color="var(--lime)" />}
              </button>
            ))}
          </div>
          {err && <p className="sub badtext" style={{ marginTop: 10 }}>{err}</p>}
        </div>

        {!fund && (
          <div className="card c12 empty">
            <ScanSearch size={34} color="var(--lime)" />
            <h2>Check the worst year before you invest</h2>
            <p className="sub">SEBI's own advice: look at a fund's worst quarter and worst year, then ask yourself how you'd react to that loss.
              X-ray shows that, plus how it would behave in our simulated crashes.</p>
          </div>
        )}

        {fund && crash && (
          <>
            <div className="card c12">
              <div className="row between">
                <div><div className="lbl">{fund.house}</div><h2 style={{ margin: "4px 0 6px", fontSize: 22, letterSpacing: "-0.02em" }}>{fund.name}</h2>
                  <div className="row" style={{ gap: 8 }}><span className="pill">{fund.category || "Mutual fund"}</span>
                    <span className={`pill ${verdict![1]}`}>{verdict![0]}</span><span className="pill">{fund.x.years.toFixed(1)} years of data</span></div></div>
                <button className="btn" onClick={add} disabled={held}><Plus size={16} />{held ? "In portfolio" : "Add to portfolio"}</button>
              </div>
            </div>
            <div className="card c3 stat delay1"><h3>1-year return</h3><div className="v">{c(fund.x.cagr.y1)}</div><div className="muted">3y {c(fund.x.cagr.y3)} · 5y {c(fund.x.cagr.y5)} a year</div></div>
            <div className="card c3 stat delay2"><h3>Worst calendar year</h3><div className="v badtext">{fund.x.worstYear ? pct(fund.x.worstYear.ret, 1) : "–"}</div>
              <div className="muted">{fund.x.worstYear?.year ?? ""} · best {fund.x.bestYear ? `${pct(fund.x.bestYear.ret, 0)} (${fund.x.bestYear.year})` : "–"}</div></div>
            <div className="card c3 stat delay3"><h3>Worst fall ever</h3><div className="v badtext"><Count value={fund.x.maxDD * 100} fmt={(v) => `${v.toFixed(1)}%`} /></div>
              <div className="muted">{fund.x.recoveryDays === null ? "not yet recovered" : `recovered in ${Math.round(fund.x.recoveryDays / 30)} months`}</div></div>
            <div className="card c3 stat delay4"><h3>In a severe crash (model)</h3><div className="v badtext">{pct(crash.troughBad)}</div>
              <div className="muted">₹1 lakh → {rs(100000 * (1 + crash.troughBad))} · moves {fund.beta.toFixed(2)}× market</div></div>

            <div className="card c8 delay2">
              <div className="row between"><h3>Growth of the fund's NAV</h3>
                <div className="seg">{(["3y", "5y", "all"] as const).map((r) => <button key={r} className={range === r ? "on" : ""} onClick={() => setRange(r)}>{r === "all" ? "All" : r.toUpperCase()}</button>)}</div></div>
              <AreaChart xLabels={series.map((s) => s.date.slice(0, 7))} fmt={(v) => `₹${v.toFixed(v < 100 ? 1 : 0)}`} height={240} fillUnder
                series={[{ values: series.map((s) => s.nav), color: "#a6d43a", label: "NAV" }]} />
            </div>
            <div className="card c4 delay3">
              <h3>How far below its peak</h3>
              <AreaChart xLabels={series.map((s) => s.date.slice(0, 7))} fmt={(v) => pct(v)} height={240}
                series={[{ values: series.map((s) => s.dd), color: "#e2475a", label: "Below peak" }]} />
            </div>
            <div className="card c12 delay4">
              <div className="row" style={{ gap: 36 }}>
                <div className="kpi"><b>{pct(fund.x.vol, 0)}</b><span>yearly swings (volatility)</span></div>
                <div className="kpi"><b className="badtext">{pct(fund.x.worst3m, 1)}</b><span>worst 3 months</span></div>
                <div className="kpi"><b className="goodtext">{pct(fund.x.upYears)}</b><span>of calendar years ended up</span></div>
                <div className="kpi"><b>{fund.x.ddPeak.slice(0, 7)} → {fund.x.ddTrough.slice(0, 7)}</b><span>its worst fall</span></div>
              </div>
              <p className="muted" style={{ marginTop: 12 }}>Past NAV history from AMFI. Returns exclude exit loads and taxes. Past performance doesn't guarantee future returns.</p>
            </div>
          </>
        )}
      </div>
    </>
  );
}
