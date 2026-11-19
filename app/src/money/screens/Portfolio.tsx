import { useEffect, useState } from "react";
import { Loader2, Plus, Search, Trash2, X } from "lucide-react";
import { fundProfile, rs, searchFunds, type Engine, type FundMatch, type Holding } from "../engine";
import { catalogue, stock, type Profile } from "../store";
import { Sheet } from "../ui";

export function PortfolioSheet({ open, onClose, eng, profile, update }:
  { open: boolean; onClose: () => void; eng: Engine; profile: Profile; update: (p: Partial<Profile>) => void }) {
  const [hs, setHs] = useState<Holding[]>(profile.holdings);
  const [tab, setTab] = useState<"basket" | "fund" | "stock">("fund");
  const [q, setQ] = useState("");
  const [matches, setMatches] = useState<FundMatch[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => { if (open) { setHs(profile.holdings); setMsg(null); } }, [open, profile.holdings]);

  useEffect(() => {
    if (tab !== "fund" || q.trim().length < 3) { setMatches([]); return; }
    const t = setTimeout(() => {
      setBusy("search");
      searchFunds(q.trim()).then((m) => setMatches(m.slice(0, 8))).catch(() => setMsg("Fund search needs an internet connection.")).finally(() => setBusy(null));
    }, 350);
    return () => clearTimeout(t);
  }, [q, tab]);

  const has = (id: string) => hs.some((h) => h.id === id);
  const add = (h: Holding) => { if (!has(h.id)) setHs((x) => [...x, { ...h, amount: h.amount || 50000 }]); };
  const addFund = async (m: FundMatch) => {
    setBusy(String(m.schemeCode)); setMsg(null);
    try {
      const f = await fundProfile(eng, m.schemeCode);
      add({ id: `mf-${m.schemeCode}`, kind: "fund", label: f.name.length > 46 ? f.name.slice(0, 45) + "…" : f.name,
        sub: `${f.category || "Mutual fund"} · moves ${f.beta.toFixed(2)}× the big-company market`, amount: 50000, beta: f.beta, residM: f.residM });
      setMsg(`Added. Measured from ${f.days} days of real NAV history: it moves about ${f.beta.toFixed(2)}× the big-company market.`);
    } catch (e) { setMsg(String((e as Error).message ?? e)); }
    setBusy(null);
  };
  const save = () => { update({ holdings: hs.filter((h) => h.amount > 0) }); onClose(); };
  const total = hs.reduce((a, h) => a + h.amount, 0);
  const stocks = eng.assets.map((a, i) => ({ a, i })).filter(({ a }) => !q || `${a.name} ${a.ticker} ${a.sector}`.toLowerCase().includes(q.toLowerCase()));

  return (
    <Sheet open={open} onClose={onClose}>
      <div className="row between"><h1 className="title" style={{ fontSize: 26 }}>Your portfolio</h1>
        <button className="btn small ghost" onClick={onClose}><X size={16} /></button></div>
      <p className="muted">Total {rs(total)}. Enter what you hold in each (rough numbers are fine).</p>
      <div className="stack" style={{ margin: "12px 0 18px" }}>
        {hs.map((h, k) => (
          <div key={h.id} className="list-row">
            <span style={{ minWidth: 0 }}><b>{h.label}</b><div className="muted">{h.sub}</div></span>
            <span className="row" style={{ gap: 6, flexWrap: "nowrap" }}>
              <input className="input" style={{ width: 120, padding: "9px 10px", fontSize: 14 }} type="number" min={0} step={1000} value={h.amount}
                onChange={(e) => setHs((x) => x.map((y, j) => (j === k ? { ...y, amount: Math.max(0, Number(e.target.value)) } : y)))} />
              <button className="btn small ghost" onClick={() => setHs((x) => x.filter((_, j) => j !== k))} aria-label="remove"><Trash2 size={14} /></button>
            </span>
          </div>
        ))}
      </div>

      <div className="row" style={{ gap: 6, marginBottom: 10 }}>
        {(["fund", "basket", "stock"] as const).map((t) => (
          <button key={t} className={`chip ${tab === t ? "on" : ""}`} onClick={() => { setTab(t); setQ(""); }}>
            {t === "fund" ? "Real mutual funds" : t === "basket" ? "Ready baskets" : "Single stocks"}</button>
        ))}
      </div>
      {tab !== "basket" && (
        <div className="row" style={{ flexWrap: "nowrap", marginBottom: 10 }}>
          <Search size={18} color="#7d86a6" />
          <input className="input" placeholder={tab === "fund" ? "Search e.g. 'Parag Parikh', 'Nifty 50 index', 'HDFC flexi'" : "Search stock or sector"}
            value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
      )}
      <div className="stack">
        {tab === "basket" && catalogue(eng).map((h) => (
          <button key={h.id} className="list-row" style={{ width: "100%", textAlign: "left" }} disabled={has(h.id)} onClick={() => add(h)}>
            <span><b>{h.label}</b><div className="muted">{h.sub}</div></span>{has(h.id) ? <span className="pill good">added</span> : <Plus size={18} />}
          </button>
        ))}
        {tab === "stock" && stocks.map(({ a, i }) => {
          const h = stock(eng, i, 0);
          return (
            <button key={a.ticker} className="list-row" style={{ width: "100%", textAlign: "left" }} disabled={has(h.id)} onClick={() => add(h)}>
              <span><b>{a.name}</b><div className="muted">{a.sector} · {a.ticker}</div></span>{has(h.id) ? <span className="pill good">added</span> : <Plus size={18} />}
            </button>
          );
        })}
        {tab === "fund" && (
          <>
            {busy === "search" && <p className="muted"><Loader2 size={14} className="spin" /> searching…</p>}
            {matches.map((m) => (
              <button key={m.schemeCode} className="list-row" style={{ width: "100%", textAlign: "left" }} disabled={!!busy || has(`mf-${m.schemeCode}`)} onClick={() => addFund(m)}>
                <span><b style={{ fontSize: 14 }}>{m.schemeName}</b><div className="muted">AMFI code {m.schemeCode}</div></span>
                {busy === String(m.schemeCode) ? <Loader2 size={18} /> : has(`mf-${m.schemeCode}`) ? <span className="pill good">added</span> : <Plus size={18} />}
              </button>
            ))}
            {q.trim().length < 3 && <p className="muted">Type at least 3 letters. We read the fund's real daily NAV history (public AMFI data via mfapi.in)
              and measure how strongly it moves with the market, then put it through our crash scenarios.</p>}
          </>
        )}
      </div>
      {msg && <p className="sub" style={{ marginTop: 12 }}>{msg}</p>}
      <div className="row between" style={{ marginTop: 18 }}>
        <span className="muted">Changes update every screen.</span>
        <button className="btn" onClick={save} disabled={hs.filter((h) => h.amount > 0).length === 0}>Save portfolio</button>
      </div>
    </Sheet>
  );
}
