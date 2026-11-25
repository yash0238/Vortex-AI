import { lazy, Suspense, useEffect, useState } from "react";
import { Activity, BarChart3, BookOpen, Database, FlaskConical, Globe2, Home, LineChart as LineIcon, Menu, Moon, Network, Scale, ShieldAlert, Sun, Wallet, Workflow, Zap } from "lucide-react";
import { Glossary, Honest, How, Overview } from "./pages/Story";
import { Benchmark, Checks, CrisisTest, DataPage, Markets, Model, Risk } from "./pages/Evidence";
import { Lab } from "./pages/Lab";
import "./legacy.css";

const MarketExplorer = lazy(() => import("./pages/MarketExplorer"));
const MutualFunds = lazy(() => import("./pages/MutualFunds"));
const Legacy = ({ children }: { children: React.ReactNode }) => (
  <><div className="callout">Side tool, separate from the research results. It needs the Python API server and an internet connection; quotes come from Yahoo Finance and are not exchange-grade.</div>
    <div className="legacy-tools"><Suspense fallback={<div>Loading…</div>}>{children}</Suspense></div></>
);

const NAV = [
  { group: "Story", items: [
    { h: "#/", label: "Overview", icon: Home, el: <Overview /> },
    { h: "#/how", label: "How it works", icon: Workflow, el: <How /> },
    { h: "#/lab", label: "Scenario lab", icon: FlaskConical, el: <Lab /> }] },
  { group: "Evidence", items: [
    { h: "#/data", label: "Real data", icon: Database, el: <DataPage /> },
    { h: "#/benchmark", label: "Benchmark", icon: BarChart3, el: <Benchmark /> },
    { h: "#/crisis", label: "Held-out crashes", icon: Zap, el: <CrisisTest /> },
    { h: "#/risk", label: "Risk backtest", icon: ShieldAlert, el: <Risk /> },
    { h: "#/model", label: "Our model, ablated", icon: Network, el: <Model /> },
    { h: "#/markets", label: "Other markets", icon: Globe2, el: <Markets /> },
    { h: "#/checks", label: "Sanity checks", icon: Activity, el: <Checks /> }] },
  { group: "Reference", items: [
    { h: "#/honest", label: "Claims and limits", icon: Scale, el: <Honest /> },
    { h: "#/glossary", label: "Glossary", icon: BookOpen, el: <Glossary /> }] },
  { group: "Side tools", items: [
    { h: "#/explorer", label: "Market explorer", icon: LineIcon, el: <Legacy><MarketExplorer /></Legacy> },
    { h: "#/funds", label: "Mutual funds", icon: Wallet, el: <Legacy><MutualFunds /></Legacy> }] },
];
const FLAT = NAV.flatMap((g) => g.items);

export default function App() {
  const [hash, setHash] = useState(window.location.hash || "#/");
  const [open, setOpen] = useState(false);
  const [theme, setTheme] = useState<string>(() => { try { return localStorage.getItem("theme") ?? "light"; } catch { return "light"; } });
  useEffect(() => { const f = () => { setHash(window.location.hash || "#/"); setOpen(false); window.scrollTo(0, 0); }; window.addEventListener("hashchange", f); return () => window.removeEventListener("hashchange", f); }, []);
  useEffect(() => { document.documentElement.dataset.theme = theme; try { localStorage.setItem("theme", theme); } catch { /* private mode */ } }, [theme]);
  const cur = FLAT.find((i) => i.h === hash) ?? FLAT[0];
  return (
    <div className="shell">
      <aside className="side" data-open={open}>
        <a className="brand" href="#/" style={{ color: "inherit", textDecoration: "none" }}><div className="brand-mark">V</div><div><b>VORTEX-AI</b><small>crisis-aware market simulator</small></div></a>
        {NAV.map((g) => (<nav className="nav-group" key={g.group}><div className="nav-title">{g.group}</div>
          {g.items.map((i) => (<a key={i.h} className="nav-item" href={i.h} aria-current={cur.h === i.h ? "page" : undefined}><i.icon />{i.label}</a>))}</nav>))}
        <div className="side-foot">
          <button className="btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}{theme === "dark" ? "Light theme" : "Dark theme"}</button>
          <span>Yashovardhan Thopte · Alok Shukla · Aaryan Singh<br />Dept. of Computer Engineering, Fr. CRCE, Mumbai</span>
        </div>
      </aside>
      <div className="main">
        <div className="topbar"><button className="btn" aria-label="Open navigation" onClick={() => setOpen(!open)}><Menu size={16} /></button><b>VORTEX-AI</b><span style={{ color: "var(--muted)", fontSize: 13 }}>{cur.label}</span></div>
        <main className="page" key={cur.h + theme}>{cur.el}</main>
      </div>
    </div>
  );
}
