import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Home as HomeIco, Target, Zap, Network, ShieldCheck, ScanSearch, Sunset, MoreHorizontal, Cloud, CloudOff, Flame } from "lucide-react";
import { diversification, historyReplay, loadEngine, simulateCrash, simulateGoal, type CrashResult, type DiversResult, type Engine,
  type GoalResult, type HistoryResult } from "./engine";
import { isoWeek, LEVEL_TITLES, levelOf, today, usePersisted, weekMissions, type Profile, type Progress } from "./store";
import { hasBackend, pullState, pushState, session } from "./api";
import { Confetti, Sheet } from "./ui";
import { LogoMark, Wordmark } from "./brand";
import { Onboarding } from "./screens/Onboarding";
import { Home } from "./screens/Home";
import { Goal } from "./screens/Goal";
import { Crash } from "./screens/Crash";
import { Diversify } from "./screens/Diversify";
import { Reality } from "./screens/Reality";
import { FundXrayScreen } from "./screens/FundXray";
import { Retire } from "./screens/Retire";
import { PortfolioSheet } from "./screens/Portfolio";
import { AccountSheet } from "./screens/Account";

export type Tab = "home" | "goal" | "crash" | "divers" | "funds" | "retire" | "reality";
export type Track = (event: string) => void;

export const BADGES = [
  { id: "goal", label: "On track", hint: "75%+ chance to reach your goal" },
  { id: "crash", label: "Crash-ready", hint: "Make a crash plan" },
  { id: "divers", label: "Truly diversified", hint: "Behave like 3+ holdings in a crash" },
  { id: "scam", label: "Scam-proof", hint: "Run a tip reality check" },
  { id: "explorer", label: "Explorer", hint: "Visit every section" },
  { id: "safety", label: "Safety net", hint: "Have an emergency fund" },
  { id: "streak", label: "7-day streak", hint: "Open the app 7 days in a row" },
  { id: "xray", label: "Fund detective", hint: "X-ray a real mutual fund" },
];

type Toast = { id: number; text: string; sub?: string };
const NAV = [["home", "Overview", HomeIco], ["goal", "Goal planner", Target], ["crash", "Crash test", Zap],
  ["divers", "Diversification", Network], ["funds", "Fund X-ray", ScanSearch], ["retire", "Retirement", Sunset],
  ["reality", "Reality check", ShieldCheck]] as const;
const MOBILE: Tab[] = ["home", "goal", "crash", "funds"];

export default function App() {
  const [eng, setEng] = useState<Engine | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [profile, setProfile] = usePersisted<Profile | null>("profile", null);
  const [prog, setProg] = usePersisted<Progress>("progress", { xp: 0, seen: [], badges: [], tipChecks: 0, quiz: {} });
  const [tab, setTab] = useState<Tab>("home");
  const [editing, setEditing] = useState(false);
  const [account, setAccount] = useState(false);
  const [more, setMore] = useState(false);
  const [fire, setFire] = useState(0);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [levelUp, setLevelUp] = useState<number | null>(null);
  const [cloud, setCloud] = useState<{ on: boolean; email?: string; version: number; status: string }>({ on: false, version: 0, status: "local" });
  const first = useRef(true);
  const lastLevel = useRef<number | null>(null);

  const toast = useCallback((text: string, sub?: string) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t.slice(-2), { id, text, sub }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3200);
  }, []);

  useEffect(() => { loadEngine().then(setEng).catch((e) => setErr(String(e))); }, []);

  // daily streak (rewards showing up, never trading)
  useEffect(() => {
    if (!profile) return;
    const d = today();
    setProg((p) => {
      const s = p.streak;
      if (s?.last === d) return p;
      const y = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
      const count = s && s.last === y ? s.count + 1 : 1;
      if (count > 1) setTimeout(() => toast(`${count}-day streak`, "+5 XP for showing up"), 900);
      return { ...p, xp: p.xp + 5, streak: { last: d, count, best: Math.max(count, s?.best ?? 0) } };
    });
  }, [profile !== null]); // eslint-disable-line react-hooks/exhaustive-deps

  const results = useMemo<Results | null>(() => {
    if (!eng || !profile || profile.holdings.length === 0) return null;
    const g = { start: profile.start, sip: profile.sip, years: profile.years, goal: profile.goal, inflation: profile.inflation, stepUp: profile.stepUp };
    const goal = simulateGoal(eng, profile.holdings, g);
    const crash = simulateCrash(eng, profile.holdings);
    const div = diversification(eng, profile.holdings);
    const hist = historyReplay(eng, profile.holdings);
    const sGoal = 35 * Math.min(1, goal.prob / 0.85);
    const bad = Math.abs(crash.troughBad);
    const sCrash = 25 * Math.max(0, Math.min(1, bad <= profile.tolerance ? 1 : 1 - (bad - profile.tolerance) / profile.tolerance));
    const sDiv = div.nHold === 0 ? 20 : 20 * Math.min(1, div.crash / Math.min(3, Math.max(div.nHold, 1)));
    const sHab = (profile.crashPlan ? 10 : 0) + (profile.emergency ? 10 : 0);
    const health = Math.round(sGoal + sCrash + sDiv + sHab);
    return { goal, crash, div, hist, health, parts: { sGoal, sCrash, sDiv, sHab } };
  }, [eng, profile]);

  // one Money Health snapshot per day (the line on the overview)
  useEffect(() => {
    if (!results) return;
    const d = today();
    setProg((p) => {
      const h = (p.history ?? []).filter((x) => x.d !== d);
      return { ...p, history: [...h, { d, h: results.health }].slice(-120) };
    });
  }, [results?.health]); // eslint-disable-line react-hooks/exhaustive-deps

  // badges
  useEffect(() => {
    if (!results || !profile) return;
    const earned = new Set(prog.badges);
    const add = (id: string, ok: boolean) => { if (ok) earned.add(id); };
    add("goal", results.goal.prob >= 0.75);
    add("crash", profile.crashPlan);
    add("divers", results.div.crash >= 3);
    add("scam", prog.tipChecks > 0);
    add("explorer", ["goal", "crash", "divers", "funds", "retire", "reality"].every((t) => prog.seen.includes(t)));
    add("safety", profile.emergency);
    add("streak", (prog.streak?.best ?? 0) >= 7);
    add("xray", (prog.missions ?? []).includes("xray") || prog.seen.includes("xray-done"));
    const fresh = [...earned].filter((b) => !prog.badges.includes(b));
    if (fresh.length) {
      setProg((p) => ({ ...p, badges: [...earned], xp: p.xp + 40 * fresh.length }));
      if (!first.current) {
        setFire((f) => f + 1);
        fresh.forEach((b) => toast(`Badge unlocked: ${BADGES.find((x) => x.id === b)?.label}`, "+40 XP"));
      }
    }
    first.current = false;
  }, [results, profile, prog.badges, prog.tipChecks, prog.seen, prog.streak, prog.missions, setProg, toast]);

  // level-up moment
  const lv = levelOf(prog.xp);
  useEffect(() => {
    if (lastLevel.current !== null && lv.level > lastLevel.current) { setLevelUp(lv.level); setFire((f) => f + 1); }
    lastLevel.current = lv.level;
  }, [lv.level]);

  // weekly missions
  const week = isoWeek();
  const missions = weekMissions(week);
  const track: Track = useCallback((event: string) => {
    setProg((p) => {
      const done = p.week === week ? p.missions ?? [] : [];
      const m = weekMissions(week).find((x) => x.id === event);
      const seen = event === "xray" && !p.seen.includes("xray-done") ? [...p.seen, "xray-done"] : p.seen;
      if (!m || done.includes(event)) return { ...p, week, missions: done, seen };
      setTimeout(() => toast(`Mission complete: ${m.label}`, `+${m.xp} XP`), 50);
      return { ...p, week, missions: [...done, event], xp: p.xp + m.xp, seen };
    });
  }, [week, setProg, toast]);

  // cloud sync (only when the backend runs and the user signed in)
  useEffect(() => {
    hasBackend().then(async (ok) => {
      const s = session.get();
      if (!ok || !s) { setCloud((c) => ({ ...c, on: false, status: ok ? "signed out" : "local" })); return; }
      try {
        const r = await pullState();
        const st = r.state as { profile?: Profile; progress?: Progress } | null;
        if (st?.profile) { setProfile(st.profile); if (st.progress) setProg(st.progress); }
        setCloud({ on: true, email: s.email, version: r.version, status: "synced" });
      } catch { setCloud((c) => ({ ...c, on: false, status: "sign in again" })); session.set(null); }
    });
  }, [account]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!cloud.on || !profile) return;
    const t = setTimeout(async () => {
      try { const r = await pushState({ profile, progress: prog }, cloud.version); setCloud((c) => ({ ...c, version: r.version, status: "synced" })); }
      catch (e) { if ((e as { status?: number }).status === 409) { const r = await pullState(); setCloud((c) => ({ ...c, version: r.version, status: "updated from another device" })); } else setCloud((c) => ({ ...c, status: "offline, will retry" })); }
    }, 1200);
    return () => clearTimeout(t);
  }, [profile, prog, cloud.on]); // eslint-disable-line react-hooks/exhaustive-deps

  const go = (t: Tab) => {
    setTab(t); setMore(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (t !== "home" && !prog.seen.includes(t)) setProg((p) => ({ ...p, seen: [...p.seen, t], xp: p.xp + 10 }));
  };

  if (err) return <div className="onb"><div className="card panel"><h1 className="title">Could not load the scenario engine</h1><p className="sub">{err}</p>
    <p className="muted">Run <code>python -m crisisbench.export_money</code> from the project folder, then reload.</p></div></div>;
  if (!eng) return (
    <div className="onb"><div className="panel stack" style={{ textAlign: "center" }}>
      <div style={{ animation: "spin 2.4s linear infinite", display: "inline-block" }}><LogoMark size={56} /></div>
      <p className="muted">Loading thousands of market futures…</p></div></div>
  );
  if (!profile) return <Onboarding eng={eng} onDone={(p) => { setProfile(p); setProg((x) => ({ ...x, xp: x.xp + 20 })); setFire((f) => f + 1); }} />;

  const update = (patch: Partial<Profile>) => setProfile({ ...profile, ...patch });
  const award = (xp: number, why?: string) => { setProg((p) => ({ ...p, xp: p.xp + xp })); if (why) toast(why, `+${xp} XP`); };
  const done = prog.week === week ? prog.missions ?? [] : [];
  const lvBox = (
    <div className="row" style={{ gap: 10, flexWrap: "nowrap" }}>
      <div className="lvchip">{lv.level}</div>
      <div style={{ flex: 1, minWidth: 90 }}><div className="row between" style={{ fontSize: 12.5 }}><span>{LEVEL_TITLES[lv.level - 1]}</span>
        <span className="muted"><Flame size={11} style={{ verticalAlign: -1, color: "var(--warn)" }} /> {prog.streak?.count ?? 1}</span></div>
        <div className="bar" style={{ marginTop: 6 }}><i style={{ width: `${lv.frac * 100}%` }} /></div></div>
    </div>
  );
  const shared = { eng, profile, update, track };

  return (
    <>
      <Confetti fire={fire} />
      <div className="app">
        <aside className="side">
          <div className="brand"><Wordmark size={30} /></div>
          {NAV.map(([k, l, Icon]) => <button key={k} className={`navi ${tab === k ? "on" : ""}`} onClick={() => go(k)}><Icon />{l}</button>)}
          <div className="foot stack">
            {lvBox}
            <div className="row" style={{ gap: 6 }}>
              <button className="btn ghost small" style={{ flex: 1 }} onClick={() => setEditing(true)}>Portfolio</button>
              <button className="btn ghost small" style={{ flex: 1 }} onClick={() => setAccount(true)} title={cloud.status}>
                {cloud.on ? <Cloud size={14} /> : <CloudOff size={14} />}{cloud.on ? "Synced" : "Sync"}</button>
            </div>
          </div>
        </aside>
        <main className="main">
          <div className="mobhead"><Wordmark size={28} /><div style={{ width: 132 }}>{lvBox}</div></div>
          {!results ? (
            <div className="card"><p className="sub">Add at least one holding to see your plan.</p><button className="btn" onClick={() => setEditing(true)}>Add holdings</button></div>
          ) : (
            <div key={tab} className="page">
              {tab === "home" && <Home {...shared} prog={prog} r={results} go={go} edit={() => setEditing(true)} award={award} setProg={setProg}
                missions={missions} done={done} toast={toast} />}
              {tab === "goal" && <Goal {...shared} r={results} />}
              {tab === "crash" && <Crash {...shared} r={results} />}
              {tab === "divers" && <Diversify {...shared} r={results} edit={() => setEditing(true)} />}
              {tab === "funds" && <FundXrayScreen {...shared} />}
              {tab === "retire" && <Retire {...shared} />}
              {tab === "reality" && <Reality eng={eng} onCheck={() => { setProg((p) => ({ ...p, tipChecks: p.tipChecks + 1, xp: p.xp + 10 })); track("tip"); }} />}
            </div>
          )}
          <p className="disclaimer">Educational simulator built on the VORTEX-AI research model (CG-NSDE). Scenarios are simulations, not predictions
            or investment advice. Fund data: AMFI via mfapi.in. Real market history is used for educational purposes only.</p>
        </main>
      </div>

      <nav className="tabbar">
        {MOBILE.map((k) => { const [, l, Icon] = NAV.find((n) => n[0] === k)!; return <button key={k} className={tab === k ? "on" : ""} onClick={() => go(k)}><Icon />{l.split(" ")[0]}</button>; })}
        <button className={!MOBILE.includes(tab) ? "on" : ""} onClick={() => setMore(true)}><MoreHorizontal />More</button>
      </nav>

      <div className="toasts">{toasts.map((t) => <div key={t.id} className="toast"><b>{t.text}</b>{t.sub && <span>{t.sub}</span>}</div>)}</div>

      <Sheet open={more} onClose={() => setMore(false)}>
        <div className="stack">
          {NAV.filter(([k]) => !MOBILE.includes(k)).map(([k, l, Icon]) => <button key={k} className="navi" onClick={() => go(k)}><Icon />{l}</button>)}
          <button className="navi" onClick={() => { setMore(false); setEditing(true); }}>Edit portfolio</button>
          <button className="navi" onClick={() => { setMore(false); setAccount(true); }}>{cloud.on ? `Synced as ${cloud.email}` : "Sign in to sync"}</button>
        </div>
      </Sheet>
      <Sheet open={levelUp !== null} onClose={() => setLevelUp(null)}>
        <div className="stack" style={{ textAlign: "center", padding: "10px 0" }}>
          <div className="levelburst">{levelUp}</div>
          <h1 className="title">Level up: {LEVEL_TITLES[(levelUp ?? 1) - 1]}</h1>
          <p className="sub">Every level is earned by building good money habits, never by trading more.</p>
          <button className="btn" onClick={() => setLevelUp(null)}>Keep going</button>
        </div>
      </Sheet>
      <PortfolioSheet open={editing} onClose={() => setEditing(false)} eng={eng} profile={profile} update={update} />
      <AccountSheet open={account} onClose={() => setAccount(false)} cloud={cloud} onSignedOut={() => setCloud({ on: false, version: 0, status: "signed out" })} />
    </>
  );
}

export type Results = {
  goal: GoalResult; crash: CrashResult; div: DiversResult; hist: HistoryResult; health: number;
  parts: { sGoal: number; sCrash: number; sDiv: number; sHab: number };
};
