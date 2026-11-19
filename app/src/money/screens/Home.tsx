import { useMemo, useState } from "react";
import { Award, ArrowRight, Compass, LifeBuoy, Network, Pencil, ShieldCheck, Target, Zap, Sparkles, History, Flame, Share2, CheckCircle2, Circle, ScanSearch } from "lucide-react";
import { historyReplay, pct, rs, sipFor, tipCheck, type Engine } from "../engine";
import { catalogue, GOALS, LEVEL_TITLES, levelOf, MISSIONS, type Profile, type Progress } from "../store";
import { AreaChart, Count, Ring, Spark, Toggle } from "../ui";
import { BADGES, type Results, type Tab, type Track } from "../App";
import { shareCard } from "../share";

const BADGE_ICONS: Record<string, typeof Award> = { goal: Target, crash: Zap, divers: Network, scam: ShieldCheck, explorer: Compass, safety: LifeBuoy, streak: Flame, xray: ScanSearch };
const KIND_COLOR: Record<string, string> = { basket: "#7480e8", fund: "#c6f432", stock: "#c47c0c", safe: "#6b707b" };

export function Home({ eng, profile, prog, r, go, edit, update, award, setProg, missions, done, track, toast }: {
  eng: Engine; profile: Profile; prog: Progress; r: Results; go: (t: Tab) => void; edit: () => void;
  update: (p: Partial<Profile>) => void; award: (xp: number, why?: string) => void; setProg: (f: (p: Progress) => Progress) => void;
  missions: typeof MISSIONS; done: string[]; track: Track; toast: (t: string, s?: string) => void;
}) {
  const hist = (prog.history ?? []).slice(-30);
  const lv = levelOf(prog.xp);
  const MISSION_TAB: Record<string, Tab> = { crash: "crash", xray: "funds", retire: "retire", quiz: "home", tip: "reality", goal: "goal", divers: "divers", gap: "crash" };
  const g = GOALS.find((x) => x.k === profile.goalKind) ?? GOALS[0];
  const total = profile.holdings.reduce((a, h) => a + h.amount, 0);
  const hour = new Date().getHours();
  const hello = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";
  const gap = r.goal.median - r.goal.goalNominal;
  const input = { start: profile.start, sip: profile.sip, years: profile.years, goal: profile.goal, inflation: profile.inflation, stepUp: profile.stepUp };
  const need = useMemo(() => (r.goal.prob >= 0.85 ? profile.sip : sipFor(eng, profile.holdings, input, 0.85)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [eng, profile.holdings, profile.sip, profile.years, profile.goal, profile.start, profile.stepUp, profile.inflation, r.goal.prob]);

  const action = r.goal.prob < 0.85
    ? { t: `Invest ${rs(need - profile.sip)} more a month`, s: `Takes your chance of reaching ${rs(profile.goal)} from ${pct(r.goal.prob)} to 85%.`,
        cta: "Apply", run: () => { update({ sip: need }); award(20); } }
    : !profile.crashPlan ? { t: "Lock in your crash plan", s: "Deciding before a crash is the best protection against panic selling.", cta: "Set plan", run: () => go("crash") }
    : !profile.emergency ? { t: "Build a 6-month emergency fund", s: "So a job loss never forces you to sell investments in a crash.", cta: "Mark done", run: () => { update({ emergency: true }); award(10); } }
    : { t: "You're on track", s: "Check back after big market moves, or stress-test a new fund.", cta: "Crash test", run: () => go("crash") };

  const step = Math.max(1, Math.floor(r.goal.months / 60));
  const labels = r.goal.fan.p50.map((_, i) => `Year ${Math.round((i * step) / 12)}`);
  const parts = [["Goal", r.parts.sGoal, 35], ["Crash safety", r.parts.sCrash, 25], ["Diversity", r.parts.sDiv, 20], ["Habits", r.parts.sHab, 20]] as const;
  const healthWord = r.health >= 75 ? "Strong" : r.health >= 50 ? "Fair" : "Needs work";

  const facts = useMemo(() => {
    const large = catalogue(eng).find((h) => h.id === "bsk-large")!;
    const hist = historyReplay(eng, [{ ...large, amount: 1 }]);
    const upDay = Array.from(eng.realEw).filter((v) => v > 0).length / eng.realEw.length;
    return { upYear: hist.upYears, upDay, tip: tipCheck(eng, 21.6, 4).share };
  }, [eng]);
  const quiz = [
    { id: "q1", q: "On any single day, how often did India's biggest companies rise?", opts: [0.5, facts.upDay, 0.8], a: facts.upDay,
      why: "Day to day it's nearly a coin flip, so short-term predictions fail." },
    { id: "q2", q: "Over a full year, how often were they up?", opts: [0.55, 0.7, facts.upYear], a: facts.upYear,
      why: "Over time, noise cancels and growth compounds." },
    { id: "q3", q: "How often did a big stock gain 22% in a month (\"5% a week\")?", opts: [0.2, 0.08, facts.tip], a: facts.tip,
      why: "Rare and never repeatable. Weekly-return promises are a scam signal." },
  ];
  const [qi, setQi] = useState(() => Math.max(0, quiz.findIndex((q) => !(q.id in prog.quiz))));
  const cur = quiz[qi % quiz.length];
  const answered = cur.id in prog.quiz;
  const answer = (v: number) => {
    if (answered) return;
    const ok = Math.abs(v - cur.a) < 1e-9;
    setProg((p) => ({ ...p, quiz: { ...p.quiz, [cur.id]: ok }, xp: p.xp + (ok ? 15 : 5) }));
    track("quiz");
  };

  return (
    <>
      <div className="pagehead">
        <div><div className="eyebrow">{hello}, {profile.name}</div><h1 className="title">Overview</h1></div>
        <button className="btn ghost small" onClick={edit}><Pencil size={14} /> Portfolio</button>
      </div>
      <div className="grid">
        <div className="card hero c8">
          <div className="row between" style={{ alignItems: "flex-start" }}>
            <div>
              <div className="lbl">{g.label} · projected in {profile.years} years</div>
              <div className="huge" style={{ marginTop: 8 }}><Count value={r.goal.median} fmt={(v) => rs(v)} /></div>
              <div className="row" style={{ marginTop: 10, gap: 10 }}>
                <span className={`delta ${gap >= 0 ? "up" : "down"}`}>{gap >= 0 ? `${rs(gap)} above goal` : `${rs(-gap)} short of goal`}</span>
                <span className="muted">Goal {rs(r.goal.goalNominal)} incl. inflation</span>
              </div>
            </div>
            <div className="kpi" style={{ textAlign: "right" }}>
              <span>Chance of success</span>
              <b style={{ fontSize: 30, color: r.goal.prob >= 0.75 ? "var(--good)" : r.goal.prob >= 0.5 ? "var(--warn)" : "var(--bad)" }}>
                <Count value={r.goal.prob * 100} fmt={(v) => `${Math.round(v)}%`} /></b>
            </div>
          </div>
          <div style={{ marginTop: 14 }}>
            <AreaChart xLabels={labels} fmt={(v) => rs(v)} height={200} hline={r.goal.goalNominal} hlabel="Goal" fillUnder
              bands={[{ lo: r.goal.fan.p10, hi: r.goal.fan.p90, color: "#74a214", opacity: 0.16 }]}
              series={[{ values: r.goal.fan.p50, color: "#a6d43a", width: 2.2, label: "Typical" }]} />
          </div>
        </div>

        <div className="card c4 delay1">
          <div className="row between"><h3>Money Health</h3>
            <div className="row" style={{ gap: 6 }}><span className="pill">{healthWord}</span>
              <button className="btn ghost small" title="Share your score" onClick={() => { shareCard({ name: profile.name, health: r.health, prob: r.goal.prob,
                level: LEVEL_TITLES[lv.level - 1], streak: prog.streak?.count ?? 1 }); toast("Score card saved", "Share it and challenge a friend"); }}><Share2 size={14} /></button></div></div>
          <div className="row" style={{ justifyContent: "center", margin: "14px 0 18px" }}>
            <Ring value={r.health / 100} size={148} stroke={11}>
              <div style={{ fontSize: 42, fontWeight: 650, letterSpacing: "-0.04em" }}><Count value={r.health} fmt={(v) => String(Math.round(v))} /></div>
              <div className="muted">of 100</div>
            </Ring>
          </div>
          <div className="stack">
            {parts.map(([l, v, m]) => (
              <div key={l} className="meter"><span>{l}</span><span className="muted">{Math.round(v)}/{m}</span><div className="bar"><i style={{ width: `${(v / m) * 100}%` }} /></div></div>
            ))}
          </div>
        </div>

        <div className="c12 delay2" style={{ animation: "rise .5s both" }}>
          <div className="action">
            <div className="row" style={{ gap: 14, flexWrap: "nowrap" }}>
              <div className="icn" style={{ background: "var(--lime)", border: 0 }}><Sparkles color="#0a0d02" /></div>
              <div><div className="t">{action.t}</div><div className="muted" style={{ fontSize: 13.5 }}>{action.s}</div></div>
            </div>
            <button className="btn" onClick={action.run}>{action.cta} <ArrowRight size={16} /></button>
          </div>
        </div>

        <div className="card tap c4 stat delay2" onClick={() => go("crash")}>
          <div className="row between"><div className="row" style={{ gap: 10 }}><div className="icn"><Zap color="#e2475a" /></div><h3>Severe crash</h3></div>
            <Spark values={r.crash.p10} color="#e2475a" width={96} height={34} /></div>
          <div className="v badtext"><Count value={-r.crash.lossBadRs} fmt={(v) => rs(v)} /></div>
          <div className="muted">Bad-case fall ({pct(r.crash.troughBad)}). Your limit: {pct(-profile.tolerance)}</div>
        </div>
        <div className="card tap c4 stat delay3" onClick={() => go("divers")}>
          <div className="row between"><div className="row" style={{ gap: 10 }}><div className="icn"><Network color="#7480e8" /></div><h3>Real diversification</h3></div></div>
          <div className="v">{r.div.nHold} <span className="muted" style={{ fontSize: 16 }}>holdings act like</span> <span className="badtext">{r.div.crash.toFixed(1)}</span></div>
          <div className="muted">in a crash ({r.div.calm.toFixed(1)} on calm days)</div>
        </div>
        <div className="card tap c4 stat delay4" onClick={() => go("crash")}>
          <div className="row between"><div className="row" style={{ gap: 10 }}><div className="icn"><History color="#c6f432" /></div><h3>Your mix in 2020</h3></div>
            <Spark values={r.hist.value.slice(-90)} color="#a6d43a" width={96} height={34} /></div>
          <div className="v">{r.hist.covidDrop !== null ? pct(r.hist.covidDrop) : "n/a"}</div>
          <div className="muted">{r.hist.covidRecover !== null ? `Recovered in ${r.hist.covidRecover} months` : "Real 2020 crash replay"}</div>
        </div>

        <div className="card c7 delay3">
          <div className="row between"><h3>This week's missions</h3><span className="muted">{done.length}/{missions.length} done · resets Monday</span></div>
          <div className="stack" style={{ marginTop: 12 }}>
            {missions.map((m) => {
              const ok = done.includes(m.id);
              return (
                <button key={m.id} className={`mission ${ok ? "done" : ""}`} onClick={() => !ok && go(MISSION_TAB[m.id] ?? "home")}>
                  {ok ? <CheckCircle2 size={20} color="var(--lime)" /> : <Circle size={20} color="var(--muted)" />}
                  <span style={{ flex: 1, textAlign: "left" }}>{m.label}</span><span className="pill">+{m.xp} XP</span>
                </button>
              );
            })}
          </div>
        </div>
        <div className="card c5 delay3">
          <div className="row between"><h3>Your streak</h3><span className="muted">best {prog.streak?.best ?? 1} days</span></div>
          <div className="row" style={{ gap: 14, margin: "12px 0 8px", flexWrap: "nowrap" }}>
            <div className="flame"><Flame size={28} /></div>
            <div><div className="big">{prog.streak?.count ?? 1} day{(prog.streak?.count ?? 1) > 1 ? "s" : ""}</div>
              <div className="muted">Check in daily. Rewards are for showing up and staying invested, never for trading.</div></div>
          </div>
          <div className="lbl" style={{ marginTop: 10 }}>Money Health over time</div>
          {hist.length > 1 ? <AreaChart xLabels={hist.map((h) => h.d.slice(5))} fmt={(v) => String(Math.round(v))} height={110}
            series={[{ values: hist.map((h) => h.h), color: "#a6d43a", label: "Health" }]} fillUnder />
            : <p className="muted" style={{ marginTop: 8 }}>Your score history builds up as you come back and improve your plan.</p>}
        </div>

        <div className="card c7 delay3">
          <div className="row between"><h3>Portfolio · {rs(total)}</h3><button className="btn ghost small" onClick={edit}>Edit</button></div>
          <div className="alloc" style={{ marginTop: 14 }}>
            {profile.holdings.map((h) => <i key={h.id} style={{ width: `${(h.amount / Math.max(total, 1)) * 100}%`, background: KIND_COLOR[h.kind] }} />)}
          </div>
          <div style={{ marginTop: 6 }}>
            {profile.holdings.map((h) => (
              <div key={h.id} className="hold">
                <span className="dot" style={{ background: KIND_COLOR[h.kind] }} />
                <span style={{ minWidth: 0 }}><div style={{ fontWeight: 550, fontSize: 14.5 }}>{h.label}</div><div className="muted">{h.sub}</div></span>
                <span className="muted">{pct(h.amount / Math.max(total, 1))}</span>
                <b style={{ fontWeight: 600 }}>{rs(h.amount)}</b>
              </div>
            ))}
          </div>
          <div className="divider" />
          <div className="row between"><span className="sub">Emergency fund set aside</span>
            <Toggle on={profile.emergency} onChange={(v) => { update({ emergency: v }); if (v) award(10); }} /></div>
        </div>

        <div className="c5" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="card delay4">
            <div className="row between"><h3>Daily quiz</h3><span className="pill">+15 XP</span></div>
            <p style={{ fontSize: 16, fontWeight: 550, margin: "12px 0", letterSpacing: "-0.01em" }}>{cur.q}</p>
            <div className="row">
              {[...cur.opts].sort((a, b) => a - b).map((o, i) => {
                const right = answered && Math.abs(o - cur.a) < 1e-9;
                return <button key={i} className={`chip ${right ? "on" : ""}`} onClick={() => answer(o)}>{o < 0.001 ? "almost never" : pct(o, o < 0.1 ? 1 : 0)}</button>;
              })}
            </div>
            {answered && <div className="row between" style={{ marginTop: 12 }}>
              <span className="muted" style={{ maxWidth: 300 }}>{prog.quiz[cur.id] ? "Correct. " : "Not quite. "}{cur.why}</span>
              <button className="btn ghost small" onClick={() => setQi(qi + 1)}>Next</button></div>}
          </div>
          <div className="card delay5">
            <div className="row between"><h3>Badges</h3><span className="muted">{prog.badges.length}/{BADGES.length}</span></div>
            <div className="row" style={{ gap: 4, marginTop: 12 }}>
              {BADGES.map((b) => {
                const Icon = BADGE_ICONS[b.id] ?? Award;
                return <div key={b.id} className={`badge ${prog.badges.includes(b.id) ? "won" : ""}`} title={b.hint}><div className="ico"><Icon /></div>{b.label}</div>;
              })}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
