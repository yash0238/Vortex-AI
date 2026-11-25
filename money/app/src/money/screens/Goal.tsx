import { useEffect, useMemo, useRef, useState } from "react";
import { Check, Sparkles, Wand2 } from "lucide-react";
import { rs, simulateGoal, sipFor, type Engine } from "../engine";
import { GOALS, type Profile } from "../store";
import { AreaChart, Count, Ring, Slider, Toggle } from "../ui";
import type { Results, Track } from "../App";
import { chanceCurve, histogram } from "../analytics";

export function Goal({ eng, profile, r, update, track }: { eng: Engine; profile: Profile; r: Results; update: (p: Partial<Profile>) => void; track: Track }) {
  const [sip, setSip] = useState(profile.sip);
  const [years, setYears] = useState(profile.years);
  const [start, setStart] = useState(profile.start);
  const [goal, setGoal] = useState(profile.goal);
  const [stepUp, setStepUp] = useState(profile.stepUp);
  const [infl, setInfl] = useState(profile.inflation > 0);
  const [res, setRes] = useState(r.goal);
  const [need, setNeed] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const moved = useRef(false);
  const [curve, setCurve] = useState<{ sip: number; prob: number }[] | null>(null);
  const g = GOALS.find((x) => x.k === profile.goalKind) ?? GOALS[0];
  const input = useMemo(() => ({ start, sip, years, goal, inflation: infl ? 0.06 : 0, stepUp }), [start, sip, years, goal, infl, stepUp]);
  // the curve depends on everything except the SIP itself (it sweeps the SIP), so moving the SIP slider doesn't recompute it
  const curveIn = useMemo(() => ({ start, sip: profile.sip, years, goal, inflation: infl ? 0.06 : 0, stepUp }), [start, profile.sip, years, goal, infl, stepUp]);
  useEffect(() => { const t = setTimeout(() => setCurve(chanceCurve(eng, profile.holdings, curveIn)), 400); return () => clearTimeout(t); }, [eng, profile.holdings, curveIn]);

  useEffect(() => {
    const t = setTimeout(() => { setRes(simulateGoal(eng, profile.holdings, input, 1500)); setNeed(null); if (moved.current) track("goal"); moved.current = true; }, 120);
    return () => clearTimeout(t);
  }, [eng, profile.holdings, input]);

  const findSip = () => {
    setBusy(true);
    setTimeout(() => { setNeed(sipFor(eng, profile.holdings, input, 0.85)); setBusy(false); }, 30);
  };
  const changed = sip !== profile.sip || years !== profile.years || start !== profile.start || goal !== profile.goal ||
    stepUp !== profile.stepUp || (infl ? 0.06 : 0) !== profile.inflation;
  const step = Math.max(1, Math.floor(res.months / 60));
  const labels = res.fan.p50.map((_, i) => { const m = i * step; return m % 12 === 0 || i === 0 ? `Yr ${Math.round(m / 12)}` : `Yr ${(m / 12).toFixed(1)}`; });
  const naiveFinal = res.naive12[res.naive12.length - 1];
  const p = res.prob;
  const tone = p >= 0.75 ? "goodtext" : p >= 0.5 ? "warntext" : "badtext";

  return (
    <>
    <div className="pagehead">
      <div><div className="eyebrow">{g.label} · {rs(goal)}{infl ? " in today's money" : ""}</div><h1 className="title">Goal planner</h1></div>
    </div>
    <div className="grid">
      <div className="card hero c12">
        <div className="row between" style={{ alignItems: "center" }}>
          <div className="row" style={{ gap: 36 }}>
            <div className="kpi"><span>Chance of success</span><b className={tone} style={{ fontSize: 52, letterSpacing: "-0.045em" }}>
              <Count value={p * 100} fmt={(v) => `${Math.round(v)}%`} /></b></div>
            <div className="kpi"><span>Typical outcome</span><b style={{ fontSize: 32 }}>{rs(res.median)}</b></div>
            <div className="kpi"><span>Bad case (1 in 10)</span><b style={{ fontSize: 32 }}>{rs(res.bad)}</b></div>
            <div className="kpi"><span>You need</span><b style={{ fontSize: 32 }}>{rs(res.goalNominal)}</b></div>
          </div>
          <Ring value={p} size={112} stroke={9} color={p >= 0.75 ? "#8fd14f" : p >= 0.5 ? "#f0b429" : "#ff6b6b"}>
            <div className="muted">1,500</div><div className="muted">futures</div>
          </Ring>
        </div>
      </div>

      <div className="card c8 delay1">
        <div className="row between"><h3>Your possible futures</h3>
          <div className="legend"><span><i style={{ background: "#74a214" }} />typical</span><span><i style={{ background: "rgba(116,162,20,.35)" }} />likely range (80%)</span>
            <span><i style={{ background: "#6b707b" }} />"12% calculator"</span><span><i style={{ background: "#c47c0c" }} />goal</span></div></div>
        <AreaChart xLabels={labels} fmt={(v) => rs(v)} hline={res.goalNominal} hlabel={`Goal ${rs(res.goalNominal)}`}
          bands={[{ lo: res.fan.p10, hi: res.fan.p90, color: "#74a214" }]}
          series={[{ values: res.naive12, color: "#6b707b", dash: "4 5", width: 1.6, label: "12% calculator" },
                   { values: res.fan.p50, color: "#a6d43a", width: 3, label: "Typical" }]} height={270} />
        <p className="muted">The goal line includes {infl ? "6% yearly inflation" : "no inflation"}: {rs(goal)} today = {rs(res.goalNominal)} in {years} years.</p>
      </div>

      <div className="card c4 delay2 stack">
        <h3>Play with your plan</h3>
        <Slider label="Monthly SIP" value={sip} min={0} max={300000} step={500} onChange={setSip} fmt={(v) => rs(v)} />
        <Slider label="Years" value={years} min={1} max={35} step={1} onChange={setYears} fmt={(v) => `${v} yrs`} />
        <Slider label="Invested today" value={start} min={0} max={10000000} step={10000} onChange={setStart} fmt={(v) => rs(v)} />
        <Slider label="Goal (today's money)" value={goal} min={100000} max={50000000} step={50000} onChange={setGoal} fmt={(v) => rs(v)} />
        <Slider label="Raise SIP every year by" value={stepUp} min={0} max={0.15} step={0.01} onChange={setStepUp} fmt={(v) => `${Math.round(v * 100)}%`} />
        <div className="row between"><span className="sub">Account for inflation (6%)</span><Toggle on={infl} onChange={setInfl} /></div>
        {changed && <button className="btn" onClick={() => update({ sip, years, start, goal, stepUp, inflation: infl ? 0.06 : 0 })}><Check size={16} /> Save this plan</button>}
      </div>

      <div className="card c6 delay3">
        <h3>Myth vs reality</h3>
        <div className="vs" style={{ marginTop: 10 }}>
          <div className="side2"><div className="muted">Typical online calculator (smooth 12%)</div><div className="big" style={{ color: "#b8bfd9" }}>{rs(naiveFinal)}</div>
            <div className="muted">"Guaranteed" path, no crashes</div></div>
          <div className="side2"><div className="muted">VORTEX realistic futures</div><div className="big gradtext">{rs(res.median)}</div>
            <div className="muted">Typical; bad case {rs(res.bad)}</div></div>
        </div>
        <p className="sub" style={{ marginTop: 10 }}>You invest {rs(res.invested)} in total. Real markets don't move in straight lines, so plan with ranges, not a single number.</p>
      </div>

      <div className="card c6 delay4 stack">
        <h3>Make it a sure thing</h3>
        <p className="sub">How much SIP would give you an 85% chance with this same mix?</p>
        {need === null ? (
          <button className="btn" onClick={findSip} disabled={busy}>{busy ? "Simulating…" : <><Wand2 size={16} /> Find my SIP</>}</button>
        ) : (
          <div className="stack">
            <div className="big"><Count value={need} fmt={(v) => rs(v)} /><span className="muted">/month</span></div>
            <p className="sub">{need <= sip ? "You're already there. Nice! 🎉" : `That's ${rs(need - sip)} more per month than now. A yearly step-up or a few extra years can close the gap too.`}</p>
            {need > sip && <button className="btn ghost" onClick={() => setSip(need)}><Sparkles size={16} /> Try it</button>}
          </div>
        )}
        <p className="muted">Assumptions: big Indian stocks grow about 10.5% a year in the long run (no stock assumed to beat
          another); safe money earns 6.5%; funds earn the safe rate plus their measured market exposure; your mix is rebalanced monthly.</p>
      </div>

      <div className="card c6 delay3">
        <div className="row between"><h3>Chance of success vs monthly SIP</h3><span className="muted">same mix, same goal</span></div>
        {curve ? (
          <div className="bars">
            {curve.map((c) => (
              <button key={c.sip} className={`barcol ${Math.abs(c.sip - sip) < 250 ? "on" : ""}`} onClick={() => setSip(c.sip)} title="Try this SIP">
                <span className="bv">{Math.round(c.prob * 100)}%</span>
                <i style={{ height: `${Math.max(4, c.prob * 150)}px`, background: c.prob >= 0.85 ? "var(--s-lime)" : c.prob >= 0.5 ? "var(--s-amber)" : "var(--s-coral)" }} />
                <span className="bl">{rs(c.sip)}</span>
              </button>
            ))}
          </div>
        ) : <div className="skeleton" style={{ height: 190 }} />}
        <p className="muted">Tap a bar to try that SIP. Green = 85%+ chance.</p>
      </div>
      <div className="card c6 delay4">
        <div className="row between"><h3>Where you could end up</h3><span className="muted">1,500 futures</span></div>
        {(() => { const h = histogram(res.finals); const mx = Math.max(...h.counts); const gi = (res.goalNominal - h.lo) / Math.max(h.hi - h.lo, 1) * h.counts.length; return (
          <div>
            <div className="hist">{h.counts.map((c, i) => <i key={i} style={{ height: `${(c / mx) * 150}px`, background: i >= gi ? "var(--s-lime)" : "#3a3f47" }} />)}</div>
            <div className="row between muted" style={{ marginTop: 6 }}><span>{rs(h.lo)}</span><span>Goal {rs(res.goalNominal)}</span><span>{rs(h.hi)}</span></div>
          </div>); })()}
        <p className="muted">Green bars are the futures where you reach your goal.</p>
      </div>
    </div>
    </>
  );
}
