import { useEffect, useState } from "react";
import { Sunset } from "lucide-react";
import { pct, rs, type Engine } from "../engine";
import { simulateSwp, type SwpResult } from "../analytics";
import type { Profile } from "../store";
import { AreaChart, Count, Ring, Slider } from "../ui";
import type { Track } from "../App";

export function Retire({ eng, profile, track }: { eng: Engine; profile: Profile; update: (p: Partial<Profile>) => void; track: Track }) {
  const [corpus, setCorpus] = useState(10000000);
  const [monthly, setMonthly] = useState(50000);
  const [years, setYears] = useState(25);
  const [res, setRes] = useState<SwpResult | null>(null);

  useEffect(() => {
    const t = setTimeout(() => { setRes(simulateSwp(eng, profile.holdings, corpus, monthly, years)); track("retire"); }, 150);
    return () => clearTimeout(t);
  }, [eng, profile.holdings, corpus, monthly, years, track]);

  const p = res?.prob ?? 0;
  const tone = p >= 0.9 ? "goodtext" : p >= 0.7 ? "warntext" : "badtext";
  const rate = (monthly * 12) / corpus;
  return (
    <>
      <div className="pagehead"><div><div className="eyebrow">Monthly withdrawals (SWP) that rise with inflation, run through realistic markets</div>
        <h1 className="title">Will my money last?</h1></div></div>
      <div className="grid">
        <div className="card hero c12">
          <div className="row between" style={{ alignItems: "center" }}>
            <div className="row" style={{ gap: 36 }}>
              <div className="kpi"><span>Chance it lasts {years} years</span><b className={tone} style={{ fontSize: 52, letterSpacing: "-0.045em" }}>
                <Count value={p * 100} fmt={(v) => `${Math.round(v)}%`} /></b></div>
              <div className="kpi"><span>Typical amount left</span><b style={{ fontSize: 30 }}>{res ? rs(res.medianEnd) : "…"}</b></div>
              <div className="kpi"><span>Safe withdrawal (90%)</span><b style={{ fontSize: 30 }} className="gradtext">{res ? `${rs(res.safeMonthly)}/mo` : "…"}</b></div>
              <div className="kpi"><span>Withdrawal rate</span><b style={{ fontSize: 30 }}>{pct(rate, 1)}</b></div>
            </div>
            <Ring value={p} size={104} stroke={9} color={p >= 0.9 ? "#8fd14f" : p >= 0.7 ? "#f0b429" : "#ff6b6b"}><Sunset size={26} color="var(--ink-2)" /></Ring>
          </div>
        </div>
        <div className="card c8 delay1">
          <div className="row between"><h3>Your retirement money over time</h3>
            <div className="legend"><span><i style={{ background: "#a6d43a" }} />typical</span><span><i style={{ background: "rgba(116,162,20,.4)" }} />80% range</span></div></div>
          {res && <AreaChart xLabels={res.p50.map((_, i) => `Year ${i}`)} fmt={(v) => rs(v)} height={260} hline={0}
            bands={[{ lo: res.p10, hi: res.p90, color: "#74a214", opacity: 0.18 }]} series={[{ values: res.p50, color: "#a6d43a", width: 2.4, label: "Typical" }]} />}
          <p className="muted">{res?.runOutMedianYear ? `In most futures the money runs out around year ${Math.round(res.runOutMedianYear)}.` :
            "The real risk is a crash in the first few years (sequence risk): withdrawing from a fallen portfolio locks in losses."}</p>
        </div>
        <div className="card c4 delay2 stack">
          <h3>Your retirement</h3>
          <Slider label="Money at retirement" value={corpus} min={1000000} max={100000000} step={500000} onChange={setCorpus} fmt={(v) => rs(v)} />
          <Slider label="Monthly withdrawal (today's ₹)" value={monthly} min={5000} max={500000} step={1000} onChange={setMonthly} fmt={(v) => rs(v)} />
          <Slider label="Years it must last" value={years} min={5} max={40} step={1} onChange={setYears} fmt={(v) => `${v} yrs`} />
          <p className="muted">Uses your current portfolio mix. Withdrawals rise 6% a year with inflation. A common rule of thumb is to keep the first-year
            withdrawal rate near 3–4% in India; see what your mix supports.</p>
        </div>
      </div>
    </>
  );
}
