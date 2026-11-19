import { useMemo, useState } from "react";
import { AlertTriangle } from "lucide-react";
import { pct, tipCheck, type Engine } from "../engine";

const PRESETS: { label: string; pct: number; weeks: 1 | 4 | 13 | 52 }[] = [
  { label: "\"5% every week\"", pct: 21.6, weeks: 4 },
  { label: "\"Double your money in 3 months\"", pct: 100, weeks: 13 },
  { label: "\"3% a day guaranteed\"", pct: 15.9, weeks: 1 },
  { label: "\"20% in a month\"", pct: 20, weeks: 4 },
  { label: "\"50% in a year\"", pct: 50, weeks: 52 },
];

const FLAGS = [
  "Guaranteed or 'fixed' returns from stocks or crypto",
  "Pressure to act now: 'only today', 'last few slots'",
  "Tips on Telegram, WhatsApp or Instagram from strangers",
  "Asking you to pay into a personal account or UPI ID",
  "Not registered with SEBI (check the adviser's registration number)",
];

export function Reality({ eng, onCheck }: { eng: Engine; onCheck: () => void }) {
  const [p, setP] = useState(21.6);
  const [weeks, setWeeks] = useState<1 | 4 | 13 | 52>(4);
  const [checked, setChecked] = useState(false);
  const res = useMemo(() => tipCheck(eng, p, weeks), [eng, p, weeks]);
  const period = { 1: "a week", 4: "a month", 13: "3 months", 52: "a year" }[weeks];
  const run = (np: number, nw: 1 | 4 | 13 | 52) => { setP(np); setWeeks(nw); if (!checked) { setChecked(true); onCheck(); } };
  const verdict = res.never ? ["Never happened", "badtext"] : res.share < 0.01 ? ["Extremely rare", "badtext"] :
    res.share < 0.1 ? ["Unusual", "warntext"] : ["Realistic", "goodtext"];

  return (
    <>
    <div className="pagehead"><div><div className="eyebrow">Test any "sure-shot" tip against 16 years of data on 47 big Indian stocks</div><h1 className="title">Reality check</h1></div></div>
    <div className="grid">

      <div className="card c7 delay1 stack">
        <h3>The claim</h3>
        <div className="row">
          {PRESETS.map((x) => <button key={x.label} className={`chip ${p === x.pct && weeks === x.weeks ? "on" : ""}`} onClick={() => run(x.pct, x.weeks)}>{x.label}</button>)}
        </div>
        <div className="row">
          <div className="field" style={{ flex: 1, minWidth: 140 }}><label>Promised gain (%)</label>
            <input className="input" type="number" value={p} min={0} onChange={(e) => run(Math.max(0, Number(e.target.value)), weeks)} /></div>
          <div className="field" style={{ flex: 1, minWidth: 140 }}><label>Within</label>
            <select className="input" value={weeks} onChange={(e) => run(p, Number(e.target.value) as 1 | 4 | 13 | 52)}>
              <option value={1}>1 week</option><option value={4}>1 month</option><option value={13}>3 months</option><option value={52}>1 year</option>
            </select></div>
        </div>
        <div className="list-row" style={{ padding: 18 }}>
          <div>
            <div className={`verdict ${verdict[1]}`}>{verdict[0]}</div>
            <div className="sub">{res.never
              ? `Not one of the 47 biggest Indian stocks ever gained ${p}% in ${period} during 2010–2026. The best ever was ${pct(res.maxSeen)}.`
              : `A big Indian stock gained ${p}% or more within ${period} in about ${res.share < 0.001 ? "less than 0.1%" : pct(res.share, 1)} of cases. And nobody knows in advance which stock or when.`}</div>
          </div>
        </div>
        <p className="muted">Even when a stock does jump, it's luck in hindsight. A promise to repeat it every week is a red flag on its own.</p>
      </div>

      <div className="card c5 delay2 stack">
        <h3>Scam red flags</h3>
        {FLAGS.map((f) => <div key={f} className="list-row"><span className="row" style={{ gap: 12, flexWrap: "nowrap" }}><AlertTriangle style={{ flex: "none" }} size={16} color="#f0b429" />{f}</span></div>)}
        <p className="muted">Lost money to a scam? Report it at the national cybercrime helpline 1930 or cybercrime.gov.in, and to SEBI SCORES.</p>
      </div>
    </div>
    </>
  );
}
