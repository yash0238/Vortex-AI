// VORTEX Money brand: the mark is three arcs spiralling into a calm centre ("plan for the storm").
export const BRAND = {
  name: "VORTEX Money",
  tagline: "Plan for the storm.",
  lime: "#c6f432",
  ink: "#0a0d02",
};

export function LogoMark({ size = 30, bg = true }: { size?: number; bg?: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" role="img" aria-label="VORTEX Money">
      {bg && <rect width="64" height="64" rx="18" fill="#c6f432" />}
      <g fill="none" stroke={bg ? "#0a0d02" : "#c6f432"} strokeLinecap="round">
        <path d="M32 9a23 23 0 1 1-21.6 15.2" strokeWidth="5.2" />
        <path d="M32 18a14 14 0 1 1-13.1 9.1" strokeWidth="5.2" opacity=".72" />
        <path d="M32 27a5 5 0 1 1-4.7 3.3" strokeWidth="5.2" opacity=".5" />
      </g>
    </svg>
  );
}

export function Wordmark({ size = 30 }: { size?: number }) {
  return (
    <span className="row" style={{ gap: 10, flexWrap: "nowrap" }}>
      <LogoMark size={size} />
      <span style={{ fontWeight: 700, fontSize: size * 0.53, letterSpacing: "-0.02em", whiteSpace: "nowrap" }}>
        VORTEX<span style={{ fontWeight: 400, color: "var(--ink-2)" }}> Money</span>
      </span>
    </span>
  );
}
