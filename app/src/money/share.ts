// Shareable Money Health card (PNG, drawn on a canvas, downloaded locally; nothing is uploaded).
export function shareCard(d: { name: string; health: number; prob: number; level: string; streak: number }) {
  const W = 1080, H = 1350, c = document.createElement("canvas");
  c.width = W; c.height = H;
  const x = c.getContext("2d");
  if (!x) return;
  const g = x.createRadialGradient(200, 0, 50, 200, 0, 1100);
  g.addColorStop(0, "rgba(198,244,50,0.22)"); g.addColorStop(1, "rgba(198,244,50,0)");
  x.fillStyle = "#08090b"; x.fillRect(0, 0, W, H); x.fillStyle = g; x.fillRect(0, 0, W, H);
  const font = (w: number, s: number) => `${w} ${s}px Inter, system-ui, sans-serif`;
  // mark
  x.fillStyle = "#c6f432"; x.beginPath(); x.roundRect(90, 90, 96, 96, 28); x.fill();
  x.strokeStyle = "#0a0d02"; x.lineCap = "round"; x.lineWidth = 8;
  [[36, 1], [22, 0.72], [8, 0.5]].forEach(([r, a]) => { x.globalAlpha = a; x.beginPath(); x.arc(138, 138, r, -Math.PI / 2, Math.PI * 1.15); x.stroke(); });
  x.globalAlpha = 1;
  x.fillStyle = "#f5f6f7"; x.font = font(700, 46); x.fillText("VORTEX", 214, 152);
  x.fillStyle = "#a4a9b3"; x.font = font(400, 46); x.fillText("Money", 214 + x.measureText("VORTEX ").width + 12, 152);
  x.fillStyle = "#a4a9b3"; x.font = font(500, 40); x.fillText(`${d.name}'s Money Health`, 90, 330);
  // ring
  const cx = W / 2, cy = 640, R = 230;
  x.lineWidth = 34; x.strokeStyle = "#1b1e23"; x.beginPath(); x.arc(cx, cy, R, 0, Math.PI * 2); x.stroke();
  const rg = x.createLinearGradient(cx - R, cy - R, cx + R, cy + R); rg.addColorStop(0, "#c6f432"); rg.addColorStop(1, "#8fd14f");
  x.strokeStyle = rg; x.beginPath(); x.arc(cx, cy, R, -Math.PI / 2, -Math.PI / 2 + (Math.PI * 2 * d.health) / 100); x.stroke();
  x.textAlign = "center"; x.fillStyle = "#f5f6f7"; x.font = font(700, 190); x.fillText(String(d.health), cx, cy + 62);
  x.fillStyle = "#6b707b"; x.font = font(500, 36); x.fillText("out of 100", cx, cy + 120);
  // stats
  const stats: [string, string][] = [[`${Math.round(d.prob * 100)}%`, "goal chance"], [d.level, "level"], [`${d.streak} days`, "streak"]];
  stats.forEach(([v, l], i) => {
    const sx = 200 + i * 340;
    x.fillStyle = "#f5f6f7"; x.font = font(700, 54); x.fillText(v, sx, 1040);
    x.fillStyle = "#6b707b"; x.font = font(500, 30); x.fillText(l, sx, 1088);
  });
  x.fillStyle = "#c6f432"; x.font = font(700, 40); x.fillText("Plan for the storm.", cx, 1230);
  x.fillStyle = "#6b707b"; x.font = font(400, 26); x.fillText("Educational simulator · not investment advice", cx, 1280);
  c.toBlob((b) => {
    if (!b) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b); a.download = "vortex-money-health.png"; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }, "image/png");
}
