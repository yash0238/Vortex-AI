interface HeatmapProps {
  data: number[][];
  xLabels?: string[];
  yLabels?: string[];
  title?: string;
  colorscale?: "rdbu" | "viridis" | "rdylbu";
  height?: number;
  maxValue?: number;
  minValue?: number;
}

const colorSchemes = {
  rdbu: { min: "#29308e", mid: "#f7f7f7", max: "#670015" },
  viridis: { min: "#440154", mid: "#21918c", max: "#fde725" },
  rdylbu: { min: "#d70026", mid: "#ffffbf", max: "#08306b" },
};

function interpolateColor(t: number, scheme: "rdbu" | "viridis" | "rdylbu"): string {
  const s = colorSchemes[scheme];
  if (t < 0.5) {
    const ratio = t * 2;
    return lerpColor(s.min, s.mid, ratio);
  } else {
    const ratio = (t - 0.5) * 2;
    return lerpColor(s.mid, s.max, ratio);
  }
}

function lerpColor(colorA: string, colorB: string, t: number): string {
  const ar = parseInt(colorA.slice(1, 3), 16);
  const ag = parseInt(colorA.slice(3, 5), 16);
  const ab = parseInt(colorA.slice(5, 7), 16);
  const br = parseInt(colorB.slice(1, 3), 16);
  const bg = parseInt(colorB.slice(3, 5), 16);
  const bb = parseInt(colorB.slice(5, 7), 16);
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const blue = Math.round(ab + (bb - ab) * t);
  return `rgb(${r},${g},${blue})`;
}

export default function Heatmap({
  data,
  xLabels,
  yLabels,
  title,
  colorscale = "rdbu",
  height = 400,
  maxValue,
  minValue,
}: HeatmapProps) {
  const actualMax = maxValue ?? data.flat().reduce((a, b) => Math.max(a, b), -Infinity);
  const actualMin = minValue ?? data.flat().reduce((a, b) => Math.min(a, b), Infinity);
  const range = actualMax - actualMin || 1;

  return (
    <div className="chart-container">
      {title && <h3 className="text-lg font-medium mb-3">{title}</h3>}
      <div
        className="overflow-auto"
        style={{ maxHeight: `${height + 40}px` }}
      >
        <table className="border-collapse text-xs">
          <thead>
            <tr>
              <th className="w-8 h-8"></th>
              {xLabels &&
                xLabels.map((label, i) => (
                  <th
                    key={i}
                    className="text-xs text-gray-600 rotate-[45deg] origin-left whitespace-nowrap"
                    style={{ width: "24px", height: "24px" }}
                  >
                    {label}
                  </th>
                ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i}>
                {yLabels && (
                  <td
                    className="text-xs text-gray-600 text-right pr-1 whitespace-nowrap"
                    style={{ width: "24px", height: "24px" }}
                  >
                    {yLabels[i]}
                  </td>
                )}
                {row.map((val, j) => {
                  const t = (val - actualMin) / range;
                  return (
                    <td
                      key={j}
                      className="border border-gray-800"
                      style={{
                        width: "24px",
                        height: "24px",
                        backgroundColor: interpolateColor(t, colorscale),
                        minWidth: "24px",
                        textAlign: "center",
                        fontSize: "10px",
                        color: t > 0.5 ? "#fff" : "#000",
                      }}
                      title={`${xLabels?.[j] || j} × ${yLabels?.[i] || i}: ${val.toFixed(3)}`}
                    >
                      {val.toFixed(1)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex justify-between text-xs text-gray-500 mt-2">
        <span>{actualMin.toFixed(2)}</span>
        <span>{actualMax.toFixed(2)}</span>
      </div>
    </div>
  );
}