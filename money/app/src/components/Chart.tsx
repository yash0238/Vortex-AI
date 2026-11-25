import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  LogarithmicScale,
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";
import type { ChartData, ChartOptions } from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  LogarithmicScale,
);

interface ChartProps {
  data: ChartData;
  options?: ChartOptions;
  height?: number;
  type?: "line" | "bar";
}

export default function Chart({
  data,
  options,
  height = 300,
  type = "line",
}: ChartProps) {
  const opts: ChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: "#9ca3af", font: { size: 12 } },
      },
    },
    scales: {
      x: { ticks: { color: "#6b7280" }, grid: { color: "#1f2937" } },
      y: { ticks: { color: "#6b7280" }, grid: { color: "#1f2937" } },
    },
    ...options,
  };

  return (
    <div style={{ height: `${height}px`, width: "100%" }}>
      {type === "bar" ? (
        <Bar data={data as any} options={opts as any} />
      ) : (
        <Line data={data as any} options={opts as any} />
      )}
    </div>
  );
}