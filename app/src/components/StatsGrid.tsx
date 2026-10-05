import { type ReactNode } from "react";

export interface StatCardProps {
  label: string;
  value: string | number;
  icon?: ReactNode;
  change?: string;
  changeType?: "increase" | "decrease" | "neutral";
  onClick?: () => void;
  subtext?: string;
  description?: string;
}

export default function StatsGrid({
  cards,
}: {
  cards: StatCardProps[];
}) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {cards.map((card, i) => (
        <StatCard key={i} {...card} />
      ))}
    </div>
  );
}

export function StatCard({
  label,
  value,
  icon,
  change,
  changeType = "neutral",
  onClick,
  subtext,
}: StatCardProps) {
  const changeColor =
    changeType === "increase"
      ? "text-green-400"
      : changeType === "decrease"
      ? "text-red-400"
      : "text-gray-400";

  return (
    <div
      className="metric-card cursor-pointer"
      onClick={onClick}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-gray-500 uppercase">{label}</p>
          <p className="text-2xl font-bold mt-1">{value}</p>
          {subtext && <p className="text-xs text-gray-600 mt-1">{subtext}</p>}
          {change && (
            <p className={`text-xs mt-1 ${changeColor}`}>
              {change}
            </p>
          )}
        </div>
        {icon && <div className="text-gray-500">{icon}</div>}
      </div>
    </div>
  );
}
