import { Loader2 } from "lucide-react";

interface LoadingStateProps {
  text?: string;
  subtext?: string;
}

export default function LoadingState({
  text = "Loading...",
  subtext,
}: LoadingStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <Loader2 size={48} className="text-primary animate-spin mb-4" />
      <p className="text-lg font-medium text-gray-300">{text}</p>
      {subtext && <p className="text-sm text-gray-500 mt-2">{subtext}</p>}
    </div>
  );
}
