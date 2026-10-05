import { type ReactNode, useState } from "react";
import {
  BarChart3,
  Database,
  GitBranch,
  Brain,
  Zap,
  BarChart2,
  Activity,
  ExternalLink,
  Wallet,
  Menu,
  Search,
  Microscope,
} from "lucide-react";

type NavItem = {
  id: string;
  label: string;
  icon: ReactNode;
  description: string;
};

const navItems: NavItem[] = [
  { id: "overview", label: "Overview", icon: <BarChart3 size={20} />, description: "Project overview and architecture" },
  { id: "data", label: "Data Pipeline", icon: <Database size={20} />, description: "NIFTY-50 data exploration" },
  { id: "gat", label: "GAT Encoder", icon: <GitBranch size={20} />, description: "Dynamic Graph Attention" },
  { id: "sde", label: "Neural SDE", icon: <Brain size={20} />, description: "Latent SDE path generation" },
  { id: "scenario", label: "Scenario Generation", icon: <Zap size={20} />, description: "Synthetic scenario sampling" },
  { id: "funds", label: "Mutual Funds", icon: <Wallet size={20} />, description: "NAV history and fund comparison" },
  { id: "market", label: "Market Explorer", icon: <Search size={20} />, description: "Search NSE/BSE stocks and watch prices" },
  { id: "research", label: "Research & Evidence", icon: <Microscope size={20} />, description: "Method, novelty, results, and limits" },
  { id: "evaluation", label: "Evaluation", icon: <BarChart2 size={20} />, description: "Statistical and discriminative metrics" },
  { id: "training", label: "Training", icon: <Activity size={20} />, description: "Loss curves and simulation" },
];

interface LayoutProps {
  children: (activePage: string) => ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const [activePage, setActivePage] = useState("overview");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden">
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-30 w-64 transform transition-transform duration-200 ease-in-out lg:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex h-screen flex-col overflow-y-auto bg-card-bg border-r border-border">
          <div className="p-4 border-b border-border">
            <h1 className="text-xl font-bold text-primary">VORTEX-AI</h1>
            <p className="text-xs text-gray-500 mt-1">CG-NSDE v1.0</p>
          </div>

          <nav className="flex-1 p-3">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  setActivePage(item.id);
                  setSidebarOpen(false);
                }}
                data-active={activePage === item.id}
                className={`dashboard-nav-item flex w-full flex-col gap-1 rounded-lg px-3 py-2 text-left transition-colors ${
                  activePage === item.id
                    ? "bg-primary/10 text-primary"
                    : "text-gray-400 hover:bg-card-bg hover:text-gray-200"
                }`}
              >
                <div className="flex items-center gap-2">
                  {item.icon}
                  <span className="font-medium">{item.label}</span>
                </div>
                <p className="text-xs opacity-70">{item.description}</p>
              </button>
            ))}
          </nav>

          <div className="p-4 border-t border-border">
            <a
              href="https://github.com/yash0238/Vortex-AI"
              target="_blank"
              rel="noopener noreferrer"
              className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-gray-400 hover:bg-card-bg hover:text-gray-200 transition-colors"
            >
              <ExternalLink size={16} />
              GitHub Repository
            </a>
          </div>
        </div>
      </aside>

      <main className="min-w-0 flex-1 overflow-y-auto lg:ml-64">
        <div className="h-12 border-b border-border bg-card-bg/30 flex items-center justify-between px-4">
          <button
            className="lg:hidden p-2 text-gray-400 hover:text-gray-200"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open navigation menu"
            title="Open navigation menu"
          >
            <Menu size={20} />
          </button>
          <h2 className="text-lg font-semibold capitalize">
            {navItems.find((i) => i.id === activePage)?.label || "Dashboard"}
          </h2>
          <div className="w-20" />
        </div>
        <div className="p-6 overflow-y-auto">{children(activePage)}</div>
      </main>
    </div>
  );
}