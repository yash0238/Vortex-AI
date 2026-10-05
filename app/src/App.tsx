import { useState, useEffect } from "react";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import DataPipeline from "./pages/DataPipeline";
import Models from "./pages/Models";
import ScenarioGeneration from "./pages/ScenarioGeneration";
import Evaluation from "./pages/Evaluation";
import Training from "./pages/Training";
import MutualFunds from "./pages/MutualFunds";
import MarketExplorer from "./pages/MarketExplorer";
import ResearchPage from "./pages/ResearchPage";
import { apiClient } from "./lib/api";
import type { DataStats } from "./types";

export default function App() {
  const [stats, setStats] = useState<DataStats | null>(null);
  const [health, setHealth] = useState<"loading" | "connected" | "disconnected">("loading");
  const [modelLoaded, setModelLoaded] = useState(false);

  useEffect(() => {
    apiClient.health()
      .then((status) => {
        setHealth("connected");
        setModelLoaded(status.model_loaded);
        return apiClient.getStats();
      })
      .then((s) => setStats(s))
      .catch(() => setHealth("disconnected"));
  }, []);

  return (
    <Layout>
      {(page) => {
        const pageProps = { stats, health, modelLoaded };
        switch (page) {
          case "overview":
            return <Overview {...pageProps} />;
          case "data":
            return <DataPipeline {...pageProps} />;
          case "gat":
            return <Models {...pageProps} tab="gat" />;
          case "sde":
            return <Models {...pageProps} tab="sde" />;
          case "scenario":
            return <ScenarioGeneration {...pageProps} />;
          case "evaluation":
            return <Evaluation {...pageProps} />;
          case "training":
            return <Training {...pageProps} />;
          case "funds":
            return <MutualFunds />;
          case "market":
            return <MarketExplorer />;
          case "research":
            return <ResearchPage />;
          default:
            return <Overview {...pageProps} />;
        }
      }}
    </Layout>
  );
}
