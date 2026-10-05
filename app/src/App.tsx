import { useState, useEffect } from "react";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import DataPipeline from "./pages/DataPipeline";
import Models from "./pages/Models";
import ScenarioGeneration from "./pages/ScenarioGeneration";
import Evaluation from "./pages/Evaluation";
import Training from "./pages/Training";
import { apiClient } from "./lib/api";
import type { DataStats } from "./types";

export default function App() {
  const [stats, setStats] = useState<DataStats | null>(null);
  const [health, setHealth] = useState<"loading" | "connected" | "disconnected">("loading");

  useEffect(() => {
    apiClient.health()
      .then(() => {
        setHealth("connected");
        return apiClient.getStats();
      })
      .then((s) => setStats(s))
      .catch(() => setHealth("disconnected"));
  }, []);

  return (
    <Layout>
      {(page) => {
        const pageProps = { stats, health };
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
          default:
            return <Overview {...pageProps} />;
        }
      }}
    </Layout>
  );
}
