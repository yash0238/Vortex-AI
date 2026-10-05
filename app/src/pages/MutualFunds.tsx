import { useEffect, useState } from "react";
import axios from "axios";
import { Activity, ExternalLink, RefreshCw, TrendingUp, Wallet } from "lucide-react";
import type { ChartData, ChartOptions } from "chart.js";
import Chart from "../components/Chart";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";

interface FundDefinition {
  id: string;
  name: string;
  category: string;
  code: number;
  color: string;
}

interface MfSearchMatch {
  schemeCode: number;
  schemeName: string;
}

const DEFAULT_SCHEMES: FundDefinition[] = [
  { id: "edelweiss", name: "Edelweiss Mid Cap", category: "Mid cap", code: 140228, color: "#38bdf8" },
  { id: "invesco", name: "Invesco India Mid Cap", category: "Mid cap", code: 120403, color: "#f97316" },
  { id: "mirae", name: "Mirae Asset Midcap", category: "Mid cap", code: 147445, color: "#a3e635" },
  { id: "hsbc", name: "HSBC Small Cap", category: "Small cap", code: 151130, color: "#fb7185" },
  { id: "bandhan", name: "Bandhan Small Cap", category: "Small cap", code: 147946, color: "#c084fc" },
] ;

const FUND_COLORS = ["#38bdf8", "#f97316", "#a3e635", "#fb7185", "#c084fc", "#facc15", "#2dd4bf"];
type Period = "1Y" | "3Y" | "5Y" | "MAX";

interface NavPoint {
  date: string;
  time: number;
  nav: number;
}

interface FundHistory {
  id: string;
  name: string;
  category: string;
  code: number;
  color: string;
  schemeName: string;
  entries: NavPoint[];
}

interface MfApiResponse {
  meta: { scheme_name: string };
  data: Array<{ date: string; nav: string }>;
  status: string;
}

function parseDate(value: string): number {
  const [day, month, year] = value.split("-").map(Number);
  return Date.UTC(year, month - 1, day);
}

function formatDate(timestamp: number): string {
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(timestamp);
}

function returnSince(fund: FundHistory, years: number): number | null {
  const latest = fund.entries[0];
  if (!latest) return null;
  const target = new Date(latest.time);
  target.setUTCFullYear(target.getUTCFullYear() - years);
  const base = fund.entries.find((point) => point.time <= target.getTime());
  return base ? ((latest.nav / base.nav) - 1) * 100 : null;
}

function annualizedVolatility(fund: FundHistory): number | null {
  const observations = fund.entries.slice(0, 253);
  if (observations.length < 3) return null;
  const dailyReturns = observations.slice(0, -1).map((point, index) =>
    point.nav / observations[index + 1].nav - 1,
  );
  const mean = dailyReturns.reduce((sum, value) => sum + value, 0) / dailyReturns.length;
  const variance = dailyReturns.reduce((sum, value) => sum + (value - mean) ** 2, 0)
    / (dailyReturns.length - 1);
  return Math.sqrt(variance * 252) * 100;
}

function chartSeries(funds: FundHistory[], visibleIds: string[], period: Period) {
  const years = period === "MAX" ? null : Number.parseInt(period, 10);
  const selectedFunds = funds.filter((fund) => visibleIds.includes(fund.id));
  const rangeEntries = selectedFunds.map((fund) => {
    const latestTime = fund.entries[0]?.time ?? 0;
    const cutoff = years === null ? 0 : latestTime - years * 365.25 * 24 * 60 * 60 * 1000;
    return fund.entries.filter((point) => point.time >= cutoff).slice().reverse();
  });
  const timestamps = [...new Set(rangeEntries.flatMap((entries) => entries.map((point) => point.time)))]
    .sort((left, right) => left - right);
  const labels = timestamps.map(formatDate);
  const datasets = selectedFunds.map((fund, index) => {
    const entries = rangeEntries[index];
    const baseNav = entries[0]?.nav ?? 1;
    const byDate = new Map(entries.map((point) => [point.time, (point.nav / baseNav) * 100]));
    return {
      label: fund.name,
      data: timestamps.map((timestamp) => byDate.get(timestamp) ?? null),
      borderColor: fund.color,
      backgroundColor: fund.color,
      borderWidth: 2,
      pointRadius: 0,
      pointHitRadius: 8,
      tension: 0.18,
      spanGaps: true,
    };
  });
  return { labels, datasets } as ChartData<"line", (number | null)[], string>;
}

export default function MutualFunds() {
  const [schemes, setSchemes] = useState<FundDefinition[]>(DEFAULT_SCHEMES);
  const [funds, setFunds] = useState<FundHistory[]>([]);
  const [visibleIds, setVisibleIds] = useState<string[]>(DEFAULT_SCHEMES.map((scheme) => scheme.id));
  const [period, setPeriod] = useState<Period>("3Y");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [fundQuery, setFundQuery] = useState("");
  const [fundSearchResults, setFundSearchResults] = useState<MfSearchMatch[]>([]);
  const [searchingFunds, setSearchingFunds] = useState(false);
  const [fundSearchError, setFundSearchError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);

    Promise.all(schemes.map(async (scheme): Promise<FundHistory> => {
      const response = await axios.get<MfApiResponse>(`https://api.mfapi.in/mf/${scheme.code}`);
      const entries = response.data.data
        .map((point) => ({ date: point.date, time: parseDate(point.date), nav: Number(point.nav) }))
        .filter((point) => Number.isFinite(point.time) && Number.isFinite(point.nav) && point.nav > 0)
        .sort((left, right) => right.time - left.time);
      if (response.data.status !== "SUCCESS" || entries.length === 0) {
        throw new Error(`No NAV history returned for ${scheme.name}.`);
      }
      return { ...scheme, schemeName: response.data.meta.scheme_name, entries };
    }))
      .then((results) => {
        if (active) setFunds(results);
      })
      .catch((requestError: unknown) => {
        if (!active) return;
        const detail = axios.isAxiosError(requestError)
          ? requestError.response?.data?.message ?? requestError.message
          : requestError instanceof Error ? requestError.message : "Unknown data error";
        setError(`NAV history could not be loaded. Check the connection and retry. ${detail}`);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [schemes, refreshKey]);

  useEffect(() => {
    let active = true;
    const term = fundQuery.trim();
    if (term.length < 2) {
      setFundSearchResults([]);
      setFundSearchError(null);
      setSearchingFunds(false);
      return () => { active = false; };
    }

    setSearchingFunds(true);
    const timer = window.setTimeout(() => {
      axios.get<MfSearchMatch[]>("https://api.mfapi.in/mf/search", { params: { q: term } })
        .then((response) => {
          if (!active) return;
          const normalizedQuery = term.toLocaleLowerCase();
          const tokens = normalizedQuery.split(/\s+/).filter(Boolean);
          const explicitFmpSearch = /\b(fmp|fixed maturity)\b/i.test(term);
          const matches = response.data
            .filter((match) => explicitFmpSearch || !/\bFMP\b|fixed maturity/i.test(match.schemeName))
            .map((match, index) => {
              const name = match.schemeName.toLocaleLowerCase();
              const tokenMatches = tokens.filter((token) => name.includes(token)).length;
              const score = (name.includes(normalizedQuery) ? 100 : 0) + tokenMatches * 10
                + (/direct plan/i.test(name) ? 2 : 0) + (/growth/i.test(name) ? 1 : 0);
              return { match, index, score };
            })
            .sort((left, right) => right.score - left.score || left.index - right.index)
            .slice(0, 12)
            .map(({ match }) => match);
          setFundSearchResults(matches);
          setFundSearchError(matches.length === 0 ? "No matching mutual fund schemes found." : null);
        })
        .catch((requestError: unknown) => {
          if (!active) return;
          setFundSearchError(axios.isAxiosError(requestError) ? requestError.message : "Fund search is unavailable.");
          setFundSearchResults([]);
        })
        .finally(() => {
          if (active) setSearchingFunds(false);
        });
    }, 300);

    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [fundQuery]);

  const chartData = chartSeries(funds, visibleIds, period);
  const chartOptions: ChartOptions<"line"> = {
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { position: "bottom", labels: { usePointStyle: true, padding: 18 } },
      tooltip: { callbacks: { label: (context) => `${context.dataset.label}: ${Number(context.parsed.y).toFixed(2)}` } },
    },
    scales: {
      x: { ticks: { maxTicksLimit: 8, maxRotation: 0 } },
      y: { title: { display: true, text: "Growth of 100 (NAV index)" } },
    },
  };

  const latestDate = funds.length
    ? Math.max(...funds.map((fund) => fund.entries[0]?.time ?? 0))
    : null;
  const latestFund = funds.reduce<FundHistory | null>((best, fund) => {
    if (!best || (fund.entries[0]?.time ?? 0) > (best.entries[0]?.time ?? 0)) return fund;
    return best;
  }, null);
  const leadingOneYear = funds
    .map((fund) => ({ fund, value: returnSince(fund, 1) }))
    .filter((item): item is { fund: FundHistory; value: number } => item.value !== null)
    .sort((left, right) => right.value - left.value)[0];

  const toggleFund = (id: string) => {
    setVisibleIds((current) => current.includes(id)
      ? current.filter((visibleId) => visibleId !== id)
      : [...current, id]);
  };

  const addFund = (match: MfSearchMatch) => {
    if (schemes.some((scheme) => scheme.code === match.schemeCode)) {
      setFundSearchError("That scheme is already in the comparison.");
      return;
    }
    if (schemes.length >= 12) {
      setFundSearchError("The comparison is limited to 12 schemes at a time.");
      return;
    }
    const name = match.schemeName
      .replace(/\s*-\s*(Direct|Regular) Plan\s*-\s*(Growth|IDCW).*$/i, "")
      .replace(/\s+-\s+(Growth|IDCW)\s+Direct$/i, "")
      .trim();
    const fund: FundDefinition = {
      id: `mf-${match.schemeCode}`,
      name: name || match.schemeName,
      category: match.schemeName,
      code: match.schemeCode,
      color: FUND_COLORS[schemes.length % FUND_COLORS.length],
    };
    setSchemes((current) => [...current, fund]);
    setVisibleIds((current) => [...current, fund.id]);
    setFundQuery("");
    setFundSearchResults([]);
    setFundSearchError(null);
  };

  if (loading && funds.length === 0) return <LoadingState text="Loading mutual fund NAV history..." />;
  if (error && funds.length === 0) {
    return <ErrorState title="NAV feed unavailable" message={error} onRetry={() => setRefreshKey((key) => key + 1)} />;
  }

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-widest text-gray-500">Market data / mutual funds</p>
          <h1 className="mt-2 text-2xl font-bold">Mutual Fund Watchlist</h1>
          <p className="mt-1 text-sm text-gray-400">Live NAV history · Five requested schemes</p>
        </div>
        <button
          type="button"
          className="btn-secondary gap-2"
          onClick={() => setRefreshKey((key) => key + 1)}
          disabled={loading}
          aria-label="Refresh mutual fund NAV data"
          title="Refresh NAV data"
        >
          <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      </header>

      {error && <p role="alert" className="text-sm text-red-300">{error}</p>}

      <section className="fund-summary-grid" aria-label="Watchlist summary">
        <div className="metric-card flex items-center gap-3">
          <Wallet className="text-sky-300" size={20} />
          <div><p className="text-xs text-gray-500">Schemes tracked</p><p className="text-lg font-semibold">{funds.length}</p></div>
        </div>
        <div className="metric-card flex items-center gap-3">
          <Activity className="text-lime-300" size={20} />
          <div><p className="text-xs text-gray-500">Most recent NAV</p><p className="text-lg font-semibold">{latestDate ? formatDate(latestDate) : "--"}</p></div>
        </div>
        <div className="metric-card flex items-center gap-3">
          <TrendingUp className="text-orange-300" size={20} />
          <div>
            <p className="text-xs text-gray-500">Top 1Y return in list</p>
            <p className="text-lg font-semibold">
              {leadingOneYear ? `${leadingOneYear.value.toFixed(1)}%` : "--"}
              {leadingOneYear && <span className="ml-2 text-xs font-normal text-gray-400">{` ${leadingOneYear.fund.name}`}</span>}
            </p>
          </div>
        </div>
      </section>

      <section className="chart-container space-y-4" aria-labelledby="performance-title">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 id="performance-title" className="text-lg font-semibold">NAV performance</h2>
            <p className="mt-1 text-xs text-gray-500">Each line is rebased to 100 at the start of its available period.</p>
          </div>
          <div className="flex rounded-md border border-border p-1" role="group" aria-label="Chart time range">
            {(["1Y", "3Y", "5Y", "MAX"] as const).map((option) => (
              <button
                key={option}
                type="button"
                aria-pressed={period === option}
                data-active={period === option}
                onClick={() => setPeriod(option)}
                className="fund-period-button text-sm"
              >
                {option}
              </button>
            ))}
          </div>
        </div>
        {visibleIds.length > 0
          ? <Chart data={chartData} options={chartOptions} height={340} />
          : <p className="py-20 text-center text-sm text-gray-500">Select at least one scheme below to draw the comparison.</p>}
      </section>

      <section className="chart-container" aria-labelledby="fund-table-title">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 id="fund-table-title" className="text-lg font-semibold">Scheme detail</h2>
            <p className="mt-1 text-xs text-gray-500">Select schemes to show or hide them on the chart.</p>
          </div>
          <a className="inline-flex items-center gap-1 text-xs text-sky-300 hover:text-sky-200" href="https://www.mfapi.in/" target="_blank" rel="noreferrer">
            NAV source: MFAPI.in <ExternalLink size={13} />
          </a>
        </div>
        <div className="fund-search-area">
          <label className="param-label" htmlFor="fund-search">Search and add any mutual fund scheme</label>
          <input
            id="fund-search"
            type="search"
            autoComplete="off"
            className="fund-search-input"
            value={fundQuery}
            onChange={(event) => setFundQuery(event.target.value)}
            placeholder="Search scheme, AMC, or category"
          />
          {searchingFunds && <p className="mt-2 text-xs text-gray-500">Searching scheme catalogue...</p>}
          {fundSearchError && <p role="alert" className="mt-2 text-xs text-amber-200">{fundSearchError}</p>}
          {fundSearchResults.length > 0 && (
            <div className="fund-search-results" role="listbox" aria-label="Matching mutual fund schemes">
              {fundSearchResults.map((match) => {
                const alreadyAdded = schemes.some((scheme) => scheme.code === match.schemeCode);
                return (
                  <div className="fund-search-result" key={match.schemeCode}>
                    <span>{match.schemeName}</span>
                    <button type="button" className="btn-secondary" disabled={alreadyAdded} onClick={() => addFund(match)}>
                      {alreadyAdded ? "Added" : "Add"}
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] border-collapse text-left text-sm">
            <thead className="text-xs uppercase text-gray-500">
              <tr className="border-b border-border">
                <th className="px-3 py-3 font-medium">Scheme</th>
                <th className="px-3 py-3 text-right font-medium">Latest NAV</th>
                <th className="px-3 py-3 text-right font-medium">As of</th>
                <th className="px-3 py-3 text-right font-medium">1Y return</th>
                <th className="px-3 py-3 text-right font-medium">1Y volatility</th>
                <th className="px-3 py-3 text-right font-medium">Since launch CAGR</th>
              </tr>
            </thead>
            <tbody>
              {funds.map((fund) => {
                const latest = fund.entries[0];
                const oldest = fund.entries[fund.entries.length - 1];
                const elapsedYears = latest && oldest ? (latest.time - oldest.time) / (365.25 * 24 * 60 * 60 * 1000) : 0;
                const cagr = elapsedYears > 0 && oldest
                  ? ((latest.nav / oldest.nav) ** (1 / elapsedYears) - 1) * 100
                  : null;
                const oneYear = returnSince(fund, 1);
                const volatility = annualizedVolatility(fund);
                return (
                  <tr key={fund.id} className="border-b border-border/70 last:border-0 hover:bg-white/[0.025]">
                    <td className="px-3 py-3">
                      <label className="flex cursor-pointer items-start gap-3">
                        <input
                          type="checkbox"
                          checked={visibleIds.includes(fund.id)}
                          onChange={() => toggleFund(fund.id)}
                          className="mt-1 accent-sky-400"
                          aria-label={`Show ${fund.name} on chart`}
                        />
                        <span>
                          <span className="block font-medium text-gray-100">{fund.name}</span>
                          <span className="mt-1 block text-xs text-gray-500">{fund.schemeName}</span>
                        </span>
                      </label>
                    </td>
                    <td className="px-3 py-3 text-right tabular-nums">{latest?.nav.toFixed(3) ?? "--"}</td>
                    <td className="px-3 py-3 text-right text-gray-400">{latest ? formatDate(latest.time) : "--"}</td>
                    <td className={`px-3 py-3 text-right tabular-nums ${oneYear !== null && oneYear >= 0 ? "text-lime-300" : "text-rose-300"}`}>
                      {oneYear === null ? "N/A" : `${oneYear.toFixed(2)}%`}
                    </td>
                    <td className="px-3 py-3 text-right tabular-nums">{volatility === null ? "N/A" : `${volatility.toFixed(2)}%`}</td>
                    <td className="px-3 py-3 text-right tabular-nums">{cagr === null ? "N/A" : `${cagr.toFixed(2)}%`}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <footer className="flex flex-wrap items-center justify-between gap-2 text-xs text-gray-500">
        <span>{latestFund ? `NAV provider history · ${latestFund.entries.length.toLocaleString()} observations per scheme where available` : "NAV provider history"}</span>
        <span>Historical NAV only. Past performance does not guarantee future results.</span>
      </footer>
    </div>
  );
}