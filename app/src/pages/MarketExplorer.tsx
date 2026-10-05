import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { Activity, RefreshCw, Search, Star, Trash2, TrendingDown, TrendingUp } from "lucide-react";
import type { ChartData, ChartOptions } from "chart.js";
import { apiClient } from "../lib/api";
import type { MarketHistory, MarketInstrument, MarketPeriod, MarketInterval, MarketQuote } from "../types";
import Chart from "../components/Chart";
import LoadingState from "../components/LoadingState";

const DEFAULT_WATCHLIST = ["^NSEI", "RELIANCE.NS", "TCS.NS"];
const ranges: Array<{ label: string; period: MarketPeriod; interval: MarketInterval }> = [
  { label: "1W", period: "5d", interval: "15m" },
  { label: "1M", period: "1mo", interval: "1d" },
  { label: "6M", period: "6mo", interval: "1d" },
  { label: "1Y", period: "1y", interval: "1d" },
  { label: "5Y", period: "5y", interval: "1wk" },
];

function loadWatchlist(): string[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem("vortex-market-watchlist") ?? "null");
    if (Array.isArray(stored)) {
      return [...new Set(stored.filter((symbol): symbol is string => typeof symbol === "string"))];
    }
  } catch {
    return DEFAULT_WATCHLIST;
  }
  return DEFAULT_WATCHLIST;
}

function loadPaperHoldings(): PaperHolding[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem("vortex-paper-holdings") ?? "[]");
    if (Array.isArray(stored)) {
      return stored.filter((item): item is PaperHolding =>
        typeof item?.symbol === "string"
        && Number.isFinite(item.units) && item.units > 0
        && Number.isFinite(item.averageCost) && item.averageCost > 0,
      );
    }
  } catch {
    return [];
  }
  return [];
}

interface PaperHolding {
  symbol: string;
  units: number;
  averageCost: number;
}

interface LocalPriceAlert {
  id: string;
  symbol: string;
  direction: "above" | "below";
  target: number;
  triggeredAt: string | null;
}

function loadPriceAlerts(): LocalPriceAlert[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem("vortex-market-alerts") ?? "[]");
    if (Array.isArray(stored)) {
      return stored.filter((item): item is LocalPriceAlert =>
        typeof item?.id === "string"
        && typeof item.symbol === "string"
        && (item.direction === "above" || item.direction === "below")
        && Number.isFinite(item.target) && item.target > 0,
      );
    }
  } catch {
    return [];
  }
  return [];
}

function formatCurrency(value: number | null, currency = "INR"): string {
  if (value === null || !Number.isFinite(value)) return "--";
  try {
    return new Intl.NumberFormat("en-IN", {
      style: "currency",
      currency,
      maximumFractionDigits: 2,
    }).format(value);
  } catch {
    return `${value.toFixed(2)} ${currency}`;
  }
}

function formatCompact(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "--";
  return new Intl.NumberFormat("en-IN", { notation: "compact", maximumFractionDigits: 2 }).format(value);
}

function displayName(symbol: string): string {
  if (symbol === "^NSEI") return "NIFTY 50";
  return symbol.replace(/\.(NS|BO)$/i, "");
}

export default function MarketExplorer() {
  const [watchlist, setWatchlist] = useState<string[]>(loadWatchlist);
  const [holdings, setHoldings] = useState<PaperHolding[]>(loadPaperHoldings);
  const [quotes, setQuotes] = useState<Record<string, MarketQuote>>({});
  const [selectedSymbol, setSelectedSymbol] = useState("^NSEI");
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<MarketInstrument[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [quoteError, setQuoteError] = useState<string | null>(null);
  const [history, setHistory] = useState<MarketHistory | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [rangeLabel, setRangeLabel] = useState("1Y");
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [holdingSymbol, setHoldingSymbol] = useState("RELIANCE.NS");
  const [holdingUnits, setHoldingUnits] = useState(1);
  const [holdingCost, setHoldingCost] = useState(0);
  const [priceAlerts, setPriceAlerts] = useState<LocalPriceAlert[]>(loadPriceAlerts);
  const [alertSymbol, setAlertSymbol] = useState("RELIANCE.NS");
  const [alertDirection, setAlertDirection] = useState<"above" | "below">("above");
  const [alertTarget, setAlertTarget] = useState(0);
  const [alertToast, setAlertToast] = useState<string | null>(null);

  const selectedRange = ranges.find((range) => range.label === rangeLabel) ?? ranges[3];
  const selectedQuote = quotes[selectedSymbol];

  useEffect(() => {
    localStorage.setItem("vortex-market-watchlist", JSON.stringify(watchlist));
  }, [watchlist]);

  useEffect(() => {
    localStorage.setItem("vortex-paper-holdings", JSON.stringify(holdings));
  }, [holdings]);

  useEffect(() => {
    localStorage.setItem("vortex-market-alerts", JSON.stringify(priceAlerts));
  }, [priceAlerts]);

  useEffect(() => {
    const timer = window.setInterval(() => setRefreshKey((key) => key + 1), 60_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const triggered = priceAlerts.filter((alert) => {
      if (alert.triggeredAt) return false;
      const price = quotes[alert.symbol]?.price;
      if (price === null || price === undefined) return false;
      return alert.direction === "above" ? price >= alert.target : price <= alert.target;
    });
    if (triggered.length === 0) return;
    const triggeredIds = new Set(triggered.map((alert) => alert.id));
    const triggeredAt = new Date().toISOString();
    setPriceAlerts((current) => current.map((alert) => triggeredIds.has(alert.id)
      ? { ...alert, triggeredAt }
      : alert));
    setAlertToast(triggered.map((alert) => `${displayName(alert.symbol)} crossed ${formatCurrency(alert.target)}`).join("; "));
  }, [priceAlerts, quotes]);

  useEffect(() => {
    let active = true;
    const timeout = window.setTimeout(async () => {
      const term = query.trim();
      if (term.length < 2) {
        setSearchResults([]);
        setSearchError(null);
        setSearching(false);
        return;
      }
      setSearching(true);
      setSearchError(null);
      try {
        const response = await apiClient.searchMarket(term, 10);
        if (active) setSearchResults(response.results);
      } catch (requestError) {
        if (active) {
          setSearchError(requestError instanceof Error ? requestError.message : "Search failed.");
          setSearchResults([]);
        }
      } finally {
        if (active) setSearching(false);
      }
    }, 300);
    return () => {
      active = false;
      window.clearTimeout(timeout);
    };
  }, [query]);

  useEffect(() => {
    let active = true;
    const symbols = [...new Set([
      ...watchlist,
      ...holdings.map((holding) => holding.symbol),
      ...priceAlerts.map((alert) => alert.symbol),
    ])];
    Promise.all(symbols.map(async (symbol) => {
      try {
        return [symbol, await apiClient.getMarketQuote(symbol)] as const;
      } catch {
        return [symbol, null] as const;
      }
    })).then((entries) => {
      if (!active) return;
      const loaded = Object.fromEntries(entries.filter((entry) => entry[1] !== null)) as Record<string, MarketQuote>;
      setQuotes(loaded);
      setQuoteError(entries.some((entry) => entry[1] === null)
        ? "Some watchlist quotes could not be refreshed."
        : null);
    });
    return () => {
      active = false;
    };
  }, [watchlist, holdings, priceAlerts, refreshKey]);

  useEffect(() => {
    let active = true;
    setLoadingHistory(true);
    setHistoryError(null);
    apiClient.getMarketHistory(selectedSymbol, selectedRange.period, selectedRange.interval)
      .then((result) => {
        if (active) setHistory(result);
      })
      .catch((requestError) => {
        if (active) {
          setHistory(null);
          setHistoryError(requestError instanceof Error ? requestError.message : "Price history unavailable.");
        }
      })
      .finally(() => {
        if (active) setLoadingHistory(false);
      });
    return () => {
      active = false;
    };
  }, [selectedSymbol, selectedRange.period, selectedRange.interval, refreshKey]);

  const selectInstrument = (instrument: MarketInstrument) => {
    setSelectedSymbol(instrument.symbol);
    setWatchlist((current) => current.includes(instrument.symbol) ? current : [...current, instrument.symbol]);
    setQuery("");
    setSearchResults([]);
  };

  const removeFromWatchlist = (symbol: string) => {
    setWatchlist((current) => current.filter((item) => item !== symbol));
    if (selectedSymbol === symbol) setSelectedSymbol("^NSEI");
  };

  const addPaperHolding = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (holdingUnits <= 0 || holdingCost <= 0) return;
    const symbol = holdingSymbol.trim().toUpperCase();
    setHoldings((current) => {
      const existing = current.find((holding) => holding.symbol === symbol);
      if (!existing) return [...current, { symbol, units: holdingUnits, averageCost: holdingCost }];
      const totalUnits = existing.units + holdingUnits;
      const averageCost = (existing.units * existing.averageCost + holdingUnits * holdingCost) / totalUnits;
      return current.map((holding) => holding.symbol === symbol
        ? { ...holding, units: totalUnits, averageCost }
        : holding);
    });
    setWatchlist((current) => current.includes(symbol) ? current : [...current, symbol]);
  };

  const addPriceAlert = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (alertTarget <= 0) return;
    const id = typeof crypto.randomUUID === "function" ? crypto.randomUUID() : `${Date.now()}-${alertSymbol}`;
    setPriceAlerts((current) => [...current, {
      id,
      symbol: alertSymbol,
      direction: alertDirection,
      target: alertTarget,
      triggeredAt: null,
    }]);
  };

  const investedValue = holdings.reduce((total, holding) => total + holding.units * holding.averageCost, 0);
  const portfolioValue = holdings.reduce((total, holding) =>
    total + (quotes[holding.symbol]?.price ?? holding.units * holding.averageCost), 0);
  const portfolioPnL = holdings.reduce((total, holding) => {
    const price = quotes[holding.symbol]?.price;
    return total + (price == null ? 0 : (price - holding.averageCost) * holding.units);
  }, 0);

  const series: ChartData<"line", number[], string> = {
    labels: history?.bars.map((bar) => new Date(bar.date).toLocaleDateString("en-IN", {
      day: selectedRange.interval === "1d" || selectedRange.interval === "15m" ? undefined : "2-digit",
      month: "short",
      year: selectedRange.period === "5y" ? "2-digit" : undefined,
      hour: selectedRange.interval === "15m" ? "2-digit" : undefined,
      minute: selectedRange.interval === "15m" ? "2-digit" : undefined,
    })) ?? [],
    datasets: [{
      label: `${displayName(selectedSymbol)} close`,
      data: history?.bars.map((bar) => bar.close ?? 0) ?? [],
      borderColor: "#38bdf8",
      backgroundColor: "rgba(56, 189, 248, 0.10)",
      borderWidth: 2,
      pointRadius: 0,
      fill: true,
      tension: 0.16,
    }],
  };
  const chartOptions: ChartOptions<"line"> = {
    interaction: { mode: "index", intersect: false },
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { maxTicksLimit: 8, maxRotation: 0 } },
      y: { title: { display: true, text: `Price (${selectedQuote?.currency ?? "INR"})` } },
    },
  };

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-widest text-gray-500">Markets / NSE & BSE</p>
          <h1 className="mt-2 text-2xl font-bold">Market Explorer</h1>
          <p className="mt-1 text-sm text-gray-400">Search an instrument, inspect its price history, and keep a local watchlist.</p>
        </div>
        <button type="button" className="btn-secondary gap-2" onClick={() => setRefreshKey((key) => key + 1)} aria-label="Refresh quotes" title="Refresh quotes">
          <RefreshCw size={16} /> Refresh
        </button>
      </header>

      <section className="chart-container" aria-label="Instrument search">
        <label className="market-search-label" htmlFor="market-search">Find a stock</label>
        <div className="market-search-input-wrap">
          <Search size={18} aria-hidden="true" />
          <input
            id="market-search"
            type="search"
            autoComplete="off"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Company name or ticker (e.g. Infosys, RELIANCE)"
            aria-label="Search NSE and BSE stocks"
            aria-expanded={searchResults.length > 0}
          />
        </div>
        {searching && <p className="mt-2 text-xs text-gray-500">Searching NSE/BSE instruments...</p>}
        {searchError && <p role="alert" className="mt-2 text-xs text-rose-300">{searchError}</p>}
        {searchResults.length > 0 && (
          <div className="market-search-results" role="listbox" aria-label="Stock search results">
            {searchResults.map((instrument) => (
              <div className="market-search-result" key={instrument.symbol}>
                <button type="button" className="market-search-select" onClick={() => selectInstrument(instrument)} role="option">
                  <span className="market-search-name">{instrument.name}</span>
                  <span className="market-search-symbol">{instrument.symbol} · {instrument.exchange}</span>
                </button>
                <button
                  type="button"
                  className="market-icon-button"
                  onClick={() => setWatchlist((current) => current.includes(instrument.symbol) ? current : [...current, instrument.symbol])}
                  aria-label={`Add ${instrument.symbol} to watchlist`}
                  title="Add to watchlist"
                >
                  <Star size={16} />
                </button>
              </div>
            ))}
          </div>
        )}
      </section>

      {quoteError && <p role="status" className="text-sm text-amber-200">{quoteError}</p>}

      <section className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(280px,1fr)]">
        <div className="chart-container space-y-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-xs text-gray-500">{selectedSymbol}</p>
              <h2 className="mt-1 text-xl font-semibold">{displayName(selectedSymbol)}</h2>
            </div>
            <div className="market-range-controls" role="group" aria-label="Price chart range">
              {ranges.map((range) => (
                <button
                  key={range.label}
                  type="button"
                  className="fund-period-button"
                  data-active={rangeLabel === range.label}
                  aria-pressed={rangeLabel === range.label}
                  onClick={() => setRangeLabel(range.label)}
                >
                  {range.label}
                </button>
              ))}
            </div>
          </div>
          {loadingHistory && !history
            ? <LoadingState text="Loading price history..." />
            : historyError
              ? <p role="alert" className="py-16 text-center text-sm text-rose-300">{historyError}</p>
              : <Chart data={series} options={chartOptions} height={330} />}
          <p className="text-xs text-gray-500">{history?.freshness_note ?? "Price history source: Yahoo Finance via yfinance."}</p>
        </div>

        <aside className="chart-container" aria-label="Selected instrument quote">
          {selectedQuote ? (
            <>
              <div className="flex items-start justify-between gap-2">
                <div>
                  <p className="text-xs text-gray-500">Latest available quote</p>
                  <p className="mt-2 text-3xl font-semibold tabular-nums">{formatCurrency(selectedQuote.price, selectedQuote.currency)}</p>
                </div>
                {selectedQuote.change_percent !== null && (
                  <span className={`inline-flex items-center gap-1 rounded px-2 py-1 text-sm ${selectedQuote.change_percent >= 0 ? "text-lime-300" : "text-rose-300"}`}>
                    {selectedQuote.change_percent >= 0 ? <TrendingUp size={15} /> : <TrendingDown size={15} />}
                    {selectedQuote.change_percent.toFixed(2)}%
                  </span>
                )}
              </div>
              <p className="mt-2 text-xs text-gray-500">Market date: {new Intl.DateTimeFormat("en-IN", { dateStyle: "medium", timeZone: "Asia/Kolkata" }).format(new Date(selectedQuote.as_of))}</p>
              <div className="market-quote-grid">
                <span>Prev close</span><strong>{formatCurrency(selectedQuote.previous_close, selectedQuote.currency)}</strong>
                <span>Open</span><strong>{formatCurrency(selectedQuote.open, selectedQuote.currency)}</strong>
                <span>Day range</span><strong>{formatCurrency(selectedQuote.day_low, selectedQuote.currency)} - {formatCurrency(selectedQuote.day_high, selectedQuote.currency)}</strong>
                <span>Volume</span><strong>{formatCompact(selectedQuote.volume)}</strong>
                <span>Market cap</span><strong>{formatCompact(selectedQuote.market_cap)}</strong>
              </div>
              <p className="mt-4 border-t border-border pt-3 text-xs text-gray-500">{selectedQuote.freshness_note}</p>
            </>
          ) : (
            <LoadingState text="Loading latest quote..." />
          )}
        </aside>
      </section>

      <section className="chart-container" aria-labelledby="watchlist-title">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <h2 id="watchlist-title" className="text-lg font-semibold">My watchlist</h2>
            <p className="mt-1 text-xs text-gray-500">Saved in this browser</p>
          </div>
          <Activity size={18} className="text-sky-300" aria-hidden="true" />
        </div>
        <div className="market-watchlist">
          {watchlist.map((symbol) => {
            const quote = quotes[symbol];
            return (
              <div className={`market-watch-row ${selectedSymbol === symbol ? "is-selected" : ""}`} key={symbol}>
                <button type="button" className="market-watch-select" onClick={() => setSelectedSymbol(symbol)}>
                  <span className="market-search-name">{displayName(symbol)}</span>
                  <span className="market-search-symbol">{symbol}</span>
                </button>
                <span className="market-watch-price">{quote ? formatCurrency(quote.price, quote.currency) : "--"}</span>
                <span className={`market-watch-change ${quote?.change_percent !== null && quote?.change_percent !== undefined && quote.change_percent >= 0 ? "positive" : "negative"}`}>
                  {quote?.change_percent == null ? "--" : `${quote.change_percent.toFixed(2)}%`}
                </span>
                <button type="button" className="market-icon-button" onClick={() => removeFromWatchlist(symbol)} aria-label={`Remove ${symbol} from watchlist`} title="Remove from watchlist">
                  <Trash2 size={15} />
                </button>
              </div>
            );
          })}
          {watchlist.length === 0 && <p className="py-4 text-sm text-gray-500">Search for an instrument to build your watchlist.</p>}
        </div>
      </section>

      <section className="chart-container" aria-labelledby="paper-portfolio-title">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 id="paper-portfolio-title" className="text-lg font-semibold">Paper portfolio</h2>
            <p className="mt-1 text-xs text-gray-500">Manual holdings saved only in this browser; no orders or broker account are connected.</p>
          </div>
          <div className="text-right">
            <p className="text-xs text-gray-500">Unrealized P&amp;L</p>
            <p className={`text-lg font-semibold tabular-nums ${portfolioPnL >= 0 ? "text-lime-300" : "text-rose-300"}`}>{formatCurrency(portfolioPnL)}</p>
          </div>
        </div>
        <form className="market-holding-form" onSubmit={addPaperHolding}>
          <label>Instrument
            <select value={holdingSymbol} onChange={(event) => setHoldingSymbol(event.target.value)}>
              {[...new Set([...watchlist, ...Object.keys(quotes)])].map((symbol) => (
                <option key={symbol} value={symbol}>{displayName(symbol)} ({symbol})</option>
              ))}
            </select>
          </label>
          <label>Units
            <input type="number" min="0.0001" step="any" value={holdingUnits} onChange={(event) => setHoldingUnits(Number(event.target.value))} required />
          </label>
          <label>Average cost / unit
            <input type="number" min="0.01" step="any" value={holdingCost || ""} onChange={(event) => setHoldingCost(Number(event.target.value))} required />
          </label>
          <button type="submit" className="btn-primary">Add holding</button>
        </form>
        <div className="market-portfolio-summary">
          <span>Cost basis <strong>{formatCurrency(investedValue)}</strong></span>
          <span>Market value <strong>{formatCurrency(portfolioValue)}</strong></span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[680px] border-collapse text-left text-sm">
            <thead className="text-xs uppercase text-gray-500">
              <tr className="border-b border-border">
                <th className="px-3 py-3">Instrument</th><th className="px-3 py-3 text-right">Units</th><th className="px-3 py-3 text-right">Avg cost</th><th className="px-3 py-3 text-right">Last price</th><th className="px-3 py-3 text-right">Value</th><th className="px-3 py-3 text-right">P&amp;L</th><th className="px-3 py-3" />
              </tr>
            </thead>
            <tbody>
              {holdings.map((holding) => {
                const price = quotes[holding.symbol]?.price ?? null;
                const value = price === null ? null : price * holding.units;
                const pnl = value === null ? null : value - holding.averageCost * holding.units;
                return (
                  <tr key={holding.symbol} className="border-b border-border/70 last:border-0">
                    <th className="px-3 py-3 font-medium">{displayName(holding.symbol)}</th>
                    <td className="px-3 py-3 text-right tabular-nums">{holding.units.toLocaleString("en-IN")}</td>
                    <td className="px-3 py-3 text-right tabular-nums">{formatCurrency(holding.averageCost)}</td>
                    <td className="px-3 py-3 text-right tabular-nums">{formatCurrency(price)}</td>
                    <td className="px-3 py-3 text-right tabular-nums">{formatCurrency(value)}</td>
                    <td className={`px-3 py-3 text-right tabular-nums ${pnl !== null && pnl >= 0 ? "text-lime-300" : "text-rose-300"}`}>{formatCurrency(pnl)}</td>
                    <td className="px-3 py-3 text-right">
                      <button type="button" className="market-icon-button" onClick={() => setHoldings((current) => current.filter((item) => item.symbol !== holding.symbol))} aria-label={`Remove ${holding.symbol} holding`} title="Remove holding">
                        <Trash2 size={15} />
                      </button>
                    </td>
                  </tr>
                );
              })}
              {holdings.length === 0 && <tr><td className="px-3 py-5 text-sm text-gray-500" colSpan={7}>Add a paper holding to track an average cost against the latest available quote.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>

      <section className="chart-container" aria-labelledby="price-alerts-title">
        <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 id="price-alerts-title" className="text-lg font-semibold">Price alerts</h2>
            <p className="mt-1 text-xs text-gray-500">Saved in this browser and checked while this page is open, on the 60-second quote refresh. Not an exchange push alert.</p>
          </div>
        </div>
        {alertToast && <p role="status" className="mb-3 text-sm text-amber-200">Alert triggered: {alertToast}</p>}
        <form className="market-holding-form" onSubmit={addPriceAlert}>
          <label>Instrument
            <select value={alertSymbol} onChange={(event) => setAlertSymbol(event.target.value)}>
              {watchlist.map((symbol) => <option key={symbol} value={symbol}>{displayName(symbol)} ({symbol})</option>)}
            </select>
          </label>
          <label>Condition
            <select value={alertDirection} onChange={(event) => setAlertDirection(event.target.value as "above" | "below")}>
              <option value="above">Price rises to</option>
              <option value="below">Price falls to</option>
            </select>
          </label>
          <label>Target price
            <input type="number" min="0.01" step="any" value={alertTarget || ""} onChange={(event) => setAlertTarget(Number(event.target.value))} required />
          </label>
          <button type="submit" className="btn-secondary">Create alert</button>
        </form>
        <div className="market-alert-list">
          {priceAlerts.map((alert) => (
            <div className="market-alert-row" key={alert.id}>
              <span><strong>{displayName(alert.symbol)}</strong> {alert.direction === "above" ? "at or above" : "at or below"} {formatCurrency(alert.target)}</span>
              <span className={alert.triggeredAt ? "text-lime-300" : "text-gray-500"}>{alert.triggeredAt ? "Triggered" : "Watching"}</span>
              <button type="button" className="market-icon-button" onClick={() => setPriceAlerts((current) => current.filter((item) => item.id !== alert.id))} aria-label={`Remove alert for ${alert.symbol}`} title="Remove alert">
                <Trash2 size={15} />
              </button>
            </div>
          ))}
          {priceAlerts.length === 0 && <p className="py-3 text-sm text-gray-500">Create an alert to track a price level while this page is open.</p>}
        </div>
      </section>
    </div>
  );
}