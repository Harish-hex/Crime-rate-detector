import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ErrorBoundary } from "./components/ErrorBoundary";
import { MetricCard } from "./components/MetricCard";
import { Panel } from "./components/Panel";
import { FilterBar } from "./components/FilterBar";
import { AlertCard } from "./components/AlertCard";
import { CrimeMap } from "./components/CrimeMap";
import { ChartSkeleton } from "./components/Skeleton";
import { useDashboard } from "./hooks/useDashboard";
import { getDataQuality } from "./api";
import {
  buildPredictionRows,
  buildRiskBreakdown,
  buildStateRows,
  buildTrendRows,
  formatNumber,
  METRIC_LABELS,
  RISK_COLORS,
  STATE_PALETTE,
} from "./utils/format";

const TOOLTIP_STYLE = {
  contentStyle: {
    background: "rgba(20, 20, 26, 0.97)",
    border: "1px solid rgba(255, 255, 255, 0.1)",
    borderRadius: 10,
    fontFamily: "'Geist', system-ui, sans-serif",
    boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
  },
  labelStyle: { color: "#f4f4f5", fontWeight: 600, fontSize: "0.82rem" },
  itemStyle: { color: "#a1a1aa", fontSize: "0.8rem" },
  cursor: { stroke: "rgba(255,255,255,0.06)", strokeWidth: 1 },
};

const AXIS_STYLE = { stroke: "#3f3f46", fontSize: 11, fontFamily: "'Geist Mono', monospace" };

function NavBar({ loading, onRefresh }) {
  return (
    <div className="nav-bar">
      <div className="nav-logo">
        <div className="nav-logo-mark" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
            <circle cx="9" cy="9" r="3.5" fill="white" fillOpacity="0.9" />
            <circle cx="9" cy="9" r="7" stroke="white" strokeOpacity="0.4" strokeWidth="1.2" />
          </svg>
        </div>
        <div>
          <div className="nav-logo-text">CrimeScope</div>
          <div className="nav-logo-sub">India Crime Intelligence</div>
        </div>
      </div>
      <div className="nav-actions">
        <div className="nav-status">
          <span className="nav-status-dot" />
          <span>Live · 2001–2024</span>
        </div>
        <button type="button" className="btn-ghost" onClick={onRefresh} disabled={loading}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
            <path d="M3 3v5h5" />
          </svg>
          {loading ? "Refreshing" : "Refresh"}
        </button>
      </div>
    </div>
  );
}

function useDataQuality() {
  const [quality, setQuality] = useState(null);
  useEffect(() => {
    getDataQuality().then(setQuality).catch(() => null);
  }, []);
  return quality;
}

function Hero() {
  const quality = useDataQuality();
  return (
    <header className="hero">
      <div>
        <p className="eyebrow">
          <span className="eyebrow-line" aria-hidden="true" />
          Intelligence Dashboard
        </p>
        <h1>
          India crime<br />
          analytics &amp; <em>forecasting</em>
        </h1>
        <p className="hero-text">
          State-level crime trends, interactive risk map, and ML-powered forecasting across
          36 states and union territories. Full 2001–2024 coverage — gaps filled with
          interpolated &amp; trend-extrapolated data.
        </p>
      </div>
      <div className="hero-aside">
        <div className="hero-stat-block">
          <div className="hero-stat-label">Dataset Coverage</div>
          <div className="hero-stat-value">864</div>
          <div className="hero-stat-sub">State × year records</div>
        </div>
        <div className="hero-stat-block">
          <div className="hero-stat-label">Synthetic Rows</div>
          <div className="hero-stat-value">
            {quality ? `${quality.synthetic_pct}%` : "—"}
          </div>
          <div className="hero-stat-sub">
            {quality ? `${quality.synthetic_rows} of ${quality.total_rows} rows` : "Loading…"}
          </div>
        </div>
      </div>
    </header>
  );
}

function App() {
  const {
    filters,
    crimeType,
    setCrimeType,
    selectedState,
    setSelectedState,
    selectedYear,
    setSelectedYear,
    range,
    setStartYear,
    setEndYear,
    summary,
    trends,
    mapData,
    alerts,
    prediction,
    filtersLoading,
    loading,
    error,
    filtersError,
    refresh,
  } = useDashboard();

  const trendRows      = useMemo(() => buildTrendRows(trends), [trends]);
  const stateRows      = useMemo(() => buildStateRows(trends?.top_states), [trends]);
  const predictionRows = useMemo(() => buildPredictionRows(prediction), [prediction]);
  const riskBreakdown  = useMemo(() => buildRiskBreakdown(mapData?.points), [mapData]);

  const metricLabel = METRIC_LABELS[crimeType] ?? crimeType;
  const forecastYear = prediction?.forecast?.[prediction.forecast.length - 1]?.year;

  return (
    <ErrorBoundary>
      <div className="app-shell">

        {/* ── Navigation ── */}
        <NavBar loading={loading} onRefresh={refresh} />

        {/* ── Hero ── */}
        <Hero />

        {/* ── Filters ── */}
        <FilterBar
          filters={filters}
          crimeType={crimeType}
          setCrimeType={setCrimeType}
          range={range}
          setStartYear={setStartYear}
          setEndYear={setEndYear}
          selectedYear={selectedYear}
          setSelectedYear={setSelectedYear}
          selectedState={selectedState}
          setSelectedState={setSelectedState}
          disabled={filtersLoading || !filters}
        />

        {/* ── Error banners ── */}
        {filtersError && (
          <div className="error-banner" role="alert">
            <svg className="error-banner-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
            Could not connect to the API: {filtersError}. Make sure the backend is running on port 8000.
          </div>
        )}
        {error && !filtersError && (
          <div className="error-banner" role="alert">
            <svg className="error-banner-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
            Dashboard data failed to load: {error}
          </div>
        )}

        {/* ── Metric Cards ── */}
        <section className="metrics-grid" aria-label="Key metrics">
          <MetricCard
            loading={loading && !summary}
            title="Filtered Total"
            value={summary ? formatNumber(summary.total_value) : "—"}
            helper={`${range.startYear}–${range.endYear} · ${metricLabel}`}
            accent="var(--data-blue)"
          />
          <MetricCard
            loading={loading && !summary}
            title="Most Affected State"
            value={summary?.highest_state ?? "—"}
            helper={summary ? `${formatNumber(summary.highest_state_value)} incidents` : "Awaiting data"}
            accent="var(--accent)"
          />
          <MetricCard
            loading={loading && !summary}
            title="Year-on-Year Change"
            value={
              summary
                ? `${summary.year_over_year_change >= 0 ? "+" : ""}${formatNumber(summary.year_over_year_change)}`
                : "—"
            }
            helper={summary ? `${summary.year_over_year_change_pct > 0 ? "+" : ""}${summary.year_over_year_change_pct}% vs prior year` : "—"}
            trend={summary?.year_over_year_change_pct}
            accent="var(--data-amber)"
          />
          <MetricCard
            loading={loading && !prediction}
            title="Forecast Confidence"
            value={prediction ? `${prediction.confidence}%` : "—"}
            helper={
              prediction
                ? `${selectedState} · ${metricLabel}${forecastYear ? ` to ${forecastYear}` : ""}`
                : "Select a state"
            }
            accent="var(--data-emerald)"
          />
        </section>

        {/* ── Dashboard Grid ── */}
        <div className="dashboard-grid">

          {/* National Trend */}
          <Panel
            title="National Trend"
            subtitle={`Historical ${metricLabel} with 5-year forecast`}
            loading={loading && !trends}
          >
            <div className="chart-wrap">
              {loading && !trends ? (
                <ChartSkeleton height={320} />
              ) : (
                <ResponsiveContainer width="100%" height={320}>
                  <AreaChart data={trendRows} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
                    <defs>
                      <linearGradient id="grad-hist" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="5%"  stopColor="#5b8af0" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#5b8af0" stopOpacity={0.02} />
                      </linearGradient>
                      <linearGradient id="grad-fore" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="5%"  stopColor="#e05c2a" stopOpacity={0.35} />
                        <stop offset="95%" stopColor="#e05c2a" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="year" {...AXIS_STYLE} />
                    <YAxis {...AXIS_STYLE} tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v)} />
                    <Tooltip formatter={(v) => formatNumber(v)} {...TOOLTIP_STYLE} />
                    <Legend wrapperStyle={{ fontFamily: "'Geist', system-ui, sans-serif", fontSize: "0.75rem" }} />
                    <Area dataKey="historical" name="Historical" type="monotone" stroke="#5b8af0" strokeWidth={2} fill="url(#grad-hist)" dot={false} />
                    <Area dataKey="forecast"   name="Forecast"   type="monotone" stroke="#e05c2a" strokeWidth={2} fill="url(#grad-fore)" strokeDasharray="5 4" dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </Panel>

          {/* State Risk Map */}
          <Panel
            title="State Risk Map"
            subtitle={`Crime distribution for ${selectedYear} · click a marker for details`}
            loading={loading && !mapData}
          >
            <div className="map-wrap">
              {mapData ? (
                <CrimeMap mapData={mapData} crimeType={crimeType} />
              ) : (
                <div className="chart-skeleton" style={{ height: 360, alignItems: "center", justifyContent: "center" }}>
                  <span style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>Loading map…</span>
                </div>
              )}
            </div>
          </Panel>

          {/* Top State Comparison */}
          <Panel
            title="Top State Comparison"
            subtitle="Historical trends continuing into forecast years"
            loading={loading && !trends}
          >
            <div className="chart-wrap">
              {loading && !trends ? (
                <ChartSkeleton height={320} />
              ) : (
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={stateRows} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="year" {...AXIS_STYLE} />
                    <YAxis {...AXIS_STYLE} tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v)} />
                    <Tooltip formatter={(v) => formatNumber(v)} {...TOOLTIP_STYLE} />
                    <Legend wrapperStyle={{ fontFamily: "'Geist', system-ui, sans-serif", fontSize: "0.75rem" }} />
                    {(trends?.top_states ?? []).map((series, index) => (
                      <Line
                        key={series.name}
                        type="monotone"
                        dataKey={series.name}
                        stroke={STATE_PALETTE[index % STATE_PALETTE.length]}
                        strokeWidth={1.8}
                        dot={false}
                        connectNulls
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </Panel>

          {/* Risk Distribution */}
          <Panel
            title="Risk Distribution"
            subtitle={`State risk spread for map year ${selectedYear}`}
            loading={loading && !mapData}
          >
            <div className="chart-wrap chart-wrap--sm">
              {loading && !mapData ? (
                <ChartSkeleton height={280} />
              ) : (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={riskBreakdown} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="risk" {...AXIS_STYLE} />
                    <YAxis {...AXIS_STYLE} allowDecimals={false} />
                    <Tooltip {...TOOLTIP_STYLE} />
                    <Bar dataKey="count" name="States" radius={[6, 6, 0, 0]}>
                      {riskBreakdown.map((item) => (
                        <Cell key={item.risk} fill={RISK_COLORS[item.risk] ?? "#52525b"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </Panel>

          {/* Prediction Workspace */}
          <Panel
            title="Prediction Workspace"
            subtitle={`5-year forecast for ${selectedState} · ${metricLabel}`}
            badge={prediction ? `${prediction.confidence}% confidence` : undefined}
            loading={loading && !prediction}
          >
            <div className="chart-wrap">
              {loading && !prediction ? (
                <ChartSkeleton height={300} />
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={predictionRows} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="year" {...AXIS_STYLE} />
                    <YAxis {...AXIS_STYLE} tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v)} />
                    <Tooltip formatter={(v) => formatNumber(v)} {...TOOLTIP_STYLE} />
                    <Legend wrapperStyle={{ fontFamily: "'Geist', system-ui, sans-serif", fontSize: "0.75rem" }} />
                    <Line type="monotone" dataKey="historical" name="Historical" stroke="#5b8af0" strokeWidth={2.2} dot={false} />
                    <Line type="monotone" dataKey="forecast"   name="Forecast"   stroke="#e05c2a" strokeWidth={2.2} strokeDasharray="5 4" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </Panel>

          {/* Alert Feed */}
          <Panel
            title="Alert Feed"
            subtitle="States with sharpest year-over-year increases"
            loading={loading && !alerts}
          >
            {loading && !alerts ? (
              <div className="alerts-grid">
                {Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className="alert-card">
                    <div className="skeleton h-3 w-24" style={{ marginBottom: 10 }} />
                    <div className="skeleton h-3 w-40" style={{ marginBottom: 8 }} />
                    <div className="skeleton h-3 w-32" />
                  </div>
                ))}
              </div>
            ) : (
              <div className="alerts-grid">
                {(alerts?.alerts ?? []).map((alert) => (
                  <AlertCard key={alert.state} alert={alert} />
                ))}
              </div>
            )}
          </Panel>

        </div>
      </div>
    </ErrorBoundary>
  );
}

export default App;
