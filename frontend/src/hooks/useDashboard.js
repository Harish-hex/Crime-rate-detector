import { useCallback, useEffect, useRef, useState } from "react";
import { getAlerts, getFilters, getMapData, getPrediction, getSummary, getTrends } from "../api";

const RETRY_DELAYS = [1000, 2000, 4000];

async function withRetry(fn, retries = RETRY_DELAYS) {
  let lastError;
  for (let attempt = 0; attempt <= retries.length; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (attempt < retries.length) {
        await new Promise((r) => setTimeout(r, retries[attempt]));
      }
    }
  }
  throw lastError;
}

export function useDashboard() {
  const [filters, setFilters] = useState(null);
  const [crimeType, setCrimeType] = useState("total_crimes");
  const [selectedState, setSelectedState] = useState("");
  const [selectedYear, setSelectedYear] = useState(2024);
  const [range, setRange] = useState({ startYear: 2001, endYear: 2024 });

  const [summary, setSummary] = useState(null);
  const [trends, setTrends] = useState(null);
  const [mapData, setMapData] = useState(null);
  const [alerts, setAlerts] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const [filtersLoading, setFiltersLoading] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [filtersError, setFiltersError] = useState("");

  const [refreshToken, setRefreshToken] = useState(0);
  const aliveRef = useRef(true);

  useEffect(() => {
    aliveRef.current = true;
    setFiltersLoading(true);
    setFiltersError("");

    withRetry(() => getFilters())
      .then((result) => {
        if (!aliveRef.current) return;
        const startYear = result.years[0];
        const endYear = result.years[result.years.length - 1];
        setFilters(result);
        setCrimeType(result.crime_types[0]);
        setSelectedState(result.states[0]);
        setSelectedYear(endYear);
        setRange({ startYear, endYear });
      })
      .catch((err) => {
        if (aliveRef.current) setFiltersError(err.message ?? "Failed to load filters");
      })
      .finally(() => {
        if (aliveRef.current) setFiltersLoading(false);
      });

    return () => {
      aliveRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!filters || !selectedState) return;
    aliveRef.current = true;
    setLoading(true);
    setError("");

    Promise.all([
      withRetry(() => getSummary({ crimeType, startYear: range.startYear, endYear: range.endYear })),
      withRetry(() => getTrends({ crimeType, startYear: range.startYear, endYear: range.endYear })),
      withRetry(() => getMapData({ crimeType, year: selectedYear })),
      withRetry(() => getAlerts({ crimeType })),
      withRetry(() => getPrediction({ state: selectedState, crimeType })),
    ])
      .then(([summaryR, trendsR, mapR, alertsR, predR]) => {
        if (!aliveRef.current) return;
        setSummary(summaryR);
        setTrends(trendsR);
        setMapData(mapR);
        setAlerts(alertsR);
        setPrediction(predR);
      })
      .catch((err) => {
        if (aliveRef.current) setError(err.message ?? "Failed to load dashboard data");
      })
      .finally(() => {
        if (aliveRef.current) setLoading(false);
      });

    const intervalId = window.setInterval(() => setRefreshToken((v) => v + 1), 60000);
    return () => {
      aliveRef.current = false;
      window.clearInterval(intervalId);
    };
  }, [filters, crimeType, range.startYear, range.endYear, selectedYear, selectedState, refreshToken]);

  const refresh = useCallback(() => setRefreshToken((v) => v + 1), []);

  const setStartYear = useCallback((nextStart) => {
    setRange((current) => ({
      startYear: nextStart,
      endYear: Math.max(nextStart, current.endYear),
    }));
  }, []);

  const setEndYear = useCallback((nextEnd) => {
    setRange((current) => ({
      startYear: Math.min(current.startYear, nextEnd),
      endYear: nextEnd,
    }));
  }, []);

  return {
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
  };
}
