export const METRIC_LABELS = {
  total_crimes: "Total Crimes",
  murder: "Murder",
  rape: "Rape",
  kidnapping: "Kidnapping",
  robbery: "Robbery",
  theft: "Theft",
  riots: "Riots",
  arson: "Arson",
};

export function formatNumber(value) {
  return new Intl.NumberFormat("en-IN").format(Math.round(value ?? 0));
}

export function formatPct(value, includeSign = true) {
  const sign = includeSign && value >= 0 ? "+" : "";
  return `${sign}${Number(value).toFixed(1)}%`;
}

export function buildTrendRows(trends) {
  if (!trends) return [];
  const rows = new Map();
  for (const item of trends.national_history) {
    rows.set(item.year, { year: item.year, historical: item.value });
  }
  for (const item of trends.national_forecast) {
    rows.set(item.year, { ...(rows.get(item.year) ?? { year: item.year }), forecast: item.value });
  }
  return Array.from(rows.values()).sort((a, b) => a.year - b.year);
}

export function buildStateRows(topStates) {
  if (!topStates?.length) return [];
  const years = new Set();
  for (const state of topStates) {
    state.history.forEach((p) => years.add(p.year));
    state.forecast.forEach((p) => years.add(p.year));
  }
  return Array.from(years)
    .sort((a, b) => a - b)
    .map((year) => {
      const row = { year };
      for (const state of topStates) {
        const hist = state.history.find((p) => p.year === year);
        const fore = state.forecast.find((p) => p.year === year);
        row[state.name] = fore?.value ?? hist?.value ?? null;
      }
      return row;
    });
}

export function buildPredictionRows(prediction) {
  if (!prediction) return [];
  const rows = prediction.historical.map((p) => ({ year: p.year, historical: p.value }));
  for (const item of prediction.forecast) {
    rows.push({ year: item.year, forecast: item.value });
  }
  return rows.sort((a, b) => a.year - b.year);
}

export function buildRiskBreakdown(points) {
  if (!points) return [];
  const counts = points.reduce((acc, item) => {
    acc[item.risk] = (acc[item.risk] ?? 0) + 1;
    return acc;
  }, {});
  return Object.entries(counts).map(([risk, count]) => ({ risk, count }));
}

export const RISK_COLORS = {
  "High Risk":    "#e05c2a",
  "Medium-High":  "#d97706",
  "Medium-Low":   "#ca8a04",
  "Low Risk":     "#16a34a",
};

export const ALERT_COLORS = {
  critical: "#e05c2a",
  warning:  "#d97706",
  watch:    "#ca8a04",
};

export const STATE_PALETTE = ["#5b8af0", "#e5667a", "#34c785", "#d4943c", "#9f7aee", "#3bb8cc"];

export function mapRadius(value) {
  if (!value || value <= 0) return 6;
  return Math.max(6, Math.min(22, Math.log10(value + 1) * 7));
}
