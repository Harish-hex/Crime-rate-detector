const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json", ...(options.headers ?? {}) },
      ...options,
    });
  } catch {
    throw new Error("Cannot reach the API server. Is the backend running on port 8000?");
  }

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = payload.detail;
    if (response.status === 429) throw new Error("Rate limit exceeded. Please wait a moment.");
    if (response.status === 503) throw new Error(`Service unavailable: ${detail ?? "Dataset not loaded."}`);
    throw new Error(detail ?? `Server error ${response.status}`);
  }

  return response.json();
}

export function getFilters() {
  return request("/filters");
}

export function getDataQuality() {
  return request("/data-quality");
}

export function getSummary({ crimeType, startYear, endYear }) {
  return request(`/summary?crime_type=${crimeType}&start_year=${startYear}&end_year=${endYear}`);
}

export function getTrends({ crimeType, startYear, endYear }) {
  return request(`/trends?crime_type=${crimeType}&start_year=${startYear}&end_year=${endYear}`);
}

export function getMapData({ crimeType, year }) {
  return request(`/map?crime_type=${crimeType}&year=${year}`);
}

export function getAlerts({ crimeType, limit = 8 }) {
  return request(`/alerts?crime_type=${crimeType}&limit=${limit}`);
}

export function getPrediction({ state, crimeType, years }) {
  return request("/predict", {
    method: "POST",
    body: JSON.stringify({ state, crime_type: crimeType, ...(years ? { years } : {}) }),
  });
}
