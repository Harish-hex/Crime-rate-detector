from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STATE_COORDS: dict[str, tuple[float, float]] = {
    "Andaman & Nicobar": (11.74, 92.66),
    "Andhra Pradesh": (15.91, 79.74),
    "Arunachal Pradesh": (28.22, 94.72),
    "Assam": (26.20, 92.94),
    "Bihar": (25.09, 85.31),
    "Chandigarh": (30.73, 76.78),
    "Chhattisgarh": (21.27, 81.87),
    "Dadra & Nagar Haveli": (20.18, 73.02),
    "Daman & Diu": (20.43, 72.84),
    "Delhi": (28.61, 77.21),
    "Goa": (15.30, 74.12),
    "Gujarat": (22.26, 71.19),
    "Haryana": (29.06, 76.09),
    "Himachal Pradesh": (31.10, 77.17),
    "Jammu & Kashmir": (33.78, 76.58),
    "Jharkhand": (23.61, 85.28),
    "Karnataka": (15.32, 75.71),
    "Kerala": (10.85, 76.27),
    "Lakshadweep": (10.56, 72.64),
    "Madhya Pradesh": (22.97, 78.65),
    "Maharashtra": (19.75, 75.71),
    "Manipur": (24.66, 93.91),
    "Meghalaya": (25.47, 91.37),
    "Mizoram": (23.16, 92.93),
    "Nagaland": (26.16, 94.56),
    "Odisha": (20.95, 85.10),
    "Puducherry": (11.94, 79.81),
    "Punjab": (31.15, 75.34),
    "Rajasthan": (27.02, 74.22),
    "Sikkim": (27.53, 88.51),
    "Tamil Nadu": (11.13, 78.66),
    "Telangana": (18.11, 79.02),
    "Tripura": (23.94, 91.99),
    "Uttar Pradesh": (26.85, 80.95),
    "Uttarakhand": (30.07, 79.02),
    "West Bengal": (22.99, 87.85),
}

CRIME_TYPE_LABELS = {
    "total_crimes": "Total Crimes",
    "murder": "Murder",
    "rape": "Rape",
    "kidnapping": "Kidnapping",
    "robbery": "Robbery",
    "theft": "Theft",
    "riots": "Riots",
    "arson": "Arson",
}


@dataclass
class ForecastResult:
    forecast: dict[int, int]
    confidence: int


class DataLoadError(Exception):
    pass


class CrimeAnalyticsService:
    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path

    @cached_property
    def dataframe(self) -> pd.DataFrame:
        if not self.dataset_path.exists():
            raise DataLoadError(
                f"Dataset not found at {self.dataset_path}. "
                "Set CRIMESCOPE_DATASET_PATH to the correct location."
            )
        try:
            df = pd.read_csv(self.dataset_path)
        except Exception as exc:
            raise DataLoadError(f"Failed to read dataset: {exc}") from exc

        required_columns = {"state", "year"}
        missing = required_columns - set(df.columns)
        if missing:
            raise DataLoadError(f"Dataset is missing required columns: {missing}")

        numeric_cols = [
            "year",
            "total_crimes",
            "murder",
            "rape",
            "kidnapping",
            "robbery",
            "theft",
            "riots",
            "arson",
        ]
        for column in numeric_cols:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

        df["state"] = df["state"].astype(str).str.strip()
        df = df.sort_values(["year", "state"]).reset_index(drop=True)

        gaps = self._detect_year_gaps(df)
        if gaps:
            logger.warning("Dataset has year gaps: %s", gaps)

        return df

    @cached_property
    def years(self) -> list[int]:
        return sorted(self.dataframe["year"].astype(int).unique().tolist())

    @cached_property
    def crime_types(self) -> list[str]:
        return [key for key in CRIME_TYPE_LABELS if key in self.dataframe.columns]

    @cached_property
    def states(self) -> list[str]:
        return sorted(self.dataframe["state"].unique().tolist())

    @cached_property
    def _states_lower(self) -> dict[str, str]:
        return {s.lower().strip(): s for s in self.states}

    def get_filters(self) -> dict:
        return {
            "years": self.years,
            "crime_types": self.crime_types,
            "states": self.states,
        }

    def get_data_quality(self) -> dict:
        gaps = self._detect_year_gaps(self.dataframe)
        synthetic_count = 0
        synthetic_pct = 0.0
        if "source" in self.dataframe.columns:
            synthetic_count = int((self.dataframe["source"].str.startswith("synthetic")).sum())
            synthetic_pct = round(synthetic_count / max(len(self.dataframe), 1) * 100, 1)
        return {
            "total_rows": len(self.dataframe),
            "year_range": [self.years[0], self.years[-1]],
            "year_gaps": gaps,
            "states_count": len(self.states),
            "synthetic_rows": synthetic_count,
            "synthetic_pct": synthetic_pct,
        }

    def get_summary(self, crime_type: str, start_year: int | None, end_year: int | None) -> dict:
        metric = self._validate_metric(crime_type)
        filtered = self._filter_years(start_year, end_year)
        total_value = int(filtered[metric].sum())
        yearly_total = filtered.groupby("year")[metric].sum().sort_index()
        state_total = filtered.groupby("state")[metric].sum().sort_values(ascending=False)
        latest_year = int(yearly_total.index[-1])
        latest_value = int(yearly_total.iloc[-1])
        previous_value = int(yearly_total.iloc[-2]) if len(yearly_total) > 1 else latest_value
        delta = latest_value - previous_value
        delta_pct = (delta / previous_value * 100) if previous_value else 0.0

        return {
            "metric": metric,
            "start_year": int(filtered["year"].min()),
            "end_year": int(filtered["year"].max()),
            "total_value": total_value,
            "yearly_average": int(round(yearly_total.mean())),
            "highest_state": str(state_total.index[0]),
            "highest_state_value": int(state_total.iloc[0]),
            "latest_year": latest_year,
            "latest_year_value": latest_value,
            "previous_year_value": previous_value,
            "year_over_year_change": int(delta),
            "year_over_year_change_pct": round(delta_pct, 2),
        }

    def get_trends(
        self,
        crime_type: str,
        start_year: int | None,
        end_year: int | None,
        top_n: int = 6,
    ) -> dict:
        metric = self._validate_metric(crime_type)
        filtered = self._filter_years(start_year, end_year)
        national = filtered.groupby("year")[metric].sum().sort_index()
        future_years = self._default_future_years()
        national_forecast = self._forecast_series(national.index.tolist(), national.values.tolist(), future_years)

        state_totals = filtered.groupby("state")[metric].sum().sort_values(ascending=False).head(top_n)
        state_series = []
        for state, total in state_totals.items():
            subset = filtered[filtered["state"] == state].sort_values("year")
            history = subset[["year", metric]]
            forecast = self._forecast_series(
                history["year"].astype(int).tolist(),
                history[metric].astype(float).tolist(),
                future_years,
            )
            state_series.append(
                {
                    "name": state,
                    "total": float(total),
                    "history": [
                        {"year": int(row.year), "value": float(row.value)}
                        for row in history.rename(columns={metric: "value"}).itertuples(index=False)
                    ],
                    "forecast": [
                        {"year": year, "value": value}
                        for year, value in forecast.forecast.items()
                    ],
                    "confidence": forecast.confidence,
                }
            )

        return {
            "metric": metric,
            "national_history": [
                {"year": int(year), "value": float(value)} for year, value in national.items()
            ],
            "national_forecast": [
                {"year": year, "value": value} for year, value in national_forecast.forecast.items()
            ],
            "top_states": state_series,
        }

    def get_map_points(
        self,
        crime_type: str,
        year: int | None,
        start_year: int | None,
        end_year: int | None,
    ) -> dict:
        metric = self._validate_metric(crime_type)
        if year is not None:
            if year not in self.years:
                raise ValueError(f"Year {year} is not available in the dataset.")
            filtered = self.dataframe[self.dataframe["year"] == year]
            mode = "year"
            start = end = year
        else:
            filtered = self._filter_years(start_year, end_year)
            mode = "range"
            start = int(filtered["year"].min())
            end = int(filtered["year"].max())

        aggregated = (
            filtered.groupby("state")[metric].sum().sort_values(ascending=False).reset_index(name="value")
        )
        q25 = float(aggregated["value"].quantile(0.25))
        q50 = float(aggregated["value"].quantile(0.50))
        q75 = float(aggregated["value"].quantile(0.75))

        forecast_year = self._default_future_years()[-1]
        points = []
        for row in aggregated.itertuples(index=False):
            if row.state not in STATE_COORDS:
                continue
            lat, lon = STATE_COORDS[row.state]
            risk, color = self._risk_for_value(int(row.value), q25, q50, q75)
            forecast = self._state_forecast(row.state, metric)
            points.append(
                {
                    "state": row.state,
                    "latitude": lat,
                    "longitude": lon,
                    "value": int(row.value),
                    "risk": risk,
                    "risk_color": color,
                    "forecast_year": forecast_year,
                    "forecast_value": forecast.forecast.get(forecast_year, 0),
                    "confidence": forecast.confidence,
                }
            )

        return {
            "metric": metric,
            "mode": mode,
            "year": year,
            "start_year": start,
            "end_year": end,
            "points": points,
        }

    def get_forecast(self, state: str, crime_type: str, years: list[int] | None = None) -> dict:
        metric = self._validate_metric(crime_type)
        resolved_state = self._resolve_state(state)
        years_to_use = years or self._default_future_years()

        if years_to_use:
            max_forecast = self.years[-1] + 20
            invalid = [y for y in years_to_use if y <= self.years[-1] or y > max_forecast]
            if invalid:
                raise ValueError(
                    f"Forecast years must be after {self.years[-1]} and no more than {max_forecast}. Invalid: {invalid}"
                )

        history_df = self.dataframe[self.dataframe["state"] == resolved_state].sort_values("year")
        forecast = self._forecast_series(
            history_df["year"].astype(int).tolist(),
            history_df[metric].astype(float).tolist(),
            years_to_use,
        )
        return {
            "state": resolved_state,
            "metric": metric,
            "historical": [
                {"year": int(row.year), "value": float(row.value)}
                for row in history_df[["year", metric]].rename(columns={metric: "value"}).itertuples(index=False)
            ],
            "forecast": [
                {"year": year, "value": value} for year, value in forecast.forecast.items()
            ],
            "confidence": forecast.confidence,
        }

    def get_alerts(self, crime_type: str, limit: int = 8) -> dict:
        metric = self._validate_metric(crime_type)
        sorted_years = sorted(self.years)
        latest_year = sorted_years[-1]

        states_with_latest = set(
            self.dataframe[self.dataframe["year"] == latest_year]["state"].unique()
        )
        if len(sorted_years) < 2:
            return {"metric": metric, "latest_year": latest_year, "alerts": []}

        previous_year = sorted_years[-2]
        states_with_previous = set(
            self.dataframe[self.dataframe["year"] == previous_year]["state"].unique()
        )
        valid_states = states_with_latest & states_with_previous

        latest = (
            self.dataframe[
                (self.dataframe["year"] == latest_year) & (self.dataframe["state"].isin(valid_states))
            ][["state", metric]]
            .rename(columns={metric: "latest"})
        )
        previous = (
            self.dataframe[
                (self.dataframe["year"] == previous_year) & (self.dataframe["state"].isin(valid_states))
            ][["state", metric]]
            .rename(columns={metric: "previous"})
        )
        merged = latest.merge(previous, on="state", how="inner")
        merged["change_pct"] = np.where(
            merged["previous"] > 0,
            (merged["latest"] - merged["previous"]) / merged["previous"] * 100,
            0,
        )
        merged = merged.sort_values(["change_pct", "latest"], ascending=[False, False]).head(limit)

        alerts = []
        for row in merged.itertuples(index=False):
            severity = "critical" if row.change_pct >= 15 else "warning" if row.change_pct >= 8 else "watch"
            direction = "rose" if row.change_pct >= 0 else "fell"
            alerts.append(
                {
                    "state": row.state,
                    "metric": metric,
                    "latest_value": int(row.latest),
                    "previous_value": int(row.previous),
                    "change_pct": round(float(row.change_pct), 2),
                    "severity": severity,
                    "message": (
                        f"{CRIME_TYPE_LABELS[metric]} {direction} "
                        f"{abs(row.change_pct):.1f}% from {previous_year} to {latest_year}."
                    ),
                }
            )

        return {"metric": metric, "latest_year": latest_year, "alerts": alerts}

    def _resolve_state(self, state: str) -> str:
        normalized = state.strip().lower()
        if normalized in self._states_lower:
            return self._states_lower[normalized]
        raise ValueError(
            f"Unknown state '{state}'. Valid states: {', '.join(self.states[:10])}..."
        )

    def _filter_years(self, start_year: int | None, end_year: int | None) -> pd.DataFrame:
        start = start_year or self.years[0]
        end = end_year or self.years[-1]
        if start > end:
            raise ValueError("start_year must be less than or equal to end_year.")
        filtered = self.dataframe[(self.dataframe["year"] >= start) & (self.dataframe["year"] <= end)]
        if filtered.empty:
            raise ValueError(f"No data available for {start}–{end}.")
        return filtered

    def _validate_metric(self, crime_type: str) -> str:
        if crime_type not in self.crime_types:
            raise ValueError(
                f"Unsupported crime_type '{crime_type}'. Valid options: {', '.join(self.crime_types)}"
            )
        return crime_type

    def _default_future_years(self) -> list[int]:
        last_year = self.years[-1]
        return list(range(last_year + 1, last_year + 6))

    def _state_forecast(self, state: str, metric: str) -> ForecastResult:
        subset = self.dataframe[self.dataframe["state"] == state].sort_values("year")
        return self._forecast_series(
            subset["year"].astype(int).tolist(),
            subset[metric].astype(float).tolist(),
            self._default_future_years(),
        )

    def _forecast_series(self, years: list[int], values: list[float], future_years: list[int]) -> ForecastResult:
        cnts = np.array(values, dtype=float)
        hist_years = np.array(years, dtype=int)

        hist_years, cnts = self._recent_consistent_window(hist_years, cnts)

        if len(cnts) > 5:
            median_val = np.median(cnts)
            if median_val > 0:
                mask = cnts > (median_val * 0.10)
                if mask.sum() >= 3:
                    cnts = cnts[mask]
                    hist_years = hist_years[mask]

        if len(cnts) == 0:
            return ForecastResult({year: 0 for year in future_years}, 35)
        if len(cnts) == 1:
            return ForecastResult({year: int(max(cnts[0], 0)) for year in future_years}, 45)

        last_val = float(max(cnts[-1], 0))
        last_year = int(hist_years[-1])

        slope, intercept = np.polyfit(hist_years, cnts, 1)
        ols_fit = slope * hist_years + intercept
        rmse = float(np.sqrt(np.mean((cnts - ols_fit) ** 2)))

        level, trend = self._holt_smooth(cnts)
        first_safe = max(float(cnts[0]), 1.0)
        cagr = (max(last_val, 1.0) / first_safe) ** (1.0 / max(len(cnts) - 1, 1)) - 1
        cagr = float(np.clip(cagr, -0.15, 0.15))

        forecasts: dict[int, int] = {}
        for year in future_years:
            steps = year - last_year
            linear = last_val + slope * steps
            holt = level + trend * steps
            growth = last_val * ((1 + cagr) ** steps)
            ensemble = 0.40 * linear + 0.35 * holt + 0.25 * growth
            bounded = max(ensemble, last_val * 0.85, 0)
            bounded = min(bounded, float(cnts.max()) * 3.0)
            forecasts[year] = int(round(bounded))

        confidence = self._confidence_score(cnts, rmse)
        return ForecastResult(forecasts, confidence)

    def _recent_consistent_window(
        self, years: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(years) < 4:
            return years, values

        year_gaps = np.diff(years)
        gap_indexes = np.where(year_gaps > 1)[0]
        if len(gap_indexes) > 0:
            last_gap = gap_indexes[-1] + 1
            if len(years[last_gap:]) >= 4:
                years = years[last_gap:]
                values = values[last_gap:]

        if len(values) >= 6:
            split_index = len(values) - min(5, len(values))
            earlier = values[:split_index]
            recent = values[split_index:]
            if len(earlier) >= 3 and len(recent) >= 3:
                earlier_median = max(float(np.median(earlier)), 1.0)
                recent_median = max(float(np.median(recent)), 1.0)
                ratio = recent_median / earlier_median
                if ratio <= 0.2 or ratio >= 5.0:
                    years = years[split_index:]
                    values = values[split_index:]

        return years, values

    def _holt_smooth(self, series: np.ndarray, alpha: float = 0.5, beta: float = 0.3) -> tuple[float, float]:
        level = float(series[0])
        trend = float(series[1] - series[0]) if len(series) > 1 else 0.0
        for value in series[1:]:
            previous_level, previous_trend = level, trend
            level = alpha * float(value) + (1 - alpha) * (previous_level + previous_trend)
            trend = beta * (level - previous_level) + (1 - beta) * previous_trend
        return level, trend

    def _confidence_score(self, series: np.ndarray, rmse: float) -> int:
        baseline = max(float(np.mean(series)), 1.0)
        volatility = float(np.std(series) / baseline)
        fit_penalty = min(rmse / baseline, 1.0)
        confidence = 92 - (volatility * 25) - (fit_penalty * 35)
        return int(np.clip(round(confidence), 45, 93))

    def _risk_for_value(self, value: int, q25: float, q50: float, q75: float) -> tuple[str, str]:
        if value >= q75:
            return "High Risk", "#ef4444"
        if value >= q50:
            return "Medium-High", "#f97316"
        if value >= q25:
            return "Medium-Low", "#facc15"
        return "Low Risk", "#22c55e"

    def _detect_year_gaps(self, df: pd.DataFrame) -> list[list[int]]:
        years = sorted(df["year"].astype(int).unique().tolist())
        gaps = []
        for i in range(len(years) - 1):
            if years[i + 1] - years[i] > 1:
                gaps.append([years[i], years[i + 1]])
        return gaps
