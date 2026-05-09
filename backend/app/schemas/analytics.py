from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class MetricPoint(BaseModel):
    year: int
    value: float


class ForecastPoint(BaseModel):
    year: int
    value: int


class TrendSeries(BaseModel):
    name: str
    total: float
    history: list[MetricPoint]
    forecast: list[ForecastPoint]
    confidence: int


class SummaryResponse(BaseModel):
    metric: str
    start_year: int
    end_year: int
    total_value: int
    yearly_average: int
    highest_state: str
    highest_state_value: int
    latest_year: int
    latest_year_value: int
    previous_year_value: int
    year_over_year_change: int
    year_over_year_change_pct: float


class FiltersResponse(BaseModel):
    years: list[int]
    crime_types: list[str]
    states: list[str]


class DataQualityResponse(BaseModel):
    total_rows: int
    year_range: list[int]
    year_gaps: list[list[int]]
    states_count: int


class MapPoint(BaseModel):
    state: str
    latitude: float
    longitude: float
    value: int
    risk: str
    risk_color: str
    forecast_year: int
    forecast_value: int
    confidence: int


class MapResponse(BaseModel):
    metric: str
    mode: Literal["year", "range"]
    year: Optional[int] = None
    start_year: int
    end_year: int
    points: list[MapPoint]


class TrendsResponse(BaseModel):
    metric: str
    national_history: list[MetricPoint]
    national_forecast: list[ForecastPoint]
    top_states: list[TrendSeries]


class PredictionRequest(BaseModel):
    state: str = Field(..., min_length=1, max_length=100, description="State or union territory name")
    crime_type: str = Field(default="total_crimes", min_length=1, max_length=50)
    years: Optional[list[int]] = Field(
        default=None,
        description="Future years to forecast",
        max_length=10,
    )


class PredictionResponse(BaseModel):
    state: str
    metric: str
    historical: list[MetricPoint]
    forecast: list[ForecastPoint]
    confidence: int


class AlertItem(BaseModel):
    state: str
    metric: str
    latest_value: int
    previous_value: int
    change_pct: float
    severity: Literal["watch", "warning", "critical"]
    message: str


class AlertsResponse(BaseModel):
    metric: str
    latest_year: int
    alerts: list[AlertItem]
