from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.dependencies import get_crime_service
from app.schemas.analytics import (
    AlertsResponse,
    DataQualityResponse,
    FiltersResponse,
    MapResponse,
    PredictionRequest,
    PredictionResponse,
    SummaryResponse,
    TrendsResponse,
)
from app.services.crime_analytics import CrimeAnalyticsService, DataLoadError

router = APIRouter()

_rate_buckets: dict[str, list[float]] = defaultdict(list)
_RATE_WINDOW = 60.0
_RATE_LIMIT = 60


def _check_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    bucket = _rate_buckets[client_ip]
    _rate_buckets[client_ip] = [t for t in bucket if now - t < _RATE_WINDOW]
    if len(_rate_buckets[client_ip]) >= _RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    _rate_buckets[client_ip].append(now)


def _handle_service_error(exc: Exception) -> None:
    if isinstance(exc, DataLoadError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/data-quality", response_model=DataQualityResponse)
async def data_quality(
    request: Request,
    service: CrimeAnalyticsService = Depends(get_crime_service),
) -> DataQualityResponse:
    _check_rate_limit(request)
    try:
        return DataQualityResponse(**service.get_data_quality())
    except Exception as exc:
        _handle_service_error(exc)


@router.get("/filters", response_model=FiltersResponse)
async def filters(
    request: Request,
    service: CrimeAnalyticsService = Depends(get_crime_service),
) -> FiltersResponse:
    _check_rate_limit(request)
    try:
        return FiltersResponse(**service.get_filters())
    except Exception as exc:
        _handle_service_error(exc)


@router.get("/summary", response_model=SummaryResponse)
async def summary(
    request: Request,
    crime_type: str = Query(default="total_crimes"),
    start_year: Optional[int] = Query(default=None, ge=1900, le=2100),
    end_year: Optional[int] = Query(default=None, ge=1900, le=2100),
    service: CrimeAnalyticsService = Depends(get_crime_service),
) -> SummaryResponse:
    _check_rate_limit(request)
    try:
        return SummaryResponse(**service.get_summary(crime_type, start_year, end_year))
    except Exception as exc:
        _handle_service_error(exc)


@router.get("/trends", response_model=TrendsResponse)
async def trends(
    request: Request,
    crime_type: str = Query(default="total_crimes"),
    start_year: Optional[int] = Query(default=None, ge=1900, le=2100),
    end_year: Optional[int] = Query(default=None, ge=1900, le=2100),
    top_n: int = Query(default=6, ge=3, le=10),
    service: CrimeAnalyticsService = Depends(get_crime_service),
) -> TrendsResponse:
    _check_rate_limit(request)
    try:
        return TrendsResponse(**service.get_trends(crime_type, start_year, end_year, top_n))
    except Exception as exc:
        _handle_service_error(exc)


@router.get("/map", response_model=MapResponse)
async def map_data(
    request: Request,
    crime_type: str = Query(default="total_crimes"),
    year: Optional[int] = Query(default=None, ge=1900, le=2100),
    start_year: Optional[int] = Query(default=None, ge=1900, le=2100),
    end_year: Optional[int] = Query(default=None, ge=1900, le=2100),
    service: CrimeAnalyticsService = Depends(get_crime_service),
) -> MapResponse:
    _check_rate_limit(request)
    try:
        return MapResponse(**service.get_map_points(crime_type, year, start_year, end_year))
    except Exception as exc:
        _handle_service_error(exc)


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    payload: PredictionRequest,
    service: CrimeAnalyticsService = Depends(get_crime_service),
) -> PredictionResponse:
    _check_rate_limit(request)
    try:
        return PredictionResponse(**service.get_forecast(payload.state, payload.crime_type, payload.years))
    except Exception as exc:
        _handle_service_error(exc)


@router.get("/alerts", response_model=AlertsResponse)
async def alerts(
    request: Request,
    crime_type: str = Query(default="total_crimes"),
    limit: int = Query(default=8, ge=1, le=20),
    service: CrimeAnalyticsService = Depends(get_crime_service),
) -> AlertsResponse:
    _check_rate_limit(request)
    try:
        return AlertsResponse(**service.get_alerts(crime_type, limit))
    except Exception as exc:
        _handle_service_error(exc)
