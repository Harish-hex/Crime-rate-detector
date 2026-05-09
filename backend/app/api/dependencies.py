from functools import lru_cache

from app.core.config import get_settings
from app.services.crime_analytics import CrimeAnalyticsService


@lru_cache
def get_crime_service() -> CrimeAnalyticsService:
    settings = get_settings()
    return CrimeAnalyticsService(settings.dataset_path)
