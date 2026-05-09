from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "CrimeScope API"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    dataset_path: Path = (
        Path(__file__).resolve().parents[3] / "india_crime_combined_2001_2024_augmented.csv"
    )
    allow_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    rate_limit_per_minute: int = 60

    model_config = {"env_prefix": "CRIMESCOPE_", "env_file": ".env", "extra": "ignore"}

    @field_validator("dataset_path", mode="before")
    @classmethod
    def resolve_dataset_path(cls, v: object) -> Path:
        path = Path(str(v))
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
