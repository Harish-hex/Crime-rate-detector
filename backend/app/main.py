import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.services.crime_analytics import DataLoadError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("CrimeScope starting — dataset: %s", settings.dataset_path)
    if not settings.dataset_path.exists():
        logger.error("Dataset file not found at %s", settings.dataset_path)
    else:
        logger.info("Dataset found, size: %s bytes", settings.dataset_path.stat().st_size)
    yield
    logger.info("CrimeScope shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

app.include_router(api_router, prefix=settings.api_prefix)


@app.exception_handler(DataLoadError)
async def data_load_error_handler(_: Request, exc: DataLoadError) -> JSONResponse:
    logger.error("Dataset error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def global_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
