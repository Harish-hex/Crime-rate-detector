# CrimeScope — India Crime Intelligence

A full-stack web application for exploring, analyzing, and forecasting crime trends across Indian states from 2001 to 2024.

## Features

- **Dashboard** — Filter by crime category and year range; summary metric cards with YoY trend indicators
- **National Trend Chart** — Historical data (2001–2024) with 5-year forecast overlay
- **State Comparison** — Top-N states plotted over time on a shared chart
- **Interactive Map** — Leaflet map with per-state risk labels, latest values, and forecast popups
- **Alert Feed** — Ranked year-over-year spikes with `watch`, `warning`, and `critical` severity levels
- **Prediction Workspace** — Per-state, per-category 5-year forecast with model confidence score
- **Dataset Coverage** — 864 state × year records (2001–2024); synthetic rows bridge NCRB publication gaps, tagged by source

## Tech Stack

| Layer | Tools |
| - | - |
| Frontend | React 18, Vite 6, Recharts, React Leaflet, Tailwind CSS v4 |
| Backend | FastAPI, Pandas, NumPy, Pydantic v2 |
| Forecasting | Weighted ensemble - trend regression, Holt smoothing, CAGR |
| Fonts | Geist, Geist Mono (Google Fonts) |

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/          # Route handlers
│   │   ├── core/         # Config (pydantic-settings)
│   │   ├── schemas/      # Request/response models
│   │   ├── services/     # Analytics engine
│   │   └── main.py
│   ├── tests/
│   │   ├── test_api.py
│   │   └── test_service.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/   # AlertCard, CrimeMap, FilterBar, MetricCard, Panel, Skeleton
│   │   ├── hooks/        # useDebounce
│   │   ├── utils/        # format.js (labels, colors, data builders)
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── india_crime_combined_2001_2024_augmented.csv   # canonical dataset (with synthetic gap-fill)
└── india_crime_combined_2001_2024.csv             # original NCRB data
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/filters` | Available states and metrics |
| GET | `/api/v1/summary` | Metric summary cards |
| GET | `/api/v1/trends` | National history + forecast |
| GET | `/api/v1/top-states` | Top states by metric |
| GET | `/api/v1/map` | State map points with risk labels |
| GET | `/api/v1/alerts` | YoY spike alerts |
| GET | `/api/v1/data-quality` | Dataset coverage and synthetic row stats |
| POST | `/api/v1/predict` | State + category forecast |

## Local Setup

### Backend

```bash
cd backend
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API runs on `http://127.0.0.1:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

UI runs on `http://localhost:5173` (or `5174` if the port is already in use).

### Tests

```bash
cd backend
pytest tests/ -v
```

## Dataset Notes

The original NCRB data has publication gaps — no state-level breakdown for 2014–2019 in most releases, and some states are missing 2020–2024 data. The augmented CSV fills these gaps using:

- **Smooth-step interpolation** — bridges 2013→2020 within each state's value range
- **Log-linear regression** — extrapolates 2020–2024 for states with missing recent data
- **Population-derived reconstruction** — Telangana pre-2014 values derived from Andhra Pradesh (46% population share)

All synthetic rows are tagged in the `source` column. Original NCRB rows retain their source tag.
