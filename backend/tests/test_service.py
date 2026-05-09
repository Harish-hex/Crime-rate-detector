import unittest
from pathlib import Path

from app.services.crime_analytics import CrimeAnalyticsService, DataLoadError

DATASET_PATH = Path(__file__).resolve().parents[2] / "india_crime_combined_2001_2024.csv"


class CrimeAnalyticsServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = CrimeAnalyticsService(DATASET_PATH)

    # --- Filters ---

    def test_filters_expose_years_metrics_and_states(self):
        filters = self.service.get_filters()
        self.assertIn(2001, filters["years"])
        self.assertIn(2024, filters["years"])
        self.assertIn("total_crimes", filters["crime_types"])
        self.assertIn("Maharashtra", filters["states"])

    def test_filters_states_are_stripped(self):
        for state in self.service.states:
            self.assertEqual(state, state.strip(), f"State '{state}' has whitespace")

    # --- State resolution ---

    def test_state_resolution_case_insensitive(self):
        result = self.service.get_forecast("delhi", "total_crimes")
        self.assertEqual(result["state"], "Delhi")

    def test_state_resolution_whitespace(self):
        result = self.service.get_forecast("  Delhi  ", "total_crimes")
        self.assertEqual(result["state"], "Delhi")

    def test_state_resolution_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.service.get_forecast("NotAState123", "total_crimes")

    # --- Summary ---

    def test_summary_returns_expected_keys(self):
        summary = self.service.get_summary("total_crimes", 2020, 2024)
        self.assertEqual(summary["metric"], "total_crimes")
        self.assertEqual(summary["start_year"], 2020)
        self.assertEqual(summary["end_year"], 2024)
        self.assertGreater(summary["total_value"], 0)

    def test_summary_invalid_range_raises(self):
        with self.assertRaises(ValueError):
            self.service.get_summary("total_crimes", 2024, 2020)

    def test_summary_invalid_crime_type_raises(self):
        with self.assertRaises(ValueError):
            self.service.get_summary("fake_crime", None, None)

    # --- Forecast ---

    def test_forecast_contains_future_years(self):
        forecast = self.service.get_forecast("Maharashtra", "total_crimes")
        years = [item["year"] for item in forecast["forecast"]]
        last_data_year = self.service.years[-1]
        expected = list(range(last_data_year + 1, last_data_year + 6))
        self.assertEqual(years, expected)
        self.assertGreaterEqual(forecast["confidence"], 45)
        self.assertLessEqual(forecast["confidence"], 93)

    def test_forecast_values_non_negative(self):
        forecast = self.service.get_forecast("Maharashtra", "total_crimes")
        for point in forecast["forecast"]:
            self.assertGreaterEqual(point["value"], 0, f"Negative forecast at year {point['year']}")

    def test_forecast_invalid_future_year_raises(self):
        last_year = self.service.years[-1]
        with self.assertRaises(ValueError):
            self.service.get_forecast("Delhi", "total_crimes", years=[last_year - 1])

    def test_forecast_prefers_recent_consistent_window(self):
        forecast = self.service.get_forecast("Delhi", "total_crimes")
        values = [item["value"] for item in forecast["forecast"]]
        self.assertLess(max(values), 5000)

    def test_forecast_all_crime_types(self):
        for crime_type in self.service.crime_types:
            result = self.service.get_forecast("Maharashtra", crime_type)
            self.assertEqual(len(result["forecast"]), 5)

    # --- Alerts ---

    def test_alerts_only_states_with_both_years(self):
        alerts = self.service.get_alerts("total_crimes")
        last_year = self.service.years[-1]
        prev_year = sorted(self.service.years)[-2]
        states_with_latest = set(
            self.service.dataframe[self.service.dataframe["year"] == last_year]["state"].unique()
        )
        states_with_prev = set(
            self.service.dataframe[self.service.dataframe["year"] == prev_year]["state"].unique()
        )
        valid = states_with_latest & states_with_prev
        for alert in alerts["alerts"]:
            self.assertIn(alert["state"], valid, f"{alert['state']} should not be in alerts")

    def test_alerts_severity_values(self):
        alerts = self.service.get_alerts("total_crimes")
        for alert in alerts["alerts"]:
            self.assertIn(alert["severity"], {"watch", "warning", "critical"})

    # --- Map ---

    def test_map_invalid_year_raises(self):
        with self.assertRaises(ValueError):
            self.service.get_map_points("total_crimes", year=1850, start_year=None, end_year=None)

    def test_map_points_have_required_fields(self):
        result = self.service.get_map_points("total_crimes", year=None, start_year=2020, end_year=2024)
        for point in result["points"]:
            self.assertIn("state", point)
            self.assertIn("latitude", point)
            self.assertIn("longitude", point)
            self.assertIn("value", point)
            self.assertIn("risk", point)
            self.assertIn("forecast_year", point)
            self.assertIn("forecast_value", point)
            self.assertGreaterEqual(point["value"], 0)

    # --- Data quality ---

    def test_data_quality_returns_expected_structure(self):
        quality = self.service.get_data_quality()
        self.assertIn("total_rows", quality)
        self.assertIn("year_range", quality)
        self.assertIn("year_gaps", quality)
        self.assertIn("states_count", quality)
        self.assertGreater(quality["total_rows"], 0)

    # --- DataLoadError ---

    def test_missing_dataset_raises_data_load_error(self):
        service = CrimeAnalyticsService(Path("/nonexistent/path.csv"))
        with self.assertRaises(DataLoadError):
            _ = service.dataframe


if __name__ == "__main__":
    unittest.main()
