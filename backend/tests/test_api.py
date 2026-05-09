import unittest

from fastapi.testclient import TestClient

from app.main import app


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_filters_endpoint(self):
        response = self.client.get("/api/v1/filters")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("years", data)
        self.assertIn("crime_types", data)
        self.assertIn("states", data)

    def test_data_quality_endpoint(self):
        response = self.client.get("/api/v1/data-quality")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_rows", data)
        self.assertIn("year_gaps", data)

    def test_summary_endpoint(self):
        response = self.client.get("/api/v1/summary?crime_type=total_crimes&start_year=2020&end_year=2024")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["metric"], "total_crimes")

    def test_summary_invalid_crime_type(self):
        response = self.client.get("/api/v1/summary?crime_type=fake_crime")
        self.assertEqual(response.status_code, 400)

    def test_trends_endpoint(self):
        response = self.client.get("/api/v1/trends?crime_type=total_crimes")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("national_history", data)
        self.assertIn("national_forecast", data)
        self.assertIn("top_states", data)

    def test_map_endpoint(self):
        response = self.client.get("/api/v1/map?crime_type=total_crimes&year=2024")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("points", data)
        for point in data["points"]:
            self.assertIn("forecast_year", point)
            self.assertIn("forecast_value", point)

    def test_map_invalid_year(self):
        response = self.client.get("/api/v1/map?crime_type=total_crimes&year=1850")
        self.assertIn(response.status_code, (400, 422))

    def test_predict_endpoint(self):
        response = self.client.post(
            "/api/v1/predict",
            json={"state": "Delhi", "crime_type": "total_crimes"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["state"], "Delhi")
        self.assertEqual(len(payload["forecast"]), 5)

    def test_predict_case_insensitive_state(self):
        response = self.client.post(
            "/api/v1/predict",
            json={"state": "delhi", "crime_type": "total_crimes"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "Delhi")

    def test_predict_invalid_state(self):
        response = self.client.post(
            "/api/v1/predict",
            json={"state": "NotAStateXYZ", "crime_type": "total_crimes"},
        )
        self.assertEqual(response.status_code, 400)

    def test_alerts_endpoint(self):
        response = self.client.get("/api/v1/alerts?crime_type=total_crimes")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("alerts", data)
        for alert in data["alerts"]:
            self.assertIn(alert["severity"], {"watch", "warning", "critical"})

    def test_alerts_limit_param(self):
        response = self.client.get("/api/v1/alerts?crime_type=total_crimes&limit=3")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertLessEqual(len(data["alerts"]), 3)


if __name__ == "__main__":
    unittest.main()
