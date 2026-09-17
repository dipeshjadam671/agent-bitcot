"""Unit tests for Weather Forecast tool, Open-Meteo live integration, and mock fallback."""

import pytest
import requests
from unittest.mock import patch, MagicMock

from tools.weather import get_weather_forecast
from data.weather_mock import get_mock_weather, normalize_month


def test_normalize_month():
    """Verify month name extraction and normalization."""
    assert normalize_month("December") == "december"
    assert normalize_month("dec") == "december"
    assert normalize_month("2026-07-15") == "july"
    assert normalize_month("in August") == "august"
    assert normalize_month("04") == "april"


def test_mock_weather_lookup():
    """Verify fallback mock data lookup table."""
    tokyo_dec = get_mock_weather("Tokyo", "December")
    assert tokyo_dec is not None
    assert tokyo_dec["city"] == "Tokyo"
    assert tokyo_dec["period"] == "December"
    assert tokyo_dec["temp_range_c"] == [4, 12]
    assert tokyo_dec["source"] == "mock"

    bangkok_apr = get_mock_weather("Bangkok", "April")
    assert bangkok_apr is not None
    assert bangkok_apr["temp_range_c"] == [27, 36]

    reykjavik_jan = get_mock_weather("Reykjavik", "January")
    assert reykjavik_jan is not None
    assert reykjavik_jan["temp_range_c"] == [-3, 2]


def test_weather_empty_city_validation():
    """Verify input validation for missing city argument."""
    res = get_weather_forecast.invoke({"city": "", "date_or_month": "December"})
    assert res["source"] == "mock"
    assert "Error: City name must be provided" in res["conditions"]


@patch("tools.weather.requests.get")
def test_weather_live_success(mock_get):
    """Verify successful live Open-Meteo API resolution and forecast parsing."""
    # 1st call: geocoding
    geo_resp = MagicMock()
    geo_resp.json.return_value = {
        "results": [
            {"latitude": 35.6895, "longitude": 139.6917, "name": "Tokyo"}
        ]
    }
    geo_resp.raise_for_status.return_value = None

    # 2nd call: forecast
    forecast_resp = MagicMock()
    forecast_resp.json.return_value = {
        "daily": {
            "temperature_2m_max": [12.0, 14.0],
            "temperature_2m_min": [4.0, 6.0],
            "weather_code": [0, 0],
        }
    }
    forecast_resp.raise_for_status.return_value = None

    mock_get.side_effect = [geo_resp, forecast_resp]

    result = get_weather_forecast.invoke({"city": "Tokyo", "date_or_month": "December"})
    assert result["source"] == "live"
    assert result["city"] == "Tokyo"
    assert result["temp_range_c"] == [5, 13]
    assert result["conditions"] == "Clear sky"


@patch("tools.weather.requests.get")
def test_weather_timeout_fallback(mock_get):
    """Verify that a request timeout gracefully falls back to mock weather."""
    mock_get.side_effect = requests.Timeout("Network socket timeout")

    result = get_weather_forecast.invoke({"city": "Tokyo", "date_or_month": "December"})
    assert result["source"] == "mock"
    assert result["city"] == "Tokyo"
    assert result["temp_range_c"] == [4, 12]
    assert "clear" in result["conditions"].lower()


@patch("tools.weather.requests.get")
def test_weather_http_error_fallback(mock_get):
    """Verify that an HTTP 500 error gracefully falls back to mock weather."""
    mock_get.side_effect = requests.HTTPError("500 Server Error")

    result = get_weather_forecast.invoke({"city": "Barcelona", "date_or_month": "May"})
    assert result["source"] == "mock"
    assert result["city"] == "Barcelona"
    assert result["temp_range_c"] == [14, 23]


@patch("tools.weather.requests.get")
def test_weather_unsupported_destination(mock_get):
    """Verify handling when city is not supported in live geocoding or mock table."""
    mock_get.side_effect = requests.RequestException("Lookup failed")

    result = get_weather_forecast.invoke({"city": "Atlantis", "date_or_month": "August"})
    assert result["source"] == "mock"
    assert "supported cities" in result["conditions"].lower()
    assert result["temp_range_c"] == [0, 0]
