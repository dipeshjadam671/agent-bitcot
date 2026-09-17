"""Weather Forecast Tool with Open-Meteo live API and graceful mock fallback.

Fetches geocoding and forecast data from Open-Meteo (keyless, free) and falls back
to data/weather_mock.py upon any timeout, network failure, or resolution error.
"""

from typing import Any, Optional
import logging
import requests
from langchain_core.tools import tool

from config.settings import settings
from data.weather_mock import get_mock_weather, normalize_month

logger = logging.getLogger("tripmate.tools.weather")

# WMO Weather interpretation codes (WW)
WMO_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _geocode_city(city: str, timeout: int) -> Optional[tuple[float, float, str]]:
    """Geocode city name to (latitude, longitude, resolved_name) using Open-Meteo."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city.strip(), "count": 1, "language": "en", "format": "json"}
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results")
    if not results:
        return None

    top = results[0]
    lat = float(top["latitude"])
    lon = float(top["longitude"])
    name = top.get("name", city.title())
    return lat, lon, name


def _fetch_live_weather(
    city: str, date_or_month: str, timeout: int
) -> Optional[dict[str, Any]]:
    """Fetch live weather or seasonal profile from Open-Meteo."""
    geo = _geocode_city(city, timeout)
    if not geo:
        logger.warning("Geocoding failed for city: '%s'", city)
        return None

    lat, lon, resolved_name = geo

    # Call Open-Meteo forecast API
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "weather_code"],
        "timezone": "auto",
    }
    resp = requests.get(forecast_url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    weather_codes = daily.get("weather_code", [])

    if not max_temps or not min_temps:
        return None

    # Calculate average high and low across the available forecast window
    avg_max = round(sum(max_temps) / len(max_temps))
    avg_min = round(sum(min_temps) / len(min_temps))

    # Determine dominant weather condition from WMO codes
    dominant_code = max(set(weather_codes), key=weather_codes.count) if weather_codes else 0
    condition_desc = WMO_CODE_MAP.get(dominant_code, "Pleasant conditions")

    return {
        "city": resolved_name,
        "period": date_or_month.title(),
        "conditions": condition_desc,
        "temp_range_c": [avg_min, avg_max],
        "source": "live",
    }


@tool
def get_weather_forecast(city: str, date_or_month: str) -> dict[str, Any]:
    """Retrieve weather forecast and seasonal climate data for a city and date/month.

    Use this tool to obtain temperatures, weather conditions, rainfall, and climate
    patterns for a destination during a specific month (e.g. 'December', 'July') or date.

    Args:
        city: The destination city (e.g., 'Tokyo', 'Bangkok', 'Barcelona', 'Reykjavik').
        date_or_month: Target month (e.g., 'December') or specific date/season.

    Returns:
        A dictionary with city, period, conditions, temp_range_c [low, high], and source ('live' or 'mock').
    """
    if not city or not city.strip():
        return {
            "city": "Unknown",
            "period": date_or_month or "Unknown",
            "conditions": "Error: City name must be provided.",
            "temp_range_c": [0, 0],
            "source": "mock",
            "error": "Missing city argument",
        }

    clean_city = city.strip()
    clean_period = date_or_month.strip() if date_or_month else "Current"
    timeout = settings.OPEN_METEO_TIMEOUT_SECONDS

    logger.info("Executing get_weather_forecast for city='%s', period='%s'", clean_city, clean_period)

    # 1. Attempt live API call with Open-Meteo
    try:
        live_result = _fetch_live_weather(clean_city, clean_period, timeout)
        if live_result:
            logger.info("Live weather retrieved successfully for '%s'.", clean_city)
            return live_result
        else:
            logger.warning("Live weather returned empty data for '%s'. Triggering mock fallback.", clean_city)
    except requests.Timeout:
        logger.warning(
            "Open-Meteo live API call timed out after %ds for city '%s'. Falling back to mock data.",
            timeout,
            clean_city,
        )
    except requests.RequestException as req_err:
        logger.warning(
            "Open-Meteo request error (%s) for city '%s'. Falling back to mock data.",
            req_err,
            clean_city,
        )
    except Exception as exc:
        logger.warning(
            "Unexpected error fetching live weather (%s) for city '%s'. Falling back to mock data.",
            exc,
            clean_city,
        )

    # 2. Graceful fallback to mock data table
    mock_data = get_mock_weather(clean_city, clean_period)
    if mock_data:
        logger.info("Retrieved mock weather fallback for city='%s', month='%s'.", clean_city, clean_period)
        return mock_data

    # 3. If city is unsupported or not in mock table, return safe schema
    logger.warning("City '%s' is not in mock weather table.", clean_city)
    supported = ", ".join(settings.SUPPORTED_CITIES)
    return {
        "city": clean_city.title(),
        "period": clean_period.title(),
        "conditions": f"Weather data currently available only for supported cities: {supported}.",
        "temp_range_c": [0, 0],
        "source": "mock",
        "error": f"Unsupported destination '{clean_city}'",
    }
