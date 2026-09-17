"""Fallback mock weather data lookup table by city and month.

Used when Open-Meteo live API calls fail, time out, or cannot resolve the city.
"""

from typing import Any, Optional
import re

# High-fidelity seasonal weather profiles for the primary destinations
# Temperature format: [low_celsius, high_celsius]
MOCK_WEATHER_DATA: dict[str, dict[str, dict[str, Any]]] = {
    "bangkok": {
        "january": {"conditions": "Warm, sunny, dry season", "temp_range_c": [22, 32]},
        "february": {"conditions": "Hot, clear skies, low humidity", "temp_range_c": [24, 33]},
        "march": {"conditions": "Very hot and humid", "temp_range_c": [26, 35]},
        "april": {"conditions": "Extremely hot, Songkran season", "temp_range_c": [27, 36]},
        "may": {"conditions": "Hot with frequent pre-monsoon showers", "temp_range_c": [26, 34]},
        "june": {"conditions": "Tropical rains, humid and overcast", "temp_range_c": [26, 33]},
        "july": {"conditions": "Monsoon downpours, high humidity", "temp_range_c": [25, 33]},
        "august": {"conditions": "Heavy afternoon rain, cloudy", "temp_range_c": [25, 33]},
        "september": {"conditions": "Peak monsoon month, torrential rains", "temp_range_c": [25, 32]},
        "october": {"conditions": "End of monsoon, passing showers", "temp_range_c": [24, 32]},
        "november": {"conditions": "Pleasantly warm, dry, gentle breeze", "temp_range_c": [23, 31]},
        "december": {"conditions": "Mild, sunny, peak pleasant weather", "temp_range_c": [21, 31]},
    },
    "barcelona": {
        "january": {"conditions": "Crisp, sunny winter days", "temp_range_c": [6, 14]},
        "february": {"conditions": "Cool and mild, occasional breeze", "temp_range_c": [7, 15]},
        "march": {"conditions": "Mild spring sunshine, fresh mornings", "temp_range_c": [9, 17]},
        "april": {"conditions": "Pleasant spring weather, brief showers", "temp_range_c": [11, 19]},
        "may": {"conditions": "Warm, sunny Mediterranean climate", "temp_range_c": [14, 23]},
        "june": {"conditions": "Sunny, warm beach weather", "temp_range_c": [18, 26]},
        "july": {"conditions": "Hot, sunny, summer peak season", "temp_range_c": [21, 29]},
        "august": {"conditions": "Hot and humid, clear skies", "temp_range_c": [21, 29]},
        "september": {"conditions": "Warm days, comfortable evenings", "temp_range_c": [18, 26]},
        "october": {"conditions": "Mild, occasional autumn showers", "temp_range_c": [14, 22]},
        "november": {"conditions": "Cool, pleasant daytime sun", "temp_range_c": [10, 17]},
        "december": {"conditions": "Chilly, clear winter days", "temp_range_c": [7, 15]},
    },
    "reykjavik": {
        "january": {"conditions": "Freezing, snowy, dark, Northern Lights season", "temp_range_c": [-3, 2]},
        "february": {"conditions": "Cold, windy, snow flurries", "temp_range_c": [-3, 2]},
        "march": {"conditions": "Crisp winter cold, lengthening daylight", "temp_range_c": [-2, 3]},
        "april": {"conditions": "Chilly spring thaw, variable conditions", "temp_range_c": [0, 6]},
        "may": {"conditions": "Cool, brisk, blooming subarctic landscape", "temp_range_c": [4, 10]},
        "june": {"conditions": "Mild, 24-hour Midnight Sun, breezy", "temp_range_c": [7, 13]},
        "july": {"conditions": "Warmest subarctic month, pleasant daylight", "temp_range_c": [9, 15]},
        "august": {"conditions": "Mild days, cool evenings, puffin season", "temp_range_c": [8, 14]},
        "september": {"conditions": "Cool autumn, colorful moss, Northern Lights return", "temp_range_c": [5, 10]},
        "october": {"conditions": "Cold, windy, rain transitioning to sleet/snow", "temp_range_c": [1, 6]},
        "november": {"conditions": "Wintry chill, early snow, short days", "temp_range_c": [-1, 4]},
        "december": {"conditions": "Cold, festive snow, minimal daylight (4 hours)", "temp_range_c": [-3, 2]},
    },
    "tokyo": {
        "january": {"conditions": "Chilly, crisp, sunny, dry winter", "temp_range_c": [2, 10]},
        "february": {"conditions": "Cold, bright, early plum blossoms", "temp_range_c": [3, 11]},
        "march": {"conditions": "Mild spring, early cherry blossoms (sakura)", "temp_range_c": [6, 14]},
        "april": {"conditions": "Pleasant, warm spring, peak cherry blossom", "temp_range_c": [10, 19]},
        "may": {"conditions": "Comfortably warm, sunny, lush greenery", "temp_range_c": [15, 23]},
        "june": {"conditions": "Tsuyu rainy season, warm and humid", "temp_range_c": [19, 26]},
        "july": {"conditions": "Hot, humid summer, evening festivals", "temp_range_c": [23, 30]},
        "august": {"conditions": "Peak summer heat, high humidity", "temp_range_c": [24, 32]},
        "september": {"conditions": "Warm, humid, occasional typhoon rains", "temp_range_c": [20, 27]},
        "october": {"conditions": "Mild, clear autumn skies (koyo foliage)", "temp_range_c": [14, 22]},
        "november": {"conditions": "Cool, crisp, vibrant autumn foliage", "temp_range_c": [9, 17]},
        "december": {"conditions": "Cold, clear, sunny winter days with Mount Fuji visible", "temp_range_c": [4, 12]},
    },
}

MONTH_NAMES: list[str] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]

MONTH_ABBREVIATIONS: dict[str, str] = {
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "may": "may",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sep": "september",
    "sept": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
}


def normalize_month(date_or_month_str: str) -> str:
    """Extract and normalize month name from a user string (e.g. 'December', '2026-12-05', 'dec', '12')."""
    cleaned = date_or_month_str.strip().lower()

    # Check for direct month name
    for m in MONTH_NAMES:
        if m in cleaned:
            return m

    # Check for abbreviations
    for abbrev, full in MONTH_ABBREVIATIONS.items():
        if re.search(rf"\b{abbrev}\b", cleaned):
            return full

    # Check for numeric month: YYYY-MM-DD or MM/DD or single/double digit
    date_match = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", cleaned)
    if date_match:
        m_idx = int(date_match.group(2))
        if 1 <= m_idx <= 12:
            return MONTH_NAMES[m_idx - 1]

    num_match = re.search(r"\b(\d{1,2})\b", cleaned)
    if num_match:
        m_idx = int(num_match.group(1))
        if 1 <= m_idx <= 12:
            return MONTH_NAMES[m_idx - 1]

    # Default to current or generic month if unparseable
    return "december"


def get_mock_weather(city: str, date_or_month: str) -> Optional[dict[str, Any]]:
    """Look up mock weather for a city and month."""
    city_key = city.strip().lower()
    month_key = normalize_month(date_or_month)

    # Match city substring if needed (e.g., 'Tokyo, Japan' -> 'tokyo')
    matched_city = None
    for supported in MOCK_WEATHER_DATA:
        if supported in city_key:
            matched_city = supported
            break

    if not matched_city:
        return None

    data = MOCK_WEATHER_DATA[matched_city].get(month_key)
    if not data:
        return None

    return {
        "city": matched_city.title(),
        "period": month_key.title(),
        "conditions": data["conditions"],
        "temp_range_c": data["temp_range_c"],
        "source": "mock",
    }
