from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import requests


try:
    from config import settings
except Exception:
    settings = None


OPEN_METEO_FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


# ============================================================
# WEATHER CODE MAPPING
# ============================================================

WEATHER_CODES: Dict[int, Dict[str, str]] = {

    0: {
        "condition": "Clear sky",
        "icon": "sun",
    },

    1: {
        "condition": "Mainly clear",
        "icon": "sun",
    },

    2: {
        "condition": "Partly cloudy",
        "icon": "cloud",
    },

    3: {
        "condition": "Overcast",
        "icon": "cloud",
    },

    45: {
        "condition": "Fog",
        "icon": "fog",
    },

    48: {
        "condition": "Depositing rime fog",
        "icon": "fog",
    },

    51: {
        "condition": "Light drizzle",
        "icon": "rain",
    },

    53: {
        "condition": "Moderate drizzle",
        "icon": "rain",
    },

    55: {
        "condition": "Dense drizzle",
        "icon": "rain",
    },

    56: {
        "condition": "Light freezing drizzle",
        "icon": "rain",
    },

    57: {
        "condition": "Dense freezing drizzle",
        "icon": "rain",
    },

    61: {
        "condition": "Slight rain",
        "icon": "rain",
    },

    63: {
        "condition": "Moderate rain",
        "icon": "rain",
    },

    65: {
        "condition": "Heavy rain",
        "icon": "rain",
    },

    66: {
        "condition": "Light freezing rain",
        "icon": "rain",
    },

    67: {
        "condition": "Heavy freezing rain",
        "icon": "rain",
    },

    71: {
        "condition": "Slight snowfall",
        "icon": "snow",
    },

    73: {
        "condition": "Moderate snowfall",
        "icon": "snow",
    },

    75: {
        "condition": "Heavy snowfall",
        "icon": "snow",
    },

    77: {
        "condition": "Snow grains",
        "icon": "snow",
    },

    80: {
        "condition": "Slight rain showers",
        "icon": "rain",
    },

    81: {
        "condition": "Moderate rain showers",
        "icon": "rain",
    },

    82: {
        "condition": "Violent rain showers",
        "icon": "rain",
    },

    85: {
        "condition": "Slight snow showers",
        "icon": "snow",
    },

    86: {
        "condition": "Heavy snow showers",
        "icon": "snow",
    },

    95: {
        "condition": "Thunderstorm",
        "icon": "storm",
    },

    96: {
        "condition": "Thunderstorm with slight hail",
        "icon": "storm",
    },

    99: {
        "condition": "Thunderstorm with heavy hail",
        "icon": "storm",
    },
}


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        if value is None:
            return default

        return float(value)

    except (TypeError, ValueError):

        return default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:

        if value is None:
            return default

        return int(
            round(
                float(value)
            )
        )

    except (TypeError, ValueError):

        return default


def _clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    return max(
        minimum,
        min(
            value,
            maximum,
        ),
    )


def _weather_info(
    code: Any,
) -> Dict[str, str]:

    code_int = _safe_int(
        code,
        0,
    )

    return WEATHER_CODES.get(
        code_int,
        {
            "condition": "Unknown",
            "icon": "cloud",
        },
    )


def _format_time(
    time_string: Optional[str],
) -> str:

    if not time_string:
        return "--:--"

    try:

        if "T" in time_string:

            return time_string.split(
                "T",
                1,
            )[1][:5]

        return time_string[:5]

    except Exception:

        return "--:--"


def _format_day(
    date_string: Optional[str],
) -> str:

    if not date_string:
        return ""

    try:

        date_obj = datetime.strptime(
            date_string,
            "%Y-%m-%d",
        )

        return date_obj.strftime(
            "%a"
        )

    except Exception:

        return ""


# ============================================================
# RAIN ICON
# ============================================================

def _rain_icon(
    weather_code: int,
    rain_probability: int,
    rain_mm: float,
) -> str:

    info = _weather_info(
        weather_code
    )

    if rain_mm >= 10:
        return "rain"

    if (
        rain_probability >= 70
        and rain_mm >= 2
    ):
        return "rain"

    if rain_probability >= 60:
        return "rain"

    if info["icon"] == "rain":
        return "rain"

    if info["icon"] == "storm":
        return "storm"

    if info["icon"] == "snow":
        return "snow"

    return info["icon"]


# ============================================================
# ALERT LEVEL
# ============================================================

def _alert_level(
    rain_probability: int,
    rain_mm: float,
    weather_code: int,
) -> str:

    if weather_code in (
        95,
        96,
        99,
    ):
        return "High"

    if rain_mm >= 20:
        return "High"

    if (
        rain_probability >= 80
        and rain_mm >= 5
    ):
        return "High"

    if (
        rain_probability >= 60
        or rain_mm >= 5
    ):
        return "Medium"

    return "Low"


# ============================================================
# REVERSE GEOCODING
# ============================================================

def _get_location_name(
    latitude: float,
    longitude: float,
    fallback_location: Optional[str] = None,
) -> str:

    try:

        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": f"{latitude},{longitude}",
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=10,
        )

        # ----------------------------------------------------
        # Open-Meteo's normal geocoding endpoint is primarily
        # name-based. If it cannot resolve the coordinates,
        # don't invent a city.
        # ----------------------------------------------------

        if response.ok:

            data = response.json()

            results = (
                data.get("results")
                or []
            )

            if results:

                result = results[0]

                name = (
                    result.get("name")
                    or result.get("city")
                    or result.get("town")
                    or result.get("village")
                )

                state = result.get(
                    "admin1"
                )

                country = result.get(
                    "country_code"
                )

                if name:

                    parts = [name]

                    if (
                        state
                        and state != name
                    ):
                        parts.append(
                            state
                        )

                    if (
                        country
                        and country != "IN"
                    ):
                        parts.append(
                            country
                        )

                    return ", ".join(
                        parts
                    )

    except Exception as exc:

        print(
            "[WEATHER] "
            "Location lookup failed:",
            exc,
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We NEVER use Mangaluru/Davangere/etc. as a fake
    # location.
    #
    # If reverse geocoding fails, report coordinates.
    # --------------------------------------------------------

    return (
        fallback_location
        or
        f"{latitude:.5f}, "
        f"{longitude:.5f}"
    )


# ============================================================
# FETCH OPEN-METEO DATA
# ============================================================

def _fetch_weather(
    latitude: float,
    longitude: float,
) -> Dict[str, Any]:

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "rain",
                "weather_code",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),

        "daily": ",".join(
            [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "apparent_temperature_max",
                "apparent_temperature_min",
                "precipitation_sum",
                "rain_sum",
                "precipitation_probability_max",
                "wind_speed_10m_max",
                "wind_direction_10m_dominant",
                "sunrise",
                "sunset",
            ]
        ),

        "hourly": ",".join(
            [
                "precipitation_probability",
                "precipitation",
                "rain",
                "weather_code",
            ]
        ),

        "timezone": "auto",

        "forecast_days": 7,

        "temperature_unit": "celsius",

        "wind_speed_unit": "ms",

        "precipitation_unit": "mm",
    }

    print(
        "\n=========================================="
    )

    print(
        "FETCHING WEATHER FROM OPEN-METEO"
    )

    print(
        "Latitude :",
        latitude,
    )

    print(
        "Longitude:",
        longitude,
    )

    print(
        "=========================================="
    )

    response = requests.get(
        OPEN_METEO_FORECAST_URL,
        params=params,
        timeout=20,
    )

    print(
        "Open-Meteo status:",
        response.status_code,
    )

    if not response.ok:

        raise Exception(
            "Open-Meteo weather failed: "
            f"{response.status_code} "
            f"{response.text}"
        )

    return response.json()


# ============================================================
# CURRENT HOURLY RAIN PROBABILITY
# ============================================================

def _get_current_hour_rain_probability(
    data: Dict[str, Any],
) -> int:

    current = (
        data.get("current")
        or {}
    )

    hourly = (
        data.get("hourly")
        or {}
    )

    current_time = current.get(
        "time"
    )

    hourly_times = (
        hourly.get("time")
        or []
    )

    probabilities = (
        hourly.get(
            "precipitation_probability"
        )
        or []
    )

    if (
        not hourly_times
        or not probabilities
    ):
        return 0

    if not current_time:
        return _safe_int(
            probabilities[0],
            0,
        )

    try:

        current_dt = datetime.fromisoformat(
            current_time.replace(
                "Z",
                "+00:00",
            )
        )

        best_index = 0

        best_difference = None

        for index, time_string in enumerate(
            hourly_times
        ):

            if index >= len(
                probabilities
            ):
                break

            try:

                forecast_dt = (
                    datetime.fromisoformat(
                        time_string.replace(
                            "Z",
                            "+00:00",
                        )
                    )
                )

                difference = abs(
                    (
                        forecast_dt
                        - current_dt
                    ).total_seconds()
                )

                if (
                    best_difference is None
                    or difference
                    < best_difference
                ):

                    best_difference = (
                        difference
                    )

                    best_index = index

            except Exception:

                continue

        return _safe_int(
            probabilities[
                best_index
            ],
            0,
        )

    except Exception:

        return _safe_int(
            probabilities[0],
            0,
        )


# ============================================================
# BUILD CURRENT WEATHER
# ============================================================

def _build_current(
    data: Dict[str, Any],
    location_name: str,
) -> Dict[str, Any]:

    current = (
        data.get("current")
        or {}
    )

    daily = (
        data.get("daily")
        or {}
    )

    temperature = _safe_float(
        current.get(
            "temperature_2m"
        )
    )

    feels_like = _safe_float(
        current.get(
            "apparent_temperature"
        )
    )

    humidity = _safe_int(
        current.get(
            "relative_humidity_2m"
        )
    )

    rain_probability = (
        _get_current_hour_rain_probability(
            data
        )
    )

    weather_code = _safe_int(
        current.get(
            "weather_code"
        ),
        0,
    )

    weather_info = _weather_info(
        weather_code
    )

    wind_speed = _safe_float(
        current.get(
            "wind_speed_10m"
        )
    )

    wind_direction = _safe_int(
        current.get(
            "wind_direction_10m"
        )
    )

    sunrise_values = (
        daily.get("sunrise")
        or []
    )

    sunset_values = (
        daily.get("sunset")
        or []
    )

    sunrise = (
        _format_time(
            sunrise_values[0]
        )
        if sunrise_values
        else "--:--"
    )

    sunset = (
        _format_time(
            sunset_values[0]
        )
        if sunset_values
        else "--:--"
    )

    return {

        "location": location_name,

        "temperature": round(
            temperature,
            1,
        ),

        "condition":
            weather_info[
                "condition"
            ],

        "feels_like": round(
            feels_like,
            1,
        ),

        "humidity": humidity,

        "rain_probability":
            _clamp(
                rain_probability,
                0,
                100,
            ),

        "wind_speed":
            f"{wind_speed:.2f} m/s",

        "wind_direction":
            f"{wind_direction}°",

        "sunrise": sunrise,

        "sunset": sunset,
    }


# ============================================================
# BUILD 7-DAY FORECAST
# ============================================================

def _build_forecast(
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:

    daily = (
        data.get("daily")
        or {}
    )

    dates = (
        daily.get("time")
        or []
    )

    weather_codes = (
        daily.get("weather_code")
        or []
    )

    max_temperatures = (
        daily.get(
            "temperature_2m_max"
        )
        or []
    )

    min_temperatures = (
        daily.get(
            "temperature_2m_min"
        )
        or []
    )

    precipitation_probability = (
        daily.get(
            "precipitation_probability_max"
        )
        or []
    )

    precipitation_sum = (
        daily.get(
            "precipitation_sum"
        )
        or []
    )

    rain_sum = (
        daily.get(
            "rain_sum"
        )
        or []
    )

    forecast: List[
        Dict[str, Any]
    ] = []

    number_of_days = min(
        7,
        len(dates),
    )

    for index in range(
        number_of_days
    ):

        date_string = dates[index]

        weather_code = (
            _safe_int(
                weather_codes[index]
            )
            if index
            < len(weather_codes)
            else 0
        )

        max_temp = (
            _safe_float(
                max_temperatures[
                    index
                ]
            )
            if index
            < len(max_temperatures)
            else 0
        )

        min_temp = (
            _safe_float(
                min_temperatures[
                    index
                ]
            )
            if index
            < len(min_temperatures)
            else 0
        )

        rain_probability = (
            _safe_int(
                precipitation_probability[
                    index
                ]
            )
            if index
            < len(
                precipitation_probability
            )
            else 0
        )

        total_precipitation = (
            _safe_float(
                precipitation_sum[
                    index
                ]
            )
            if index
            < len(
                precipitation_sum
            )
            else 0
        )

        rain_amount = (
            _safe_float(
                rain_sum[index]
            )
            if index
            < len(rain_sum)
            else 0
        )

        rainfall_mm = max(
            rain_amount,
            total_precipitation,
        )

        weather_info = _weather_info(
            weather_code
        )

        icon = _rain_icon(
            weather_code,
            rain_probability,
            rainfall_mm,
        )

        forecast.append(
            {

                "day":
                    _format_day(
                        date_string
                    ),

                "temp":
                    round(
                        max_temp,
                        1,
                    ),

                "min_temp":
                    round(
                        min_temp,
                        1,
                    ),

                "max_temp":
                    round(
                        max_temp,
                        1,
                    ),

                "rain":
                    int(
                        _clamp(
                            rain_probability,
                            0,
                            100,
                        )
                    ),

                "rain_mm":
                    round(
                        rainfall_mm,
                        1,
                    ),

                "precipitation_mm":
                    round(
                        total_precipitation,
                        1,
                    ),

                "icon": icon,

                "condition":
                    weather_info[
                        "condition"
                    ],
            }
        )

    return forecast


# ============================================================
# BUILD ALERTS
# ============================================================

def _build_alerts(
    forecast: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    alerts: List[
        Dict[str, Any]
    ] = []

    for day in forecast:

        day_name = day.get(
            "day",
            "",
        )

        rain_probability = (
            _safe_int(
                day.get("rain"),
                0,
            )
        )

        rain_mm = (
            _safe_float(
                day.get("rain_mm"),
                0,
            )
        )

        if rain_mm >= 20:

            alerts.append(
                {

                    "level": "High",

                    "title":
                        f"Heavy Rain Expected on "
                        f"{day_name}",

                    "body":
                        f"Approximately "
                        f"{rain_mm:.1f} mm of rain "
                        f"is forecast with a "
                        f"{rain_probability}% "
                        f"precipitation probability.",
                }
            )

        elif (
            rain_probability >= 80
            and rain_mm >= 5
        ):

            alerts.append(
                {

                    "level": "High",

                    "title":
                        f"High Rain Chance on "
                        f"{day_name}",

                    "body":
                        f"{rain_probability}% chance "
                        f"of precipitation with "
                        f"approximately "
                        f"{rain_mm:.1f} mm expected.",
                }
            )

        elif (
            rain_probability >= 60
            and rain_mm >= 2
        ):

            alerts.append(
                {

                    "level": "Medium",

                    "title":
                        f"Rain Possible on "
                        f"{day_name}",

                    "body":
                        f"{rain_probability}% chance "
                        f"of precipitation with "
                        f"approximately "
                        f"{rain_mm:.1f} mm expected.",
                }
            )

    return alerts


# ============================================================
# MAIN WEATHER FUNCTION
# ============================================================

def get_weather(
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    location: Optional[str] = None,
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # NO DEFAULT CITY
    # --------------------------------------------------------

    if lat is None or lng is None:

        raise ValueError(
            "GPS coordinates are required. "
            "The browser/device must provide "
            "latitude and longitude."
        )

    latitude = _safe_float(
        lat,
        float("nan"),
    )

    longitude = _safe_float(
        lng,
        float("nan"),
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if (
        latitude != latitude
        or longitude != longitude
    ):
        raise ValueError(
            "Invalid GPS coordinates."
        )

    if not (
        -90
        <= latitude
        <= 90
    ):
        raise ValueError(
            "Invalid latitude."
        )

    if not (
        -180
        <= longitude
        <= 180
    ):
        raise ValueError(
            "Invalid longitude."
        )

    # --------------------------------------------------------
    # LOG EXACT GPS
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "WEATHER REQUEST"
    )

    print(
        "=========================================="
    )

    print(
        "Latitude :",
        latitude,
    )

    print(
        "Longitude:",
        longitude,
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # LOCATION NAME
    # --------------------------------------------------------
    #
    # Coordinates determine weather.
    #
    # location is ONLY an optional display name.
    #
    # --------------------------------------------------------

    location_name = _get_location_name(
        latitude,
        longitude,
        location,
    )

    print(
        "Resolved location:",
        location_name,
    )

    # --------------------------------------------------------
    # OPEN-METEO
    # --------------------------------------------------------

    data = _fetch_weather(
        latitude,
        longitude,
    )

    # --------------------------------------------------------
    # BUILD RESPONSE
    # --------------------------------------------------------

    current = _build_current(
        data,
        location_name,
    )

    forecast = _build_forecast(
        data,
    )

    alerts = _build_alerts(
        forecast,
    )

    result = {

        "current": current,

        "forecast": forecast,

        "alerts": alerts,
    }

    print(
        "\n========== WEATHER RESULT =========="
    )

    print(
        "Location:",
        current["location"],
    )

    print(
        "Temperature:",
        current["temperature"],
    )

    print(
        "Condition:",
        current["condition"],
    )

    print(
        "Humidity:",
        current["humidity"],
    )

    print(
        "Rain Chance:",
        current[
            "rain_probability"
        ],
    )

    print(
        "Forecast days:",
        len(forecast),
    )

    print(
        "====================================\n"
    )

    return result


# ============================================================
# LIVE WEATHER
# ============================================================

def get_live_weather(
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    location: Optional[str] = None,
) -> Dict[str, Any]:

    return get_weather(
        lat=lat,
        lng=lng,
        location=location,
    )


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================

fetch_weather = get_weather

get_current_weather = (
    get_live_weather
)