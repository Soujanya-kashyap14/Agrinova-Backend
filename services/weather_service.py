from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


try:
    from config import settings
except Exception:
    settings = None


OPEN_METEO_FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

OPENWEATHER_CURRENT_URL = (
    "https://api.openweathermap.org/data/2.5/weather"
)

OPENWEATHER_FORECAST_URL = (
    "https://api.openweathermap.org/data/2.5/forecast"
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
# FORWARD GEOCODING (name -> coordinates)
# ============================================================

def _geocode_location(
    location: str,
) -> Optional[Dict[str, Any]]:

    # This app is exclusively for Indian farmers, but the geocoding
    # API has no concept of that and will happily match a common
    # Indian place name to an obscure town anywhere else in the
    # world - e.g. "Mangalore" resolved to a village of 421 people
    # in Tasmania, Australia, ahead of Mangaluru, India (population
    # ~500,000), because the API ranks by relevance/exact-match, not
    # population, and doesn't know which country actually matters
    # here. Bias to India first; only fall back to an unrestricted
    # worldwide search if nothing in India matches at all.
    for country_code in ("IN", None):

        try:

            params = {
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json",
            }

            if country_code:
                params["countryCode"] = country_code

            response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params=params,
                timeout=10,
            )

            if not response.ok:
                continue

            data = response.json()

            results = (
                data.get("results")
                or []
            )

            if results:
                break

        except Exception:
            continue

    else:
        return None

    try:

        result = results[0]

        latitude = result.get("latitude")
        longitude = result.get("longitude")

        if latitude is None or longitude is None:
            return None

        name = (
            result.get("name")
            or result.get("city")
            or result.get("town")
            or result.get("village")
            or location
        )

        state = result.get("admin1")
        country = result.get("country_code")

        parts = [name]

        if state and state != name:
            parts.append(state)

        if country and country != "IN":
            parts.append(country)

        return {
            "latitude": _safe_float(latitude),
            "longitude": _safe_float(longitude),
            "name": ", ".join(parts),
        }

    except Exception as exc:

        print(
            "[WEATHER] "
            "Forward geocoding failed:",
            exc,
        )

        return None


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
# OPENWEATHER CURRENT CONDITIONS
# ============================================================
#
# OpenWeatherMap's current-weather endpoint is backed by real
# weather station observations where available ("base":
# "stations"), which tends to track a farmer's own thermometer
# more closely than Open-Meteo's pure grid-model interpolation.
#
# It is used for "right now" fields only. Rain probability and
# the 7-day forecast still come from Open-Meteo, since
# OpenWeatherMap's free current-weather endpoint has no
# precipitation-probability field.
# ============================================================

def _fetch_openweather_current(
    latitude: float,
    longitude: float,
) -> Optional[Dict[str, Any]]:

    api_key = getattr(
        settings,
        "openweather_api_key",
        "",
    ) if settings else ""

    if not api_key:
        return None

    try:

        response = requests.get(
            OPENWEATHER_CURRENT_URL,
            params={
                "lat": latitude,
                "lon": longitude,
                "appid": api_key,
                "units": "metric",
            },
            timeout=10,
        )

        if not response.ok:

            print(
                "[WEATHER] OpenWeather request failed:",
                response.status_code,
                response.text,
            )

            return None

        return response.json()

    except Exception as exc:

        print(
            "[WEATHER] OpenWeather request error:",
            exc,
        )

        return None


def _build_current_from_openweather(
    ow_data: Dict[str, Any],
    rain_probability: int,
    location_name: str,
) -> Optional[Dict[str, Any]]:

    main = ow_data.get("main") or {}
    wind = ow_data.get("wind") or {}
    sys_info = ow_data.get("sys") or {}
    weather_list = ow_data.get("weather") or []

    if "temp" not in main:
        return None

    timezone_offset = _safe_int(
        ow_data.get("timezone"),
        0,
    )

    def _format_unix(value: Any) -> str:

        try:

            timestamp = int(value) + timezone_offset

            return datetime.fromtimestamp(
                timestamp,
                tz=timezone.utc,
            ).strftime("%H:%M")

        except Exception:

            return "--:--"

    condition = (
        weather_list[0].get("description", "").title()
        if weather_list
        else "Unknown"
    )

    ow_name = str(ow_data.get("name") or "").strip()
    country = (
        (ow_data.get("sys") or {}).get("country") or ""
    ).strip()

    resolved_location = location_name

    if ow_name:

        resolved_location = (
            f"{ow_name}, {country}"
            if country and country != "IN"
            else ow_name
        )

    return {

        "location": resolved_location,

        "temperature": round(
            _safe_float(main.get("temp")),
            1,
        ),

        "condition": condition or "Unknown",

        "feels_like": round(
            _safe_float(main.get("feels_like")),
            1,
        ),

        "humidity": _safe_int(main.get("humidity")),

        "rain_probability": _clamp(
            rain_probability,
            0,
            100,
        ),

        "wind_speed":
            f"{_safe_float(wind.get('speed')):.2f} m/s",

        "wind_direction":
            f"{_safe_int(wind.get('deg'))}°",

        "sunrise": _format_unix(sys_info.get("sunrise")),

        "sunset": _format_unix(sys_info.get("sunset")),
    }


# ============================================================
# OPENWEATHER FORECAST (FOR RAIN-PROBABILITY BLENDING)
# ============================================================
#
# Open-Meteo's daily precipitation_probability_max is a single
# model's estimate. OpenWeatherMap's 3-hourly forecast is
# produced by an independent model/pipeline. Averaging two
# independent forecasts (a small multi-model ensemble) is a
# standard way to reduce single-model bias and generally
# improves rain-probability calibration versus either model
# alone. OpenWeatherMap's free tier only covers ~5 days, so
# days 6-7 keep using Open-Meteo alone.
# ============================================================

def _fetch_openweather_forecast(
    latitude: float,
    longitude: float,
) -> Optional[Dict[str, Any]]:

    api_key = getattr(
        settings,
        "openweather_api_key",
        "",
    ) if settings else ""

    if not api_key:
        return None

    try:

        response = requests.get(
            OPENWEATHER_FORECAST_URL,
            params={
                "lat": latitude,
                "lon": longitude,
                "appid": api_key,
                "units": "metric",
            },
            timeout=15,
        )

        if not response.ok:

            print(
                "[WEATHER] OpenWeather forecast "
                "request failed:",
                response.status_code,
                response.text,
            )

            return None

        return response.json()

    except Exception as exc:

        print(
            "[WEATHER] OpenWeather forecast "
            "request error:",
            exc,
        )

        return None


def _aggregate_openweather_daily(
    ow_forecast: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Group OpenWeatherMap's 3-hour steps into per-local-date
    rain probability (max pop for the day) and rain totals.
    """

    city = ow_forecast.get("city") or {}

    tz_offset = _safe_int(
        city.get("timezone"),
        0,
    )

    buckets: Dict[str, Dict[str, Any]] = {}

    for entry in (ow_forecast.get("list") or []):

        dt = entry.get("dt")

        if dt is None:
            continue

        try:

            local_dt = datetime.fromtimestamp(
                int(dt) + tz_offset,
                tz=timezone.utc,
            )

        except Exception:
            continue

        date_key = local_dt.strftime("%Y-%m-%d")

        bucket = buckets.setdefault(
            date_key,
            {"pop_values": [], "rain_mm": 0.0, "slots": 0},
        )

        pop = entry.get("pop")

        if pop is not None:
            bucket["pop_values"].append(
                _safe_float(pop) * 100
            )

        rain_block = entry.get("rain") or {}
        snow_block = entry.get("snow") or {}

        bucket["rain_mm"] += _safe_float(
            rain_block.get("3h"), 0.0
        ) + _safe_float(
            snow_block.get("3h"), 0.0
        )

        bucket["slots"] += 1

    daily: Dict[str, Dict[str, float]] = {}

    for date_key, bucket in buckets.items():

        pop_values = bucket["pop_values"]

        daily[date_key] = {
            "rain_probability":
                max(pop_values) if pop_values else None,
            "rain_mm": bucket["rain_mm"],
            # A day needs most of its 8 three-hour slots present
            # before its rainfall TOTAL is trustworthy enough to
            # blend in (a partial day undercounts total mm even
            # though its peak probability is still informative).
            "full_day": bucket["slots"] >= 6,
        }

    return daily


# ============================================================
# BUILD 7-DAY FORECAST
# ============================================================

def _build_forecast(
    data: Dict[str, Any],
    owm_daily: Optional[Dict[str, Dict[str, float]]] = None,
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

        # ----------------------------------------------------
        # BLEND WITH OPENWEATHER (INDEPENDENT MODEL)
        # ----------------------------------------------------

        owm_day = (
            owm_daily.get(date_string)
            if owm_daily
            else None
        )

        if owm_day:

            owm_probability = owm_day.get(
                "rain_probability"
            )

            if owm_probability is not None:
                rain_probability = round(
                    (rain_probability + owm_probability) / 2
                )

            if owm_day.get("full_day"):
                rainfall_mm = (
                    rainfall_mm + owm_day.get("rain_mm", 0.0)
                ) / 2

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

        # ------------------------------------------------------
        # HIGH-PROBABILITY / LOW-VOLUME DAYS
        #
        # A near-certain drizzle (e.g. 85% chance, 1 mm) still
        # ruins pesticide/fertilizer spraying and harvest timing
        # even though it never clears the rainfall-amount
        # thresholds above. Farmers plan around the CHANCE of
        # rain, not only the volume, so flag high probability on
        # its own.
        # ------------------------------------------------------

        elif rain_probability >= 85:

            alerts.append(
                {

                    "level": "Medium",

                    "title":
                        f"High Chance of Rain on "
                        f"{day_name}",

                    "body":
                        f"{rain_probability}% chance "
                        f"of precipitation, though only "
                        f"light rainfall "
                        f"(~{rain_mm:.1f} mm) is expected. "
                        f"Avoid spraying and plan harvesting "
                        f"around this.",
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
    #
    # If GPS coordinates are missing but a place name was
    # supplied (e.g. the farmer asked "weather in Davangere"),
    # resolve that name to coordinates via forward geocoding
    # instead of failing outright.
    # --------------------------------------------------------

    geocoded_name: Optional[str] = None

    if lat is None or lng is None:

        if not location:

            raise ValueError(
                "GPS coordinates are required. "
                "The browser/device must provide "
                "latitude and longitude."
            )

        geocoded = _geocode_location(location)

        if not geocoded:

            raise ValueError(
                f"Could not resolve location '{location}' "
                "to coordinates."
            )

        lat = geocoded["latitude"]
        lng = geocoded["longitude"]
        geocoded_name = geocoded["name"]

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

    location_name = geocoded_name or _get_location_name(
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

    # --------------------------------------------------------
    # OPENWEATHER OVERRIDE
    #
    # When available, prefer OpenWeatherMap's station-based
    # current conditions over Open-Meteo's grid-model estimate
    # for "right now" fields. Rain probability keeps coming
    # from Open-Meteo, which is the only one of the two that
    # supplies it. If OpenWeatherMap is unavailable or the
    # response is malformed, the Open-Meteo current block
    # computed above is used unchanged.
    # --------------------------------------------------------

    ow_data = _fetch_openweather_current(
        latitude,
        longitude,
    )

    if ow_data:

        ow_current = _build_current_from_openweather(
            ow_data,
            current["rain_probability"],
            location_name,
        )

        if ow_current:

            print(
                "Using OpenWeatherMap for "
                "current conditions."
            )

            current = ow_current

    # --------------------------------------------------------
    # OPENWEATHER FORECAST BLEND (RAIN ACCURACY)
    # --------------------------------------------------------

    owm_daily = None

    ow_forecast = _fetch_openweather_forecast(
        latitude,
        longitude,
    )

    if ow_forecast:

        try:
            owm_daily = _aggregate_openweather_daily(
                ow_forecast
            )

            print(
                "Blending OpenWeatherMap forecast "
                "into",
                len(owm_daily),
                "day(s) of rain prediction.",
            )

        except Exception as exc:

            print(
                "[WEATHER] OpenWeather forecast "
                "aggregation failed:",
                exc,
            )

            owm_daily = None

    forecast = _build_forecast(
        data,
        owm_daily,
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