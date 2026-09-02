"""Weather-related Pydantic models."""

from typing import Literal

from pydantic import BaseModel, Field


class WeatherDayForecast(BaseModel):
    day: str

    temp: float

    rain: float = Field(
        ...,
        ge=0,
        le=100,
    )

    icon: Literal[
        "sun",
        "rain",
        "cloud",
    ]


class WeatherAlert(BaseModel):
    level: Literal[
        "High",
        "Medium",
        "Low",
    ]

    title: str

    body: str


class WeatherCurrent(BaseModel):
    location: str

    temperature: float

    condition: str

    feels_like: float

    humidity: float

    rain_probability: float

    wind_speed: str

    wind_direction: str

    sunrise: str

    sunset: str


class WeatherResponse(BaseModel):
    current: WeatherCurrent

    forecast: list[WeatherDayForecast]

    alerts: list[WeatherAlert]