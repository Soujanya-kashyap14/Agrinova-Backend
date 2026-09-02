"""Market price and AI price prediction service."""

import math
from typing import Optional

from models.price import (
    BestDayRecommendation,
    MarketPriceResponse,
    MonthTrendItem,
    PricePredictionResponse,
    PriceSummary,
    WeekForecastItem,
)

WEEK_FORECAST_DATA = [
    {"day": "Mon", "price": 2450, "low": 2380, "high": 2510},
    {"day": "Tue", "price": 2520, "low": 2440, "high": 2600},
    {"day": "Wed", "price": 2585, "low": 2490, "high": 2670},
    {"day": "Thu", "price": 2640, "low": 2540, "high": 2740},
    {"day": "Fri", "price": 2710, "low": 2600, "high": 2820},
    {"day": "Sat", "price": 2665, "low": 2560, "high": 2770},
    {"day": "Sun", "price": 2590, "low": 2480, "high": 2700},
]


def _generate_month_trend() -> list[MonthTrendItem]:
    return [
        MonthTrendItem(
            day=str(i + 1),
            price=round(2300 + math.sin(i / 3.4) * 180 + i * 9 + (i % 4) * 22),
        )
        for i in range(30)
    ]


def get_market_prices(crop: str = "Tomato") -> MarketPriceResponse:
    """Return current and tomorrow market prices for a crop."""
    today = 2450.0
    tomorrow = 2520.0
    change_pct = round(((tomorrow - today) / today) * 100, 1)

    return MarketPriceResponse(
        crop=crop,
        unit="quintal",
        currency="INR",
        today_price=today,
        tomorrow_price=tomorrow,
        tomorrow_change_pct=change_pct,
    )


def get_price_prediction(
    crop: str = "Tomato",
    location: Optional[str] = None,
) -> PricePredictionResponse:
    """
    Return AI price forecast (dummy data for now).

    TODO: Replace with ML model trained on historical mandi arrivals.
    """
    _ = location

    week_forecast = [WeekForecastItem(**item) for item in WEEK_FORECAST_DATA]
    month_trend = _generate_month_trend()

    today_price = 2450.0
    tomorrow_price = 2520.0
    change_pct = round(((tomorrow_price - today_price) / today_price) * 100, 1)

    summary = PriceSummary(
        today_price=today_price,
        tomorrow_price=tomorrow_price,
        tomorrow_change_pct=change_pct,
        best_selling_day="Friday",
        best_selling_price=2710.0,
        model_confidence=94.0,
        crop=crop,
    )

    best_day = BestDayRecommendation(
        day="Friday",
        price=2710.0,
        narrative=(
            f"Arrivals dip mid-week while retail demand rises before the weekend. "
            f"Holding your Grade 1 {crop.lower()} until Friday could add about "
            f"₹260 per quintal versus selling today."
        ),
    )

    return PricePredictionResponse(
        summary=summary,
        week_forecast=week_forecast,
        month_trend=month_trend,
        best_day=best_day,
    )
