"""Market price and price prediction models."""

from pydantic import BaseModel, Field


class WeekForecastItem(BaseModel):
    day: str
    price: float
    low: float
    high: float


class MonthTrendItem(BaseModel):
    day: str
    price: float


class PriceSummary(BaseModel):
    today_price: float
    tomorrow_price: float
    tomorrow_change_pct: float
    best_selling_day: str
    best_selling_price: float
    model_confidence: float = Field(..., ge=0, le=100)
    crop: str = "Tomato"
    unit: str = "quintal"
    currency: str = "INR"


class BestDayRecommendation(BaseModel):
    day: str
    price: float
    narrative: str


class MarketPriceResponse(BaseModel):
    crop: str
    unit: str
    currency: str
    today_price: float
    tomorrow_price: float
    tomorrow_change_pct: float


class PricePredictionResponse(BaseModel):
    summary: PriceSummary
    week_forecast: list[WeekForecastItem]
    month_trend: list[MonthTrendItem]
    best_day: BestDayRecommendation


class PredictionHistoryItem(BaseModel):
    crop: str
    predicted_price: float
    actual_price: float
    accurate: bool
    date: str


class PredictionHistoryResponse(BaseModel):
    predictions: list[PredictionHistoryItem]
