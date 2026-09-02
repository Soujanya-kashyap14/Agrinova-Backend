"""Market price and price prediction routes."""

from typing import Optional

from fastapi import APIRouter, Query

from models.price import MarketPriceResponse, PricePredictionResponse
from services.price_service import get_market_prices, get_price_prediction

router = APIRouter(prefix="/price", tags=["Price"])


@router.get("/market", response_model=MarketPriceResponse)
async def get_live_market_prices(
    crop: str = Query("Tomato", description="Crop name"),
) -> MarketPriceResponse:
    """Get current and tomorrow market prices for a crop."""
    return get_market_prices(crop=crop)


@router.get("/predict", response_model=PricePredictionResponse)
async def get_price_forecast(
    crop: str = Query("Tomato", description="Crop name"),
    location: Optional[str] = Query(None, description="Farm or mandi location"),
) -> PricePredictionResponse:
    """Get AI price forecast for 7 days and 30-day trend."""
    return get_price_prediction(crop=crop, location=location)
