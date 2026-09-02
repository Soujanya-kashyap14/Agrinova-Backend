"""
Weather API routes for EcoAgri Intelligence.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from services.weather_service import get_weather


router = APIRouter(
    prefix="/weather",
    tags=["Weather"],
)


@router.get("")
async def weather_endpoint(
    lat: Optional[float] = Query(
        default=None,
        description="Latitude from browser/device location",
    ),

    lng: Optional[float] = Query(
        default=None,
        description="Longitude from browser/device location",
    ),

    location: Optional[str] = Query(
        default=None,
        description="Fallback location name",
    ),
):
    """
    Get current weather and a 7-day forecast.

    Browser/device latitude and longitude are preferred.
    """

    try:

        print("\n==========================================")
        print("WEATHER API REQUEST")
        print("==========================================")
        print("Latitude :", lat)
        print("Longitude:", lng)
        print("Location :", location)
        print("==========================================")

        result = get_weather(
            lat=lat,
            lng=lng,
            location=location,
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        print(
            "[WEATHER ERROR]",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )