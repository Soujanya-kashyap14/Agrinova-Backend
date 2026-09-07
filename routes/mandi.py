"""
AgriWise Intelligence
Mandi Routes
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from services.mandi_service import get_mandi_status, get_nearby_mandi_response


router = APIRouter(prefix="/mandis", tags=["Mandis"])


@router.get("/nearby")
async def nearby_mandis(
    lat: Optional[float] = Query(default=None, description="User GPS latitude"),
    lng: Optional[float] = Query(default=None, description="User GPS longitude"),
    location: Optional[str] = Query(
        default=None,
        description="Place name such as Mysore or Bangalore Rural",
    ),
    sort_by: str = Query(
        default="distance",
        description="Sort by distance or rating",
    ),
) -> dict[str, Any]:
    sort_value = (sort_by or "distance").strip().lower()
    if sort_value not in {"distance", "rating"}:
        raise HTTPException(status_code=400, detail="sort_by must be distance or rating.")

    print("[MANDI ROUTE] nearby tomato mandis")
    print(f"[MANDI ROUTE] lat={lat} lng={lng} location={location} sort_by={sort_value}")

    try:
        response = await get_nearby_mandi_response(
            latitude=lat,
            longitude=lng,
            location=location,
            sort_by=sort_value,
        )
        print(f"[MANDI ROUTE] returning {response.get('mandi_count', 0)} mandis")
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(f"[MANDI ROUTE] Service error: {exc!r}")
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load nearby mandi data: {exc}",
        )


@router.get("/status")
async def mandi_status() -> dict[str, Any]:
    try:
        return get_mandi_status()
    except Exception as exc:
        print(f"[MANDI STATUS] Error: {exc!r}")
        raise HTTPException(
            status_code=500,
            detail="Unable to read mandi catalogue status.",
        )
