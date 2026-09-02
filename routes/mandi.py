"""
AgriWise Intelligence
Mandi Routes
"""

from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from services.mandi_service import (
    get_nearby_mandi_response,
    get_mandi_status,
)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/mandis",
    tags=["Mandis"],
)


# ============================================================
# NEARBY KARNATAKA MANDIS
# ============================================================

@router.get(
    "/nearby",
)
async def nearby_mandis(
    lat: float = Query(
        ...,
        description="User GPS latitude",
    ),

    lng: float = Query(
        ...,
        description="User GPS longitude",
    ),
) -> dict[str, Any]:

    print(
        "======================================================================"
    )

    print(
        "[MANDI ROUTE] ALL KARNATAKA TOMATO MANDI SEARCH"
    )

    print(
        f"[MANDI ROUTE] Latitude : {lat}"
    )

    print(
        f"[MANDI ROUTE] Longitude: {lng}"
    )

    print(
        "[MANDI ROUTE] Radius   : NONE"
    )

    print(
        "[MANDI ROUTE] Limit    : NONE"
    )

    print(
        "[MANDI ROUTE] Sorting  : nearest -> farthest"
    )

    print(
        "======================================================================"
    )

    # --------------------------------------------------------
    # Validate coordinates
    # --------------------------------------------------------

    if not math.isfinite(lat):
        raise HTTPException(
            status_code=400,
            detail="Invalid latitude.",
        )

    if not math.isfinite(lng):
        raise HTTPException(
            status_code=400,
            detail="Invalid longitude.",
        )

    if lat < -90 or lat > 90:
        raise HTTPException(
            status_code=400,
            detail="Latitude must be between -90 and 90.",
        )

    if lng < -180 or lng > 180:
        raise HTTPException(
            status_code=400,
            detail="Longitude must be between -180 and 180.",
        )

    # --------------------------------------------------------
    # Service
    # --------------------------------------------------------

    try:

        response = (
            await get_nearby_mandi_response(
                latitude=lat,
                longitude=lng,
            )
        )

        print(
            "======================================================================"
        )

        print(
            "[MANDI ROUTE] Search complete"
        )

        print(
            f"[MANDI ROUTE] Returning "
            f"{response.get('mandi_count', 0)} mandis"
        )

        print(
            "[MANDI ROUTE] No radius restriction"
        )

        print(
            "[MANDI ROUTE] No result limit"
        )

        print(
            "======================================================================"
        )

        return response

    except ValueError as exc:

        print(
            f"[MANDI ROUTE] Validation error: {exc!r}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        print(
            f"[MANDI ROUTE] Service error: {exc!r}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to load Karnataka mandi "
                f"data: {str(exc)}"
            ),
        )


# ============================================================
# MANDI STATUS
# ============================================================

@router.get(
    "/status",
)
async def mandi_status() -> dict[str, Any]:

    try:

        return get_mandi_status()

    except Exception as exc:

        print(
            f"[MANDI STATUS] Error: {exc!r}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to read mandi catalogue status."
            ),
        )