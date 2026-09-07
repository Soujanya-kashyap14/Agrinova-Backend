"""
EcoAgri Intelligence
Mandi response models.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MandiItem(BaseModel):
    name: str

    city: str

    distance: float = Field(
        ...,
        ge=0,
    )

    price: float = Field(
        0,
        ge=0,
    )

    rating: float = Field(
        0,
        ge=0,
        le=5,
    )

    travel: str

    trend: float = 0.0

    crops: list[str]

    latitude: float | None = None

    longitude: float | None = None

    state: str | None = None

    district: str | None = None

    market_id: int | str | None = None

    source: str = "CEDA / AGMARKNET"


class MandiListResponse(BaseModel):
    mandis: list[MandiItem]

    mandi_count: int

    # Kept for frontend/API compatibility.
    # It is NOT used as a search filter anymore.
    radius_km: float = 0.0

    region: str

    last_updated: str

    searched_latitude: float | None = None

    searched_longitude: float | None = None

    location_label: str | None = None

    search_mode: str | None = None

    sorted_by: str | None = None

    sort_order: str | None = None

    farthest_distance_km: float | None = None

    ceda_live: bool = False

    source: str = "CEDA / AGMARKNET"