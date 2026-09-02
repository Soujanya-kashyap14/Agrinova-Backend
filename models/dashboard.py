"""Dashboard profile and earnings models."""

from pydantic import BaseModel


class UserStats(BaseModel):
    analyses: int
    grade1_pct: str
    markets: int


class UserProfileResponse(BaseModel):
    initials: str
    name: str
    location: str
    farm_size: str
    stats: UserStats
    season_earnings: float
    season_change_pct: float


class EarningsItem(BaseModel):
    month: str
    earnings: float


class EarningsResponse(BaseModel):
    earnings: list[EarningsItem]
