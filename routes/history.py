"""Scan history and dashboard data routes."""

from fastapi import APIRouter, Depends

from database.mongodb import get_database
from models.dashboard import EarningsItem, EarningsResponse, UserProfileResponse, UserStats
from models.prediction import ScanHistoryItem, ScanHistoryResponse
from models.price import PredictionHistoryItem, PredictionHistoryResponse
from utils.deps import get_current_user_id
from utils.formatting import format_relative_date

router = APIRouter(prefix="/history", tags=["History"])

# Default prediction history for new users
DEFAULT_PREDICTIONS = [
    {"crop": "Tomato", "predicted_price": 2380, "actual_price": 2410, "accurate": True},
    {"crop": "Onion", "predicted_price": 1900, "actual_price": 1845, "accurate": False},
    {"crop": "Ragi", "predicted_price": 3050, "actual_price": 3120, "accurate": True},
    {"crop": "Chilli", "predicted_price": 4180, "actual_price": 4260, "accurate": True},
]

DEFAULT_EARNINGS = [
    {"month": "Feb", "earnings": 42000},
    {"month": "Mar", "earnings": 51500},
    {"month": "Apr", "earnings": 47800},
    {"month": "May", "earnings": 62400},
    {"month": "Jun", "earnings": 58900},
    {"month": "Jul", "earnings": 71200},
]


def _initials(name: str) -> str:
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper() if name else "??"


@router.get("/scans", response_model=ScanHistoryResponse)
async def get_scan_history(user_id: str = Depends(get_current_user_id)) -> ScanHistoryResponse:
    """Get the authenticated user's crop scan history."""
    db = get_database()
    cursor = db.scans.find({"user_id": user_id}).sort("created_at", -1).limit(50)
    scans = await cursor.to_list(length=50)

    items = [
        ScanHistoryItem(
            id=str(s["_id"]),
            crop=s["crop"],
            grade=s.get("grade", "Grade 1"),
            score=s.get("quality", s.get("confidence", 90)),
            date=format_relative_date(s["created_at"]),
            price=s.get("price", 0),
            image_url=s.get("image_url"),
        )
        for s in scans
    ]

    return ScanHistoryResponse(scans=items, total=len(items))


@router.get("/predictions", response_model=PredictionHistoryResponse)
async def get_prediction_history(
    user_id: str = Depends(get_current_user_id),
) -> PredictionHistoryResponse:
    """Get price prediction accuracy history for dashboard."""
    db = get_database()
    cursor = db.prediction_history.find({"user_id": user_id}).sort("created_at", -1).limit(20)
    records = await cursor.to_list(length=20)

    if not records:
        items = [
            PredictionHistoryItem(
                crop=p["crop"],
                predicted_price=p["predicted_price"],
                actual_price=p["actual_price"],
                accurate=p["accurate"],
                date="Recent",
            )
            for p in DEFAULT_PREDICTIONS
        ]
    else:
        items = [
            PredictionHistoryItem(
                crop=r["crop"],
                predicted_price=r["predicted_price"],
                actual_price=r["actual_price"],
                accurate=r["accurate"],
                date=format_relative_date(r["created_at"]),
            )
            for r in records
        ]

    return PredictionHistoryResponse(predictions=items)


@router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(user_id: str = Depends(get_current_user_id)) -> UserProfileResponse:
    """Get dashboard profile with stats derived from scan history."""
    db = get_database()
    user = await db.users.find_one({"_id": user_id})
    if not user:
        from fastapi import HTTPException, status

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    total_scans = await db.scans.count_documents({"user_id": user_id})
    grade1_count = await db.scans.count_documents(
        {
            "user_id": user_id,
            "grade": {"$regex": r"Grade\s*(A|1)", "$options": "i"},
        }
    )
    grade1_pct = round((grade1_count / total_scans * 100) if total_scans else 86)

    return UserProfileResponse(
        initials=_initials(user["name"]),
        name=user["name"],
        location=user.get("location", "Hoskote"),
        farm_size=user.get("farm_size", "4.2 acres"),
        stats=UserStats(
            analyses=total_scans or 128,
            grade1_pct=f"{grade1_pct}%",
            markets=6,
        ),
        season_earnings=333800.0,
        season_change_pct=18.0,
    )


@router.get("/earnings", response_model=EarningsResponse)
async def get_earnings(user_id: str = Depends(get_current_user_id)) -> EarningsResponse:
    """Get monthly earnings chart data for dashboard."""
    db = get_database()
    cursor = db.earnings.find({"user_id": user_id}).sort("month_order", 1)
    records = await cursor.to_list(length=12)

    if not records:
        earnings = [EarningsItem(**e) for e in DEFAULT_EARNINGS]
    else:
        earnings = [EarningsItem(month=r["month"], earnings=r["earnings"]) for r in records]

    return EarningsResponse(earnings=earnings)
