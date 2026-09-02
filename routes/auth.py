"""Authentication routes: signup and login with JWT."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from database.mongodb import get_database
from models.auth import AuthResponse, LoginRequest, SignupRequest, UserResponse
from utils.deps import get_current_user_id
from utils.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["Authentication"])

def _user_to_response(user: dict) -> UserResponse:
    return UserResponse(
        id=str(user["_id"]),
        name=user["name"],
        email=user.get("email"),
        phone=user.get("phone"),
        village=user.get("village", ""),
        district=user.get("district", ""),
        state=user.get("state", ""),
        landArea=user.get("landArea", 0),
        location=user.get("location", ""),
        farm_size=user.get("farm_size", ""),
    )


def _build_auth_response(user: dict) -> AuthResponse:
    token = create_access_token({"sub": str(user["_id"])})
    return AuthResponse(access_token=token, user=_user_to_response(user))


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(body: SignupRequest) -> AuthResponse:
    """Register a new farmer account with email or phone. Password is always required."""
    if not body.email and not body.phone:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either email or phone is required.",
        )

    db = get_database()

    if body.email:
        existing = await db.users.find_one({"email": body.email.lower()})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists.",
            )

    if body.phone:
        existing = await db.users.find_one({"phone": body.phone})
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this phone number already exists.",
            )

    user_id = str(uuid.uuid4())
    user_doc = {
    "_id": user_id,
    "name": body.name.strip(),
    "email": body.email.lower() if body.email else None,
    "phone": body.phone,
    "password_hash": hash_password(body.password),

    "village": body.village,
    "district": body.district,
    "state": body.state,
    "landArea": body.landArea,

    "location": f"{body.village}, {body.district}, {body.state}",
    "farm_size": f"{body.landArea} acres",

    "created_at": datetime.now(timezone.utc),
    "updated_at": datetime.now(timezone.utc),
}

    await db.users.insert_one(user_doc)
    return _build_auth_response(user_doc)


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest) -> AuthResponse:
    """Authenticate with email or phone. Password is always required and verified."""
    if not body.email and not body.phone:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either email or phone is required.",
        )

    db = get_database()
    query = {"email": body.email.lower()} if body.email else {"phone": body.phone}
    user = await db.users.find_one(query)

    if not user or not user.get("password_hash") or not verify_password(
        body.password, user["password_hash"]
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid phone/email or password.",
        )

    return _build_auth_response(user)


@router.get("/me", response_model=UserResponse)
async def get_me(user_id: str = Depends(get_current_user_id)) -> UserResponse:
    """Get current authenticated user profile."""
    db = get_database()
    user = await db.users.find_one({"_id": user_id})
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return _user_to_response(user)
