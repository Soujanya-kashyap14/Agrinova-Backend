"""Authentication-related Pydantic models."""

from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=14)
    password: str = Field(..., min_length=8, max_length=128)

    village: str
    district: str
    state: str
    landArea: float


class LoginRequest(BaseModel):
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=14)
    password: str = Field(..., min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None

    village: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    landArea: Optional[float] = None

    location: Optional[str] = None
    farm_size: Optional[str] = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class MessageResponse(BaseModel):
    message: str