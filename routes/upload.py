"""Image upload routes."""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from config import get_settings
from database.mongodb import get_database
from models.prediction import UploadGalleryItem, UploadResponse
from utils.deps import get_current_user_id, get_optional_user_id

router = APIRouter(prefix="/upload", tags=["Upload"])


def _ensure_upload_dir() -> Path:
    settings = get_settings()
    upload_path = Path(settings.upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(
    file: UploadFile = File(...),
    user_id: str | None = Depends(get_optional_user_id),
) -> UploadResponse:
    """Upload a crop image for AI grading."""
    settings = get_settings()

    if file.content_type not in settings.allowed_image_types_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(settings.allowed_image_types_list)}",
        )

    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb} MB.",
        )

    upload_dir = _ensure_upload_dir()
    ext = Path(file.filename or "image.jpg").suffix or ".jpg"
    upload_id = str(uuid.uuid4())
    filename = f"{upload_id}{ext}"
    file_path = upload_dir / filename

    with open(file_path, "wb") as f:
        f.write(content)

    image_url = f"/uploads/{filename}"

    if user_id:
        db = get_database()
        await db.uploads.insert_one(
            {
                "_id": upload_id,
                "user_id": user_id,
                "filename": filename,
                "original_name": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(content),
                "image_url": image_url,
                "alt": file.filename or "Crop upload",
                "created_at": datetime.now(timezone.utc),
            }
        )

    return UploadResponse(
        upload_id=upload_id,
        filename=filename,
        image_url=image_url,
    )


@router.get("/gallery", response_model=list[UploadGalleryItem])
async def get_upload_gallery(user_id: str = Depends(get_current_user_id)) -> list[UploadGalleryItem]:
    """Get user's uploaded crop images for dashboard gallery."""
    db = get_database()
    cursor = db.uploads.find({"user_id": user_id}).sort("created_at", -1).limit(8)
    uploads = await cursor.to_list(length=8)

    return [
        UploadGalleryItem(
            id=str(u["_id"]),
            src=u["image_url"],
            alt=u.get("alt", "Crop upload"),
            created_at=u["created_at"],
        )
        for u in uploads
    ]
