"""
Crop prediction routes.
"""

from datetime import datetime, timezone
from pathlib import Path
import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from config import get_settings
from database.mongodb import get_database
from models.prediction import CropPredictionResponse
from services.ai_service import predict_crop
from utils.deps import get_optional_user_id


router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


# ============================================================
# SAVE SCAN
# ============================================================

async def _save_scan(
    user_id: str | None,
    prediction: CropPredictionResponse,
    image_url: str | None,
) -> str:

    scan_id = str(
        uuid.uuid4()
    )

    db = get_database()

    scan_doc = {
        "_id": scan_id,

        "user_id": user_id,

        "crop": prediction.crop,

        "grade": prediction.grade,

        "confidence": prediction.confidence,

        "freshness": prediction.freshness,

        "quality": prediction.quality,

        "price": prediction.price,

        "best_market":
            prediction.best_market.model_dump(),

        "recommendation":
            prediction.recommendation,

        "image_url": image_url,

        "is_tomato":
            prediction.is_tomato,

        "tomato_verified":
            prediction.tomato_verified,

        "tomato_confidence":
            prediction.tomato_confidence,

        "analysis_status":
            prediction.analysis_status,

        "rejection_reason":
            prediction.rejection_reason,

        "class_name":
            prediction.class_name,

        "maturity":
            prediction.maturity,

        "maturity_confidence":
            prediction.maturity_confidence,

        "created_at":
            datetime.now(timezone.utc),
    }

    await db.scans.insert_one(
        scan_doc
    )

    return scan_id


# ============================================================
# PREDICT FROM FILE
# ============================================================

@router.post(
    "/crop",
    response_model=CropPredictionResponse,
)
async def predict_from_file(
    file: UploadFile | None = File(
        default=None
    ),
    user_id: str | None = Depends(
        get_optional_user_id
    ),
) -> CropPredictionResponse:

    settings = get_settings()

    image_path: str | None = None

    image_url: str | None = None

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    if file and file.filename:

        content = await file.read()

        if not content:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        upload_dir = Path(
            settings.upload_dir
        )

        upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        ext = (
            Path(
                file.filename
            ).suffix.lower()
            or ".jpg"
        )

        filename = (
            f"{uuid.uuid4()}{ext}"
        )

        image_path = str(
            upload_dir / filename
        )

        with open(
            image_path,
            "wb",
        ) as output_file:

            output_file.write(
                content
            )

        image_url = (
            f"/uploads/{filename}"
        )

    else:

        raise HTTPException(
            status_code=400,
            detail="Please upload an image using the 'file' field.",
        )

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    try:

        # IMPORTANT:
        # predict_crop is synchronous.
        prediction_data = predict_crop(
            image_path
        )

        prediction = (
            CropPredictionResponse(
                **prediction_data
            )
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        print(
            "[PREDICTION ERROR]",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Crop analysis failed: "
                f"{str(exc)}"
            ),
        ) from exc

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    try:

        scan_id = await _save_scan(
            user_id,
            prediction,
            image_url,
        )

    except Exception as exc:

        print(
            "[SCAN SAVE ERROR]",
            repr(exc),
        )

        scan_id = None

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    prediction.image_url = image_url

    # CropPredictionResponse does not need scan_id
    # unless your frontend explicitly requires it.

    if scan_id:

        # Pydantic model allows this only if the field exists.
        # We intentionally don't assign it here because the
        # response model you supplied does not define scan_id.
        pass

    return prediction


# ============================================================
# PREDICT FROM EXISTING UPLOAD
# ============================================================

@router.post(
    "/crop/{upload_id}",
    response_model=CropPredictionResponse,
)
async def predict_from_upload_id(
    upload_id: str,
    user_id: str | None = Depends(
        get_optional_user_id
    ),
) -> CropPredictionResponse:

    settings = get_settings()

    db = get_database()

    upload = await db.uploads.find_one(
        {
            "_id": upload_id
        }
    )

    if not upload:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload not found.",
        )

    image_path = str(
        Path(
            settings.upload_dir
        )
        / upload["filename"]
    )

    if not Path(
        image_path
    ).exists():

        raise HTTPException(
            status_code=404,
            detail="Uploaded image file not found.",
        )

    try:

        prediction_data = predict_crop(
            image_path
        )

        prediction = (
            CropPredictionResponse(
                **prediction_data
            )
        )

    except Exception as exc:

        print(
            "[PREDICTION ERROR]",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Crop analysis failed: "
                f"{str(exc)}"
            ),
        ) from exc

    image_url = upload.get(
        "image_url"
    )

    try:

        await _save_scan(
            user_id,
            prediction,
            image_url,
        )

    except Exception as exc:

        print(
            "[SCAN SAVE ERROR]",
            repr(exc),
        )

    prediction.image_url = image_url

    return prediction