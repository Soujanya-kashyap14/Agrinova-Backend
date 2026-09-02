"""
AgriWise Intelligence
Prediction Response Models
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================
# BEST MARKET
# ============================================================

class BestMarket(BaseModel):
    name: str = "Not recommended"
    distance_km: float = 0.0
    travel_time: str = "N/A"


# ============================================================
# UPLOAD RESPONSE
# ============================================================

class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    image_url: str


# ============================================================
# UPLOAD GALLERY ITEM
# ============================================================

class UploadGalleryItem(BaseModel):
    id: str
    src: str
    alt: str = "Crop upload"
    created_at: datetime


# ============================================================
# SCAN HISTORY
# ============================================================

class ScanHistoryItem(BaseModel):
    id: str
    crop: str = "Tomato"
    grade: str = "Rejected"
    score: float = 0.0
    date: str = "Recent"
    price: float = 0.0
    image_url: Optional[str] = None


class ScanHistoryResponse(BaseModel):
    scans: List[ScanHistoryItem] = Field(
        default_factory=list
    )
    total: int = 0


# ============================================================
# CROP PREDICTION RESPONSE
# ============================================================

class CropPredictionResponse(BaseModel):
    """
    Final tomato AI response.

    Pipeline:

        Image
          |
          +--> Maturity evidence
          |
          +--> Tomato similarity evidence
          |
          +--> STRICT TOMATO GATE
                    |
             +------+------+
             |             |
          REJECTED       VERIFIED
             |             |
             |        Fresh / Rotten
             |             |
             |           Quality
             |             |
             |            Grade
             |             |
             |          Maturity
    """

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image_url: Optional[str] = None

    # --------------------------------------------------------
    # BASIC CROP
    # --------------------------------------------------------

    crop: str = "Tomato"

    # --------------------------------------------------------
    # GRADE / QUALITY
    # --------------------------------------------------------

    grade: str = "Rejected"

    confidence: float = 0.0

    freshness: float = 0.0

    quality: float = 0.0

    price: float = 0.0

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    best_market: BestMarket = Field(
        default_factory=BestMarket
    )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    recommendation: str = ""

    # --------------------------------------------------------
    # TOMATO VALIDATION
    # --------------------------------------------------------

    is_tomato: bool = False

    tomato_verified: bool = False

    tomato_confidence: float = 0.0

    tomato_verification_reason: Optional[str] = None

    analysis_status: str = "REJECTED"

    rejection_reason: Optional[str] = None

    # --------------------------------------------------------
    # FRESH / ROTTEN
    # --------------------------------------------------------

    class_name: Optional[str] = None

    class_probabilities: Dict[str, float] = Field(
        default_factory=dict
    )

    classifier_rotten_probability: float = 0.0

    similarity_rotten_score: float = 0.0

    combined_rotten_probability: float = 0.0

    # --------------------------------------------------------
    # SIMILARITY
    # --------------------------------------------------------

    nearest_similarity: Optional[float] = None

    top_similar_images: List[Dict[str, Any]] = Field(
        default_factory=list
    )

    rotten_neighbors: int = 0

    fresh_neighbors: int = 0

    total_neighbors: int = 0

    rotten_ratio: float = 0.0

    fresh_ratio: float = 0.0

    # --------------------------------------------------------
    # MATURITY
    # --------------------------------------------------------

    maturity: str = "Unknown"

    maturity_confidence: float = 0.0

    maturity_probabilities: Dict[str, float] = Field(
        default_factory=dict
    )

    # Backward compatibility
    ripeness: Optional[str] = None

    # --------------------------------------------------------
    # DEFECT
    # --------------------------------------------------------

    defect_detected: bool = False

    defect_type: Optional[str] = None

    # --------------------------------------------------------
    # REJECTION
    # --------------------------------------------------------

    visual_rejection: bool = False

    rejection_reasons: List[str] = Field(
        default_factory=list
    )