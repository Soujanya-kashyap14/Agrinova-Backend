"""
AgriWise / EcoAgri Intelligence
AI Service

This file is intentionally synchronous.

The crop_classifier owns:
    - model loading
    - tomato verification
    - nearest-neighbour matching
    - quality prediction
    - maturity prediction
    - grading

This service only:
    - receives the image path
    - calls crop_classifier.predict()
    - normalizes the returned dictionary
    - guarantees numeric API fields
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict

from services.crop_classifier import crop_classifier


# ============================================================
# STARTUP
# ============================================================

print()
print("=" * 70)
print("AGRIWISE AI SERVICE")
print("=" * 70)

print(
    "[AI SERVICE] Quality model:",
    getattr(
        crop_classifier,
        "quality_model",
        None,
    ) is not None,
)

print(
    "[AI SERVICE] Maturity model:",
    getattr(
        crop_classifier,
        "maturity_model",
        None,
    ) is not None,
)

print(
    "[AI SERVICE] Feature model:",
    getattr(
        crop_classifier,
        "feature_model",
        None,
    ) is not None,
)

print(
    "[AI SERVICE] Nearest-neighbour database:",
    getattr(
        crop_classifier,
        "reference_embeddings",
        None,
    ) is not None,
)

print(
    "[AI SERVICE] Reference images:",
    len(
        getattr(
            crop_classifier,
            "reference_paths",
            [],
        )
    ),
)

print("=" * 70)
print()


# ============================================================
# LOAD AI MODELS
# ============================================================

def load_ai_models() -> bool:

    """
    Reload all AI models.

    Returns True when the crop classifier is ready.
    """

    try:

        print()
        print(
            "[AI SERVICE] Loading AI models..."
        )

        loaded = crop_classifier.load_model()

        print(
            "[AI SERVICE] Models loaded:",
            loaded,
        )

        return bool(loaded)

    except Exception as exc:

        print(
            "[AI SERVICE] Model loading failed:",
            exc,
        )

        traceback.print_exc()

        return False


# ============================================================
# SAFE FLOAT
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    """
    Convert a value to float safely.

    This prevents values such as:
        "Unknown"
        None
        ""
    from reaching Pydantic float fields.
    """

    try:

        if value is None:
            return default

        if isinstance(value, str):

            value = value.strip()

            if value.lower() in {
                "",
                "unknown",
                "none",
                "null",
                "nan",
            }:

                return default

        result = float(value)

        if result != result:
            return default

        return result

    except Exception:

        return default


# ============================================================
# SAFE STRING
# ============================================================

def _safe_string(
    value: Any,
    default: str,
) -> str:

    if value is None:

        return default

    text = str(value).strip()

    if not text:

        return default

    return text


# ============================================================
# NORMALIZE RESULT
# ============================================================

def _normalize_result(
    result: Dict[str, Any],
) -> Dict[str, Any]:

    """
    Make the classifier response safe for the FastAPI
    Pydantic response model.

    In particular:
        quality -> float
        freshness -> float
        confidence -> float
        maturity_confidence -> float
        price -> float
    """

    normalized = dict(result)

    # --------------------------------------------------------
    # Strings
    # --------------------------------------------------------

    is_tomato = bool(
        normalized.get(
            "is_tomato",
            False,
        )
    )

    normalized["crop"] = _safe_string(
        normalized.get("crop"),
        "Tomato" if is_tomato else "Unknown",
    )

    normalized["quality_label"] = _safe_string(
        normalized.get("quality_label"),
        "Unknown",
    )

    normalized["maturity"] = _safe_string(
        normalized.get("maturity"),
        "Unknown",
    )

    normalized["grade"] = _safe_string(
        normalized.get("grade"),
        "Rejected",
    )

    normalized["recommendation"] = _safe_string(
        normalized.get("recommendation"),
        "Please upload a clear tomato image.",
    )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    normalized["quality"] = _safe_float(
        normalized.get("quality"),
        0.0,
    )

    normalized["freshness"] = _safe_float(
        normalized.get("freshness"),
        0.0,
    )

    normalized["confidence"] = _safe_float(
        normalized.get("confidence"),
        0.0,
    )

    normalized["maturity_confidence"] = _safe_float(
        normalized.get("maturity_confidence"),
        0.0,
    )

    normalized["maturity_probability"] = _safe_float(
        normalized.get("maturity_probability"),
        0.0,
    )

    normalized["price"] = _safe_float(
        normalized.get("price"),
        0.0,
    )

    normalized["nearest_similarity"] = _safe_float(
        normalized.get("nearest_similarity"),
        0.0,
    )

    normalized["tomato_similarity"] = _safe_float(
        normalized.get("tomato_similarity"),
        0.0,
    )

    normalized["tomato_threshold"] = _safe_float(
        normalized.get("tomato_threshold"),
        0.0,
    )

    normalized["imagenet_confidence"] = _safe_float(
        normalized.get("imagenet_confidence"),
        0.0,
    )

    # --------------------------------------------------------
    # Boolean fields
    # --------------------------------------------------------

    normalized["is_tomato"] = bool(
        normalized.get(
            "is_tomato",
            False,
        )
    )

    normalized["tomato_verified"] = bool(
        normalized.get(
            "tomato_verified",
            is_tomato,
        )
    )

    normalized["tomato_verification_reason"] = _safe_string(
        normalized.get(
            "tomato_verification_reason",
            normalized.get("verification_message"),
        ),
        "",
    )

    normalized["rejection_reason"] = _safe_string(
        normalized.get(
            "rejection_reason",
            normalized.get("verification_message")
            if not is_tomato
            else "",
        ),
        "",
    )

    # --------------------------------------------------------
    # Maturity probabilities
    # --------------------------------------------------------

    maturity_probabilities = (
        normalized.get(
            "maturity_probabilities"
        )
    )

    if not isinstance(
        maturity_probabilities,
        dict,
    ):

        maturity_probabilities = {}

    normalized[
        "maturity_probabilities"
    ] = {

        "mature": _safe_float(
            maturity_probabilities.get(
                "mature",
                0.0,
            ),
            0.0,
        ),

        "immature": _safe_float(
            maturity_probabilities.get(
                "immature",
                0.0,
            ),
            0.0,
        ),
    }

    # --------------------------------------------------------
    # Best market
    # --------------------------------------------------------

    best_market = normalized.get(
        "best_market"
    )

    if not isinstance(
        best_market,
        dict,
    ):

        best_market = {}

    normalized["best_market"] = {

        "name": _safe_string(
            best_market.get("name"),
            "Available through Nearby Mandi",
        ),

        "distance_km": _safe_float(
            best_market.get(
                "distance_km",
                0.0,
            ),
            0.0,
        ),

        "travel_time": _safe_string(
            best_market.get(
                "travel_time"
            ),
            "N/A",
        ),
    }

    # --------------------------------------------------------
    # Keep a useful reason
    # --------------------------------------------------------

    normalized["reason"] = _safe_string(
        normalized.get("reason"),
        "",
    )

    return normalized


# ============================================================
# MAIN PREDICTION
# ============================================================

def predict_crop(
    image_path: str,
) -> Dict[str, Any]:

    """
    Predict a tomato image.

    IMPORTANT:
    This function is synchronous.
    """

    print()
    print("=" * 70)
    print("AGRIWISE AI PREDICTION")
    print("=" * 70)

    print(
        f"[AI SERVICE] Predicting: {image_path}"
    )

    try:

        # ----------------------------------------------------
        # File check
        # ----------------------------------------------------

        path = Path(image_path)

        if not path.exists():

            print(
                "[AI SERVICE] Image file not found:",
                path,
            )

            result = {
                "crop": "Unknown",
                "is_tomato": False,
                "tomato_verified": False,
                "quality_label": "Unknown",
                "quality": 0.0,
                "freshness": 0.0,
                "confidence": 0.0,
                "maturity": "Unknown",
                "maturity_confidence": 0.0,
                "maturity_probability": 0.0,
                "maturity_probabilities": {
                    "mature": 0.0,
                    "immature": 0.0,
                },
                "grade": "Rejected",
                "price": 0.0,
                "best_market": {
                    "name": (
                        "Available through Nearby Mandi"
                    ),
                    "distance_km": 0.0,
                    "travel_time": "N/A",
                },
                "recommendation": (
                    "The uploaded image could not be found."
                ),
                "reason": (
                    "Image file not found."
                ),
            }

            return _normalize_result(
                result
            )

        # ----------------------------------------------------
        # Check classifier readiness
        # ----------------------------------------------------

        ready = bool(
            getattr(
                crop_classifier,
                "models_loaded",
                False,
            )
        )

        if not ready:

            print(
                "[AI SERVICE] Classifier not ready."
            )

            print(
                "[AI SERVICE] Attempting model load..."
            )

            ready = load_ai_models()

        if not ready:

            print(
                "[AI SERVICE] Models could not be loaded."
            )

            result = {
                "crop": "Unknown",
                "is_tomato": False,
                "tomato_verified": False,
                "quality_label": "Unknown",
                "quality": 0.0,
                "freshness": 0.0,
                "confidence": 0.0,
                "maturity": "Unknown",
                "maturity_confidence": 0.0,
                "maturity_probability": 0.0,
                "maturity_probabilities": {
                    "mature": 0.0,
                    "immature": 0.0,
                },
                "grade": "Rejected",
                "price": 0.0,
                "best_market": {
                    "name": (
                        "Available through Nearby Mandi"
                    ),
                    "distance_km": 0.0,
                    "travel_time": "N/A",
                },
                "recommendation": (
                    "AI models are not loaded. "
                    "Please restart the backend."
                ),
                "reason": (
                    "AI models are not loaded."
                ),
            }

            return _normalize_result(
                result
            )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        result = crop_classifier.predict(
            str(path)
        )

        if not isinstance(
            result,
            dict,
        ):

            print(
                "[AI SERVICE] Classifier returned "
                "an invalid result."
            )

            result = {
                "crop": "Unknown",
                "is_tomato": False,
                "tomato_verified": False,
                "quality_label": "Unknown",
                "quality": 0.0,
                "freshness": 0.0,
                "confidence": 0.0,
                "maturity": "Unknown",
                "maturity_confidence": 0.0,
                "maturity_probability": 0.0,
                "maturity_probabilities": {
                    "mature": 0.0,
                    "immature": 0.0,
                },
                "grade": "Rejected",
                "price": 0.0,
                "best_market": {
                    "name": (
                        "Available through Nearby Mandi"
                    ),
                    "distance_km": 0.0,
                    "travel_time": "N/A",
                },
                "recommendation": (
                    "Unable to analyse the image."
                ),
                "reason": (
                    "Invalid classifier response."
                ),
            }

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        normalized = _normalize_result(
            result
        )

        print()
        print("=" * 70)
        print("AI SERVICE - PREDICTION COMPLETED")
        print("=" * 70)

        print(
            f"[AI SERVICE] Crop: "
            f"{normalized['crop']}"
        )

        print(
            f"[AI SERVICE] Tomato verified: "
            f"{normalized['tomato_verified']}"
        )

        print(
            f"[AI SERVICE] Quality label: "
            f"{normalized['quality_label']}"
        )

        print(
            f"[AI SERVICE] Quality: "
            f"{normalized['quality']:.2f} %"
        )

        print(
            f"[AI SERVICE] Freshness: "
            f"{normalized['freshness']:.2f} %"
        )

        print(
            f"[AI SERVICE] Maturity: "
            f"{normalized['maturity']}"
        )

        print(
            f"[AI SERVICE] Maturity confidence: "
            f"{normalized['maturity_confidence']:.2f} %"
        )

        print(
            f"[AI SERVICE] Grade: "
            f"{normalized['grade']}"
        )

        print(
            f"[AI SERVICE] Tomato similarity: "
            f"{normalized['nearest_similarity']:.4f}"
        )

        print(
            f"[AI SERVICE] Best market: "
            f"{normalized['best_market']}"
        )

        print("=" * 70)

        return normalized

    except Exception as exc:

        print()
        print(
            "[PREDICTION ERROR]",
            exc,
        )

        traceback.print_exc()

        # ----------------------------------------------------
        # NEVER return strings in numeric fields.
        # ----------------------------------------------------

        error_result = {

            "crop": "Unknown",

            "is_tomato": False,

            "tomato_verified": False,

            "nearest_similarity": 0.0,

            "tomato_similarity": 0.0,

            "tomato_threshold": 0.0,

            "imagenet_label": None,

            "imagenet_confidence": 0.0,

            "quality_label": "Unknown",

            "quality": 0.0,

            "freshness": 0.0,

            "confidence": 0.0,

            "maturity": "Unknown",

            "maturity_confidence": 0.0,

            "maturity_probability": 0.0,

            "maturity_probabilities": {
                "mature": 0.0,
                "immature": 0.0,
            },

            "grade": "Rejected",

            "price": 0.0,

            "best_market": {
                "name": (
                    "Available through Nearby Mandi"
                ),
                "distance_km": 0.0,
                "travel_time": "N/A",
            },

            "recommendation": (
                "Unable to analyse the uploaded image."
            ),

            "reason": str(exc),
        }

        return _normalize_result(
            error_result
        )


# ============================================================
# SYNC WRAPPER
# ============================================================

def predict_crop_sync(
    image_path: str,
) -> Dict[str, Any]:

    """
    Backward-compatible synchronous wrapper.

    Existing code can continue calling:
        predict_crop_sync(image_path)
    """

    return predict_crop(
        image_path
    )