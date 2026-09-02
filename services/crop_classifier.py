"""
AgriNova
Tomato Crop Classifier

FINAL PIPELINE
==============

Uploaded Image
      |
      v
Tomato Similarity Gate
      |
      +---- NOT TOMATO ----> REJECT
      |
      v
Tomato Verifier
      |
      +---- NOT TOMATO ----> REJECT
      |
      v
Tomato
      |
      +----> Quality Model
      |
      +----> Maturity Model
      |
      v
Grade A / B / C

IMPORTANT
---------
The tomato quality model was trained only for:

    Fresh
    Rotten

Therefore the quality model must NEVER be used to
decide whether an arbitrary image is a tomato.

The tomato similarity index was generated using:

    EfficientNetB0
    include_top=False
    ImageNet weights
    224x224 input
    1280-dimensional embeddings

The saved similarity index is:

    tomato_quality/tomato_similarity_index.npz

with:

    embeddings
    labels
    paths

The new binary verifier is:

    tomato_verifier.keras
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import traceback

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications import EfficientNetB0

# Used only for loading old Lambda layers that were serialized
# with a function named preprocess_input.
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess_input,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

TRAINED_MODELS_DIR = (
    BASE_DIR / "trained_models"
)


# ============================================================
# QUALITY MODEL
# ============================================================

QUALITY_MODEL_PATH = (
    TRAINED_MODELS_DIR
    / "tomato_quality.keras"
)


# ============================================================
# MATURITY MODEL
# ============================================================

MATURITY_MODEL_PATH = (
    TRAINED_MODELS_DIR
    / "tomato_maturity.keras"
)


# ============================================================
# NEW TOMATO / NON-TOMATO VERIFIER
# ============================================================

TOMATO_VERIFIER_PATH = (
    TRAINED_MODELS_DIR
    / "tomato_verifier.keras"
)


# ============================================================
# SIMILARITY INDEX
# ============================================================

SIMILARITY_INDEX_PATH = (
    TRAINED_MODELS_DIR
    / "tomato_quality"
    / "tomato_similarity_index.npz"
)


# Compatibility fallback
OLD_SIMILARITY_PATH = (
    TRAINED_MODELS_DIR
    / "tomato_verifier.npz"
)


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_SIZE = (
    224,
    224,
)


# ------------------------------------------------------------
# Tomato similarity threshold
# ------------------------------------------------------------

TOMATO_SIMILARITY_THRESHOLD = 0.35


# ------------------------------------------------------------
# Strong similarity
# ------------------------------------------------------------

STRONG_TOMATO_SIMILARITY = 0.60


# ------------------------------------------------------------
# Binary verifier threshold
# ------------------------------------------------------------

TOMATO_THRESHOLD = 0.50


# ------------------------------------------------------------
# Quality
# ------------------------------------------------------------

ROTTEN_THRESHOLD = 0.50


# ------------------------------------------------------------
# Maturity
# ------------------------------------------------------------

MATURITY_THRESHOLD = 0.50


# ------------------------------------------------------------
# Grade
# ------------------------------------------------------------

GRADE_A_MIN = 85.0
GRADE_B_MIN = 65.0
GRADE_C_MIN = 40.0


# ------------------------------------------------------------
# Similarity embedding size
# ------------------------------------------------------------

EXPECTED_FEATURE_SIZE = 1280


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        if value is None:
            return default

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (
        TypeError,
        ValueError,
    ):

        return default


def _clip_probability(
    value: Any,
) -> float:

    value = _safe_float(
        value,
        0.0,
    )

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


# ============================================================
# CLASSIFIER
# ============================================================

class CropClassifier:

    # ========================================================
    # INIT
    # ========================================================

    def __init__(self) -> None:

        # ----------------------------------------------------
        # Compatibility paths
        # ----------------------------------------------------

        self.model_path = (
            QUALITY_MODEL_PATH
        )

        self.quality_model_path = (
            QUALITY_MODEL_PATH
        )

        self.maturity_model_path = (
            MATURITY_MODEL_PATH
        )

        self.verifier_path = (
            TOMATO_VERIFIER_PATH
        )

        self.similarity_index_path = (
            SIMILARITY_INDEX_PATH
        )

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        self.quality_model = None

        self.maturity_model = None

        self.tomato_verifier = None

        self.feature_model = None

        self.feature_extractor = None

        # ----------------------------------------------------
        # Similarity database
        # ----------------------------------------------------

        self.similarity_embeddings = None

        self.similarity_labels = None

        self.similarity_paths = None

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        self.is_loaded = False

        self.models_loaded = False

        self.quality_loaded = False

        self.maturity_loaded = False

        self.verifier_loaded = False

        self.similarity_loaded = False

        self.feature_model_loaded = False

        # ----------------------------------------------------
        # Last values
        # ----------------------------------------------------

        self.last_tomato_probability = 0.0

        self.last_tomato_similarity = 0.0

        # ----------------------------------------------------
        # Startup information
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("AGRINOVA TOMATO CLASSIFIER")
        print("=" * 70)

        print(
            "[QUALITY MODEL]",
            QUALITY_MODEL_PATH,
        )

        print(
            "[QUALITY EXISTS]",
            QUALITY_MODEL_PATH.exists(),
        )

        print(
            "[MATURITY MODEL]",
            MATURITY_MODEL_PATH,
        )

        print(
            "[MATURITY EXISTS]",
            MATURITY_MODEL_PATH.exists(),
        )

        print(
            "[TOMATO VERIFIER]",
            TOMATO_VERIFIER_PATH,
        )

        print(
            "[VERIFIER EXISTS]",
            TOMATO_VERIFIER_PATH.exists(),
        )

        print(
            "[SIMILARITY INDEX]",
            SIMILARITY_INDEX_PATH,
        )

        print(
            "[SIMILARITY EXISTS]",
            SIMILARITY_INDEX_PATH.exists(),
        )

        print("=" * 70)

    # ========================================================
    # KERAS MODEL LOADER
    # ========================================================

    def _load_keras_model(
        self,
        path: Path,
    ):

        if not path.exists():

            print(
                "[MODEL] NOT FOUND:",
                path,
            )

            return None

        print()
        print(
            "[MODEL] Loading:",
            path,
        )

        try:

            model = keras.models.load_model(
                path,
                custom_objects={
                    "preprocess_input":
                        mobilenet_preprocess_input,
                },
                compile=False,
                safe_mode=False,
            )

            print(
                "[MODEL] Loaded successfully:",
                path.name,
            )

            return model

        except TypeError:

            # ------------------------------------------------
            # Compatibility with older Keras versions
            # ------------------------------------------------

            try:

                model = keras.models.load_model(
                    path,
                    custom_objects={
                        "preprocess_input":
                            mobilenet_preprocess_input,
                    },
                    compile=False,
                )

                print(
                    "[MODEL] Loaded successfully:",
                    path.name,
                )

                return model

            except Exception as exc:

                print(
                    "[MODEL ERROR]",
                    path.name,
                )

                print(
                    repr(exc),
                )

                traceback.print_exc()

                return None

        except Exception as exc:

            print(
                "[MODEL ERROR]",
                path.name,
            )

            print(
                repr(exc),
            )

            traceback.print_exc()

            return None

    # ========================================================
    # LOAD SIMILARITY INDEX
    # ========================================================

    def _load_similarity_index(self) -> bool:

        print()
        print(
            "[SIMILARITY] Loading tomato reference index..."
        )

        path = (
            self.similarity_index_path
        )

        # ----------------------------------------------------
        # Primary path
        # ----------------------------------------------------

        if not path.exists():

            print(
                "[SIMILARITY] Primary index not found."
            )

            # Compatibility fallback
            if OLD_SIMILARITY_PATH.exists():

                print(
                    "[SIMILARITY] Trying old index:",
                    OLD_SIMILARITY_PATH,
                )

                path = OLD_SIMILARITY_PATH

            else:

                print(
                    "[SIMILARITY] No similarity index found."
                )

                self.similarity_loaded = False

                return False

        try:

            data = np.load(
                path,
                allow_pickle=True,
            )

            # ------------------------------------------------
            # EMBEDDINGS
            # ------------------------------------------------

            if "embeddings" not in data:

                raise ValueError(
                    "Similarity index does not contain "
                    "'embeddings'."
                )

            embeddings = (
                np.asarray(
                    data["embeddings"],
                    dtype=np.float32,
                )
            )

            # ------------------------------------------------
            # LABELS
            # ------------------------------------------------

            if "labels" in data:

                labels = (
                    np.asarray(
                        data["labels"],
                        dtype=np.int32,
                    )
                )

            else:

                labels = np.zeros(
                    len(embeddings),
                    dtype=np.int32,
                )

            # ------------------------------------------------
            # PATHS
            # ------------------------------------------------

            if "paths" in data:

                paths = (
                    np.asarray(
                        data["paths"]
                    )
                )

            else:

                paths = np.asarray(
                    [
                        ""
                        for _ in range(
                            len(embeddings)
                        )
                    ]
                )

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

            if embeddings.ndim != 2:

                raise ValueError(
                    "Similarity embeddings must be 2D."
                )

            if embeddings.shape[1] != EXPECTED_FEATURE_SIZE:

                raise ValueError(
                    "Expected 1280-dimensional "
                    "EfficientNetB0 features, "
                    f"got {embeddings.shape[1]}."
                )

            if len(embeddings) == 0:

                raise ValueError(
                    "Similarity index is empty."
                )

            if len(labels) != len(embeddings):

                raise ValueError(
                    "Labels and embeddings have "
                    "different lengths."
                )

            # ------------------------------------------------
            # Normalize reference vectors
            # ------------------------------------------------

            norms = np.linalg.norm(
                embeddings,
                axis=1,
                keepdims=True,
            )

            embeddings = (
                embeddings
                /
                np.maximum(
                    norms,
                    1e-12,
                )
            )

            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            self.similarity_embeddings = (
                embeddings
            )

            self.similarity_labels = (
                labels
            )

            self.similarity_paths = (
                paths
            )

            self.similarity_index_path = path

            self.similarity_loaded = True

            print(
                "[SIMILARITY] LOADED SUCCESSFULLY"
            )

            print(
                "[SIMILARITY] Shape:",
                embeddings.shape,
            )

            print(
                "[SIMILARITY] References:",
                len(embeddings),
            )

            print(
                "[SIMILARITY] Feature size:",
                embeddings.shape[1],
            )

            return True

        except Exception as exc:

            print(
                "[SIMILARITY ERROR]",
                repr(exc),
            )

            self.similarity_embeddings = None

            self.similarity_labels = None

            self.similarity_paths = None

            self.similarity_loaded = False

            return False

    # ========================================================
    # BUILD EFFICIENTNET FEATURE EXTRACTOR
    # ========================================================

    def _build_feature_extractor(self) -> bool:

        """
        Build the SAME feature extractor used by the
        tomato similarity index.

        Training configuration:

            EfficientNetB0
            include_top=False
            weights="imagenet"
            input_shape=(224,224,3)
            GlobalAveragePooling2D

        Result:

            1280-dimensional feature vector.
        """

        print()
        print(
            "[FEATURE] Building EfficientNetB0..."
        )

        try:

            self.feature_extractor = (
                EfficientNetB0(
                    include_top=False,
                    weights="imagenet",
                    input_shape=(
                        224,
                        224,
                        3,
                    ),
                    pooling="avg",
                )
            )

            self.feature_extractor.trainable = False

            # Compatibility alias
            self.feature_model = (
                self.feature_extractor
            )

            # ------------------------------------------------
            # Verify output
            # ------------------------------------------------

            output_shape = (
                self.feature_extractor.output_shape
            )

            print(
                "[FEATURE] Output shape:",
                output_shape,
            )

            if (
                len(output_shape) != 2
                or
                output_shape[-1]
                !=
                EXPECTED_FEATURE_SIZE
            ):

                print(
                    "[FEATURE ERROR] Expected:",
                    "(None, 1280)",
                )

                print(
                    "[FEATURE ERROR] Received:",
                    output_shape,
                )

                self.feature_extractor = None

                self.feature_model = None

                self.feature_model_loaded = False

                return False

            self.feature_model_loaded = True

            print(
                "[FEATURE] EfficientNetB0 ready."
            )

            print(
                "[FEATURE] 1280-dimensional extractor ready."
            )

            return True

        except Exception as exc:

            print(
                "[FEATURE ERROR]",
                repr(exc),
            )

            traceback.print_exc()

            self.feature_extractor = None

            self.feature_model = None

            self.feature_model_loaded = False

            return False

    # ========================================================
    # LOAD EVERYTHING
    # ========================================================

    def load_model(self) -> bool:

        print()
        print("=" * 70)
        print("LOADING AGRINOVA AI MODELS")
        print("=" * 70)

        # ----------------------------------------------------
        # Reset status
        # ----------------------------------------------------

        self.is_loaded = False

        self.models_loaded = False

        self.quality_loaded = False

        self.maturity_loaded = False

        self.verifier_loaded = False

        self.similarity_loaded = False

        self.feature_model_loaded = False

        # ====================================================
        # QUALITY
        # ====================================================

        print()
        print(
            "[LOAD] Loading quality model..."
        )

        self.quality_model = (
            self._load_keras_model(
                QUALITY_MODEL_PATH
            )
        )

        if self.quality_model is not None:

            self.quality_loaded = True

            print(
                "[OK] Quality model loaded."
            )

            try:

                print(
                    "[QUALITY] Input:",
                    self.quality_model.input_shape,
                )

                print(
                    "[QUALITY] Output:",
                    self.quality_model.output_shape,
                )

            except Exception:
                pass

        # ====================================================
        # SIMILARITY INDEX
        # ====================================================

        self._load_similarity_index()

        # ====================================================
        # EFFICIENTNET FEATURE EXTRACTOR
        # ====================================================

        self._build_feature_extractor()

        # ====================================================
        # MATURITY
        # ====================================================

        print()
        print(
            "[LOAD] Loading maturity model..."
        )

        self.maturity_model = (
            self._load_keras_model(
                MATURITY_MODEL_PATH
            )
        )

        if self.maturity_model is not None:

            self.maturity_loaded = True

            print(
                "[OK] Maturity model loaded."
            )

            try:

                print(
                    "[MATURITY] Input:",
                    self.maturity_model.input_shape,
                )

                print(
                    "[MATURITY] Output:",
                    self.maturity_model.output_shape,
                )

            except Exception:
                pass

        # ====================================================
        # TOMATO VERIFIER
        # ====================================================

        print()
        print(
            "[LOAD] Loading tomato/non-tomato verifier..."
        )

        self.tomato_verifier = (
            self._load_keras_model(
                TOMATO_VERIFIER_PATH
            )
        )

        if self.tomato_verifier is not None:

            self.verifier_loaded = True

            print(
                "[OK] Tomato verifier loaded."
            )

            try:

                print(
                    "[VERIFIER] Input:",
                    self.tomato_verifier.input_shape,
                )

                print(
                    "[VERIFIER] Output:",
                    self.tomato_verifier.output_shape,
                )

            except Exception:
                pass

        # ====================================================
        # FINAL STATUS
        # ====================================================

        self.is_loaded = (
            self.quality_loaded
            and
            self.maturity_loaded
            and
            self.verifier_loaded
            and
            self.similarity_loaded
            and
            self.feature_model_loaded
        )

        self.models_loaded = (
            self.is_loaded
        )

        print()
        print("=" * 70)

        if self.is_loaded:

            print(
                "[SUCCESS] ALL AGRINOVA AI COMPONENTS READY"
            )

        else:

            print(
                "[WARNING] AGRINOVA AI LOADED PARTIALLY"
            )

        print(
            "Quality:",
            self.quality_loaded,
        )

        print(
            "Maturity:",
            self.maturity_loaded,
        )

        print(
            "Tomato verifier:",
            self.verifier_loaded,
        )

        print(
            "Similarity index:",
            self.similarity_loaded,
        )

        print(
            "EfficientNet feature extractor:",
            self.feature_model_loaded,
        )

        print("=" * 70)

        return self.is_loaded

    # ========================================================
    # IMAGE LOADING
    # ========================================================

    def _load_image(
        self,
        image_path: str,
    ) -> np.ndarray:

        path = Path(
            image_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = (
            tf.keras.utils.load_img(
                path,
                target_size=IMAGE_SIZE,
                color_mode="rgb",
            )
        )

        array = (
            tf.keras.utils.img_to_array(
                image
            )
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Models were trained using raw 0-255 pixels.
        # ----------------------------------------------------

        array = np.asarray(
            array,
            dtype=np.float32,
        )

        return array

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def _extract_feature(
        self,
        image: np.ndarray,
    ) -> Optional[np.ndarray]:

        if self.feature_extractor is None:

            return None

        try:

            batch = np.expand_dims(
                image,
                axis=0,
            )

            features = (
                self.feature_extractor.predict(
                    batch,
                    verbose=0,
                )
            )

            features = np.asarray(
                features,
                dtype=np.float32,
            )

            # ------------------------------------------------
            # Expected:
            #
            # (1,1280)
            # ------------------------------------------------

            if features.ndim == 4:

                features = np.mean(
                    features,
                    axis=(
                        1,
                        2,
                    ),
                )

            elif features.ndim > 2:

                features = (
                    features.reshape(
                        features.shape[0],
                        -1,
                    )
                )

            if features.ndim != 2:

                return None

            if features.shape[1] != EXPECTED_FEATURE_SIZE:

                print(
                    "[FEATURE ERROR] Dimension mismatch:",
                    features.shape,
                )

                return None

            feature = (
                features[0]
            )

            norm = np.linalg.norm(
                feature
            )

            if norm <= 1e-12:

                return None

            feature = (
                feature / norm
            )

            return feature.astype(
                np.float32
            )

        except Exception as exc:

            print(
                "[FEATURE ERROR]",
                repr(exc),
            )

            return None

    # ========================================================
    # SIMILARITY ANALYSIS
    # ========================================================

    def _calculate_tomato_similarity(
        self,
        image: np.ndarray,
    ) -> Dict[str, Any]:

        result = {

            "available":
                False,

            "nearest_similarity":
                0.0,

            "top_matches":
                [],

            "meaningful_neighbors":
                0,

            "fresh_neighbors":
                0,

            "rotten_neighbors":
                0,

            "fresh_ratio":
                0.0,

            "rotten_ratio":
                0.0,
        }

        if not self.similarity_loaded:

            return result

        feature = (
            self._extract_feature(
                image
            )
        )

        if feature is None:

            return result

        try:

            similarities = (
                np.dot(
                    self.similarity_embeddings,
                    feature,
                )
            )

            if similarities.size == 0:

                return result

            # ------------------------------------------------
            # Top 10 references
            # ------------------------------------------------

            top_k = min(
                10,
                len(similarities),
            )

            indices = np.argsort(
                similarities
            )[
                -top_k:
            ][::-1]

            top_matches = []

            meaningful_neighbors = 0

            fresh_neighbors = 0

            rotten_neighbors = 0

            for index in indices:

                similarity = float(
                    similarities[index]
                )

                # ------------------------------------------------
                # Label
                #
                # 0 = fresh
                # 1 = rotten
                # ------------------------------------------------

                label = int(
                    self.similarity_labels[
                        index
                    ]
                )

                if label == 1:

                    class_name = "rotten"

                else:

                    class_name = "fresh"

                if (
                    similarity
                    >=
                    TOMATO_SIMILARITY_THRESHOLD
                ):

                    meaningful_neighbors += 1

                if class_name == "fresh":

                    fresh_neighbors += 1

                else:

                    rotten_neighbors += 1

                reference_path = ""

                if (
                    self.similarity_paths is not None
                    and
                    index
                    <
                    len(
                        self.similarity_paths
                    )
                ):

                    reference_path = str(
                        self.similarity_paths[
                            index
                        ]
                    )

                top_matches.append(
                    {
                        "similarity":
                            round(
                                similarity,
                                4,
                            ),

                        "class_name":
                            class_name,

                        "label":
                            label,

                        "path":
                            reference_path,
                    }
                )

            nearest_similarity = float(
                similarities[
                    indices[0]
                ]
            )

            total_neighbors = (
                fresh_neighbors
                +
                rotten_neighbors
            )

            if total_neighbors > 0:

                fresh_ratio = (
                    fresh_neighbors
                    /
                    total_neighbors
                )

                rotten_ratio = (
                    rotten_neighbors
                    /
                    total_neighbors
                )

            else:

                fresh_ratio = 0.0

                rotten_ratio = 0.0

            self.last_tomato_similarity = (
                nearest_similarity
            )

            print()
            print(
                "[TOMATO SIMILARITY]"
            )

            print(
                "Nearest similarity:",
                f"{nearest_similarity:.4f}",
            )

            print(
                "Threshold:",
                TOMATO_SIMILARITY_THRESHOLD,
            )

            print(
                "Meaningful neighbors:",
                meaningful_neighbors,
            )

            print(
                "Fresh references:",
                fresh_neighbors,
            )

            print(
                "Rotten references:",
                rotten_neighbors,
            )

            return {

                "available":
                    True,

                "nearest_similarity":
                    round(
                        nearest_similarity,
                        4,
                    ),

                "top_matches":
                    top_matches,

                "meaningful_neighbors":
                    meaningful_neighbors,

                "fresh_neighbors":
                    fresh_neighbors,

                "rotten_neighbors":
                    rotten_neighbors,

                "fresh_ratio":
                    round(
                        fresh_ratio,
                        4,
                    ),

                "rotten_ratio":
                    round(
                        rotten_ratio,
                        4,
                    ),
            }

        except Exception as exc:

            print(
                "[SIMILARITY ERROR]",
                repr(exc),
            )

            return result

    # ========================================================
    # BINARY TOMATO VERIFIER
    # ========================================================

    def _run_tomato_verifier(
        self,
        image: np.ndarray,
    ) -> Dict[str, Any]:

        result = {

            "available":
                False,

            "tomato_probability":
                0.0,

            "non_tomato_probability":
                1.0,

            "tomato_confidence":
                0.0,

            "verified":
                False,
        }

        if self.tomato_verifier is None:

            return result

        try:

            batch = np.expand_dims(
                image,
                axis=0,
            )

            prediction = (
                self.tomato_verifier.predict(
                    batch,
                    verbose=0,
                )
            )

            prediction = np.asarray(
                prediction,
                dtype=np.float32,
            )

            if prediction.size == 0:

                return result

            probability = _clip_probability(
                prediction.reshape(
                    -1
                )[0]
            )

            self.last_tomato_probability = (
                probability
            )

            verified = (
                probability
                >=
                TOMATO_THRESHOLD
            )

            confidence = (
                probability
                if verified
                else
                1.0 - probability
            )

            print()
            print(
                "[BINARY TOMATO VERIFIER]"
            )

            print(
                "Tomato probability:",
                f"{probability * 100.0:.2f}%",
            )

            print(
                "Non-tomato probability:",
                f"{(1.0 - probability) * 100.0:.2f}%",
            )

            print(
                "Decision:",
                "TOMATO"
                if verified
                else "NON-TOMATO",
            )

            return {

                "available":
                    True,

                "tomato_probability":
                    round(
                        probability,
                        6,
                    ),

                "non_tomato_probability":
                    round(
                        1.0 - probability,
                        6,
                    ),

                "tomato_confidence":
                    round(
                        confidence,
                        6,
                    ),

                "verified":
                    verified,
            }

        except Exception as exc:

            print(
                "[VERIFIER ERROR]",
                repr(exc),
            )

            traceback.print_exc()

            return result

    # ========================================================
    # FINAL TOMATO GATE
    # ========================================================

    def _verify_tomato(
        self,
        similarity_result: Dict[str, Any],
        verifier_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        similarity_available = bool(
            similarity_result.get(
                "available",
                False,
            )
        )

        nearest_similarity = _safe_float(
            similarity_result.get(
                "nearest_similarity",
                0.0,
            )
        )

        meaningful_neighbors = int(
            similarity_result.get(
                "meaningful_neighbors",
                0,
            )
        )

        verifier_available = bool(
            verifier_result.get(
                "available",
                False,
            )
        )

        verifier_probability = _safe_float(
            verifier_result.get(
                "tomato_probability",
                0.0,
            )
        )

        verifier_verified = bool(
            verifier_result.get(
                "verified",
                False,
            )
        )

        # ====================================================
        # SAFETY-FIRST DECISION
        # ====================================================
        #
        # Similarity index is the main tomato-domain check.
        #
        # The binary verifier is additional evidence.
        #
        # If both are available:
        #
        #   Strong similarity + verifier
        #       -> tomato
        #
        #   Strong similarity alone
        #       -> tomato
        #
        #   Weak similarity
        #       -> reject
        #
        # This prevents the Fresh/Rotten model from seeing
        # arbitrary vegetables.
        # ====================================================

        strong_similarity = (
            nearest_similarity
            >=
            STRONG_TOMATO_SIMILARITY
        )

        normal_similarity = (
            nearest_similarity
            >=
            TOMATO_SIMILARITY_THRESHOLD
        )

        enough_neighbors = (
            meaningful_neighbors >= 3
        )

        verified = False

        reason = ""

        # ----------------------------------------------------
        # Similarity available
        # ----------------------------------------------------

        if similarity_available:

            # Strong visual match
            if strong_similarity:

                verified = True

                if (
                    verifier_available
                    and
                    verifier_verified
                ):

                    reason = (
                        "Strong tomato visual similarity "
                        "and tomato verifier both passed."
                    )

                else:

                    reason = (
                        "Strong visual similarity to the "
                        "trained tomato reference database."
                    )

            # Moderate match + several references
            elif (
                normal_similarity
                and
                enough_neighbors
            ):

                # If verifier is available, require it too.
                if verifier_available:

                    if verifier_verified:

                        verified = True

                        reason = (
                            "Tomato similarity and binary "
                            "tomato verification both passed."
                        )

                    else:

                        verified = False

                        reason = (
                            "Tomato similarity was moderate, "
                            "but the binary tomato verifier "
                            "rejected the image."
                        )

                else:

                    verified = True

                    reason = (
                        "Multiple tomato reference images "
                        "support the uploaded image."
                    )

            else:

                verified = False

                reason = (
                    "The uploaded image does not have "
                    "sufficient similarity to the trained "
                    "tomato reference images."
                )

        # ----------------------------------------------------
        # Similarity unavailable
        # ----------------------------------------------------

        else:

            # Do NOT silently trust quality model.
            # Only use the binary verifier as fallback.

            if (
                verifier_available
                and
                verifier_verified
            ):

                verified = True

                reason = (
                    "Tomato similarity index unavailable; "
                    "binary tomato verifier passed."
                )

            else:

                verified = False

                reason = (
                    "Tomato verification failed because "
                    "there is insufficient tomato evidence."
                )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        if similarity_available:

            tomato_confidence = (
                nearest_similarity
            )

        elif verifier_available:

            tomato_confidence = (
                verifier_probability
            )

        else:

            tomato_confidence = 0.0

        tomato_confidence = float(
            np.clip(
                tomato_confidence,
                0.0,
                1.0,
            )
        )

        print()
        print("=" * 70)
        print("FINAL TOMATO VERIFICATION")
        print("=" * 70)

        print(
            "Nearest similarity:",
            f"{nearest_similarity:.4f}",
        )

        print(
            "Similarity threshold:",
            TOMATO_SIMILARITY_THRESHOLD,
        )

        print(
            "Strong threshold:",
            STRONG_TOMATO_SIMILARITY,
        )

        print(
            "Meaningful neighbors:",
            meaningful_neighbors,
        )

        print(
            "Verifier probability:",
            f"{verifier_probability:.4f}",
        )

        print(
            "FINAL DECISION:",
            "TOMATO"
            if verified
            else "NON-TOMATO",
        )

        print(
            "Reason:",
            reason,
        )

        print("=" * 70)

        return {

            "is_tomato":
                verified,

            "tomato_verified":
                verified,

            "verified":
                verified,

            "tomato_probability":
                verifier_probability,

            "tomato_confidence":
                tomato_confidence,

            "nearest_similarity":
                nearest_similarity,

            "meaningful_neighbors":
                meaningful_neighbors,

            "reason":
                reason,
        }

    # ========================================================
    # QUALITY PREDICTION
    # ========================================================

    def _predict_quality(
        self,
        image: np.ndarray,
    ) -> Dict[str, Any]:

        if self.quality_model is None:

            raise RuntimeError(
                "Quality model is not loaded."
            )

        batch = np.expand_dims(
            image,
            axis=0,
        )

        prediction = (
            self.quality_model.predict(
                batch,
                verbose=0,
            )
        )

        prediction = np.asarray(
            prediction,
            dtype=np.float32,
        )

        if prediction.size == 0:

            raise ValueError(
                "Quality model returned empty prediction."
            )

        value = _clip_probability(
            prediction.reshape(
                -1
            )[0]
        )

        # Existing model:
        #
        # 0 = Fresh
        # 1 = Rotten

        rotten_probability = value

        fresh_probability = (
            1.0
            -
            rotten_probability
        )

        if (
            rotten_probability
            >=
            ROTTEN_THRESHOLD
        ):

            quality_label = "Rotten"

        else:

            quality_label = "Fresh"

        freshness = (
            fresh_probability
            *
            100.0
        )

        quality = freshness

        confidence = (
            max(
                fresh_probability,
                rotten_probability,
            )
            *
            100.0
        )

        print()
        print(
            "[QUALITY]"
        )

        print(
            "Quality:",
            quality_label,
        )

        print(
            "Fresh probability:",
            f"{fresh_probability * 100.0:.2f}%",
        )

        print(
            "Rotten probability:",
            f"{rotten_probability * 100.0:.2f}%",
        )

        print(
            "Freshness:",
            f"{freshness:.2f}%",
        )

        return {

            "quality_label":
                quality_label,

            "quality":
                round(
                    quality,
                    2,
                ),

            "freshness":
                round(
                    freshness,
                    2,
                ),

            "confidence":
                round(
                    confidence,
                    2,
                ),

            "fresh_probability":
                round(
                    fresh_probability * 100.0,
                    2,
                ),

            "rotten_probability":
                round(
                    rotten_probability * 100.0,
                    2,
                ),

            "classifier_rotten_probability":
                round(
                    rotten_probability,
                    6,
                ),
        }

    # ========================================================
    # MATURITY PREDICTION
    # ========================================================

    def _predict_maturity(
        self,
        image: np.ndarray,
    ) -> Dict[str, Any]:

        if self.maturity_model is None:

            return {

                "maturity":
                    "Unknown",

                "maturity_label":
                    "Unknown",

                "maturity_confidence":
                    0.0,

                "mature_probability":
                    0.0,

                "immature_probability":
                    0.0,

                "maturity_probability":
                    0.0,

                "maturity_probabilities":
                    {
                        "mature": 0.0,
                        "immature": 0.0,
                    },

                "ripeness":
                    "Unknown",
            }

        batch = np.expand_dims(
            image,
            axis=0,
        )

        prediction = (
            self.maturity_model.predict(
                batch,
                verbose=0,
            )
        )

        prediction = np.asarray(
            prediction,
            dtype=np.float32,
        )

        if prediction.size == 0:

            raise ValueError(
                "Maturity model returned empty prediction."
            )

        value = _clip_probability(
            prediction.reshape(
                -1
            )[0]
        )

        # Existing maturity model:
        #
        # 0 = immature
        # 1 = mature

        mature_probability = value

        immature_probability = (
            1.0
            -
            mature_probability
        )

        if (
            mature_probability
            >=
            MATURITY_THRESHOLD
        ):

            maturity = "Mature"

            confidence = (
                mature_probability
            )

        else:

            maturity = "Immature"

            confidence = (
                immature_probability
            )

        print()
        print(
            "[MATURITY]"
        )

        print(
            "Maturity:",
            maturity,
        )

        print(
            "Confidence:",
            f"{confidence * 100.0:.2f}%",
        )

        return {

            "maturity":
                maturity,

            "maturity_label":
                maturity,

            "maturity_confidence":
                round(
                    confidence * 100.0,
                    2,
                ),

            "mature_probability":
                round(
                    mature_probability,
                    6,
                ),

            "immature_probability":
                round(
                    immature_probability,
                    6,
                ),

            "maturity_probability":
                round(
                    mature_probability,
                    6,
                ),

            "maturity_probabilities":
                {
                    "mature":
                        round(
                            mature_probability,
                            6,
                        ),

                    "immature":
                        round(
                            immature_probability,
                            6,
                        ),
                },

            "ripeness":
                maturity,
        }

    # ========================================================
    # GRADE
    # ========================================================

    def _calculate_grade(
        self,
        quality_result: Dict[str, Any],
        maturity_result: Dict[str, Any],
    ) -> str:

        quality_label = str(
            quality_result.get(
                "quality_label",
                "Unknown",
            )
        )

        freshness = _safe_float(
            quality_result.get(
                "freshness",
                0.0,
            )
        )

        maturity = str(
            maturity_result.get(
                "maturity",
                "Unknown",
            )
        )

        # ----------------------------------------------------
        # Rotten tomato
        # ----------------------------------------------------

        if quality_label == "Rotten":

            return "Rejected"

        # ----------------------------------------------------
        # Unknown
        # ----------------------------------------------------

        if quality_label == "Unknown":

            return "Rejected"

        # ----------------------------------------------------
        # Immature
        # ----------------------------------------------------

        if maturity == "Immature":

            if freshness >= GRADE_B_MIN:

                return "Grade B"

            if freshness >= GRADE_C_MIN:

                return "Grade C"

            return "Rejected"

        # ----------------------------------------------------
        # Mature + Fresh
        # ----------------------------------------------------

        if freshness >= GRADE_A_MIN:

            return "Grade A"

        if freshness >= GRADE_B_MIN:

            return "Grade B"

        if freshness >= GRADE_C_MIN:

            return "Grade C"

        return "Rejected"

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    def _recommendation(
        self,
        quality_result: Dict[str, Any],
        maturity_result: Dict[str, Any],
        grade: str,
    ) -> str:

        quality_label = str(
            quality_result.get(
                "quality_label",
                "Unknown",
            )
        )

        freshness = _safe_float(
            quality_result.get(
                "freshness",
                0.0,
            )
        )

        maturity = str(
            maturity_result.get(
                "maturity",
                "Unknown",
            )
        )

        maturity_confidence = _safe_float(
            maturity_result.get(
                "maturity_confidence",
                0.0,
            )
        )

        if quality_label == "Rotten":

            return (
                "The uploaded image contains a tomato, "
                "but it appears rotten or has significant "
                "quality defects. It is not recommended "
                "for sale as fresh produce."
            )

        if maturity == "Immature":

            return (
                "The tomato was verified successfully. "
                f"Freshness is {freshness:.1f}% and the "
                f"tomato is immature with "
                f"{maturity_confidence:.1f}% confidence. "
                f"Classification: {grade}."
            )

        if grade == "Grade A":

            return (
                "The tomato was verified successfully "
                "and classified as fresh, mature and "
                "high quality. Grade A."
            )

        if grade == "Grade B":

            return (
                "The tomato was verified successfully. "
                "It is suitable for sale but has lower "
                "quality than Grade A."
            )

        if grade == "Grade C":

            return (
                "The tomato was verified successfully, "
                "but its quality score is relatively low. "
                "Further inspection is recommended."
            )

        return (
            "The tomato was verified, but it does not "
            "meet the required quality threshold."
        )

    # ========================================================
    # REJECTED RESULT
    # ========================================================

    def _rejected_result(
        self,
        verification: Dict[str, Any],
    ) -> Dict[str, Any]:

        probability = _safe_float(
            verification.get(
                "tomato_probability",
                0.0,
            )
        )

        confidence = _safe_float(
            verification.get(
                "tomato_confidence",
                0.0,
            )
        )

        similarity = _safe_float(
            verification.get(
                "nearest_similarity",
                0.0,
            )
        )

        reason = str(
            verification.get(
                "reason",
                "The uploaded image does not appear "
                "to be a tomato.",
            )
        )

        print()
        print("=" * 70)
        print("[RESULT] NON-TOMATO / REJECTED")
        print("=" * 70)

        print(
            "Tomato probability:",
            f"{probability * 100.0:.2f}%",
        )

        print(
            "Tomato similarity:",
            f"{similarity:.4f}",
        )

        print(
            "Reason:",
            reason,
        )

        print("=" * 70)

        return {

            # ------------------------------------------------
            # Crop
            # ------------------------------------------------

            "crop":
                "Unknown",

            "crop_name":
                "Unknown",

            "crop_type":
                "unknown",

            # ------------------------------------------------
            # Tomato verification
            # ------------------------------------------------

            "is_tomato":
                False,

            "tomato_verified":
                False,

            "tomato_probability":
                round(
                    probability,
                    6,
                ),

            "non_tomato_probability":
                round(
                    1.0 - probability,
                    6,
                ),

            "tomato_confidence":
                round(
                    confidence,
                    6,
                ),

            "tomato_threshold":
                TOMATO_THRESHOLD,

            "nearest_similarity":
                round(
                    similarity,
                    4,
                ),

            "similarity":
                round(
                    similarity,
                    4,
                ),

            "similarity_threshold":
                TOMATO_SIMILARITY_THRESHOLD,

            # ------------------------------------------------
            # Analysis
            # ------------------------------------------------

            "analysis_status":
                "REJECTED",

            "visual_rejection":
                True,

            "rejection_reason":
                reason,

            "rejection_reasons":
                [
                    reason
                ],

            # ------------------------------------------------
            # Grade
            # ------------------------------------------------

            "grade":
                "Rejected",

            "grade_label":
                "Rejected",

            "confidence":
                0.0,

            # ------------------------------------------------
            # Quality
            # ------------------------------------------------

            "quality":
                0.0,

            "quality_score":
                0.0,

            "quality_label":
                "Unknown",

            "freshness":
                0.0,

            "fresh_probability":
                0.0,

            "rotten_probability":
                0.0,

            # ------------------------------------------------
            # Maturity
            # ------------------------------------------------

            "maturity":
                "Unknown",

            "maturity_label":
                "Unknown",

            "maturity_confidence":
                0.0,

            "mature_probability":
                0.0,

            "immature_probability":
                0.0,

            "maturity_probability":
                0.0,

            "maturity_probabilities":
                {
                    "mature":
                        0.0,

                    "immature":
                        0.0,
                },

            "ripeness":
                "Unknown",

            # ------------------------------------------------
            # Defects
            # ------------------------------------------------

            "defect_detected":
                False,

            "defect_type":
                None,

            # ------------------------------------------------
            # Similarity compatibility
            # ------------------------------------------------

            "top_similar_images":
                [],

            "rotten_neighbors":
                0,

            "fresh_neighbors":
                0,

            "total_neighbors":
                0,

            "rotten_ratio":
                0.0,

            "fresh_ratio":
                0.0,

            # ------------------------------------------------
            # Market compatibility
            # ------------------------------------------------

            "price":
                0.0,

            "best_market":
                {
                    "name":
                        "Not recommended",

                    "distance_km":
                        0.0,

                    "travel_time":
                        "N/A",
                },

            # ------------------------------------------------
            # Recommendation
            # ------------------------------------------------

            "recommendation":
                (
                    "The uploaded image does not appear "
                    "to contain a tomato. Tomato quality "
                    "and maturity analysis was not performed. "
                    "Please upload a clear tomato image."
                ),

            # ------------------------------------------------
            # Class compatibility
            # ------------------------------------------------

            "class_name":
                None,

            "class_probabilities":
                {},

            "classifier_rotten_probability":
                0.0,

            "similarity_rotten_score":
                0.0,

            "combined_rotten_probability":
                0.0,

            # ------------------------------------------------
            # Verification object
            # ------------------------------------------------

            "tomato_verification":
                {
                    "verified":
                        False,

                    "tomato_probability":
                        probability,

                    "tomato_confidence":
                        confidence,

                    "nearest_similarity":
                        similarity,

                    "threshold":
                        TOMATO_THRESHOLD,

                    "reason":
                        reason,
                },
        }

    # ========================================================
    # MAIN PREDICTION
    # ========================================================

    def predict(
        self,
        image_path: str,
    ) -> Dict[str, Any]:

        """
        SYNCHRONOUS prediction.

        This function must remain synchronous because
        existing AgriNova backend code calls it directly.
        """

        # ----------------------------------------------------
        # Validate path
        # ----------------------------------------------------

        if not image_path:

            return self._rejected_result(
                {
                    "reason":
                        "No image was provided.",
                }
            )

        image_file = Path(
            image_path
        )

        if not image_file.exists():

            return self._rejected_result(
                {
                    "reason":
                        "Image file was not found.",
                }
            )

        print()
        print("=" * 70)
        print("AGRINOVA TOMATO IMAGE ANALYSIS")
        print("=" * 70)

        print(
            "[IMAGE]",
            image_file,
        )

        # ====================================================
        # LOAD MODELS
        # ====================================================

        if not self.models_loaded:

            print(
                "[AI] Models are not ready."
            )

            print(
                "[AI] Loading models..."
            )

            if not self.load_model():

                raise RuntimeError(
                    "Agrinova AI models could not be loaded."
                )

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        try:

            image = (
                self._load_image(
                    str(image_file)
                )
            )

        except Exception as exc:

            print(
                "[IMAGE ERROR]",
                repr(exc),
            )

            return self._rejected_result(
                {
                    "reason":
                        "The uploaded image could not be read.",
                }
            )

        # ====================================================
        # STEP 1
        # TOMATO SIMILARITY
        # ====================================================

        similarity_result = (
            self._calculate_tomato_similarity(
                image
            )
        )

        # ====================================================
        # STEP 2
        # BINARY TOMATO VERIFIER
        # ====================================================

        verifier_result = (
            self._run_tomato_verifier(
                image
            )
        )

        # ====================================================
        # STEP 3
        # FINAL TOMATO GATE
        # ====================================================

        verification = (
            self._verify_tomato(
                similarity_result,
                verifier_result,
            )
        )

        # ====================================================
        # REJECT NON-TOMATO
        # ====================================================

        if not verification["is_tomato"]:

            print()
            print(
                "[TOMATO GATE] ❌ NON-TOMATO"
            )

            print(
                "[TOMATO GATE] Quality model will NOT run."
            )

            print(
                "[TOMATO GATE] Maturity model will NOT run."
            )

            return self._rejected_result(
                verification
            )

        # ====================================================
        # VERIFIED TOMATO
        # ====================================================

        print()
        print(
            "[TOMATO GATE] ✅ TOMATO VERIFIED"
        )

        # ====================================================
        # STEP 4
        # QUALITY
        # ====================================================

        quality_result = (
            self._predict_quality(
                image
            )
        )

        # ====================================================
        # STEP 5
        # MATURITY
        # ====================================================

        maturity_result = (
            self._predict_maturity(
                image
            )
        )

        # ====================================================
        # STEP 6
        # GRADE
        # ====================================================

        grade = (
            self._calculate_grade(
                quality_result,
                maturity_result,
            )
        )

        # ====================================================
        # STATUS
        # ====================================================

        quality_label = str(
            quality_result.get(
                "quality_label",
                "Unknown",
            )
        )

        if (
            quality_label
            ==
            "Rotten"
        ):

            analysis_status = (
                "REJECTED"
            )

        elif grade in (
            "Grade A",
            "Grade B",
            "Grade C",
        ):

            analysis_status = (
                "ACCEPTED"
            )

        else:

            analysis_status = (
                "REJECTED"
            )

        # ====================================================
        # RECOMMENDATION
        # ====================================================

        recommendation = (
            self._recommendation(
                quality_result,
                maturity_result,
                grade,
            )
        )

        # ====================================================
        # SIMILARITY VALUES
        # ====================================================

        nearest_similarity = _safe_float(
            similarity_result.get(
                "nearest_similarity",
                0.0,
            )
        )

        top_matches = (
            similarity_result.get(
                "top_matches",
                [],
            )
        )

        fresh_neighbors = int(
            similarity_result.get(
                "fresh_neighbors",
                0,
            )
        )

        rotten_neighbors = int(
            similarity_result.get(
                "rotten_neighbors",
                0,
            )
        )

        total_neighbors = (
            fresh_neighbors
            +
            rotten_neighbors
        )

        # ====================================================
        # QUALITY VALUES
        # ====================================================

        quality = _safe_float(
            quality_result.get(
                "quality",
                0.0,
            )
        )

        freshness = _safe_float(
            quality_result.get(
                "freshness",
                0.0,
            )
        )

        confidence = _safe_float(
            quality_result.get(
                "confidence",
                0.0,
            )
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        result = {

            # ------------------------------------------------
            # Crop
            # ------------------------------------------------

            "crop":
                "Tomato",

            "crop_name":
                "Tomato",

            "crop_type":
                "tomato",

            # ------------------------------------------------
            # Tomato verification
            # ------------------------------------------------

            "is_tomato":
                True,

            "tomato_verified":
                True,

            "tomato_probability":
                verification.get(
                    "tomato_probability",
                    0.0,
                ),

            "non_tomato_probability":
                1.0
                -
                _safe_float(
                    verification.get(
                        "tomato_probability",
                        0.0,
                    )
                ),

            "tomato_confidence":
                verification.get(
                    "tomato_confidence",
                    0.0,
                ),

            "tomato_threshold":
                TOMATO_THRESHOLD,

            "tomato_verification_reason":
                verification.get(
                    "reason",
                    "",
                ),

            # ------------------------------------------------
            # Grade
            # ------------------------------------------------

            "grade":
                grade,

            "grade_label":
                grade,

            # ------------------------------------------------
            # Quality
            # ------------------------------------------------

            "confidence":
                confidence,

            "quality":
                quality,

            "quality_score":
                quality,

            "quality_label":
                quality_label,

            "freshness":
                freshness,

            "fresh_probability":
                quality_result.get(
                    "fresh_probability",
                    0.0,
                ),

            "rotten_probability":
                quality_result.get(
                    "rotten_probability",
                    0.0,
                ),

            # ------------------------------------------------
            # Maturity
            # ------------------------------------------------

            "maturity":
                maturity_result.get(
                    "maturity",
                    "Unknown",
                ),

            "maturity_label":
                maturity_result.get(
                    "maturity_label",
                    "Unknown",
                ),

            "maturity_confidence":
                maturity_result.get(
                    "maturity_confidence",
                    0.0,
                ),

            "mature_probability":
                maturity_result.get(
                    "mature_probability",
                    0.0,
                ),

            "immature_probability":
                maturity_result.get(
                    "immature_probability",
                    0.0,
                ),

            "maturity_probability":
                maturity_result.get(
                    "maturity_probability",
                    0.0,
                ),

            "maturity_probabilities":
                maturity_result.get(
                    "maturity_probabilities",
                    {},
                ),

            "ripeness":
                maturity_result.get(
                    "ripeness",
                    "Unknown",
                ),

            "maturity_message":
                (
                    "The tomato has reached full maturity."
                    if
                    maturity_result.get(
                        "maturity"
                    )
                    ==
                    "Mature"
                    else
                    "The tomato is still immature."
                ),

            # ------------------------------------------------
            # Similarity
            # ------------------------------------------------

            "nearest_similarity":
                round(
                    nearest_similarity,
                    4,
                ),

            "similarity":
                round(
                    nearest_similarity,
                    4,
                ),

            "similarity_threshold":
                TOMATO_SIMILARITY_THRESHOLD,

            "top_similar_images":
                top_matches,

            # ------------------------------------------------
            # Neighbours
            # ------------------------------------------------

            "rotten_neighbors":
                rotten_neighbors,

            "fresh_neighbors":
                fresh_neighbors,

            "total_neighbors":
                total_neighbors,

            "rotten_ratio":
                similarity_result.get(
                    "rotten_ratio",
                    0.0,
                ),

            "fresh_ratio":
                similarity_result.get(
                    "fresh_ratio",
                    0.0,
                ),

            # ------------------------------------------------
            # Defects
            # ------------------------------------------------

            "defect_detected":
                quality_label
                ==
                "Rotten",

            "defect_type":
                (
                    "Rotten / quality defect"
                    if quality_label
                    ==
                    "Rotten"
                    else None
                ),

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            "analysis_status":
                analysis_status,

            "visual_rejection":
                False,

            "rejection_reason":
                (
                    "Quality below required threshold."
                    if analysis_status
                    ==
                    "REJECTED"
                    else None
                ),

            "rejection_reasons":
                [],

            # ------------------------------------------------
            # Market compatibility
            # ------------------------------------------------

            "price":
                0.0,

            "best_market":
                {
                    "name":
                        "Available through Nearby Mandi",

                    "distance_km":
                        0.0,

                    "travel_time":
                        "N/A",
                },

            # ------------------------------------------------
            # Recommendation
            # ------------------------------------------------

            "recommendation":
                recommendation,

            # ------------------------------------------------
            # Compatibility
            # ------------------------------------------------

            "class_name":
                (
                    "rotten"
                    if quality_label
                    ==
                    "Rotten"
                    else
                    "fresh"
                ),

            "class_probabilities":
                {
                    "fresh":
                        _safe_float(
                            quality_result.get(
                                "fresh_probability",
                                0.0,
                            )
                        )
                        /
                        100.0,

                    "rotten":
                        _safe_float(
                            quality_result.get(
                                "rotten_probability",
                                0.0,
                            )
                        )
                        /
                        100.0,
                },

            "classifier_rotten_probability":
                quality_result.get(
                    "classifier_rotten_probability",
                    0.0,
                ),

            "similarity_rotten_score":
                similarity_result.get(
                    "rotten_ratio",
                    0.0,
                ),

            "combined_rotten_probability":
                quality_result.get(
                    "classifier_rotten_probability",
                    0.0,
                ),

            # ------------------------------------------------
            # Verification object
            # ------------------------------------------------

            "tomato_verification":
                {
                    "verified":
                        True,

                    "tomato_probability":
                        verification.get(
                            "tomato_probability",
                            0.0,
                        ),

                    "tomato_confidence":
                        verification.get(
                            "tomato_confidence",
                            0.0,
                        ),

                    "nearest_similarity":
                        nearest_similarity,

                    "threshold":
                        TOMATO_THRESHOLD,

                    "similarity_threshold":
                        TOMATO_SIMILARITY_THRESHOLD,

                    "reason":
                        verification.get(
                            "reason",
                            "",
                        ),
                },
        }

        # ====================================================
        # LOG RESULT
        # ====================================================

        print()
        print("=" * 70)
        print("FINAL AGRINOVA RESULT")
        print("=" * 70)

        print(
            "Tomato:",
            True,
        )

        print(
            "Tomato similarity:",
            f"{nearest_similarity:.4f}",
        )

        print(
            "Quality:",
            quality_label,
        )

        print(
            "Freshness:",
            f"{freshness:.2f}%",
        )

        print(
            "Maturity:",
            maturity_result.get(
                "maturity",
                "Unknown",
            ),
        )

        print(
            "Grade:",
            grade,
        )

        print(
            "Status:",
            analysis_status,
        )

        print("=" * 70)

        return result

    # ========================================================
    # STATUS
    # ========================================================

    def status(
        self,
    ) -> Dict[str, Any]:

        return {

            "is_loaded":
                self.is_loaded,

            "models_loaded":
                self.models_loaded,

            "quality_loaded":
                self.quality_loaded,

            "maturity_loaded":
                self.maturity_loaded,

            "verifier_loaded":
                self.verifier_loaded,

            "similarity_loaded":
                self.similarity_loaded,

            "feature_model_loaded":
                self.feature_model_loaded,

            "quality_model_path":
                str(
                    QUALITY_MODEL_PATH
                ),

            "maturity_model_path":
                str(
                    MATURITY_MODEL_PATH
                ),

            "tomato_verifier_path":
                str(
                    TOMATO_VERIFIER_PATH
                ),

            "similarity_index_path":
                str(
                    self.similarity_index_path
                ),

            "tomato_threshold":
                TOMATO_THRESHOLD,

            "similarity_threshold":
                TOMATO_SIMILARITY_THRESHOLD,

            "feature_dimension":
                EXPECTED_FEATURE_SIZE,
        }

    # ========================================================
    # COMPATIBILITY
    # ========================================================

    get_status = status


# ============================================================
# SINGLETON
# ============================================================

crop_classifier = (
    CropClassifier()
)


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def predict_crop(
    image_path: str,
) -> Dict[str, Any]:

    return crop_classifier.predict(
        image_path
    )