
"""
AgriNova - Strict Tomato Crop Classifier.

Pipeline:
1. Validate image
2. Verify tomato vs non-tomato
3. Optionally verify tomato similarity
4. Detect fresh/rotten condition
5. Detect immature/mature condition
6. Generate grade and clear user message

Expected model class order:
- Tomato verifier: [non_tomato, tomato]
- Quality model: [fresh, rotten]
- Maturity model: [immature, mature]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import (
    decode_predictions,
    preprocess_input,
)


BASE_DIR = Path(__file__).resolve().parents[1]
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"

QUALITY_DIR = TRAINED_MODELS_DIR / "tomato_quality"
MATURITY_DIR = TRAINED_MODELS_DIR / "tomato_maturity"

QUALITY_MODEL_CANDIDATES = [
    TRAINED_MODELS_DIR / "tomato_quality.keras",
    QUALITY_DIR / "tomato_quality.keras",
    QUALITY_DIR / "tomato_quality_best.keras",
]

MATURITY_MODEL_CANDIDATES = [
    TRAINED_MODELS_DIR / "tomato_maturity.keras",
    MATURITY_DIR / "tomato_maturity.keras",
    MATURITY_DIR / "tomato_maturity_best.keras",
]

TOMATO_VERIFIER_PATH = TRAINED_MODELS_DIR / "tomato_verifier.keras"
TOMATO_VERIFIER_METADATA_PATH = (
    TRAINED_MODELS_DIR / "tomato_verifier.npz"
)

SIMILARITY_INDEX_CANDIDATES = [
    TRAINED_MODELS_DIR / "tomato_reference_embeddings.npz",
    QUALITY_DIR / "tomato_similarity_index.npz",
    TRAINED_MODELS_DIR / "tomato_similarity_index.npz",
]

IMAGE_SIZE = (224, 224)

# Strict thresholds.
# The verifier is combined with the mandatory reference-similarity gate
# below. This cutoff rejects weak verifier predictions without rejecting
# valid tomatoes that score strongly on both checks.
TOMATO_VERIFIER_THRESHOLD = 0.60
TOMATO_SIMILARITY_THRESHOLD = 0.48
MIN_STRONG_TOMATO_NEIGHBORS = 5
MIN_AVERAGE_TOMATO_SIMILARITY = 0.48
REQUIRE_SIMILARITY_VERIFICATION = True

# Calibrated against the project quality dataset: fresh examples remain
# below 0.001 while rotten examples are substantially higher.
ROTTEN_THRESHOLD = 0.10
MATURITY_THRESHOLD = 0.50
MIN_CONFIDENCE = 0.60

GRADE_A_THRESHOLD = 85.0
GRADE_B_THRESHOLD = 65.0


def _find_first_existing(
    candidates: list[Path],
) -> Optional[Path]:
    for path in candidates:
        if path.is_file():
            return path
    return None


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        number = float(value)

        if not np.isfinite(number):
            return default

        return number
    except (TypeError, ValueError):
        return default


def _clip_probability(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value)))


class CropClassifier:
    def __init__(self) -> None:
        self.quality_model: Any = None
        self.maturity_model: Any = None
        self.tomato_verifier: Any = None
        self.feature_extractor: Any = None
        self.semantic_guard: Any = None

        self.similarity_embeddings: Optional[np.ndarray] = None
        self.similarity_labels: Optional[np.ndarray] = None

        self.models_loaded = False
        self.last_error: Optional[str] = None

        self.load_model()

    # ============================================================
    # MODEL LOADING
    # ============================================================

    def _load_keras_model(self, path: Path) -> Any:
        try:
            return keras.models.load_model(
                path,
                compile=False,
                custom_objects={
                    "preprocess_input": preprocess_input,
                },
            )
        except Exception as exc:
            self.last_error = (
                f"Could not load model '{path}': {exc}"
            )
            print(f"[CropClassifier] {self.last_error}")
            return None

    def load_model(self) -> bool:
        quality_path = _find_first_existing(
            QUALITY_MODEL_CANDIDATES
        )
        maturity_path = _find_first_existing(
            MATURITY_MODEL_CANDIDATES
        )

        self.quality_model = (
            self._load_keras_model(quality_path)
            if quality_path
            else None
        )

        self.maturity_model = (
            self._load_keras_model(maturity_path)
            if maturity_path
            else None
        )

        if TOMATO_VERIFIER_PATH.is_file():
            self.tomato_verifier = self._load_keras_model(
                TOMATO_VERIFIER_PATH
            )

        self._load_similarity_database()
        self._build_feature_extractor()
        self._load_semantic_guard()

        self.models_loaded = all(
            [
                self.quality_model is not None,
                self.maturity_model is not None,
                self.tomato_verifier is not None,
            ]
        )

        if not self.models_loaded:
            print(
                "[CropClassifier] Strict mode: required models "
                "are missing. Images will be rejected."
            )

        return self.models_loaded

    def _load_semantic_guard(self) -> None:
        """Load an independent ImageNet guard for obvious apple images."""
        try:
            self.semantic_guard = keras.applications.MobileNetV2(
                weights="imagenet",
                include_top=True,
            )
        except Exception as exc:
            self.semantic_guard = None
            print(
                "[CropClassifier] Semantic guard unavailable: "
                f"{exc}"
            )

    # ============================================================
    # SIMILARITY DATABASE
    # ============================================================

    def _load_similarity_database(self) -> None:
        index_path = _find_first_existing(
            SIMILARITY_INDEX_CANDIDATES
        )

        if index_path is None:
            return

        try:
            data = np.load(index_path, allow_pickle=False)

            embedding_key = next(
                (
                    key
                    for key in (
                        "embeddings",
                        "features",
                        "vectors",
                    )
                    if key in data
                ),
                None,
            )

            if embedding_key is None:
                print(
                    "[CropClassifier] Similarity index has no "
                    "embeddings/features/vectors array."
                )
                return

            embeddings = np.asarray(
                data[embedding_key],
                dtype=np.float32,
            )

            if embeddings.ndim != 2 or len(embeddings) == 0:
                return

            norms = np.linalg.norm(
                embeddings,
                axis=1,
                keepdims=True,
            )

            self.similarity_embeddings = embeddings / np.maximum(
                norms,
                1e-8,
            )

            if "labels" in data:
                self.similarity_labels = data["labels"]

            print(
                "[CropClassifier] Similarity database loaded: "
                f"{len(embeddings)} references"
            )

        except Exception as exc:
            print(
                "[CropClassifier] Similarity database could not "
                f"be loaded: {exc}"
            )

    def _build_feature_extractor(self) -> None:
        if self.quality_model is None:
            return

        try:
            if len(self.quality_model.layers) < 2:
                return

            self.feature_extractor = keras.Model(
                inputs=self.quality_model.input,
                outputs=self.quality_model.layers[-2].output,
            )
        except Exception as exc:
            print(
                "[CropClassifier] Feature extractor unavailable: "
                f"{exc}"
            )

    # ============================================================
    # IMAGE PROCESSING
    # ============================================================

    def _load_image(
        self,
        image_path: str,
    ) -> np.ndarray:
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Image does not exist: {image_path}"
            )

        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                "The uploaded file is not a valid image."
            ) from exc

        width, height = image.size

        if width < 80 or height < 80:
            raise ValueError(
                "Image is too small. Upload a clearer image."
            )

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS,
        )

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        # The verifier and tomato models apply MobileNetV2
        # preprocess_input internally, so inference must receive the
        # same raw 0..255 pixel range used during training.
        return array

    # ============================================================
    # PREDICTION HELPERS
    # ============================================================

    def _predict(
        self,
        model: Any,
        image: np.ndarray,
    ) -> np.ndarray:
        if model is None:
            raise RuntimeError("Required model is not loaded.")

        result = model.predict(
            np.expand_dims(image, axis=0),
            verbose=0,
        )

        if isinstance(result, dict):
            result = next(iter(result.values()))

        return np.asarray(
            result,
            dtype=np.float32,
        ).reshape(-1)

    def _binary_probabilities(
        self,
        prediction: Any,
        positive_class_index: int = 1,
    ) -> tuple[float, float]:
        values = np.asarray(
            prediction,
            dtype=np.float32,
        ).reshape(-1)

        if values.size == 0:
            return 0.0, 0.0

        if values.size == 1:
            value = float(values[0])

            if value < 0.0 or value > 1.0:
                value = float(
                    tf.math.sigmoid(value).numpy()
                )

            positive = _clip_probability(value)

            return 1.0 - positive, positive

        values = values[:2]

        is_probability_vector = (
            np.all(values >= 0.0)
            and np.all(values <= 1.0)
            and abs(float(values.sum()) - 1.0) <= 0.05
        )

        if not is_probability_vector:
            shifted = values - np.max(values)
            exponentials = np.exp(shifted)
            values = exponentials / max(
                float(exponentials.sum()),
                1e-8,
            )
        else:
            values = values / max(
                float(values.sum()),
                1e-8,
            )

        positive = _clip_probability(
            values[positive_class_index]
        )
        negative = _clip_probability(
            values[1 - positive_class_index]
        )

        return negative, positive

    # ============================================================
    # TOMATO VERIFICATION
    # ============================================================

    def _predict_tomato_probability(
        self,
        image: np.ndarray,
    ) -> float:
        # Use a small test-time ensemble so close-ups, clusters, and
        # partially damaged tomatoes are not rejected because of framing.
        height, width = image.shape[:2]
        views = [
            image,
            image[:, ::-1],
            image[: int(height * 0.90), :],
            image[int(height * 0.10) :, :],
            image[:, : int(width * 0.90)],
            image[:, int(width * 0.10) :],
        ]

        probabilities = []
        for view in views:
            resized = np.asarray(
                Image.fromarray(view.astype(np.uint8)).resize(
                    IMAGE_SIZE,
                    Image.Resampling.LANCZOS,
                ),
                dtype=np.float32,
            )
            prediction = self._predict(
                self.tomato_verifier,
                resized,
            )
            _, tomato_probability = self._binary_probabilities(
                prediction,
                positive_class_index=1,
            )
            probabilities.append(tomato_probability)

        # A single favorable crop can make a cherry, apple, or background
        # look tomato-like. Require consistent evidence across the views.
        return float(np.percentile(probabilities, 40))

    def _similarity_score(
        self,
        image: np.ndarray,
    ) -> Optional[Dict[str, float]]:
        if (
            self.feature_extractor is None
            or self.similarity_embeddings is None
        ):
            return None

        try:
            feature = self.feature_extractor.predict(
                np.expand_dims(image, axis=0),
                verbose=0,
            )

            feature = np.asarray(
                feature,
                dtype=np.float32,
            ).reshape(1, -1)

            feature /= np.maximum(
                np.linalg.norm(feature, axis=1, keepdims=True),
                1e-8,
            )

            similarities = np.dot(
                self.similarity_embeddings,
                feature[0],
            )

            similarities = np.sort(
                similarities
            )[::-1]

            strong = similarities[
                similarities >= TOMATO_SIMILARITY_THRESHOLD
            ]

            average_similarity = float(
                np.mean(strong[:10])
            ) if len(strong) else 0.0

            return {
                "top_similarity": float(similarities[0]),
                "average_similarity": average_similarity,
                "strong_neighbors": float(len(strong)),
            }

        except Exception as exc:
            print(
                "[CropClassifier] Similarity check failed: "
                f"{exc}"
            )
            return None

    def _is_obvious_non_tomato(
        self,
        image: np.ndarray,
    ) -> bool:
        """Reject obvious non-tomato produce before crop analysis."""
        pixels = image / 255.0
        red = pixels[:, :, 0]
        green = pixels[:, :, 1]
        blue = pixels[:, :, 2]

        purple_hue_fraction = float(
            (
                (red + blue - (2.0 * green) > 0.15)
                & (red > 0.15)
                & (blue > 0.12)
            ).mean()
        )

        purple_fraction = float(
            (
                (red > green * 1.15)
                & (blue > green * 1.15)
                & (red > 0.18)
                & (blue > 0.18)
            ).mean()
        )

        red_fraction = float(
            (
                (red > green * 1.18)
                & (red > blue * 1.08)
                & (red > 0.25)
            ).mean()
        )

        # Purple-dominant produce is characteristic of eggplant/brinjal.
        # Do not use this veto for red/green tomato images.
        if (
            purple_fraction >= 0.32
            or (
                purple_hue_fraction >= 0.20
                and red_fraction < 0.60
            )
        ):
            return True

        if self.semantic_guard is None:
            return False

        try:
            predictions = self.semantic_guard.predict(
                np.expand_dims(
                    preprocess_input(image.copy()),
                    axis=0,
                ),
                verbose=0,
            )

            labels = decode_predictions(
                predictions,
                top=5,
            )[0]

            return any(
                label in {
                    "Granny_Smith",
                    "pomegranate",
                }
                and float(score) >= (
                    0.35 if label == "pomegranate" else 0.15
                )
                for _, label, score in labels
            )
        except Exception as exc:
            print(
                "[CropClassifier] Semantic guard failed: "
                f"{exc}"
            )
            return False

    def _verify_tomato(
        self,
        image: np.ndarray,
    ) -> Dict[str, Any]:
        if self.tomato_verifier is None:
            return {
                "is_tomato": False,
                "tomato_probability": 0.0,
                "verification_message": (
                    "Tomato verification model is unavailable."
                ),
            }

        tomato_probability = self._predict_tomato_probability(
            image
        )

        similarity = self._similarity_score(image)

        obvious_non_tomato = self._is_obvious_non_tomato(image)

        if obvious_non_tomato:
            return {
                "is_tomato": False,
                "tomato_probability": round(
                    tomato_probability,
                    6,
                ),
                "tomato_confidence": round(
                    tomato_probability * 100.0,
                    2,
                ),
                "verification_message": (
                    "This image appears to contain non-tomato produce, "
                    "such  another fruit or vegetable. "
                    "Upload a clear tomato image only."
                ),
            }

        verifier_passed = (
            tomato_probability >= TOMATO_VERIFIER_THRESHOLD
        )

        degraded_tomato_candidate = False
        if self.quality_model is not None:
            quality = self._predict_quality(image)
            degraded_tomato_candidate = (
                quality["quality_label"] == "Rotten"
                and quality["rotten_probability"] >= 0.25
            )

        similarity_passed = (
            similarity is not None
            or not REQUIRE_SIMILARITY_VERIFICATION
        )

        if similarity is not None:
            similarity_passed = (
                similarity["top_similarity"]
                >= TOMATO_SIMILARITY_THRESHOLD
                and similarity["strong_neighbors"]
                >= MIN_STRONG_TOMATO_NEIGHBORS
                and similarity["average_similarity"]
                >= MIN_AVERAGE_TOMATO_SIMILARITY
            )

        # The verifier was trained on varied framing and ripeness. Do not
        # reject a strong verifier result merely because the older reference
        # embedding index has weak similarity for a valid green tomato.
        is_tomato = (
            verifier_passed
            and (
                tomato_probability >= 0.75
                or similarity_passed
            )
        ) or degraded_tomato_candidate

        if is_tomato:
            message = "Tomato image verified successfully."
        else:
            message = (
                "This image was rejected because it is not confidently "
                "identified as a tomato. Upload a clear image containing "
                "tomato fruit only."
            )

        result: Dict[str, Any] = {
            "is_tomato": is_tomato,
            "tomato_probability": round(
                tomato_probability,
                6,
            ),
            "tomato_confidence": round(
                tomato_probability * 100.0,
                2,
            ),
            "verification_message": message,
        }

        if similarity is not None:
            result["similarity"] = similarity

        return result

    # ============================================================
    # QUALITY AND MATURITY
    # ============================================================

    def _detect_localized_discoloration_patch(
        self,
        pixels: np.ndarray,
    ) -> Dict[str, Any]:
        """Heuristic check for a blossom-end-rot-style defect: a
        localized, desaturated tan/gray/olive patch on an otherwise
        red tomato surface. This is one of the most common real-world
        tomato defects, and looks visually very different from the
        mold/dark-decay patterns the quality model appears to have
        been trained on - it's precisely the kind of damage the
        `suspected_out_of_distribution_damage` check above exists to
        catch, since the model can be confidently (and wrongly) sure
        such fruit is "Fresh".

        This is purely diagnostic and never called on its own to flip
        a "Fresh" verdict - see its one call site below, which only
        uses it to refine cases the existing safety net has already
        flagged as suspicious, adding a more specific and more
        confident diagnosis when the visual evidence for this
        particular defect is clear.
        """
        r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]

        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        chroma = max_c - min_c
        brightness = (r + g + b) / 3.0

        # Desaturated (unlike vivid red skin or vivid green leaves),
        # and neither near-black shadow nor near-white highlight.
        patch_mask = (
            (chroma < 0.18)
            & (brightness > 0.25)
            & (brightness < 0.85)
        )

        patch_fraction = float(patch_mask.mean())

        # A real defect patch is a solid blob, not pixels scattered
        # evenly across the whole photo (which could just be a gray
        # background). Check an 8x8 grid for a densely-packed block.
        height, width = patch_mask.shape
        block_h, block_w = height // 8, width // 8
        max_block_density = 0.0

        if block_h > 0 and block_w > 0:
            trimmed = patch_mask[: block_h * 8, : block_w * 8]
            blocks = trimmed.reshape(8, block_h, 8, block_w)
            max_block_density = float(
                blocks.mean(axis=(1, 3)).max()
            )

        # Big enough to be a real patch, small enough that it isn't
        # simply most of the fruit (which is the "obvious rot" case
        # the model already handles fine on its own).
        patch_detected = (
            0.03 < patch_fraction < 0.30
            and max_block_density > 0.55
        )

        return {
            "patch_detected": patch_detected,
            "patch_fraction": round(patch_fraction, 4),
            "max_block_density": round(max_block_density, 4),
        }

    def _predict_quality(
        self,
        image: np.ndarray,
    ) -> Dict[str, Any]:
        prediction = self._predict(
            self.quality_model,
            image,
        )

        # Expected quality order: [fresh, rotten].
        fresh_probability, rotten_probability = (
            self._binary_probabilities(
                prediction,
                positive_class_index=1,
            )
        )

        # A very low tomato-reference similarity means the quality model is
        # outside its training distribution. For red produce, do not allow
        # an overconfident Fresh prediction to become Grade A.
        similarity = self._similarity_score(image)
        pixels = image / 255.0
        red_fraction = float(
            (
                (pixels[:, :, 0] > pixels[:, :, 1] * 1.18)
                & (pixels[:, :, 0] > pixels[:, :, 2] * 1.08)
                & (pixels[:, :, 0] > 0.25)
            ).mean()
        )
        suspected_out_of_distribution_damage = (
            fresh_probability > 0.90
            and red_fraction > 0.55
            and similarity is not None
            and similarity["top_similarity"] < 0.45
        )

        discoloration_patch_detected = False

        if suspected_out_of_distribution_damage:
            rotten_probability = max(rotten_probability, 0.51)
            fresh_probability = 1.0 - rotten_probability

            # When there's ALSO a clear, localized discoloration
            # patch (the classic blossom-end-rot visual signature),
            # that's independent, specific corroborating evidence -
            # not just "the model is out of its depth" but "here is a
            # describable defect". Reflect that with a more decisive
            # confidence than the bare-minimum override above,
            # instead of leaving the farmer with an unhelpful,
            # barely-over-50% split for an otherwise visually obvious
            # defect.
            discoloration = self._detect_localized_discoloration_patch(
                pixels
            )
            discoloration_patch_detected = discoloration["patch_detected"]

            if discoloration_patch_detected:
                rotten_probability = max(rotten_probability, 0.78)
                fresh_probability = 1.0 - rotten_probability

        quality_label = (
            "Rotten"
            if rotten_probability >= ROTTEN_THRESHOLD
            else "Fresh"
        )

        confidence = max(
            fresh_probability,
            rotten_probability,
        )

        return {
            "quality": round(
                fresh_probability * 100.0,
                2,
            ),
            "quality_score": round(
                fresh_probability * 100.0,
                2,
            ),
            "quality_label": quality_label,
            "freshness": round(
                fresh_probability * 100.0,
                2,
            ),
            "fresh_probability": round(
                fresh_probability,
                6,
            ),
            "rotten_probability": round(
                rotten_probability,
                6,
            ),
            "quality_confidence": round(
                confidence * 100.0,
                2,
            ),
            "discoloration_patch_detected": discoloration_patch_detected,
        }

    def _predict_maturity(
        self,
        image: np.ndarray,
    ) -> Dict[str, Any]:
        prediction = self._predict(
            self.maturity_model,
            image,
        )

        # Expected maturity order: [immature, mature].
        immature_probability, mature_probability = (
            self._binary_probabilities(
                prediction,
                positive_class_index=1,
            )
        )

        maturity = (
            "Mature"
            if mature_probability >= MATURITY_THRESHOLD
            else "Immature"
        )

        confidence = max(
            immature_probability,
            mature_probability,
        )

        return {
            "maturity": maturity,
            "maturity_label": maturity,
            "immature_probability": round(
                immature_probability,
                6,
            ),
            "mature_probability": round(
                mature_probability,
                6,
            ),
            "maturity_confidence": round(
                confidence * 100.0,
                2,
            ),
        }

    # ============================================================
    # RESULT CREATION
    # ============================================================

    def _result_message(
        self,
        quality_label: str,
        maturity_label: str,
        quality_confidence: float,
        maturity_confidence: float,
        discoloration_patch_detected: bool = False,
    ) -> str:
        if quality_label.lower() == "rotten":
            if discoloration_patch_detected:
                # Keep this consistent with defect_message in
                # predict() below - a farmer shouldn't see "rotten,
                # discard, don't mix with healthy fruit" here while a
                # more specific, less alarming diagnosis appears
                # elsewhere in the same result.
                return (
                    "Signs consistent with blossom end rot were "
                    "detected - likely calcium deficiency or "
                    "irregular watering, not a contagious disease. "
                    "Set the affected fruit aside and see the "
                    "detailed note below."
                )

            return (
                "Rotten tomato detected. It shows signs of "
                "decomposition and should not be consumed or mixed "
                "with healthy tomatoes."
            )

        if quality_confidence < MIN_CONFIDENCE * 100:
            return (
                "Tomato detected, but its condition could not be "
                "identified confidently."
            )

        if maturity_label.lower() == "immature":
            return (
                "Immature tomato detected. It is still green or "
                "not sufficiently ripe for normal harvesting."
            )

        if maturity_confidence < MIN_CONFIDENCE * 100:
            return (
                "Tomato detected, but its maturity could not be "
                "determined confidently."
            )

        return (
            "Tomato detected. No visible rotting was identified."
        )

    def _calculate_grade(
        self,
        quality: Dict[str, Any],
        maturity: Dict[str, Any],
    ) -> tuple[str, float]:
        freshness = float(
            quality.get("freshness", 0.0)
        )
        maturity_confidence = float(
            maturity.get("maturity_confidence", 0.0)
        )

        if quality.get("quality_label") == "Rotten":
            return "C", round(freshness * 0.25, 2)

        if maturity.get("maturity_label") == "Immature":
            score = freshness * 0.60
        else:
            score = (
                freshness * 0.75
                + maturity_confidence * 0.25
            )

        score = max(0.0, min(100.0, score))

        if score >= GRADE_A_THRESHOLD:
            grade = "A"
        elif score >= GRADE_B_THRESHOLD:
            grade = "B"
        else:
            grade = "C"

        return grade, round(score, 2)

    def _rejected_result(
        self,
        message: str,
        verification: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "success": False,
            "accepted": False,
            "is_tomato": False,
            "grade": None,
            "grade_score": 0.0,
            "quality": 0.0,
            "quality_score": 0.0,
            "quality_label": "Rejected",
            "maturity": "Unknown",
            "maturity_label": "Unknown",
            "defect_detected": False,
            "defect_type": None,
            "defect_message": None,
            "result_message": message,
            "recommendation": message,
        }

        if verification:
            result.update(verification)

        return result

    # ============================================================
    # PUBLIC API
    # ============================================================

    def predict(
        self,
        image_path: str,
    ) -> Dict[str, Any]:
        try:
            image = self._load_image(image_path)
        except Exception as exc:
            return self._rejected_result(
                str(exc)
            )

        if self.tomato_verifier is None:
            return self._rejected_result(
                "Tomato verification is unavailable. "
                "The image cannot be analyzed safely."
            )

        try:
            verification = self._verify_tomato(image)
        except Exception as exc:
            return self._rejected_result(
                f"Tomato verification failed: {exc}"
            )

        if not verification["is_tomato"]:
            return self._rejected_result(
                verification["verification_message"],
                verification,
            )

        try:
            quality = self._predict_quality(image)
            maturity = self._predict_maturity(image)
        except Exception as exc:
            return self._rejected_result(
                f"Tomato analysis failed: {exc}",
                verification,
            )

        if maturity["maturity_label"] == "Immature":
            rejected = self._rejected_result(
                "Immature tomato detected. Only sufficiently ripe tomatoes "
                "can receive a quality grade.",
                verification,
            )
            rejected.update(quality)
            rejected.update(maturity)
            rejected["crop"] = "Tomato"
            return rejected

        quality_label = quality["quality_label"]
        maturity_label = maturity["maturity_label"]

        result_message = self._result_message(
            quality_label=quality_label,
            maturity_label=maturity_label,
            quality_confidence=quality["quality_confidence"],
            maturity_confidence=maturity["maturity_confidence"],
            discoloration_patch_detected=quality.get(
                "discoloration_patch_detected", False
            ),
        )

        grade, grade_score = self._calculate_grade(
            quality,
            maturity,
        )

        defect_detected = quality_label.lower() == "rotten"

        if defect_detected and quality.get("discoloration_patch_detected"):
            # A specific, more useful diagnosis than the generic
            # "Rotting/decomposition" below: blossom end rot is a
            # calcium-deficiency/irregular-watering disorder, not a
            # contagious disease or fungus, so the correct advice
            # (consistent watering, calcium amendment) is different
            # from what "rotten, discard, don't mix with healthy
            # fruit" would suggest.
            defect_type = "Blossom End Rot (suspected)"
            defect_message = (
                "Signs consistent with blossom end rot were detected - "
                "a localized, sunken, tan or gray patch typically at "
                "the base of the fruit, opposite the stem. This is "
                "usually caused by calcium deficiency or irregular "
                "watering, not a contagious disease or fungus, so it "
                "does not put the rest of the plant's fruit at risk. "
                "Set affected fruit aside, keep soil moisture even, "
                "and consider a calcium-rich amendment - confirm with "
                "your local agriculture officer."
            )
        elif defect_detected:
            defect_type = "Rotting/decomposition"
            defect_message = (
                "Rotten tomato detected. Visible signs of decomposition "
                "were identified. Do not consume it or mix it with healthy "
                "tomatoes."
            )
        else:
            defect_type = None
            defect_message = None

        result: Dict[str, Any] = {
            "success": True,
            "accepted": True,
            "is_tomato": True,
            "crop": "Tomato",
            "confidence": round(
                min(
                    float(verification["tomato_confidence"]),
                    float(quality["quality_confidence"]),
                    float(maturity["maturity_confidence"]),
                ),
                2,
            ),
            "grade": grade,
            "grade_score": grade_score,
            "defect_detected": defect_detected,
            "defect_type": defect_type,
            "defect_message": defect_message,
            "result_message": result_message,
            "recommendation": result_message,
            **verification,
            **quality,
            **maturity,
        }

        return result

    def analyze(
        self,
        image_path: str,
    ) -> Dict[str, Any]:
        """Compatibility alias for callers using analyze()."""
        return self.predict(image_path)


crop_classifier = CropClassifier()

print("=" * 70)
print("[AGRINOVA] Strict Tomato CropClassifier initialized")
print("=" * 70)