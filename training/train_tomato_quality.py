"""
EcoAgri Intelligence
Tomato AI Training Pipeline

Dataset expected:

backend/
└── datasets/
    └── tomato/
        ├── maturity/
        │   ├── immature/
        │   └── mature/
        │
        └── quality/
            ├── fresh/
            └── rotten/

Creates:

backend/trained_models/
    tomato_quality.keras
    tomato_maturity.keras
    tomato_verifier.npz
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

EPOCHS_HEAD = 8
EPOCHS_FINE = 8

BASE_DIR = Path(__file__).resolve().parents[1]

# Add manually reviewed images captured outside the original datasets here:
# tomato_generalization/quality/fresh, tomato_generalization/quality/rotten
# tomato_generalization/maturity/immature, tomato_generalization/maturity/mature
GENERALIZATION_DIR = BASE_DIR / "datasets" / "tomato_generalization"

DATASET_DIR = (
    BASE_DIR
    / "datasets"
    / "tomato"
)

QUALITY_DIR = DATASET_DIR / "quality"
MATURITY_DIR = DATASET_DIR / "maturity"

MODEL_DIR = (
    BASE_DIR
    / "trained_models"
)

QUALITY_MODEL_PATH = (
    MODEL_DIR
    / "tomato_quality.keras"
)

MATURITY_MODEL_PATH = (
    MODEL_DIR
    / "tomato_maturity.keras"
)

VERIFIER_PATH = (
    MODEL_DIR
    / "tomato_verifier.npz"
)

METADATA_PATH = (
    MODEL_DIR
    / "tomato_ai_metadata.json"
)


# ============================================================
# PRINT HELPERS
# ============================================================

def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset() -> None:

    banner("CHECKING TOMATO DATASET")

    print("Dataset:")
    print(DATASET_DIR)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"\nTomato dataset does not exist:\n{DATASET_DIR}\n"
        )

    required = [
        QUALITY_DIR / "fresh",
        QUALITY_DIR / "rotten",
        MATURITY_DIR / "mature",
        MATURITY_DIR / "immature",
    ]

    for folder in required:

        if not folder.exists():

            raise FileNotFoundError(
                f"\nMissing dataset folder:\n{folder}"
            )

    for folder in required:

        images = list_images(folder)

        print(
            f"{folder.relative_to(DATASET_DIR)} : "
            f"{len(images)} images"
        )

        if len(images) == 0:

            raise RuntimeError(
                f"No images found in {folder}"
            )


# ============================================================
# IMAGE DISCOVERY
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def list_images(folder: Path) -> list[Path]:

    result = []

    for path in folder.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):
            result.append(path)

    return sorted(result)


def prepare_training_directory(
    directory: Path,
    model_name: str,
) -> Path:
    """Merge the base dataset with manually reviewed field images."""
    merged = MODEL_DIR / f"_merged_{model_name}"

    if merged.exists():
        shutil.rmtree(merged)

    for class_name in sorted(
        path.name
        for path in directory.iterdir()
        if path.is_dir()
    ):
        destination = merged / class_name
        destination.mkdir(parents=True, exist_ok=True)

        sources = [directory / class_name]
        extra = GENERALIZATION_DIR / model_name / class_name
        if extra.is_dir():
            sources.append(extra)

        image_index = 0
        for source in sources:
            for image_path in list_images(source):
                destination_path = (
                    destination
                    / f"{image_index:08d}{image_path.suffix.lower()}"
                )
                shutil.copy2(image_path, destination_path)
                image_index += 1

    return merged


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(
    directory: Path,
):
    """
    Loads a two-class dataset.

    Keras automatically uses alphabetical
    class ordering.
    """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.20,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True,
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.20,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    class_names = train_ds.class_names

    print()
    print("Classes:", class_names)

    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    validation_ds = validation_ds.prefetch(
        tf.data.AUTOTUNE
    )

    return (
        train_ds,
        validation_ds,
        class_names,
    )


# ============================================================
# MODEL
# ============================================================

def build_model() -> tuple[models.Model, tf.keras.Model]:

    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom((-0.15, 0.25)),
            layers.RandomTranslation(0.12, 0.12),
            layers.RandomContrast(0.20),
            layers.RandomBrightness(0.18),
        ],
        name="augmentation",
    )

    base = MobileNetV2(
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3,
        ),
        include_top=False,
        weights="imagenet",
    )

    base.trainable = False

    inputs = layers.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3,
        )
    )

    x = data_augmentation(inputs)

    x = layers.Lambda(
        preprocess_input
    )(x)

    x = base(
        x,
        training=False,
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    model = models.Model(
        inputs,
        outputs,
    )

    return model, base


# ============================================================
# TRAIN TWO-CLASS MODEL
# ============================================================

def train_binary_model(
    directory: Path,
    model_path: Path,
    model_name: str,
):

    banner(
        f"TRAINING {model_name.upper()} MODEL"
    )

    training_directory = prepare_training_directory(
        directory,
        model_name="quality" if "quality" in model_name else "maturity",
    )

    (
        train_ds,
        validation_ds,
        class_names,
    ) = load_dataset(training_directory)

    model, base = build_model()

    class_counts = {
        class_name: len(list_images(training_directory / class_name))
        for class_name in class_names
    }
    total_images = sum(class_counts.values())
    class_weight = {
        index: total_images / (
            len(class_names) * class_counts[class_name]
        )
        for index, class_name in enumerate(class_names)
    }

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    print()
    print("Training classifier head...")

    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # --------------------------------------------------------
    # Fine tuning
    # --------------------------------------------------------

    print()
    print("Fine tuning MobileNetV2...")

    base.trainable = True

    # Keep most of the backbone frozen.
    fine_tune_from = max(
        0,
        len(base.layers) - 35
    )

    for layer in base.layers[
        :fine_tune_from
    ]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS_FINE,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    model.save(model_path)

    print()
    print("Saved:")
    print(model_path)

    return class_names


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

def build_feature_extractor():

    base = MobileNetV2(
        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3,
        ),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    base.trainable = False

    return base


# ============================================================
# LOAD IMAGE FOR FEATURE EXTRACTION
# ============================================================

def load_single_image(
    image_path: Path,
) -> np.ndarray:

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
    )

    array = tf.keras.utils.img_to_array(
        image
    )

    array = preprocess_input(
        array
    )

    return array


# ============================================================
# NORMALIZE
# ============================================================

def normalize_vector(
    vector: np.ndarray,
) -> np.ndarray:

    norm = np.linalg.norm(
        vector
    )

    if norm <= 1e-8:
        return vector

    return vector / norm


# ============================================================
# TOMATO VERIFIER
# ============================================================

def train_tomato_verifier():

    banner(
        "BUILDING TOMATO VERIFICATION MODEL"
    )

    print(
        "Using all tomato images as positive references."
    )

    tomato_images = []

    # Quality images
    tomato_images.extend(
        list_images(
            QUALITY_DIR / "fresh"
        )
    )

    tomato_images.extend(
        list_images(
            QUALITY_DIR / "rotten"
        )
    )

    # Maturity images
    tomato_images.extend(
        list_images(
            MATURITY_DIR / "mature"
        )
    )

    tomato_images.extend(
        list_images(
            MATURITY_DIR / "immature"
        )
    )

    # Remove duplicates
    tomato_images = list(
        dict.fromkeys(
            tomato_images
        )
    )

    print(
        "Tomato reference images:",
        len(tomato_images)
    )

    if len(tomato_images) < 5:

        raise RuntimeError(
            "At least 5 tomato images are required "
            "to build the tomato verifier."
        )

    extractor = build_feature_extractor()

    features = []

    for index, image_path in enumerate(
        tomato_images,
        start=1,
    ):

        try:

            image = load_single_image(
                image_path
            )

            batch = np.expand_dims(
                image,
                axis=0,
            )

            feature = extractor.predict(
                batch,
                verbose=0,
            )[0]

            feature = normalize_vector(
                feature
            )

            features.append(
                feature
            )

        except Exception as exc:

            print(
                "Skipped:",
                image_path,
                "->",
                exc,
            )

        if index % 25 == 0:

            print(
                f"Processed {index}/"
                f"{len(tomato_images)}"
            )

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if len(features) < 5:

        raise RuntimeError(
            "Not enough valid tomato images "
            "for verifier."
        )

    # --------------------------------------------------------
    # Tomato centroid
    # --------------------------------------------------------

    centroid = np.mean(
        features,
        axis=0,
    )

    centroid = normalize_vector(
        centroid
    )

    # --------------------------------------------------------
    # Calculate reference similarities
    # --------------------------------------------------------

    similarities = []

    for feature in features:

        similarity = float(
            np.dot(
                feature,
                centroid,
            )
        )

        similarities.append(
            similarity
        )

    similarities = np.asarray(
        similarities,
        dtype=np.float32,
    )

    # Use a conservative threshold.
    #
    # We don't want to reject genuine tomatoes
    # too aggressively.
    #
    # The threshold is based on the lower
    # tail of the tomato reference distribution.

    percentile_threshold = float(
        np.percentile(
            similarities,
            5,
        )
    )

    threshold = max(
        0.45,
        min(
            0.72,
            percentile_threshold - 0.05,
        ),
    )

    print()
    print(
        "Tomato similarity minimum:",
        float(similarities.min()),
    )

    print(
        "Tomato similarity average:",
        float(similarities.mean()),
    )

    print(
        "Tomato verifier threshold:",
        threshold,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez(
        VERIFIER_PATH,
        centroid=centroid,
        threshold=np.asarray(
            threshold,
            dtype=np.float32,
        ),
        reference_count=np.asarray(
            len(features),
            dtype=np.int32,
        ),
    )

    print()
    print(
        "Saved tomato verifier:"
    )

    print(
        VERIFIER_PATH
    )

    return threshold


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "TOMATO QUALITY TRAINING"
    )

    print(
        "TensorFlow:",
        tf.__version__,
    )

    print(
        "Python:",
        __import__("sys").version,
    )

    print(
        "Dataset:",
        DATASET_DIR,
    )

    # --------------------------------------------------------
    # Check folders
    # --------------------------------------------------------

    check_dataset()

    # --------------------------------------------------------
    # Model directory
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Remove ONLY old tomato models
    # --------------------------------------------------------

    old_models = [
        QUALITY_MODEL_PATH,
        MATURITY_MODEL_PATH,
        VERIFIER_PATH,
        METADATA_PATH,
    ]

    for path in old_models:

        if path.exists():

            print(
                "Removing old model:",
                path.name,
            )

            path.unlink()

    # --------------------------------------------------------
    # Train quality
    # --------------------------------------------------------

    quality_classes = train_binary_model(
        QUALITY_DIR,
        QUALITY_MODEL_PATH,
        "tomato quality",
    )

    # --------------------------------------------------------
    # Train maturity
    # --------------------------------------------------------

    maturity_classes = train_binary_model(
        MATURITY_DIR,
        MATURITY_MODEL_PATH,
        "tomato maturity",
    )

    # --------------------------------------------------------
    # Tomato verifier
    # --------------------------------------------------------

    verifier_threshold = (
        train_tomato_verifier()
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "image_size": [
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
        ],

        "quality_classes":
            quality_classes,

        "maturity_classes":
            maturity_classes,

        "verifier_threshold":
            verifier_threshold,

        "dataset_structure":
            "datasets/tomato/"
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    banner(
        "TRAINING COMPLETE"
    )

    print()
    print(
        "Quality model:"
    )
    print(
        QUALITY_MODEL_PATH
    )

    print()
    print(
        "Maturity model:"
    )
    print(
        MATURITY_MODEL_PATH
    )

    print()
    print(
        "Tomato verifier:"
    )
    print(
        VERIFIER_PATH
    )

    print()
    print(
        "Metadata:"
    )
    print(
        METADATA_PATH
    )

    print()
    print(
        "The AI pipeline is ready."
    )


if __name__ == "__main__":
    main()