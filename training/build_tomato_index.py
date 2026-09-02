"""
Build a visual similarity index for the tomato quality dataset.

Creates embeddings for all Fresh and Rotten tomato images using
the same EfficientNetB0 backbone used by the trained classifier.

Output:
    trained_models/tomato_quality/tomato_similarity_index.npz
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "datasets" / "tomato" / "quality"
MODEL_PATH = (
    BASE_DIR
    / "trained_models"
    / "tomato_quality"
    / "tomato_quality.keras"
)

OUTPUT_PATH = (
    BASE_DIR
    / "trained_models"
    / "tomato_quality"
    / "tomato_similarity_index.npz"
)

IMAGE_SIZE = (224, 224)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("AGRIWISE - TOMATO VISUAL SIMILARITY INDEX")
print("=" * 70)

print("Dataset:", DATASET_DIR)
print("Model:", MODEL_PATH)
print("Output:", OUTPUT_PATH)


if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Tomato quality model not found: {MODEL_PATH}"
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained tomato quality model...")

classifier_model = keras.models.load_model(
    MODEL_PATH
)

print("Classifier loaded.")
print("Input shape:", classifier_model.input_shape)
print("Output shape:", classifier_model.output_shape)


# ============================================================
# BUILD EFFICIENTNET FEATURE EXTRACTOR
# ============================================================

print("\nBuilding EfficientNetB0 feature extractor...")

# Find the EfficientNetB0 backbone inside the trained model.
base_model = None

for layer in classifier_model.layers:
    if isinstance(
        layer,
        keras.Model,
    ):
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

if base_model is None:
    raise RuntimeError(
        "Could not find EfficientNetB0 backbone inside the trained model."
    )

print("Backbone:", base_model.name)


# Create a feature extractor from the EfficientNet output.
#
# This gives a 1280-dimensional visual representation
# before the classifier's final layers.

feature_extractor = keras.Model(
    inputs=base_model.input,
    outputs=base_model.output,
)


# ============================================================
# FIND DATASET IMAGES
# ============================================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

image_paths = []
labels = []


for class_name, label in [
    ("fresh", 0),
    ("rotten", 1),
]:

    class_dir = DATASET_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Missing class directory: {class_dir}"
        )

    files = sorted(
        [
            p
            for p in class_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in extensions
        ]
    )

    print(
        f"{class_name.capitalize():8s}: {len(files)} images"
    )

    for path in files:
        image_paths.append(str(path))
        labels.append(label)


print("\nTotal images:", len(image_paths))


if not image_paths:
    raise RuntimeError(
        "No images found in the tomato quality dataset."
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path: str) -> np.ndarray:

    image = keras.utils.load_img(
        path,
        target_size=IMAGE_SIZE,
    )

    image = keras.utils.img_to_array(
        image
    )

    # IMPORTANT:
    #
    # DO NOT divide by 255.
    #
    # Your trained EfficientNet model expects raw
    # 0-255 image values because EfficientNetB0 contains
    # its own preprocessing.

    return image.astype(
        np.float32
    )


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating visual embeddings...")

BATCH_SIZE = 32

all_embeddings = []

for start in range(
    0,
    len(image_paths),
    BATCH_SIZE,
):

    end = min(
        start + BATCH_SIZE,
        len(image_paths),
    )

    batch_paths = image_paths[
        start:end
    ]

    batch_images = np.stack(
        [
            load_image(path)
            for path in batch_paths
        ]
    )

    embeddings = feature_extractor.predict(
        batch_images,
        verbose=0,
    )

    # Global-average-pool if necessary.
    if embeddings.ndim == 4:
        embeddings = np.mean(
            embeddings,
            axis=(1, 2),
        )

    all_embeddings.append(
        embeddings.astype(
            np.float32
        )
    )

    print(
        f"Processed {end}/{len(image_paths)}"
    )


embeddings = np.concatenate(
    all_embeddings,
    axis=0,
)


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

print("\nNormalizing embeddings...")

norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True,
)

norms = np.maximum(
    norms,
    1e-12,
)

embeddings = (
    embeddings / norms
).astype(
    np.float32
)


# ============================================================
# SAVE INDEX
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

np.savez_compressed(
    OUTPUT_PATH,
    embeddings=embeddings,
    labels=np.asarray(
        labels,
        dtype=np.int8,
    ),
    paths=np.asarray(
        image_paths,
    ),
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SIMILARITY INDEX CREATED")
print("=" * 70)

print("Images:", len(image_paths))
print(
    "Embedding shape:",
    embeddings.shape,
)

print(
    "Fresh:",
    int(np.sum(np.asarray(labels) == 0)),
)

print(
    "Rotten:",
    int(np.sum(np.asarray(labels) == 1)),
)

print("Saved to:")
print(OUTPUT_PATH)

print("=" * 70)