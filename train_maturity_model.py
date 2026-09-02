from pathlib import Path
import json
import shutil

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\USER\Downloads\TomatoMaturityExtracted"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "trained_models"
    / "tomato_maturity"
)

MODEL_PATH = OUTPUT_DIR / "tomato_maturity.keras"
CLASS_NAMES_PATH = OUTPUT_DIR / "class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS_INITIAL = 15
EPOCHS_FINE_TUNE = 10

SEED = 42


# ============================================================
# FIND DATASET
# ============================================================

def find_dataset_root() -> Path:
    """
    Finds the directory containing:

        Augment Dataset/
            Immature/
            Mature/

        Original Dataset/
            Immature/
            Mature/
    """

    candidates = [
        DATASET_ROOT,
        DATASET_ROOT / "Tomato Maturity Detection Dataset",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        augment = candidate / "Augment Dataset"
        original = candidate / "Original Dataset"

        if augment.exists() or original.exists():
            return candidate

    for directory in DATASET_ROOT.rglob("*"):
        if not directory.is_dir():
            continue

        augment = directory / "Augment Dataset"
        original = directory / "Original Dataset"

        if augment.exists() or original.exists():
            return directory

    raise FileNotFoundError(
        f"Could not find Tomato Maturity Detection Dataset under:\n"
        f"{DATASET_ROOT}"
    )


# ============================================================
# COLLECT IMAGES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def collect_images(dataset_root: Path):
    """
    Collects images from BOTH:

        Augment Dataset
        Original Dataset

    Classes:

        Immature
        Mature
    """

    class_dirs = {
        "Immature": [],
        "Mature": [],
    }

    sources = [
        dataset_root / "Augment Dataset",
        dataset_root / "Original Dataset",
    ]

    for source in sources:

        if not source.exists():
            print(
                f"[DATASET] Source not found, skipping: {source}"
            )
            continue

        for class_name in class_dirs:

            class_dir = source / class_name

            if not class_dir.exists():
                print(
                    f"[DATASET] Class directory not found: "
                    f"{class_dir}"
                )
                continue

            for file in class_dir.rglob("*"):

                if (
                    file.is_file()
                    and file.suffix.lower()
                    in IMAGE_EXTENSIONS
                ):
                    class_dirs[class_name].append(file)

    return class_dirs


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TOMATO MATURITY CLASSIFIER TRAINING")
    print("=" * 70)

    print(
        f"[DATASET] Requested root:\n{DATASET_ROOT}"
    )

    dataset_root = find_dataset_root()

    print(
        f"[DATASET] Actual dataset root:\n{dataset_root}"
    )

    images = collect_images(dataset_root)

    immature_count = len(images["Immature"])
    mature_count = len(images["Mature"])

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"[DATASET] Immature images: {immature_count}"
    )

    print(
        f"[DATASET] Mature images:   {mature_count}"
    )

    print(
        f"[DATASET] Total images:    "
        f"{immature_count + mature_count}"
    )

    if immature_count == 0:
        raise RuntimeError(
            "No Immature images were found."
        )

    if mature_count == 0:
        raise RuntimeError(
            "No Mature images were found."
        )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CREATE TEMPORARY TRAINING DATASET
    # --------------------------------------------------------

    temp_dataset = (
        OUTPUT_DIR
        / "_training_dataset"
    )

    if temp_dataset.exists():
        print(
            "[DATASET] Removing previous temporary dataset..."
        )

        shutil.rmtree(temp_dataset)

    temp_dataset.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # COPY DATA INTO STANDARD CLASS DIRECTORIES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREPARING DATASET")
    print("=" * 70)

    for class_name in [
        "Immature",
        "Mature",
    ]:

        destination = (
            temp_dataset
            / class_name
        )

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, image_path in enumerate(
            images[class_name]
        ):

            destination_file = (
                destination
                / f"{class_name.lower()}_{index:06d}"
                f"{image_path.suffix.lower()}"
            )

            shutil.copy2(
                image_path,
                destination_file,
            )

    print(
        "[DATASET] Dataset preparation complete."
    )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING TENSORFLOW DATASET")
    print("=" * 70)

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        temp_dataset,
        validation_split=0.20,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        temp_dataset,
        validation_split=0.20,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    class_names = train_dataset.class_names

    print(
        f"[MODEL] Classes: {class_names}"
    )

    with open(
        CLASS_NAMES_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            class_names,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = (
        train_dataset
        .prefetch(AUTOTUNE)
    )

    validation_dataset = (
        validation_dataset
        .prefetch(AUTOTUNE)
    )

    # --------------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------------

    augmentation = keras.Sequential(
        [
            layers.RandomFlip(
                "horizontal"
            ),

            layers.RandomRotation(
                0.08
            ),

            layers.RandomZoom(
                0.10
            ),

            layers.RandomContrast(
                0.10
            ),
        ],
        name="maturity_augmentation",
    )

    # --------------------------------------------------------
    # BASE MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING EFFICIENTNETB0")
    print("=" * 70)

    base_model = (
        tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(
                224,
                224,
                3,
            ),
        )
    )

    base_model.trainable = False

    inputs = keras.Input(
        shape=(
            224,
            224,
            3,
        ),
        name="image",
    )

    x = augmentation(inputs)

    x = tf.keras.applications.efficientnet.preprocess_input(
        x
    )

    x = base_model(
        x,
        training=False,
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(
        0.30
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="maturity_probability",
    )(x)

    model = keras.Model(
        inputs,
        outputs,
        name="tomato_maturity_classifier",
    )

    # --------------------------------------------------------
    # INITIAL TRAINING
    # --------------------------------------------------------

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            keras.metrics.AUC(
                name="auc"
            ),
        ],
    )

    print()
    print("=" * 70)
    print("INITIAL TRAINING")
    print("=" * 70)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
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

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS_INITIAL,
        callbacks=callbacks,
    )

    # --------------------------------------------------------
    # FINE TUNING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINE-TUNING EFFICIENTNETB0")
    print("=" * 70)

    base_model.trainable = True

    # Keep early EfficientNet layers frozen.
    fine_tune_from = 180

    for layer in base_model.layers[
        :fine_tune_from
    ]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-5
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            keras.metrics.AUC(
                name="auc"
            ),
        ],
    )

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=callbacks,
    )

    # --------------------------------------------------------
    # FINAL SAVE
    # --------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"[MODEL] Saved to:\n{MODEL_PATH}"
    )

    print(
        f"[CLASSES] Saved to:\n{CLASS_NAMES_PATH}"
    )

    print()
    print(
        "[IMPORTANT] Existing tomato-quality model was NOT modified."
    )

    print(
        "[IMPORTANT] This is a separate tomato-maturity model."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()