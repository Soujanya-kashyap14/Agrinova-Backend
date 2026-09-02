import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

# Your downloaded dataset
FRUIT_DATASET = Path(
    r"C:\Users\USER\Downloads\fruit_dataset"
)

# Your existing tomato dataset
EXISTING_TOMATO_DATASET = (
    PROJECT_DIR / "datasets" / "tomato"
)

# Output model
MODEL_DIR = PROJECT_DIR / "trained_models"

MODEL_PATH = (
    MODEL_DIR / "tomato_verifier.keras"
)

# Training parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

# Maximum number of images from each dataset
# We use all available images up to these limits.
MAX_TRAIN_IMAGES = 50000
MAX_TEST_IMAGES = 15000


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_image(path: Path):
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_images(folder: Path):
    if not folder.exists():
        return []

    return [
        p
        for p in folder.rglob("*")
        if is_image(p)
    ]


def find_dataset_split(root: Path, split_name: str):
    """
    Automatically finds the directory containing class folders.

    Handles structures such as:

        fruit_dataset/train/train/Apple
        fruit_dataset/train/Apple

    """

    possible_roots = [
        root / split_name,
        root / split_name / split_name,
    ]

    for candidate in possible_roots:

        if not candidate.exists():
            continue

        class_dirs = [
            p
            for p in candidate.iterdir()
            if p.is_dir()
        ]

        if len(class_dirs) >= 10:
            return candidate

    # Recursive fallback
    for candidate in root.rglob("*"):

        if not candidate.is_dir():
            continue

        name = candidate.name.lower()

        if name != split_name.lower():
            continue

        class_dirs = [
            p
            for p in candidate.iterdir()
            if p.is_dir()
        ]

        if len(class_dirs) >= 10:
            return candidate

    return None


def is_tomato_class(class_name: str):
    """
    Any class whose name starts with Tomato
    is treated as a tomato class.

    Examples:
        Tomato
        Tomato 1
        Tomato 2
        Tomato Cherry Red
        Tomato Heart
        Tomato Maroon
        Tomato Yellow
    """

    name = class_name.strip().lower()

    return name.startswith("tomato")


# ============================================================
# DISCOVER DATASET
# ============================================================

def discover_dataset():

    print()
    print("=" * 70)
    print("DISCOVERING FRUIT / VEGETABLE DATASET")
    print("=" * 70)

    print()
    print("Dataset:")
    print(FRUIT_DATASET)

    if not FRUIT_DATASET.exists():

        raise FileNotFoundError(
            f"""
Dataset not found:

{FRUIT_DATASET}

Make sure the extracted dataset is located at:

C:\\Users\\USER\\Downloads\\fruit_dataset
"""
        )

    train_dir = find_dataset_split(
        FRUIT_DATASET,
        "train"
    )

    test_dir = find_dataset_split(
        FRUIT_DATASET,
        "test"
    )

    if train_dir is None:

        raise RuntimeError(
            "Could not find the training directory."
        )

    if test_dir is None:

        raise RuntimeError(
            "Could not find the testing directory."
        )

    print()
    print("TRAIN DIRECTORY:")
    print(train_dir)

    print()
    print("TEST DIRECTORY:")
    print(test_dir)

    return train_dir, test_dir


# ============================================================
# COLLECT BINARY IMAGES
# ============================================================

def collect_binary_images(split_dir):

    tomato_images = []
    non_tomato_images = []

    class_dirs = [
        p
        for p in split_dir.iterdir()
        if p.is_dir()
    ]

    print()
    print(f"Classes found: {len(class_dirs)}")

    if len(class_dirs) == 0:

        raise RuntimeError(
            f"No class folders found in:\n{split_dir}"
        )

    print()
    print("-" * 70)

    for class_dir in sorted(class_dirs):

        class_name = class_dir.name

        images = get_images(class_dir)

        if is_tomato_class(class_name):

            tomato_images.extend(images)

            print(
                f"[TOMATO]     "
                f"{class_name:<35} "
                f"{len(images):>6} images"
            )

        else:

            non_tomato_images.extend(images)

            print(
                f"[NON-TOMATO] "
                f"{class_name:<35} "
                f"{len(images):>6} images"
            )

    print("-" * 70)

    return tomato_images, non_tomato_images


# ============================================================
# ADD EXISTING TOMATO DATASET
# ============================================================

def add_existing_tomato_images(tomato_images):

    if not EXISTING_TOMATO_DATASET.exists():

        print()
        print(
            "[WARNING] Existing tomato dataset not found:"
        )

        print(
            EXISTING_TOMATO_DATASET
        )

        return tomato_images

    existing_images = get_images(
        EXISTING_TOMATO_DATASET
    )

    print()
    print(
        f"Existing tomato dataset images: "
        f"{len(existing_images)}"
    )

    tomato_images.extend(existing_images)

    return tomato_images


# ============================================================
# BALANCE DATASET
# ============================================================

def balance_dataset(
    tomato_images,
    non_tomato_images
):

    random.shuffle(tomato_images)
    random.shuffle(non_tomato_images)

    print()
    print("=" * 70)
    print("DATASET BALANCING")
    print("=" * 70)

    print()
    print(
        f"Available tomato images: "
        f"{len(tomato_images)}"
    )

    print(
        f"Available non-tomato images: "
        f"{len(non_tomato_images)}"
    )

    # Use balanced number of images.
    number_of_images = min(
        len(tomato_images),
        len(non_tomato_images)
    )

    number_of_images = min(
        number_of_images,
        MAX_TRAIN_IMAGES
    )

    tomato_images = tomato_images[
        :number_of_images
    ]

    non_tomato_images = non_tomato_images[
        :number_of_images
    ]

    print()
    print(
        f"Using {number_of_images} "
        f"tomato images"
    )

    print(
        f"Using {number_of_images} "
        f"non-tomato images"
    )

    return tomato_images, non_tomato_images


# ============================================================
# CREATE DATASET FROM FILE PATHS
# ============================================================

def make_tf_dataset(
    tomato_images,
    non_tomato_images,
    shuffle=True
):

    paths = []
    labels = []

    # Tomato = 1
    for image_path in tomato_images:

        paths.append(str(image_path))
        labels.append(1.0)

    # Non-Tomato = 0
    for image_path in non_tomato_images:

        paths.append(str(image_path))
        labels.append(0.0)

    paths = np.array(paths)
    labels = np.array(
        labels,
        dtype=np.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    def load_image(path, label):

        image = tf.io.read_file(path)

        image = tf.image.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        image.set_shape(
            [None, None, 3]
        )

        image = tf.image.resize(
            image,
            IMAGE_SIZE
        )

        image = tf.cast(
            image,
            tf.float32
        )

        return image, label

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=min(
                len(paths),
                10000
            ),
            seed=SEED
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# SPLIT DATA
# ============================================================

def split_images(
    tomato_images,
    non_tomato_images
):

    random.shuffle(tomato_images)
    random.shuffle(non_tomato_images)

    count = min(
        len(tomato_images),
        len(non_tomato_images)
    )

    tomato_images = tomato_images[:count]
    non_tomato_images = non_tomato_images[:count]

    n = len(tomato_images)

    train_end = int(
        n * 0.70
    )

    val_end = int(
        n * 0.85
    )

    train_tomato = tomato_images[
        :train_end
    ]

    val_tomato = tomato_images[
        train_end:val_end
    ]

    test_tomato = tomato_images[
        val_end:
    ]

    train_non_tomato = non_tomato_images[
        :train_end
    ]

    val_non_tomato = non_tomato_images[
        train_end:val_end
    ]

    test_non_tomato = non_tomato_images[
        val_end:
    ]

    return (
        train_tomato,
        train_non_tomato,
        val_tomato,
        val_non_tomato,
        test_tomato,
        test_non_tomato,
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    print()
    print("=" * 70)
    print("BUILDING TOMATO VERIFIER")
    print("=" * 70)

    base_model = (
        tf.keras.applications.MobileNetV2(
            input_shape=(
                224,
                224,
                3
            ),
            include_top=False,
            weights="imagenet"
        )
    )

    base_model.trainable = False

    inputs = keras.Input(
        shape=(224, 224, 3),
        name="image"
    )

    # MobileNetV2 preprocessing
    x = (
        tf.keras.applications
        .mobilenet_v2
        .preprocess_input(inputs)
    )

    # Data augmentation
    x = layers.RandomFlip(
        "horizontal"
    )(x)

    x = layers.RandomRotation(
        0.08
    )(x)

    x = layers.RandomZoom(
        0.10
    )(x)

    x = layers.RandomContrast(
        0.10
    )(x)

    x = base_model(
        x,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(
        0.30
    )(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.20
    )(x)

    output = layers.Dense(
        1,
        activation="sigmoid",
        name="tomato_probability"
    )(x)

    model = keras.Model(
        inputs,
        output
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            keras.metrics.Precision(
                name="precision"
            ),
            keras.metrics.Recall(
                name="recall"
            ),
            keras.metrics.AUC(
                name="auc"
            ),
        ]
    )

    return model


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    # --------------------------------------------------------
    # FIND DATASET
    # --------------------------------------------------------

    train_dir, test_dir = discover_dataset()

    # --------------------------------------------------------
    # COLLECT TRAINING IMAGES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COLLECTING TRAINING IMAGES")
    print("=" * 70)

    train_tomato, train_non_tomato = (
        collect_binary_images(train_dir)
    )

    # Add your existing tomato dataset
    train_tomato = add_existing_tomato_images(
        train_tomato
    )

    # --------------------------------------------------------
    # COLLECT TEST IMAGES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COLLECTING TEST IMAGES")
    print("=" * 70)

    test_tomato, test_non_tomato = (
        collect_binary_images(test_dir)
    )

    # --------------------------------------------------------
    # BALANCE TRAINING DATA
    # --------------------------------------------------------

    train_tomato, train_non_tomato = (
        balance_dataset(
            train_tomato,
            train_non_tomato
        )
    )

    # --------------------------------------------------------
    # LIMIT TEST DATA
    # --------------------------------------------------------

    random.shuffle(test_tomato)
    random.shuffle(test_non_tomato)

    test_count = min(
        len(test_tomato),
        len(test_non_tomato),
        MAX_TEST_IMAGES
    )

    test_tomato = test_tomato[
        :test_count
    ]

    test_non_tomato = test_non_tomato[
        :test_count
    ]

    # --------------------------------------------------------
    # SPLIT TRAIN INTO TRAIN / VALIDATION
    # --------------------------------------------------------

    (
        train_tomato,
        train_non_tomato,
        val_tomato,
        val_non_tomato,
        _,
        _
    ) = split_images(
        train_tomato,
        train_non_tomato
    )

    print()
    print("=" * 70)
    print("FINAL SPLIT")
    print("=" * 70)

    print()
    print(
        f"Training tomato: "
        f"{len(train_tomato)}"
    )

    print(
        f"Training non-tomato: "
        f"{len(train_non_tomato)}"
    )

    print(
        f"Validation tomato: "
        f"{len(val_tomato)}"
    )

    print(
        f"Validation non-tomato: "
        f"{len(val_non_tomato)}"
    )

    print(
        f"Test tomato: "
        f"{len(test_tomato)}"
    )

    print(
        f"Test non-tomato: "
        f"{len(test_non_tomato)}"
    )

    # --------------------------------------------------------
    # CREATE TF DATASETS
    # --------------------------------------------------------

    train_ds = make_tf_dataset(
        train_tomato,
        train_non_tomato,
        shuffle=True
    )

    val_ds = make_tf_dataset(
        val_tomato,
        val_non_tomato,
        shuffle=False
    )

    test_ds = make_tf_dataset(
        test_tomato,
        test_non_tomato,
        shuffle=False
    )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = build_model()

    model.summary()

    # --------------------------------------------------------
    # MODEL DIRECTORY
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    callbacks = [

        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    model = keras.models.load_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST")
    print("=" * 70)

    results = model.evaluate(
        test_ds,
        return_dict=True,
        verbose=1
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for name, value in results.items():

        print(
            f"{name:12s}: "
            f"{value:.4f}"
        )

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()
    print("MODEL:")
    print(MODEL_PATH)

    print()
    print("The verifier predicts:")

    print(
        "1.0 = TOMATO"
    )

    print(
        "0.0 = NON-TOMATO"
    )

    print()
    print("Next step:")
    print(
        "Connect this verifier to crop_classifier.py"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()