from pathlib import Path

import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ],
    name="data_augmentation",
)


# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():
    """
    Load train, validation and test datasets.
    """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Save class names before map/prefetch changes the dataset object
    class_names = train_ds.class_names

    return train_ds, validation_ds, test_ds, class_names


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_datasets(train_ds, validation_ds, test_ds):

    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(
        lambda images, labels: (
            normalization(images),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    validation_ds = validation_ds.map(
        lambda images, labels: (
            normalization(images),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    test_ds = test_ds.map(
        lambda images, labels: (
            normalization(images),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return train_ds, validation_ds, test_ds


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

def optimize_datasets(train_ds, validation_ds, test_ds):

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, validation_ds, test_ds


# ============================================================
# COMPLETE DATA PIPELINE
# ============================================================

def get_data():

    train_ds, validation_ds, test_ds, class_names = load_datasets()

    train_ds, validation_ds, test_ds = preprocess_datasets(
        train_ds,
        validation_ds,
        test_ds,
    )

    # Apply augmentation ONLY to training data
    train_ds = train_ds.map(
        lambda images, labels: (
            data_augmentation(images, training=True),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds, validation_ds, test_ds = optimize_datasets(
        train_ds,
        validation_ds,
        test_ds,
    )

    return train_ds, validation_ds, test_ds, class_names


# ============================================================
# TEST PIPELINE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI WasteWise - Data Pipeline Test")
    print("=" * 60)

    print("\nDataset directories:")
    print(f"Train      : {TRAIN_DIR}")
    print(f"Validation : {VALIDATION_DIR}")
    print(f"Test       : {TEST_DIR}")

    train_ds, validation_ds, test_ds, class_names = get_data()

    print("\nClass names:")
    print(class_names)

    print("\nNumber of classes:")
    print(len(class_names))

    print("\nChecking first training batch...")

    images, labels = next(iter(train_ds))

    print(f"Image batch shape : {images.shape}")
    print(f"Label batch shape : {labels.shape}")

    print(
        f"Minimum pixel value : "
        f"{tf.reduce_min(images).numpy():.4f}"
    )

    print(
        f"Maximum pixel value : "
        f"{tf.reduce_max(images).numpy():.4f}"
    )

    print("\n" + "=" * 60)
    print("DATA PIPELINE TEST PASSED")
    print("=" * 60)