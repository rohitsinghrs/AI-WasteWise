from pathlib import Path
from collections import Counter
from PIL import Image
import hashlib
import os

# ============================================================
# AI WasteWise - Dataset Verification Script
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".gif",
}

SPLIT_NAMES = {
    "train",
    "val",
    "validation",
    "test",
}


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def get_image_files(folder):
    """Return all supported image files recursively."""
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def calculate_hash(file_path):
    """Calculate MD5 hash for duplicate-content detection."""
    md5 = hashlib.md5()

    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                md5.update(chunk)

        return md5.hexdigest()

    except Exception:
        return None


def verify_image(file_path):
    """Check whether an image can actually be opened."""
    try:
        with Image.open(file_path) as img:
            img.verify()

        return True, ""

    except Exception as e:
        return False, str(e)


# ------------------------------------------------------------
# Dataset structure detection
# ------------------------------------------------------------

def detect_structure():
    folders = [
        p for p in DATASET_DIR.iterdir()
        if p.is_dir()
    ]

    folder_names = {p.name.lower() for p in folders}

    has_splits = bool(folder_names & SPLIT_NAMES)

    if has_splits:
        return "SPLIT"

    return "CLASS"


# ------------------------------------------------------------
# CLASS-FOLDER DATASET
# ------------------------------------------------------------

def check_class_dataset():
    print("\n" + "=" * 70)
    print("CLASS-FOLDER DATASET VERIFICATION")
    print("=" * 70)

    class_folders = [
        p for p in DATASET_DIR.iterdir()
        if p.is_dir()
    ]

    if not class_folders:
        print("\n❌ No class folders found.")
        print(f"Expected folders inside: {DATASET_DIR}")
        return

    print(f"\nDataset location: {DATASET_DIR}")
    print(f"Classes detected: {len(class_folders)}")

    print("\nClasses:")
    for folder in sorted(class_folders):
        print(f"  - {folder.name}")

    print("\n" + "-" * 70)
    print("IMAGE COUNTS")
    print("-" * 70)

    total_images = 0
    class_counts = {}

    for class_folder in sorted(class_folders):

        images = get_image_files(class_folder)

        class_counts[class_folder.name] = len(images)
        total_images += len(images)

        print(f"{class_folder.name:20} : {len(images):6} images")

    print("-" * 70)
    print(f"{'TOTAL':20} : {total_images:6} images")

    # --------------------------------------------------------
    # Empty classes
    # --------------------------------------------------------

    empty_classes = [
        name for name, count in class_counts.items()
        if count == 0
    ]

    print("\n" + "-" * 70)
    print("EMPTY CLASS CHECK")
    print("-" * 70)

    if empty_classes:
        print("❌ Empty classes:")
        for name in empty_classes:
            print(f"   - {name}")
    else:
        print("✅ No empty classes.")

    # --------------------------------------------------------
    # Corrupted images
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CORRUPTED IMAGE CHECK")
    print("-" * 70)

    corrupted = []

    all_images = get_image_files(DATASET_DIR)

    for image_path in all_images:
        valid, error = verify_image(image_path)

        if not valid:
            corrupted.append((image_path, error))

    if corrupted:
        print(f"❌ Corrupted/unreadable images: {len(corrupted)}")

        for path, error in corrupted[:20]:
            print(f"\n   {path}")
            print(f"   Error: {error}")

        if len(corrupted) > 20:
            print(f"\n   ... and {len(corrupted) - 20} more.")

    else:
        print("✅ No corrupted images found.")

    # --------------------------------------------------------
    # Duplicate filename check
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DUPLICATE FILENAME CHECK")
    print("-" * 70)

    filename_map = {}

    for image_path in all_images:
        filename = image_path.name.lower()

        filename_map.setdefault(filename, []).append(image_path)

    duplicate_filenames = {
        name: paths
        for name, paths in filename_map.items()
        if len(paths) > 1
    }

    if duplicate_filenames:

        print(
            f"⚠️ Duplicate filenames found: "
            f"{len(duplicate_filenames)}"
        )

        for name, paths in list(duplicate_filenames.items())[:20]:

            print(f"\n   {name}")

            for path in paths:
                print(f"      {path}")

    else:
        print("✅ No duplicate filenames found.")

    # --------------------------------------------------------
    # Duplicate image content check
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DUPLICATE IMAGE CONTENT CHECK")
    print("-" * 70)

    hash_map = {}

    for image_path in all_images:

        file_hash = calculate_hash(image_path)

        if file_hash:
            hash_map.setdefault(file_hash, []).append(image_path)

    duplicate_images = {
        h: paths
        for h, paths in hash_map.items()
        if len(paths) > 1
    }

    if duplicate_images:

        print(
            f"⚠️ Duplicate image contents found: "
            f"{len(duplicate_images)} groups"
        )

        for paths in list(duplicate_images.values())[:10]:

            print("\n   Duplicate group:")

            for path in paths:
                print(f"      {path}")

    else:
        print("✅ No duplicate image contents found.")

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CLASS BALANCE CHECK")
    print("-" * 70)

    if class_counts:

        maximum = max(class_counts.values())
        minimum = min(class_counts.values())

        print(f"Maximum class size : {maximum}")
        print(f"Minimum class size : {minimum}")

        if minimum > 0:

            imbalance_ratio = maximum / minimum

            print(f"Imbalance ratio    : {imbalance_ratio:.2f}")

            if imbalance_ratio <= 1.5:
                print("✅ Classes are relatively balanced.")

            elif imbalance_ratio <= 3:
                print("⚠️ Moderate class imbalance.")

            else:
                print("⚠️ Significant class imbalance.")

    # --------------------------------------------------------
    # Train / Validation / Test readiness
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAIN / VALIDATION / TEST READINESS")
    print("=" * 70)

    if total_images == 0:

        print("❌ Dataset contains no images.")
        return

    print("\nRecommended split:")
    print("  Train      : 70%")
    print("  Validation : 15%")
    print("  Test       : 15%")

    train_count = int(total_images * 0.70)
    val_count = int(total_images * 0.15)
    test_count = total_images - train_count - val_count

    print("\nApproximate image distribution:")
    print(f"  Train      : {train_count}")
    print(f"  Validation : {val_count}")
    print(f"  Test       : {test_count}")

    if len(class_folders) >= 2:
        print("\n✅ Enough classes detected for classification.")

    if total_images >= 100:
        print("✅ Dataset has enough images to begin preprocessing.")

    else:
        print("⚠️ Dataset is very small. More images are recommended.")

    print("\n" + "=" * 70)
    print("DATASET VERIFICATION COMPLETE")
    print("=" * 70)


# ------------------------------------------------------------
# PRE-SPLIT DATASET
# ------------------------------------------------------------

def check_split_dataset():

    print("\n" + "=" * 70)
    print("TRAIN / VALIDATION / TEST DATASET VERIFICATION")
    print("=" * 70)

    split_folders = {}

    for folder in DATASET_DIR.iterdir():

        if folder.is_dir():

            name = folder.name.lower()

            if name in SPLIT_NAMES:
                split_folders[name] = folder

    print(f"\nDataset location: {DATASET_DIR}")

    for split_name, split_folder in split_folders.items():

        images = get_image_files(split_folder)

        print("\n" + "-" * 50)
        print(f"{split_name.upper()}")
        print("-" * 50)

        print(f"Total images: {len(images)}")

        class_counts = Counter(
            image_path.parent.name
            for image_path in images
        )

        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name:20} : {count}")

    # --------------------------------------------------------
    # Corrupted files
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CORRUPTED IMAGE CHECK")
    print("-" * 70)

    all_images = get_image_files(DATASET_DIR)

    corrupted = []

    for image_path in all_images:

        valid, error = verify_image(image_path)

        if not valid:
            corrupted.append((image_path, error))

    if corrupted:

        print(f"❌ Corrupted images: {len(corrupted)}")

        for path, error in corrupted[:20]:
            print(f"   {path}")

    else:

        print("✅ No corrupted images found.")

    # --------------------------------------------------------
    # Duplicate filenames
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DUPLICATE FILENAME CHECK")
    print("-" * 70)

    filename_map = {}

    for image_path in all_images:

        filename_map.setdefault(
            image_path.name.lower(),
            []
        ).append(image_path)

    duplicate_filenames = {
        name: paths
        for name, paths in filename_map.items()
        if len(paths) > 1
    }

    if duplicate_filenames:

        print(
            f"⚠️ Duplicate filenames: "
            f"{len(duplicate_filenames)}"
        )

    else:

        print("✅ No duplicate filenames found.")

    # --------------------------------------------------------
    # Split readiness
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT READINESS")
    print("=" * 70)

    normalized = set(split_folders.keys())

    has_train = "train" in normalized
    has_validation = (
        "validation" in normalized
        or "val" in normalized
    )
    has_test = "test" in normalized

    if has_train and has_validation and has_test:

        print("✅ Train / Validation / Test folders detected.")

    else:

        print("⚠️ Missing one or more dataset splits.")

        if not has_train:
            print("   ❌ Train folder missing.")

        if not has_validation:
            print("   ❌ Validation/Val folder missing.")

        if not has_test:
            print("   ❌ Test folder missing.")

    print("\n" + "=" * 70)
    print("DATASET VERIFICATION COMPLETE")
    print("=" * 70)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("\n")
    print("=" * 70)
    print("AI WASTEWISE DATASET VERIFICATION")
    print("=" * 70)

    print(f"\nDataset path:")
    print(DATASET_DIR)

    if not DATASET_DIR.exists():

        print("\n❌ Dataset folder does not exist.")

        print("\nCreate it using:")
        print("  data/raw/")

        return

    structure = detect_structure()

    print(f"\nDetected dataset structure: {structure}")

    if structure == "SPLIT":
        check_split_dataset()

    else:
        check_class_dataset()


if __name__ == "__main__":
    main()