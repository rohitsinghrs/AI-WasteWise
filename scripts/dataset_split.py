from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "split"

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


def sha256_file(path: Path) -> str:
    """Return SHA-256 hash of an image."""
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def collect_images():
    """Collect images grouped by class."""
    class_images = defaultdict(list)

    for class_dir in sorted(RAW_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        if class_dir.name.startswith(".") or class_dir.name == "__MACOSX":
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                class_images[class_dir.name].append(image_path)

    return dict(class_images)


def build_duplicate_groups(class_images):
    """
    Group exact duplicate images using SHA-256.

    A duplicate group can contain images from different classes.
    Such a group will always be assigned to one split.
    """
    hash_groups = defaultdict(list)

    for class_name, paths in class_images.items():
        for path in paths:
            digest = sha256_file(path)

            hash_groups[digest].append(
                {
                    "class": class_name,
                    "path": path,
                }
            )

    return hash_groups


def create_units(hash_groups):
    """
    Create indivisible groups.

    Every image is its own unit unless it belongs to an exact
    duplicate group. Duplicate groups become one unit.
    """
    units = []

    for digest, items in hash_groups.items():
        units.append(
            {
                "hash": digest,
                "items": items,
            }
        )

    return units


def split_units_by_class(units):
    """
    Assign duplicate units to a primary class for proportional
    allocation.

    For cross-class duplicates, the group remains indivisible.
    """
    class_units = defaultdict(list)

    for unit in units:
        classes = sorted({item["class"] for item in unit["items"]})

        # For normal images there is one class.
        # For cross-class duplicates, use the first class only
        # for deciding the target split; the entire group moves together.
        primary_class = classes[0]

        class_units[primary_class].append(unit)

    return class_units


def calculate_targets(total):
    """Calculate target image counts for each split."""
    train = round(total * TRAIN_RATIO)
    val = round(total * VAL_RATIO)
    test = total - train - val

    return {
        "train": train,
        "validation": val,
        "test": test,
    }


def allocate_units(units, target_counts):
    """
    Greedily allocate indivisible duplicate units to splits.

    The objective is to stay as close as possible to the requested
    70/15/15 image proportions while never separating duplicates.
    """
    units = sorted(
        units,
        key=lambda u: len(u["items"]),
        reverse=True,
    )

    assignments = {
        "train": [],
        "validation": [],
        "test": [],
    }

    current_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    split_order = ["train", "validation", "test"]

    for unit in units:
        size = len(unit["items"])

        best_split = min(
            split_order,
            key=lambda split: (
                current_counts[split] / max(target_counts[split], 1)
            ),
        )

        assignments[best_split].append(unit)
        current_counts[best_split] += size

    return assignments, current_counts


def flatten_assignments(assignments):
    """Convert split units into image-level rows."""
    rows = []

    for split, units in assignments.items():
        for unit in units:
            for item in unit["items"]:
                rows.append(
                    {
                        "split": split,
                        "class": item["class"],
                        "filename": item["path"].name,
                        "source_path": str(
                            item["path"].relative_to(PROJECT_ROOT)
                        ),
                        "sha256": unit["hash"],
                        "duplicate_group_size": len(unit["items"]),
                    }
                )

    return rows


def write_csv(path, rows):
    """Write dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def copy_images(rows):
    """
    Copy images from data/raw into data/processed.

    data/raw is never modified.
    """
    for row in rows:
        source = PROJECT_ROOT / row["source_path"]

        destination_dir = (
            PROCESSED_DIR
            / row["split"]
            / row["class"]
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = destination_dir / row["filename"]

        shutil.copy2(source, destination)


def write_summary(rows):
    """Create split/class summary CSV."""
    counts = defaultdict(int)

    for row in rows:
        counts[(row["split"], row["class"])] += 1

    summary_rows = []

    classes = sorted({row["class"] for row in rows})

    for split in ["train", "validation", "test"]:
        total = sum(
            counts[(split, class_name)]
            for class_name in classes
        )

        for class_name in classes:
            count = counts[(split, class_name)]

            percentage = (
                count / total * 100
                if total
                else 0
            )

            summary_rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "image_count": count,
                    "percentage_within_split": round(
                        percentage,
                        2,
                    ),
                }
            )

    write_csv(
        REPORT_DIR / "split_summary.csv",
        summary_rows,
    )


def verify_no_duplicate_leakage(rows):
    """Ensure one exact duplicate group never appears in multiple splits."""
    hash_splits = defaultdict(set)

    for row in rows:
        hash_splits[row["sha256"]].add(row["split"])

    leaked = {
        digest: splits
        for digest, splits in hash_splits.items()
        if len(splits) > 1
    }

    if leaked:
        raise RuntimeError(
            "Duplicate leakage detected! "
            f"{len(leaked)} duplicate groups span multiple splits."
        )


def main():
    print("=" * 60)
    print("AI WasteWise - Dataset Split")
    print("=" * 60)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {RAW_DIR}"
        )

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError(
            "Train/validation/test ratios must add up to 1.0."
        )

    random.seed(SEED)

    class_images = collect_images()

    total_images = sum(
        len(paths)
        for paths in class_images.values()
    )

    if total_images == 0:
        raise RuntimeError(
            "No images found in data/raw."
        )

    print("\nClasses:")
    for class_name, paths in sorted(class_images.items()):
        print(
            f"  {class_name:<15}: {len(paths):>4}"
        )

    print(f"\nTotal images: {total_images}")

    hash_groups = build_duplicate_groups(
        class_images
    )

    units = create_units(hash_groups)

    duplicate_groups = [
        unit
        for unit in units
        if len(unit["items"]) > 1
    ]

    cross_class_groups = [
        unit
        for unit in duplicate_groups
        if len(
            {item["class"] for item in unit["items"]}
        ) > 1
    ]

    print(
        f"Exact duplicate groups: "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Cross-class duplicate groups: "
        f"{len(cross_class_groups)}"
    )

    target_counts = calculate_targets(
        total_images
    )

    print("\nTarget split:")
    for split, count in target_counts.items():
        print(
            f"  {split:<12}: {count:>4}"
        )

    # Shuffle units deterministically before allocation.
    random.shuffle(units)

    assignments, actual_counts = allocate_units(
        units,
        target_counts,
    )

    rows = flatten_assignments(
        assignments
    )

    verify_no_duplicate_leakage(rows)

    # Clean only the generated processed directory.
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    if REPORT_DIR.exists():
        shutil.rmtree(REPORT_DIR)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nCopying images...")
    copy_images(rows)

    # Split-specific CSVs.
    for split in ["train", "validation", "test"]:
        split_rows = [
            row
            for row in rows
            if row["split"] == split
        ]

        write_csv(
            REPORT_DIR / f"{split}.csv",
            split_rows,
        )

    write_csv(
        REPORT_DIR / "all_splits.csv",
        rows,
    )

    write_summary(rows)

    print("\nACTUAL SPLIT")
    print("-" * 35)

    for split in ["train", "validation", "test"]:
        count = sum(
            1
            for row in rows
            if row["split"] == split
        )

        percentage = (
            count / total_images * 100
        )

        print(
            f"{split:<12}: "
            f"{count:>4} images "
            f"({percentage:>6.2f}%)"
        )

    print("\nOUTPUTS")
    print("-" * 35)
    print(
        f"Processed dataset: "
        f"{PROCESSED_DIR.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Split reports: "
        f"{REPORT_DIR.relative_to(PROJECT_ROOT)}"
    )

    print("\nGenerated CSV files:")
    print("  - train.csv")
    print("  - validation.csv")
    print("  - test.csv")
    print("  - all_splits.csv")
    print("  - split_summary.csv")

    print("\nDuplicate leakage check: PASSED")
    print("data/raw was left untouched.")
    print("\nDataset split completed successfully.")


if __name__ == "__main__":
    main()