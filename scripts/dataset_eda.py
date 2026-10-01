"""
AI WasteWise - Dataset EDA
Run from the project root:
    python scripts/dataset_eda.py

Expected dataset layout:
    data/raw/
        cardboard/
        glass/
        metal/
        paper/
        plastic/
        trash/

Outputs:
    reports/eda/
        class_distribution.png
        image_dimensions.png
        sample_images.png
        duplicate_content.csv
        image_inventory.csv
        class_summary.csv
        eda_summary.txt
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports" / "eda"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SAMPLE_PER_CLASS = 5

# Known classes from the current TrashNet dataset.
EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_images() -> tuple[list[dict], dict[str, list[Path]]]:
    class_images: dict[str, list[Path]] = {}
    inventory: list[dict] = []

    for class_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        if class_dir.name.startswith(".") or class_dir.name == "__MACOSX":
            continue

        paths = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        class_images[class_dir.name] = paths

        for path in paths:
            row = {
                "class": class_dir.name,
                "filename": path.name,
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "width": "",
                "height": "",
                "mode": "",
                "format": "",
                "file_size_bytes": path.stat().st_size,
                "sha256": "",
                "readable": False,
                "error": "",
            }

            try:
                with Image.open(path) as img:
                    img.verify()

                with Image.open(path) as img:
                    row["width"], row["height"] = img.size
                    row["mode"] = img.mode
                    row["format"] = img.format

                row["sha256"] = sha256_file(path)
                row["readable"] = True

            except (UnidentifiedImageError, OSError, ValueError) as exc:
                row["error"] = str(exc)

            inventory.append(row)

    return inventory, class_images


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_duplicates(inventory: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)

    for row in inventory:
        if row["readable"] and row["sha256"]:
            groups[row["sha256"]].append(row)

    duplicate_rows = []

    for digest, items in groups.items():
        classes = sorted({item["class"] for item in items})

        if len(items) > 1:
            for item in items:
                duplicate_rows.append(
                    {
                        "sha256": digest,
                        "class": item["class"],
                        "filename": item["filename"],
                        "relative_path": item["relative_path"],
                        "group_size": len(items),
                        "classes_in_group": "|".join(classes),
                        "cross_class": len(classes) > 1,
                    }
                )

    return duplicate_rows


def class_statistics(class_images: dict[str, list[Path]]) -> list[dict]:
    counts = {name: len(paths) for name, paths in class_images.items()}
    total = sum(counts.values())

    rows = []
    max_count = max(counts.values(), default=0)
    min_count = min(counts.values(), default=0)

    for class_name in sorted(counts):
        count = counts[class_name]
        rows.append(
            {
                "class": class_name,
                "image_count": count,
                "percentage": round((count / total) * 100, 2) if total else 0,
                "relative_to_largest": round(count / max_count, 4)
                if max_count else 0,
                "shortfall_from_largest": max_count - count,
            }
        )

    return rows


def plot_class_distribution(stats: list[dict]) -> None:
    names = [r["class"] for r in stats]
    counts = [r["image_count"] for r in stats]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, counts)
    plt.title("AI WasteWise - Dataset Class Distribution")
    plt.xlabel("Waste Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=20)

    for bar, value in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            str(value),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "class_distribution.png", dpi=180)
    plt.close()


def plot_dimensions(inventory: list[dict]) -> None:
    widths = [r["width"] for r in inventory if r["readable"]]
    heights = [r["height"] for r in inventory if r["readable"]]

    if not widths:
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.35, s=14)
    plt.title("AI WasteWise - Image Dimensions")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "image_dimensions.png", dpi=180)
    plt.close()


def plot_samples(class_images: dict[str, list[Path]]) -> None:
    classes = sorted(class_images)

    if not classes:
        return

    rows = len(classes)
    cols = SAMPLE_PER_CLASS

    fig, axes = plt.subplots(
        rows, cols, figsize=(15, max(3, rows * 3))
    )

    if rows == 1:
        axes = np.array([axes])

    for row_idx, class_name in enumerate(classes):
        samples = class_images[class_name][:SAMPLE_PER_CLASS]

        for col_idx in range(cols):
            ax = axes[row_idx, col_idx]
            ax.axis("off")

            if col_idx >= len(samples):
                continue

            path = samples[col_idx]

            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    ax.imshow(img)

                ax.set_title(path.name, fontsize=8)
            except Exception:
                ax.set_title("Unreadable", fontsize=8)

            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=10)

    fig.suptitle("AI WasteWise - Sample Images by Class", fontsize=14)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "sample_images.png", dpi=180)
    plt.close()


def imbalance_summary(stats: list[dict]) -> dict:
    counts = [r["image_count"] for r in stats]

    if not counts:
        return {
            "total": 0,
            "max": 0,
            "min": 0,
            "ratio": 0,
            "status": "No images found",
        }

    maximum = max(counts)
    minimum = min(counts)
    ratio = maximum / minimum if minimum else math.inf

    if ratio >= 3:
        status = "Significant class imbalance"
    elif ratio >= 1.5:
        status = "Moderate class imbalance"
    else:
        status = "Relatively balanced"

    return {
        "total": sum(counts),
        "max": maximum,
        "min": minimum,
        "ratio": ratio,
        "status": status,
    }


def write_summary(
    stats: list[dict],
    inventory: list[dict],
    duplicates: list[dict],
) -> None:
    readable = sum(1 for r in inventory if r["readable"])
    unreadable = len(inventory) - readable
    imbalance = imbalance_summary(stats)

    duplicate_groups = defaultdict(list)
    for row in duplicates:
        duplicate_groups[row["sha256"]].append(row)

    cross_class_groups = sum(
        1
        for items in duplicate_groups.values()
        if len({item["class"] for item in items}) > 1
    )

    summary_path = REPORT_DIR / "eda_summary.txt"

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("AI WasteWise - Dataset EDA Summary\n")
        f.write("=" * 42 + "\n\n")

        f.write(f"Dataset location: {RAW_DIR}\n")
        f.write(f"Classes detected: {len(stats)}\n")
        f.write(f"Total images: {imbalance['total']}\n")
        f.write(f"Readable images: {readable}\n")
        f.write(f"Unreadable images: {unreadable}\n\n")

        f.write("Class distribution\n")
        f.write("-" * 20 + "\n")
        for row in stats:
            f.write(
                f"{row['class']:<15} "
                f"{row['image_count']:>5} images "
                f"({row['percentage']:>6.2f}%)\n"
            )

        f.write("\nImbalance analysis\n")
        f.write("-" * 20 + "\n")
        f.write(f"Maximum class size: {imbalance['max']}\n")
        f.write(f"Minimum class size: {imbalance['min']}\n")
        f.write(f"Imbalance ratio: {imbalance['ratio']:.2f}\n")
        f.write(f"Assessment: {imbalance['status']}\n")

        f.write("\nDuplicate-content analysis\n")
        f.write("-" * 28 + "\n")
        f.write(f"Duplicate image rows: {len(duplicates)}\n")
        f.write(f"Duplicate groups: {len(duplicate_groups)}\n")
        f.write(f"Cross-class duplicate groups: {cross_class_groups}\n")

        if cross_class_groups:
            f.write("\nCross-class duplicate groups:\n")
            for digest, items in duplicate_groups.items():
                classes = {item["class"] for item in items}
                if len(classes) > 1:
                    f.write(f"\nSHA256: {digest}\n")
                    for item in items:
                        f.write(
                            f"  {item['class']}/{item['filename']}\n"
                        )

        f.write("\nExpected TrashNet classes\n")
        f.write("-" * 27 + "\n")
        missing = [c for c in EXPECTED_CLASSES if c not in {r["class"] for r in stats}]
        extra = [c for c in {r["class"] for r in stats} if c not in EXPECTED_CLASSES]
        f.write(f"Missing expected classes: {missing or 'None'}\n")
        f.write(f"Unexpected classes: {sorted(extra) or 'None'}\n")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {RAW_DIR}\n"
            "Expected: data/raw/<class_name>/<image files>"
        )

    print("=" * 60)
    print("AI WasteWise - Dataset EDA")
    print("=" * 60)
    print(f"Dataset: {RAW_DIR}")
    print(f"Reports: {REPORT_DIR}\n")

    inventory, class_images = collect_images()

    if not inventory:
        raise RuntimeError("No supported image files were found in data/raw.")

    stats = class_statistics(class_images)
    duplicates = analyze_duplicates(inventory)

    save_csv(REPORT_DIR / "image_inventory.csv", inventory)
    save_csv(REPORT_DIR / "class_summary.csv", stats)
    save_csv(REPORT_DIR / "duplicate_content.csv", duplicates)

    plot_class_distribution(stats)
    plot_dimensions(inventory)
    plot_samples(class_images)
    write_summary(stats, inventory, duplicates)

    imbalance = imbalance_summary(stats)

    print("CLASS DISTRIBUTION")
    print("-" * 30)
    for row in stats:
        print(
            f"{row['class']:<15}: "
            f"{row['image_count']:>4} images "
            f"({row['percentage']:.2f}%)"
        )

    print("\nDATASET SUMMARY")
    print("-" * 30)
    print(f"Total images       : {imbalance['total']}")
    print(f"Readable images    : {sum(r['readable'] for r in inventory)}")
    print(f"Unreadable images  : {sum(not r['readable'] for r in inventory)}")
    print(f"Duplicate rows     : {len(duplicates)}")
    print(f"Imbalance ratio    : {imbalance['ratio']:.2f}")
    print(f"Imbalance analysis : {imbalance['status']}")

    print("\nOUTPUTS")
    print("-" * 30)
    for path in sorted(REPORT_DIR.iterdir()):
        if path.is_file():
            print(f"- {path.relative_to(PROJECT_ROOT)}")

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()
