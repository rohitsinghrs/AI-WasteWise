from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "wastewise_mobilenetv2.keras"

REPORT_DIR = PROJECT_ROOT / "reports" / "evaluation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("AI WasteWise - Model Evaluation")
print("=" * 60)

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = test_ds.class_names

print(f"\nClasses: {class_names}")
print(f"Number of classes: {len(class_names)}")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0,
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1,
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
)

print("\nClassification Report:")
print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = REPORT_DIR / "classification_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:

    file.write("AI WasteWise - Model Evaluation\n")
    file.write("=" * 60 + "\n\n")

    file.write(
        f"Test Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Test Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write("-" * 60 + "\n")

    file.write(report)

    file.write("\n\nConfusion Matrix\n")
    file.write("-" * 60 + "\n")

    file.write(
        np.array2string(cm)
    )


print("\nReport saved to:")
print(report_path)

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETE")
print("=" * 60)