from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "wastewise_mobilenetv2.keras"


# ============================================================
# Model Configuration
# ============================================================

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]


# ============================================================
# Waste Disposal Recommendations
# ============================================================

RECOMMENDATIONS = {
    "cardboard": (
        "Flatten the cardboard if possible and place it in the "
        "appropriate paper/cardboard recycling stream. Keep it dry and clean."
    ),

    "glass": (
        "Keep glass separate and place it in the designated glass-recycling "
        "stream where available. Handle broken glass carefully."
    ),

    "metal": (
        "Empty and rinse the item if needed, then place it in the "
        "appropriate recyclable metal stream where accepted."
    ),

    "paper": (
        "Keep paper dry and clean and place it in the appropriate "
        "paper-recycling stream."
    ),

    "plastic": (
        "Empty and clean the plastic item if needed, then place it in the "
        "appropriate plastic-recycling stream where accepted."
    ),

    "trash": (
        "Place it in general/residual waste unless your local waste system "
        "provides a separate option. Check local waste guidelines when unsure."
    ),
}


# ============================================================
# Load Model
# ============================================================

print("Loading AI WasteWise model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# Prediction Function
# ============================================================

def predict_waste(image_path):
    """
    Predict the waste category from an image.

    Returns:
        dict:
            class           -> Top predicted class
            confidence      -> Confidence of top prediction
            recommendation  -> Disposal recommendation
            top_predictions -> Top 3 predictions with confidence
    """

    # --------------------------------------------------------
    # Load Image
    # --------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    # --------------------------------------------------------
    # Resize Image
    # --------------------------------------------------------

    image = image.resize(IMAGE_SIZE)

    # --------------------------------------------------------
    # Convert Image To NumPy Array
    #
    # IMPORTANT:
    # The trained model expects raw pixel values (0-255).
    # --------------------------------------------------------

    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # --------------------------------------------------------
    # Model Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # Get Top 3 Predictions
    # --------------------------------------------------------

    top_indices = np.argsort(predictions)[::-1][:3]

    # --------------------------------------------------------
    # Top Prediction
    # --------------------------------------------------------

    predicted_index = int(top_indices[0])

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        predictions[predicted_index]
    ) * 100

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    recommendation = RECOMMENDATIONS[predicted_class]

    # --------------------------------------------------------
    # Build Top 3 Prediction List
    # --------------------------------------------------------

    top_predictions = []

    for index in top_indices:

        index = int(index)

        top_predictions.append(
            {
                "class": CLASS_NAMES[index],
                "confidence": float(predictions[index]) * 100,
            }
        )

    # --------------------------------------------------------
    # Return Result
    # --------------------------------------------------------

    return {
        "class": predicted_class,
        "confidence": confidence,
        "recommendation": recommendation,
        "top_predictions": top_predictions,
    }


# ============================================================
# Command Line Testing
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Check Image Argument
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print("\nUsage:")

        print(
            "python src/predict.py <image_path>"
        )

        print("\nExample:")

        print(
            "python src/predict.py "
            "data/raw/plastic/plastic1.jpg"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Get Image Path
    # --------------------------------------------------------

    image_path = sys.argv[1]

    # --------------------------------------------------------
    # Check File Exists
    # --------------------------------------------------------

    if not Path(image_path).exists():

        print("\nERROR: Image not found:")

        print(image_path)

        sys.exit(1)

    # --------------------------------------------------------
    # Run Prediction
    # --------------------------------------------------------

    result = predict_waste(image_path)

    # --------------------------------------------------------
    # Display Result
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print("AI WASTEWISE PREDICTION")

    print("=" * 60)

    print(
        f"\nWaste Category : "
        f"{result['class'].upper()}"
    )

    print(
        f"Confidence     : "
        f"{result['confidence']:.2f}%"
    )

    # --------------------------------------------------------
    # Top 3 Predictions
    # --------------------------------------------------------

    print("\nTop 3 Predictions:")

    for rank, prediction in enumerate(
        result["top_predictions"],
        start=1
    ):

        print(
            f"{rank}. "
            f"{prediction['class'].upper():<12} "
            f"{prediction['confidence']:.2f}%"
        )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    print("\nRecommendation:")

    print(result["recommendation"])

    print("\n" + "=" * 60)