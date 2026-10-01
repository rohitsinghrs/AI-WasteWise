from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from PIL import Image


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "wastewise_mobilenetv2.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]


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


print("Loading AI WasteWise model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")


def predict_waste(image_path):
    """
    Predict the waste category from an image.
    """

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    # IMPORTANT:
    # The trained model expects raw pixel values (0-255).
    image_array = np.array(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(predictions[0][predicted_index]) * 100

    recommendation = RECOMMENDATIONS[predicted_class]

    return {
        "class": predicted_class,
        "confidence": confidence,
        "recommendation": recommendation,
    }


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("python src/predict.py <image_path>")
        print("\nExample:")
        print("python src/predict.py data/raw/plastic/plastic1.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"\nERROR: Image not found:")
        print(image_path)
        sys.exit(1)

    result = predict_waste(image_path)

    print("\n" + "=" * 60)
    print("AI WASTEWISE PREDICTION")
    print("=" * 60)

    print(f"\nWaste Category : {result['class'].upper()}")
    print(f"Confidence     : {result['confidence']:.2f}%")

    print("\nRecommendation:")
    print(result["recommendation"])

    print("\n" + "=" * 60)