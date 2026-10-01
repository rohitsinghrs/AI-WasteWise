from pathlib import Path
import sys
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_waste


app = FastAPI(
    title="AI WasteWise API",
    description="AI-powered waste classification and disposal recommendation API",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "AI WasteWise API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg",
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, JPEG, or PNG image."
        )

    temp_path = None

    try:

        contents = await file.read()

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp_file:

            temp_file.write(contents)
            temp_path = temp_file.name

        # Check whether uploaded file is a valid image
        image = Image.open(temp_path)
        image.verify()

        # Run AI model
        result = predict_waste(temp_path)

        return {
            "success": True,
            "filename": file.filename,
            "waste_category": result["class"],
            "confidence": round(result["confidence"], 2),
            "recommendation": result["recommendation"],
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:

        if temp_path:
            Path(temp_path).unlink(missing_ok=True)