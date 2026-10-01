import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_waste


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI WasteWise",
    page_icon="♻️",
    layout="centered",
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("♻️ AI WasteWise")

st.subheader("AI-Powered Waste Segregation Assistant")

st.write(
    "Upload an image of a waste item and AI WasteWise will "
    "identify its category and provide a responsible disposal recommendation."
)


# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"],
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch",
    )

    if st.button("🔍 Analyze Waste", type="primary"):

        with st.spinner("AI is analyzing the image..."):

            # Save uploaded image temporarily in memory
            import tempfile

            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False
            ) as temp_file:

                image.save(temp_file.name)

                result = predict_waste(temp_file.name)

        # ------------------------------------------
        # Result
        # ------------------------------------------

        st.success("Analysis completed!")

        st.divider()

        st.subheader("🤖 Prediction")

        st.write(
            f"### Waste Category: **{result['class'].upper()}**"
        )

        confidence = result["confidence"]

        st.metric(
            label="AI Confidence",
            value=f"{confidence:.2f}%"
        )

        # ------------------------------------------
        # Confidence warning
        # ------------------------------------------

        if confidence < 60:

            st.warning(
                "⚠️ The AI confidence is relatively low. "
                "Please manually verify the item before disposal."
            )

        elif confidence < 80:

            st.info(
                "ℹ️ The prediction has moderate confidence."
            )

        else:

            st.success(
                "✅ The prediction has high confidence."
            )

        # ------------------------------------------
        # Recommendation
        # ------------------------------------------

        st.divider()

        st.subheader("♻️ Responsible Disposal Recommendation")

        st.info(result["recommendation"])


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI WasteWise | AICTE × IBM SkillsBuild Machine Learning & Applied AI Internship 2026"
)