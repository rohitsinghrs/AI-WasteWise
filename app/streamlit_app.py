import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# Add Project Root To Python Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

from src.predict import predict_waste

from src.database import (
    initialize_database,
    save_prediction,
    get_predictions,
    get_total_predictions,
    get_average_confidence,
    get_category_distribution,
    clear_predictions,
)


# ============================================================
# Initialize Database
# ============================================================

initialize_database()


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AI WasteWise",
    page_icon="♻️",
    layout="centered",
)


# ============================================================
# Header
# ============================================================

st.title("♻️ AI WasteWise")

st.subheader(
    "AI-Powered Waste Segregation Assistant"
)

st.write(
    "Upload an image of a waste item and AI WasteWise "
    "will identify its category and provide a responsible "
    "disposal recommendation."
)


# ============================================================
# Dashboard Refresh
# ============================================================

if st.button(
    "🔄 Refresh Dashboard",
    use_container_width=True,
):
    st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

st.divider()

st.header("📊 WasteWise Analytics Dashboard")


# ============================================================
# Get Database Statistics
# ============================================================

total_predictions = get_total_predictions()

average_confidence = get_average_confidence()

category_distribution = get_category_distribution()


# ============================================================
# Dashboard Metrics
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        label="🔢 Total Predictions",
        value=total_predictions,
    )


with col2:

    st.metric(
        label="🎯 Average Confidence",
        value=f"{average_confidence:.2f}%",
    )


with col3:

    if category_distribution:

        most_detected = max(
            category_distribution,
            key=category_distribution.get,
        )

        st.metric(
            label="🏆 Most Detected",
            value=most_detected.upper(),
        )

    else:

        st.metric(
            label="🏆 Most Detected",
            value="N/A",
        )


# ============================================================
# No Data Message
# ============================================================

if total_predictions == 0:

    st.info(
        "📭 No predictions have been recorded yet. "
        "Upload a waste image and run an analysis first."
    )


else:

    # ========================================================
    # Category Distribution
    # ========================================================

    st.divider()

    st.subheader(
        "♻️ Category Distribution"
    )

    st.caption(
        "Number of predictions recorded for each waste category."
    )


    if category_distribution:

        category_df = pd.DataFrame(
            {
                "Category": list(
                    category_distribution.keys()
                ),
                "Predictions": list(
                    category_distribution.values()
                ),
            }
        )

        category_df = category_df.set_index(
            "Category"
        )

        st.bar_chart(
            category_df
        )


    # ========================================================
    # Category Details
    # ========================================================

    st.subheader(
        "📋 Category Statistics"
    )

    category_table = pd.DataFrame(
        [
            {
                "Category": category.upper(),
                "Predictions": count,
                "Percentage": (
                    count / total_predictions
                ) * 100,
            }
            for category, count
            in category_distribution.items()
        ]
    )

    if not category_table.empty:

        category_table["Percentage"] = (
            category_table["Percentage"]
            .round(2)
            .astype(str)
            + "%"
        )

        st.dataframe(
            category_table,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # Prediction History
    # ========================================================

    st.divider()

    st.subheader(
        "📜 Prediction History"
    )


    predictions = get_predictions(
        limit=100
    )


    if predictions:

        history_df = pd.DataFrame(
            predictions
        )


        # ----------------------------------------------------
        # Convert timestamp
        # ----------------------------------------------------

        history_df["timestamp"] = pd.to_datetime(
            history_df["timestamp"]
        )


        # ----------------------------------------------------
        # Confidence Analysis
        # ----------------------------------------------------

        low_confidence_count = len(
            history_df[
                history_df["confidence"] < 60
            ]
        )

        high_confidence_count = len(
            history_df[
                history_df["confidence"] >= 80
            ]
        )


        metric1, metric2 = st.columns(2)


        with metric1:

            st.metric(
                "⚠️ Low Confidence (<60%)",
                low_confidence_count,
            )


        with metric2:

            st.metric(
                "✅ High Confidence (≥80%)",
                high_confidence_count,
            )


        # ====================================================
        # Confidence Over Predictions
        # ====================================================

        st.subheader(
            "📈 Confidence Analysis"
        )

        confidence_df = (
            history_df
            .sort_values("id")
            .reset_index(drop=True)
        )

        # Use sequential positions (1, 2, 3...) instead of database IDs.
        confidence_df["Prediction Number"] = range(
            1, len(confidence_df) + 1
        )

        # Matplotlib lets us explicitly control both axes.
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            confidence_df["Prediction Number"],
            confidence_df["confidence"],
            marker="o",
            linewidth=2,
        )
        ax.set_xlabel("Prediction Number")
        ax.set_ylabel("Confidence (%)")
        ax.set_title("Model Confidence for Each Prediction")
        ax.set_ylim(0, 100)
        ax.set_xlim(1, max(2, len(confidence_df)))
        ax.set_xticks(confidence_df["Prediction Number"].tolist())
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


        # ====================================================
        # Predictions Over Time
        # ====================================================

        st.subheader(
            "🕐 Predictions Over Time"
        )


        timeline_df = (
            history_df
            .set_index("timestamp")
            .resample("D")
            .size()
        )

        if not timeline_df.empty:
            # Add missing calendar days with a count of zero.
            full_dates = pd.date_range(
                start=timeline_df.index.min().normalize(),
                end=timeline_df.index.max().normalize(),
                freq="D",
            )

            timeline_df = timeline_df.reindex(
                full_dates,
                fill_value=0,
            )

            timeline_df.index.name = "Date"
            timeline_df = timeline_df.to_frame(
                name="Predictions"
            )

            st.line_chart(
                timeline_df,
                y="Predictions",
                y_label="Predictions per day",
            )


        # ====================================================
        # Detailed History Table
        # ====================================================

        st.subheader(
            "📋 Detailed Prediction History"
        )


        display_df = history_df[
            [
                "id",
                "timestamp",
                "image_name",
                "category",
                "confidence",
                "recommendation",
            ]
        ].copy()


        display_df["category"] = (
            display_df["category"]
            .str.upper()
        )


        display_df["confidence"] = (
            display_df["confidence"]
            .round(2)
        )


        display_df = display_df.rename(
            columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "image_name": "Image",
                "category": "Category",
                "confidence": "Confidence (%)",
                "recommendation": "Recommendation",
            }
        )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# Upload Image
# ============================================================

st.divider()

st.header(
    "🔍 Analyze New Waste"
)


uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


# ============================================================
# Prediction
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Open Image
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # Display Image
    # --------------------------------------------------------

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch",
    )


    # --------------------------------------------------------
    # Analyze Button
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyze Waste",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "🤖 AI is analyzing the image..."
        ):

            # ------------------------------------------------
            # Temporary Image
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False,
            ) as temp_file:

                image.save(
                    temp_file.name
                )

                temp_image_path = (
                    temp_file.name
                )


            # ------------------------------------------------
            # Run Prediction
            # ------------------------------------------------

            result = predict_waste(
                temp_image_path
            )


        # ====================================================
        # Save Prediction
        # ====================================================

        prediction_id = save_prediction(
            image_name=uploaded_file.name,
            category=result["class"],
            confidence=result["confidence"],
            recommendation=result["recommendation"],
        )


        # ====================================================
        # Analysis Completed
        # ====================================================

        st.success(
            f"✅ Analysis completed! "
            f"Prediction #{prediction_id} saved."
        )


        st.divider()


        # ====================================================
        # Main Prediction
        # ====================================================

        st.subheader(
            "🤖 Prediction"
        )


        st.write(
            f"### Waste Category: "
            f"**{result['class'].upper()}**"
        )


        # ====================================================
        # Confidence
        # ====================================================

        confidence = result[
            "confidence"
        ]


        st.metric(
            label="AI Confidence",
            value=f"{confidence:.2f}%",
        )


        # ====================================================
        # Confidence Warning
        # ====================================================

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


        # ====================================================
        # Top 3 Predictions
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Top 3 AI Predictions"
        )


        st.caption(
            "These are the three categories the AI "
            "considered most likely."
        )


        for rank, prediction in enumerate(
            result["top_predictions"],
            start=1,
        ):

            category = (
                prediction["class"]
                .upper()
            )

            probability = (
                prediction["confidence"]
            )


            st.write(
                f"**{rank}. {category}**"
            )


            st.progress(
                min(
                    probability / 100,
                    1.0,
                )
            )


            st.caption(
                f"{probability:.2f}% probability"
            )


        # ====================================================
        # Recommendation
        # ====================================================

        st.divider()

        st.subheader(
            "♻️ Responsible Disposal Recommendation"
        )


        st.info(
            result["recommendation"]
        )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "AI WasteWise | "
    "AICTE × IBM SkillsBuild "
    "Machine Learning & Applied AI Internship 2026"
)