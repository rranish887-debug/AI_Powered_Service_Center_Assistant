import os
import re
import sys
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


MODEL_PATH = "models/ticket_classifier.pkl"
DATA_PATH = "data/clean_service_center_data.csv"


st.set_page_config(
    page_title="Ticket Classification",
    page_icon="🎫",
    layout="wide"
)


def clean_text(text: str) -> str:
    """Clean text. This is needed because model was trained with this function."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model():
    """Load trained ticket classification model."""
    if not os.path.exists(MODEL_PATH):
        return None

    # Important fix:
    # The trained pickle model needs clean_text() while loading.
    sys.modules["__main__"].clean_text = clean_text

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    """Load dataset."""
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def detect_priority(complaint_type: str) -> str:
    """Detect priority automatically based on complaint type."""
    high_priority = [
        "Motherboard Issue",
        "Charging Issue",
        "Network Issue"
    ]

    medium_priority = [
        "Battery Issue",
        "Display Issue",
        "Speaker Issue"
    ]

    if complaint_type in high_priority:
        return "High"
    if complaint_type in medium_priority:
        return "Medium"
    return "Low"


def get_recommendation(complaint_type: str) -> str:
    """Give technician recommendation based on complaint type."""
    recommendations = {
        "Battery Issue": "Perform battery health check and inspect charging cycle.",
        "Display Issue": "Inspect display panel, touch sensor, and screen connector.",
        "Software Issue": "Perform software diagnosis, reset, or OS reinstall.",
        "Charging Issue": "Check charging port, adapter, cable, and charging IC.",
        "Camera Issue": "Inspect camera module and camera app permissions.",
        "Speaker Issue": "Check speaker, microphone, and audio IC.",
        "Network Issue": "Check SIM slot, WiFi module, Bluetooth, and network IC.",
        "Motherboard Issue": "Perform deep hardware diagnosis and motherboard inspection."
    }

    return recommendations.get(
        complaint_type,
        "Assign technician for detailed diagnosis."
    )


def main():
    st.title("🎫 Service Ticket Classification")

    st.write(
        "Enter a customer complaint. The ML model will classify the complaint type automatically."
    )

    model = load_model()
    df = load_data()

    if model is None:
        st.error("Model not found. Please run `python train_models.py` first.")
        st.stop()

    complaint = st.text_area(
        "Enter Customer Complaint",
        placeholder="Example: My phone battery drains very fast and heats while charging",
        height=150
    )

    predict_button = st.button(
        "🔍 Classify Complaint",
        use_container_width=True
    )

    if predict_button:
        if complaint.strip() == "":
            st.warning("Please enter a complaint.")
        else:
            prediction = model.predict([complaint])[0]
            priority = detect_priority(prediction)
            recommendation = get_recommendation(prediction)

            st.success("Prediction completed successfully!")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted Complaint Type", prediction)

            with col2:
                st.metric("Auto Detected Priority", priority)

            st.subheader("🧾 Ticket Summary")

            st.json(
                {
                    "Customer Complaint": complaint,
                    "Predicted Complaint Type": prediction,
                    "Suggested Priority": priority,
                    "Recommended Action": recommendation
                }
            )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([complaint])[0]
                classes = model.classes_

                prob_df = pd.DataFrame(
                    {
                        "Complaint Type": classes,
                        "Confidence": probabilities
                    }
                )

                prob_df = prob_df.sort_values(
                    by="Confidence",
                    ascending=False
                )

                prob_df["Confidence"] = prob_df["Confidence"].round(4)

                st.subheader("📊 Model Confidence")

                fig = px.bar(
                    prob_df,
                    x="Complaint Type",
                    y="Confidence",
                    text="Confidence",
                    title="Prediction Confidence Score"
                )

                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(prob_df, use_container_width=True)

    st.markdown("---")

    st.subheader("📌 Sample Complaints for Testing")

    sample_complaints = [
        "Battery drains very fast",
        "Screen is flickering and touch is not working",
        "Device is not charging properly",
        "Camera app crashes when opened",
        "Speaker sound is very low",
        "WiFi is not connecting",
        "Device is completely dead",
        "Apps are crashing frequently"
    ]

    for sample in sample_complaints:
        st.code(sample)

    if df is not None:
        st.markdown("---")
        st.subheader("📊 Complaint Type Distribution")

        complaint_count = df["Complaint_Type"].value_counts().reset_index()
        complaint_count.columns = ["Complaint Type", "Count"]

        fig = px.bar(
            complaint_count,
            x="Complaint Type",
            y="Count",
            text="Count",
            title="Complaint Types in Dataset"
        )

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()