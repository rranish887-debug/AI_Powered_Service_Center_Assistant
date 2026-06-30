import os
import re
import sys

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


REPAIR_MODEL_PATH = "models/repair_prediction.pkl"
TICKET_MODEL_PATH = "models/ticket_classifier.pkl"
DATA_PATH = "data/clean_service_center_data.csv"


st.set_page_config(
    page_title="Repair Time Prediction",
    page_icon="⏱️",
    layout="wide"
)


def clean_text(text: str) -> str:
    """Clean text. This is needed because the saved ticket model uses clean_text."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_repair_model():
    """Load trained repair prediction model."""
    if not os.path.exists(REPAIR_MODEL_PATH):
        return None
    return joblib.load(REPAIR_MODEL_PATH)


@st.cache_resource
def load_ticket_model():
    """Load trained ticket classification model."""
    if not os.path.exists(TICKET_MODEL_PATH):
        return None

    # Fix for pickle clean_text loading issue
    sys.modules["__main__"].clean_text = clean_text

    return joblib.load(TICKET_MODEL_PATH)


@st.cache_data
def load_data():
    """Load dataset."""
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def main():
    st.title("⏱️ Repair Time Prediction")

    st.write(
        "Predict estimated repair completion time using complaint type, brand, priority, warranty, parts, and repair cost."
    )

    repair_model = load_repair_model()
    ticket_model = load_ticket_model()
    df = load_data()

    if repair_model is None:
        st.error("Repair model not found. Please run `python train_models.py` first.")
        st.stop()

    st.subheader("📝 Enter Service Details")

    complaint_text = st.text_area(
        "Customer Complaint",
        placeholder="Example: Device is not charging and battery drains fast",
        height=120
    )

    complaint_types = [
        "Battery Issue",
        "Display Issue",
        "Software Issue",
        "Charging Issue",
        "Camera Issue",
        "Speaker Issue",
        "Network Issue",
        "Motherboard Issue"
    ]

    brands = [
        "Dell", "HP", "Lenovo", "Apple", "Asus",
        "Samsung", "OnePlus", "Vivo", "Oppo",
        "LG", "Sony", "Mi", "Whirlpool", "Godrej",
        "Bosch", "Voltas", "Blue Star"
    ]

    priorities = ["Low", "Medium", "High"]
    warranty_statuses = ["Under Warranty", "Out of Warranty"]

    parts = [
        "Battery",
        "Charging IC",
        "No Part",
        "Display Panel",
        "Touch Screen",
        "Screen Glass",
        "OS Reinstall",
        "Software Reset",
        "Charging Port",
        "USB Board",
        "Adapter",
        "Camera Module",
        "Camera Lens",
        "Speaker",
        "Microphone",
        "Audio IC",
        "Network IC",
        "WiFi Module",
        "SIM Slot",
        "Motherboard",
        "Power IC",
        "Processor Board"
    ]

    col1, col2 = st.columns(2)

    with col1:
        auto_detect = st.checkbox(
            "Auto detect complaint type from complaint text",
            value=True
        )

        if auto_detect and complaint_text.strip() and ticket_model is not None:
            complaint_type = ticket_model.predict([complaint_text])[0]
            st.success(f"Detected Complaint Type: {complaint_type}")
        else:
            complaint_type = st.selectbox("Complaint Type", complaint_types)

        brand = st.selectbox("Brand", brands)
        priority = st.selectbox("Priority", priorities)

    with col2:
        warranty_status = st.selectbox("Warranty Status", warranty_statuses)
        parts_replaced = st.selectbox("Parts Replaced", parts)

        repair_cost = st.number_input(
            "Estimated Repair Cost",
            min_value=0,
            max_value=50000,
            value=2500,
            step=100
        )

    predict_button = st.button(
        "⏱️ Predict Repair Time",
        use_container_width=True
    )

    if predict_button:
        input_data = pd.DataFrame(
            [
                {
                    "Complaint_Type": complaint_type,
                    "Brand": brand,
                    "Priority": priority,
                    "Warranty_Status": warranty_status,
                    "Parts_Replaced": parts_replaced,
                    "Repair_Cost": repair_cost
                }
            ]
        )

        predicted_days = repair_model.predict(input_data)[0]
        predicted_days = max(1, round(float(predicted_days), 2))

        st.success("Repair time prediction completed successfully!")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Predicted Repair Time", f"{predicted_days} Days")

        with col_b:
            st.metric("Priority", priority)

        with col_c:
            st.metric("Warranty Status", warranty_status)

        st.subheader("🧾 Prediction Summary")

        st.json(
            {
                "Complaint Type": complaint_type,
                "Brand": brand,
                "Priority": priority,
                "Warranty Status": warranty_status,
                "Parts Replaced": parts_replaced,
                "Repair Cost": repair_cost,
                "Predicted Repair Time": f"{predicted_days} Days"
            }
        )

        if predicted_days <= 2:
            st.info("This repair is expected to be completed quickly.")
        elif predicted_days <= 5:
            st.warning("This repair may take a moderate amount of time.")
        else:
            st.error("This repair may take longer due to hardware complexity or part replacement.")

    if df is not None:
        st.markdown("---")
        st.subheader("📊 Repair Time Analysis")

        col_x, col_y = st.columns(2)

        with col_x:
            fig = px.box(
                df,
                x="Complaint_Type",
                y="Repair_Time",
                title="Repair Time by Complaint Type"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_y:
            avg_time = (
                df.groupby("Priority")["Repair_Time"]
                .mean()
                .round(2)
                .reset_index()
            )

            fig = px.bar(
                avg_time,
                x="Priority",
                y="Repair_Time",
                text="Repair_Time",
                title="Average Repair Time by Priority"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📁 Recent Repair Records")

        st.dataframe(
            df[
                [
                    "Complaint_Type",
                    "Brand",
                    "Priority",
                    "Warranty_Status",
                    "Parts_Replaced",
                    "Repair_Cost",
                    "Repair_Time"
                ]
            ].head(50),
            use_container_width=True
        )


if __name__ == "__main__":
    main()