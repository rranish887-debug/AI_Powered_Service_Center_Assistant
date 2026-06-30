import os

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = "data/clean_service_center_data.csv"
METRICS_PATH = "models/model_metrics.csv"


st.set_page_config(
    page_title="AI Service Center Assistant",
    page_icon="🛠️",
    layout="wide"
)


@st.cache_data
def load_data():
    """Load dataset."""
    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH)
    df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])
    df["Service_Request_Date"] = pd.to_datetime(df["Service_Request_Date"])
    return df


@st.cache_data
def load_metrics():
    """Load model metrics."""
    if not os.path.exists(METRICS_PATH):
        return None

    return pd.read_csv(METRICS_PATH)


def main():
    st.title("🛠️ AI-Powered Service Center Assistant")
    st.write(
        "Service Data Analysis • Ticket Classification • Repair Time Prediction • Sentiment Analysis • AI Chatbot"
    )

    df = load_data()

    if df is None:
        st.error("Dataset not found. Please run `python data.py` first.")
        return

    st.success("Dataset loaded successfully!")

    total_tickets = len(df)
    closed_tickets = df[df["Status"] == "Closed"].shape[0]
    avg_repair_time = round(df["Repair_Time"].mean(), 2)
    avg_rating = round(df["Customer_Rating"].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Tickets", total_tickets)
    col2.metric("Closed Tickets", closed_tickets)
    col3.metric("Average Repair Time", f"{avg_repair_time} Days")
    col4.metric("Average Rating", f"{avg_rating}/5")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["📊 Dashboard", "🤖 Model Results", "📁 Dataset"]
    )

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            complaint_count = df["Complaint_Type"].value_counts().reset_index()
            complaint_count.columns = ["Complaint Type", "Count"]

            fig = px.bar(
                complaint_count,
                x="Complaint Type",
                y="Count",
                title="Complaint Type Count",
                text="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            status_count = df["Status"].value_counts().reset_index()
            status_count.columns = ["Status", "Count"]

            fig = px.pie(
                status_count,
                names="Status",
                values="Count",
                title="Ticket Status Distribution",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)

        col_c, col_d = st.columns(2)

        with col_c:
            warranty_count = df["Warranty_Status"].value_counts().reset_index()
            warranty_count.columns = ["Warranty Status", "Count"]

            fig = px.pie(
                warranty_count,
                names="Warranty Status",
                values="Count",
                title="Warranty Status Analysis",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_d:
            priority_count = df["Priority"].value_counts().reset_index()
            priority_count.columns = ["Priority", "Count"]

            fig = px.bar(
                priority_count,
                x="Priority",
                y="Count",
                title="Priority Wise Requests",
                text="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        metrics_df = load_metrics()

        if metrics_df is None:
            st.warning("Model metrics not found. Please run `python train_models.py`.")
        else:
            st.subheader("Model Performance")
            st.dataframe(metrics_df, use_container_width=True)

            classification_df = metrics_df[
                metrics_df["Module"] == "Ticket Classification"
            ]

            repair_df = metrics_df[
                metrics_df["Module"] == "Repair Time Prediction"
            ]

            col1, col2 = st.columns(2)

            with col1:
                if not classification_df.empty:
                    fig = px.bar(
                        classification_df,
                        x="Model",
                        y="F1_Score",
                        title="Ticket Classification F1 Score",
                        text="F1_Score"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if not repair_df.empty:
                    fig = px.bar(
                        repair_df,
                        x="Model",
                        y="R2_Score",
                        title="Repair Prediction R² Score",
                        text="R2_Score"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Dataset",
            data=csv,
            file_name="service_center_data.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.info(
        "Next pages: Data Analysis, Ticket Classification, Repair Prediction, Sentiment Analysis, and AI Assistant."
    )


if __name__ == "__main__":
    main()