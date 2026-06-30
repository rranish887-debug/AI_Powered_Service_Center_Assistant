import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA_PATH = "data/clean_service_center_data.csv"


st.set_page_config(
    page_title="Service Data Analysis",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_data():
    """Load service center dataset."""
    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH)
    df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])
    df["Service_Request_Date"] = pd.to_datetime(df["Service_Request_Date"])
    return df


def filter_data(df):
    """Apply sidebar filters."""
    st.sidebar.header("🔎 Filters")

    product = st.sidebar.selectbox(
        "Product",
        ["All"] + sorted(df["Product"].unique().tolist())
    )

    brand = st.sidebar.selectbox(
        "Brand",
        ["All"] + sorted(df["Brand"].unique().tolist())
    )

    status = st.sidebar.selectbox(
        "Status",
        ["All"] + sorted(df["Status"].unique().tolist())
    )

    priority = st.sidebar.selectbox(
        "Priority",
        ["All"] + sorted(df["Priority"].unique().tolist())
    )

    filtered_df = df.copy()

    if product != "All":
        filtered_df = filtered_df[filtered_df["Product"] == product]

    if brand != "All":
        filtered_df = filtered_df[filtered_df["Brand"] == brand]

    if status != "All":
        filtered_df = filtered_df[filtered_df["Status"] == status]

    if priority != "All":
        filtered_df = filtered_df[filtered_df["Priority"] == priority]

    return filtered_df


def show_basic_eda(df):
    """Show basic dataset information."""
    st.subheader("📌 Basic Dataset Information")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    with st.expander("📋 Data Types"):
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values
        })
        st.dataframe(dtype_df, use_container_width=True)

    with st.expander("📊 Summary Statistics"):
        st.dataframe(df.describe(include="all"), use_container_width=True)


def show_search(df):
    """Search customer records."""
    st.subheader("🔍 Search Customer Records")

    search_text = st.text_input(
        "Search by Customer ID, Name, Brand, Product, Complaint, or Status"
    )

    if search_text:
        result = df[
            df.astype(str).apply(
                lambda row: row.str.contains(
                    search_text,
                    case=False,
                    na=False
                ).any(),
                axis=1
            )
        ]

        st.success(f"Found {len(result)} records")
        st.dataframe(result, use_container_width=True)


def show_charts(df):
    """Show EDA charts."""
    st.subheader("📊 Interactive Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        complaint_count = df["Complaint_Type"].value_counts().reset_index()
        complaint_count.columns = ["Complaint Type", "Count"]

        fig = px.bar(
            complaint_count,
            x="Complaint Type",
            y="Count",
            text="Count",
            title="1. Top Complaint Categories"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        product_count = df["Product"].value_counts().reset_index()
        product_count.columns = ["Product", "Count"]

        fig = px.pie(
            product_count,
            names="Product",
            values="Count",
            title="2. Product Distribution",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        brand_count = df["Brand"].value_counts().reset_index()
        brand_count.columns = ["Brand", "Count"]

        fig = px.bar(
            brand_count,
            x="Brand",
            y="Count",
            text="Count",
            title="3. Brand-wise Complaints"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        priority_count = df["Priority"].value_counts().reset_index()
        priority_count.columns = ["Priority", "Count"]

        fig = px.pie(
            priority_count,
            names="Priority",
            values="Count",
            title="4. Priority-wise Requests",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)

    with col5:
        status_count = df["Status"].value_counts().reset_index()
        status_count.columns = ["Status", "Count"]

        fig = px.bar(
            status_count,
            x="Status",
            y="Count",
            text="Count",
            title="5. Ticket Status Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        warranty_count = df["Warranty_Status"].value_counts().reset_index()
        warranty_count.columns = ["Warranty Status", "Count"]

        fig = px.pie(
            warranty_count,
            names="Warranty Status",
            values="Count",
            title="6. Warranty Status Analysis",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    col7, col8 = st.columns(2)

    with col7:
        fig = px.histogram(
            df,
            x="Repair_Cost",
            nbins=30,
            title="7. Repair Cost Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col8:
        fig = px.histogram(
            df,
            x="Repair_Time",
            nbins=10,
            title="8. Repair Time Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    col9, col10 = st.columns(2)

    with col9:
        rating_count = df["Customer_Rating"].value_counts().sort_index().reset_index()
        rating_count.columns = ["Rating", "Count"]

        fig = px.bar(
            rating_count,
            x="Rating",
            y="Count",
            text="Count",
            title="9. Customer Rating Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col10:
        sentiment_count = df["Sentiment"].value_counts().reset_index()
        sentiment_count.columns = ["Sentiment", "Count"]

        fig = px.pie(
            sentiment_count,
            names="Sentiment",
            values="Count",
            title="10. Sentiment Distribution",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    col11, col12 = st.columns(2)

    with col11:
        monthly_data = (
            df.groupby(df["Service_Request_Date"].dt.to_period("M"))
            .size()
            .reset_index(name="Requests")
        )
        monthly_data["Service_Request_Date"] = monthly_data[
            "Service_Request_Date"
        ].astype(str)

        fig = px.line(
            monthly_data,
            x="Service_Request_Date",
            y="Requests",
            markers=True,
            title="11. Monthly Service Requests"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col12:
        technician_count = df["Technician"].value_counts().reset_index()
        technician_count.columns = ["Technician", "Tickets"]

        fig = px.bar(
            technician_count,
            x="Technician",
            y="Tickets",
            text="Tickets",
            title="12. Technician Ticket Count"
        )
        st.plotly_chart(fig, use_container_width=True)

    col13, col14 = st.columns(2)

    with col13:
        technician_time = (
            df.groupby("Technician")["Repair_Time"]
            .mean()
            .round(2)
            .reset_index()
        )

        fig = px.bar(
            technician_time,
            x="Technician",
            y="Repair_Time",
            text="Repair_Time",
            title="13. Technician Average Repair Time"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col14:
        brand_cost = (
            df.groupby("Brand")["Repair_Cost"]
            .mean()
            .round(2)
            .reset_index()
        )

        fig = px.bar(
            brand_cost,
            x="Brand",
            y="Repair_Cost",
            text="Repair_Cost",
            title="14. Brand-wise Average Repair Cost"
        )
        st.plotly_chart(fig, use_container_width=True)

    col15, col16 = st.columns(2)

    with col15:
        complaint_priority = (
            df.groupby(["Complaint_Type", "Priority"])
            .size()
            .reset_index(name="Count")
        )

        fig = px.bar(
            complaint_priority,
            x="Complaint_Type",
            y="Count",
            color="Priority",
            barmode="group",
            title="15. Complaint Type vs Priority"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col16:
        fig = px.box(
            df,
            x="Warranty_Status",
            y="Repair_Cost",
            title="16. Repair Cost by Warranty Status"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("17. Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                text=corr.round(2).values,
                texttemplate="%{text}",
                colorscale="Blues"
            )
        )

        fig.update_layout(title="Correlation Between Numerical Columns")
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📊 Service Data Analysis")

    st.write(
        "This page performs Exploratory Data Analysis on service center data."
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])
        df["Service_Request_Date"] = pd.to_datetime(df["Service_Request_Date"])
    else:
        df = load_data()

    if df is None:
        st.error("Dataset not found. Please run `python data.py` first.")
        st.stop()

    filtered_df = filter_data(df)

    st.success(f"Showing {filtered_df.shape[0]} records")

    show_basic_eda(filtered_df)

    st.subheader("📁 Dataset Preview")
    st.dataframe(filtered_df.head(100), use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Data",
        data=csv_data,
        file_name="filtered_service_center_data.csv",
        mime="text/csv"
    )

    show_search(filtered_df)
    show_charts(filtered_df)


if __name__ == "__main__":
    main()