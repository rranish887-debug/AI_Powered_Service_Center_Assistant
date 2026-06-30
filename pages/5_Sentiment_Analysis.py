import os

# Force Hugging Face Transformers to use PyTorch only.
# This avoids TensorFlow / Keras 3 compatibility errors.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import Dict, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from textblob import TextBlob


DATA_PATH = "data/clean_service_center_data.csv"


st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)


@st.cache_data
def load_data():
    """Load dataset."""
    if not os.path.exists(DATA_PATH):
        return None

    return pd.read_csv(DATA_PATH)


def textblob_sentiment(text: str) -> Tuple[str, float]:
    """Analyze sentiment using TextBlob."""
    polarity = TextBlob(str(text)).sentiment.polarity

    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    confidence = round(abs(polarity), 3)

    return sentiment, confidence


def vader_sentiment(text: str) -> Tuple[str, float]:
    """Analyze sentiment using NLTK VADER."""
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")

        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(str(text))
        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        confidence = round(abs(compound), 3)

        return sentiment, confidence

    except Exception as error:
        return f"Unavailable: {str(error)[:80]}", 0.0


@st.cache_resource
def load_transformer_model():
    """
    Load Hugging Face sentiment model using PyTorch only.

    The first time this runs, it may take time because the model
    may need to download from Hugging Face.
    """
    try:
        from transformers import pipeline

        sentiment_pipeline = pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt",
            device=-1
        )

        return sentiment_pipeline, ""

    except Exception as error:
        return None, str(error)


def transformer_sentiment(text: str) -> Tuple[str, float]:
    """Analyze sentiment using Hugging Face Transformers."""
    model, error_message = load_transformer_model()

    if model is None:
        return f"Unavailable: {error_message[:80]}", 0.0

    try:
        result = model(str(text))[0]

        label = result["label"]
        score = round(float(result["score"]), 3)

        if label.upper() == "POSITIVE":
            sentiment = "Positive"
        elif label.upper() == "NEGATIVE":
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return sentiment, score

    except Exception as error:
        return f"Unavailable: {str(error)[:80]}", 0.0


def compare_sentiment_methods(text: str) -> Dict[str, Dict[str, object]]:
    """Compare TextBlob, NLTK VADER, and Hugging Face results."""
    tb_sentiment, tb_confidence = textblob_sentiment(text)
    vader_result, vader_confidence = vader_sentiment(text)
    hf_sentiment, hf_confidence = transformer_sentiment(text)

    return {
        "TextBlob": {
            "Sentiment": tb_sentiment,
            "Confidence": tb_confidence
        },
        "NLTK VADER": {
            "Sentiment": vader_result,
            "Confidence": vader_confidence
        },
        "Hugging Face": {
            "Sentiment": hf_sentiment,
            "Confidence": hf_confidence
        }
    }


def get_sentiment_emoji(sentiment: str) -> str:
    """Return emoji based on sentiment."""
    if sentiment == "Positive":
        return "😊"

    if sentiment == "Negative":
        return "😞"

    if sentiment == "Neutral":
        return "😐"

    return "⚠️"


def main():
    """Main Sentiment Analysis page."""
    st.title("😊 Customer Sentiment Analysis")

    st.write(
        "Analyze customer feedback using TextBlob, NLTK VADER, "
        "and Hugging Face Transformers."
    )

    df = load_data()

    tab1, tab2, tab3 = st.tabs(
        [
            "📝 Single Feedback Analysis",
            "📊 Dataset Sentiment Dashboard",
            "⚖️ Method Comparison"
        ]
    )

    with tab1:
        st.subheader("📝 Analyze Customer Feedback")

        feedback_text = st.text_area(
            "Enter Customer Feedback",
            placeholder="Example: The service was very good and repair completed on time",
            height=150
        )

        analyze_button = st.button(
            "🔍 Analyze Sentiment",
            width="stretch"
        )

        if analyze_button:
            if feedback_text.strip() == "":
                st.warning("Please enter customer feedback.")
            else:
                results = compare_sentiment_methods(feedback_text)

                st.success("Sentiment analysis completed!")

                col1, col2, col3 = st.columns(3)

                with col1:
                    sentiment = results["TextBlob"]["Sentiment"]
                    confidence = results["TextBlob"]["Confidence"]

                    st.metric(
                        f"{get_sentiment_emoji(sentiment)} TextBlob",
                        sentiment,
                        f"Confidence: {confidence}"
                    )

                with col2:
                    sentiment = results["NLTK VADER"]["Sentiment"]
                    confidence = results["NLTK VADER"]["Confidence"]

                    st.metric(
                        f"{get_sentiment_emoji(sentiment)} NLTK VADER",
                        sentiment,
                        f"Confidence: {confidence}"
                    )

                with col3:
                    sentiment = results["Hugging Face"]["Sentiment"]
                    confidence = results["Hugging Face"]["Confidence"]

                    st.metric(
                        f"{get_sentiment_emoji(sentiment)} Hugging Face",
                        sentiment,
                        f"Confidence: {confidence}"
                    )

                result_df = pd.DataFrame(results).T.reset_index()
                result_df.columns = ["Method", "Sentiment", "Confidence"]

                st.subheader("📋 Result Table")
                st.dataframe(result_df, width="stretch")

                fig = px.bar(
                    result_df,
                    x="Method",
                    y="Confidence",
                    color="Sentiment",
                    text="Confidence",
                    title="Sentiment Confidence by Method"
                )

                st.plotly_chart(fig, width="stretch")

                hf_result = results["Hugging Face"]["Sentiment"]

                if str(hf_result).startswith("Unavailable"):
                    with st.expander("Why Hugging Face is unavailable?"):
                        st.warning(
                            "Hugging Face may be unavailable because the model "
                            "is not downloaded yet, internet is not available, "
                            "or the system has low RAM/package conflicts."
                        )
                        st.code(hf_result)

        st.markdown("---")
        st.subheader("📌 Sample Feedback for Testing")

        samples = [
            "Good service and quick repair",
            "Repair took too much time",
            "Service was okay",
            "Technician explained the issue clearly",
            "Not satisfied with service",
            "Issue solved but delay happened",
            "It is not good!",
            "I am very happy with the repair service",
            "The technician did not solve my problem"
        ]

        for sample in samples:
            st.code(sample)

    with tab2:
        st.subheader("📊 Dataset Sentiment Dashboard")

        if df is None:
            st.error("Dataset not found. Please run `python data.py` first.")
            st.stop()

        total_feedback = len(df)
        positive_count = df[df["Sentiment"] == "Positive"].shape[0]
        neutral_count = df[df["Sentiment"] == "Neutral"].shape[0]
        negative_count = df[df["Sentiment"] == "Negative"].shape[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Feedback", total_feedback)
        col2.metric("Positive", positive_count)
        col3.metric("Neutral", neutral_count)
        col4.metric("Negative", negative_count)

        col_a, col_b = st.columns(2)

        with col_a:
            sentiment_count = df["Sentiment"].value_counts().reset_index()
            sentiment_count.columns = ["Sentiment", "Count"]

            fig = px.pie(
                sentiment_count,
                names="Sentiment",
                values="Count",
                title="Sentiment Distribution",
                hole=0.4
            )

            st.plotly_chart(fig, width="stretch")

        with col_b:
            fig = px.bar(
                sentiment_count,
                x="Sentiment",
                y="Count",
                color="Sentiment",
                text="Count",
                title="Sentiment Count"
            )

            st.plotly_chart(fig, width="stretch")

        col_c, col_d = st.columns(2)

        with col_c:
            rating_sentiment = (
                df.groupby(["Customer_Rating", "Sentiment"])
                .size()
                .reset_index(name="Count")
            )

            fig = px.bar(
                rating_sentiment,
                x="Customer_Rating",
                y="Count",
                color="Sentiment",
                barmode="group",
                title="Customer Rating vs Sentiment"
            )

            st.plotly_chart(fig, width="stretch")

        with col_d:
            complaint_sentiment = (
                df.groupby(["Complaint_Type", "Sentiment"])
                .size()
                .reset_index(name="Count")
            )

            fig = px.bar(
                complaint_sentiment,
                x="Complaint_Type",
                y="Count",
                color="Sentiment",
                barmode="group",
                title="Complaint Type vs Sentiment"
            )

            st.plotly_chart(fig, width="stretch")

        st.subheader("📁 Feedback Records")

        st.dataframe(
            df[
                [
                    "Customer_ID",
                    "Complaint_Type",
                    "Customer_Rating",
                    "Feedback",
                    "Sentiment"
                ]
            ].head(100),
            width="stretch"
        )

    with tab3:
        st.subheader("⚖️ Compare Methods on Dataset Sample")

        if df is None:
            st.error("Dataset not found.")
            st.stop()

        sample_size = st.slider(
            "Select number of feedback samples",
            min_value=5,
            max_value=50,
            value=10
        )

        run_button = st.button(
            "Run Comparison on Sample",
            width="stretch"
        )

        if run_button:
            sample_df = df.sample(sample_size, random_state=42).copy()

            textblob_results = []
            vader_results = []
            huggingface_results = []

            for feedback in sample_df["Feedback"]:
                tb_sent, _ = textblob_sentiment(feedback)
                vd_sent, _ = vader_sentiment(feedback)
                hf_sent, _ = transformer_sentiment(feedback)

                textblob_results.append(tb_sent)
                vader_results.append(vd_sent)
                huggingface_results.append(hf_sent)

            sample_df["TextBlob_Result"] = textblob_results
            sample_df["VADER_Result"] = vader_results
            sample_df["HuggingFace_Result"] = huggingface_results

            st.dataframe(
                sample_df[
                    [
                        "Feedback",
                        "Sentiment",
                        "TextBlob_Result",
                        "VADER_Result",
                        "HuggingFace_Result"
                    ]
                ],
                width="stretch"
            )

            comparison_data = pd.DataFrame(
                {
                    "Method": [
                        "Original Dataset",
                        "TextBlob",
                        "NLTK VADER",
                        "Hugging Face"
                    ],
                    "Positive": [
                        (sample_df["Sentiment"] == "Positive").sum(),
                        (sample_df["TextBlob_Result"] == "Positive").sum(),
                        (sample_df["VADER_Result"] == "Positive").sum(),
                        (sample_df["HuggingFace_Result"] == "Positive").sum()
                    ],
                    "Neutral": [
                        (sample_df["Sentiment"] == "Neutral").sum(),
                        (sample_df["TextBlob_Result"] == "Neutral").sum(),
                        (sample_df["VADER_Result"] == "Neutral").sum(),
                        (sample_df["HuggingFace_Result"] == "Neutral").sum()
                    ],
                    "Negative": [
                        (sample_df["Sentiment"] == "Negative").sum(),
                        (sample_df["TextBlob_Result"] == "Negative").sum(),
                        (sample_df["VADER_Result"] == "Negative").sum(),
                        (sample_df["HuggingFace_Result"] == "Negative").sum()
                    ]
                }
            )

            melted_df = comparison_data.melt(
                id_vars="Method",
                var_name="Sentiment",
                value_name="Count"
            )

            fig = px.bar(
                melted_df,
                x="Method",
                y="Count",
                color="Sentiment",
                barmode="group",
                title="Sentiment Method Comparison"
            )

            st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()