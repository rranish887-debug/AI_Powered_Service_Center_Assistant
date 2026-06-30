import os
import re
from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import (
    CountVectorizer,
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from wordcloud import WordCloud


DATA_PATH = "data/clean_service_center_data.csv"


st.set_page_config(
    page_title="Complaint Analysis",
    page_icon="📝",
    layout="wide"
)


@st.cache_data
def load_data():
    """Load dataset."""
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def clean_text(text: str) -> str:
    """Clean complaint text."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [
        word for word in text.split()
        if word not in ENGLISH_STOP_WORDS and len(word) > 2
    ]

    return " ".join(words)


def get_top_words(texts: List[str], top_n: int = 20) -> pd.DataFrame:
    """Get most frequent words."""
    all_words = " ".join(texts).split()
    word_counts = Counter(all_words)

    top_words = word_counts.most_common(top_n)

    return pd.DataFrame(top_words, columns=["Word", "Count"])


def get_ngrams(texts: List[str], ngram_range: tuple, top_n: int = 20) -> pd.DataFrame:
    """Generate bigrams or trigrams."""
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(texts)
    counts = matrix.sum(axis=0).A1

    ngram_df = pd.DataFrame(
        {
            "Phrase": vectorizer.get_feature_names_out(),
            "Count": counts
        }
    )

    return ngram_df.sort_values(by="Count", ascending=False).head(top_n)


def get_tfidf_keywords(texts: List[str], top_n: int = 20) -> pd.DataFrame:
    """Get important keywords using TF-IDF."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500
    )

    matrix = vectorizer.fit_transform(texts)
    scores = matrix.sum(axis=0).A1

    tfidf_df = pd.DataFrame(
        {
            "Keyword": vectorizer.get_feature_names_out(),
            "TF-IDF Score": scores
        }
    )

    return tfidf_df.sort_values(
        by="TF-IDF Score",
        ascending=False
    ).head(top_n)


def create_wordcloud(texts: List[str]) -> WordCloud:
    """Create word cloud object."""
    combined_text = " ".join(texts)

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        max_words=100,
        colormap="viridis"
    ).generate(combined_text)

    return wordcloud


def perform_lda(texts: List[str], num_topics: int = 4, num_words: int = 8) -> pd.DataFrame:
    """Perform topic modeling using LDA."""
    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=1000
    )

    doc_term_matrix = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42
    )

    lda.fit(doc_term_matrix)

    words = vectorizer.get_feature_names_out()

    topics = []

    for topic_index, topic in enumerate(lda.components_):
        top_word_indices = topic.argsort()[-num_words:][::-1]
        top_words = [words[i] for i in top_word_indices]

        topics.append(
            {
                "Topic": f"Topic {topic_index + 1}",
                "Top Words": ", ".join(top_words)
            }
        )

    return pd.DataFrame(topics)


def main():
    st.title("📝 Complaint Analysis")

    st.write(
        "Analyze customer complaints using NLP techniques like word frequency, n-grams, TF-IDF, and topic modeling."
    )

    df = load_data()

    if df is None:
        st.error("Dataset not found. Please run `python data.py` first.")
        st.stop()

    df["Clean_Complaint"] = df["Complaint"].apply(clean_text)

    st.sidebar.header("🔎 Filters")

    complaint_options = ["All"] + sorted(df["Complaint_Type"].unique().tolist())
    selected_complaint_type = st.sidebar.selectbox(
        "Select Complaint Type",
        complaint_options
    )

    product_options = ["All"] + sorted(df["Product"].unique().tolist())
    selected_product = st.sidebar.selectbox(
        "Select Product",
        product_options
    )

    filtered_df = df.copy()

    if selected_complaint_type != "All":
        filtered_df = filtered_df[
            filtered_df["Complaint_Type"] == selected_complaint_type
        ]

    if selected_product != "All":
        filtered_df = filtered_df[
            filtered_df["Product"] == selected_product
        ]

    st.success(f"Showing {filtered_df.shape[0]} complaint records")

    cleaned_texts = filtered_df["Clean_Complaint"].dropna().tolist()

    if len(cleaned_texts) == 0:
        st.warning("No complaint text available for analysis.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Complaint Overview",
            "☁️ Word Cloud & Top Words",
            "🔤 N-Grams & TF-IDF",
            "🧠 Topic Modeling"
        ]
    )

    with tab1:
        st.subheader("📊 Complaint Frequency Analysis")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Complaints", len(filtered_df))
        col2.metric("Complaint Types", filtered_df["Complaint_Type"].nunique())
        col3.metric("Products", filtered_df["Product"].nunique())
        col4.metric("Brands", filtered_df["Brand"].nunique())

        col_a, col_b = st.columns(2)

        with col_a:
            complaint_count = (
                filtered_df["Complaint_Type"]
                .value_counts()
                .reset_index()
            )
            complaint_count.columns = ["Complaint Type", "Count"]

            fig = px.bar(
                complaint_count,
                x="Complaint Type",
                y="Count",
                text="Count",
                title="Complaint Type Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            product_complaints = (
                filtered_df["Product"]
                .value_counts()
                .reset_index()
            )
            product_complaints.columns = ["Product", "Count"]

            fig = px.pie(
                product_complaints,
                names="Product",
                values="Count",
                title="Product-wise Complaints",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)

        col_c, col_d = st.columns(2)

        with col_c:
            brand_complaints = (
                filtered_df["Brand"]
                .value_counts()
                .reset_index()
            )
            brand_complaints.columns = ["Brand", "Count"]

            fig = px.bar(
                brand_complaints,
                x="Brand",
                y="Count",
                text="Count",
                title="Brand-wise Complaints"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_d:
            priority_complaints = (
                filtered_df["Priority"]
                .value_counts()
                .reset_index()
            )
            priority_complaints.columns = ["Priority", "Count"]

            fig = px.pie(
                priority_complaints,
                names="Priority",
                values="Count",
                title="Priority-wise Complaints",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)

        filtered_df["Service_Request_Date"] = pd.to_datetime(
            filtered_df["Service_Request_Date"]
        )

        trend_df = (
            filtered_df
            .groupby(filtered_df["Service_Request_Date"].dt.to_period("M"))
            .size()
            .reset_index(name="Complaint Count")
        )

        trend_df["Service_Request_Date"] = trend_df[
            "Service_Request_Date"
        ].astype(str)

        fig = px.line(
            trend_df,
            x="Service_Request_Date",
            y="Complaint Count",
            markers=True,
            title="Monthly Complaint Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("☁️ Complaint Word Cloud")

        wordcloud = create_wordcloud(cleaned_texts)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")

        st.pyplot(fig)

        st.subheader("🔝 Top Complaint Words")

        top_words_df = get_top_words(cleaned_texts, top_n=20)

        fig = px.bar(
            top_words_df,
            x="Word",
            y="Count",
            text="Count",
            title="Top 20 Complaint Words"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_words_df, use_container_width=True)

    with tab3:
        col_x, col_y = st.columns(2)

        with col_x:
            st.subheader("🔤 Top Bi-grams")

            bigram_df = get_ngrams(
                cleaned_texts,
                ngram_range=(2, 2),
                top_n=20
            )

            fig = px.bar(
                bigram_df,
                x="Phrase",
                y="Count",
                text="Count",
                title="Top 20 Bi-grams"
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(bigram_df, use_container_width=True)

        with col_y:
            st.subheader("🔤 Top Tri-grams")

            trigram_df = get_ngrams(
                cleaned_texts,
                ngram_range=(3, 3),
                top_n=20
            )

            fig = px.bar(
                trigram_df,
                x="Phrase",
                y="Count",
                text="Count",
                title="Top 20 Tri-grams"
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(trigram_df, use_container_width=True)

        st.subheader("⭐ Important Keywords using TF-IDF")

        tfidf_df = get_tfidf_keywords(cleaned_texts, top_n=20)

        fig = px.bar(
            tfidf_df,
            x="Keyword",
            y="TF-IDF Score",
            text="TF-IDF Score",
            title="Top TF-IDF Keywords"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tfidf_df, use_container_width=True)

    with tab4:
        st.subheader("🧠 Topic Modeling using LDA")

        num_topics = st.slider(
            "Select number of topics",
            min_value=2,
            max_value=8,
            value=4
        )

        topic_df = perform_lda(
            cleaned_texts,
            num_topics=num_topics,
            num_words=8
        )

        st.dataframe(topic_df, use_container_width=True)

        for _, row in topic_df.iterrows():
            st.info(f"{row['Topic']}: {row['Top Words']}")

        st.markdown("---")

        st.subheader("📁 Complaint Records")

        st.dataframe(
            filtered_df[
                [
                    "Customer_ID",
                    "Product",
                    "Brand",
                    "Complaint",
                    "Complaint_Type",
                    "Priority"
                ]
            ].head(100),
            use_container_width=True
        )


if __name__ == "__main__":
    main()