import streamlit as st


st.set_page_config(
    page_title="About Project",
    page_icon="ℹ️",
    layout="wide"
)


def main():
    st.title("ℹ️ About Project")

    st.markdown(
        """
        # 🛠️ AI-Powered Service Center Assistant

        This project is a complete AI and Machine Learning based system for service center management.

        It helps service centers to:

        - Analyze service data
        - Classify customer complaints
        - Predict repair completion time
        - Analyze customer feedback sentiment
        - Understand complaint patterns
        - Provide automated AI-based customer support

        ---
        """
    )

    st.subheader("🎯 Project Objective")

    st.write(
        """
        The main objective of this project is to automate service center operations
        using Machine Learning, Natural Language Processing, and AI chatbot technology.
        """
    )

    st.subheader("📌 Modules Implemented")

    modules = {
        "Service Data Analysis": "EDA, filters, search, charts, data quality checks",
        "Ticket Classification": "ML model to classify customer complaints",
        "Complaint Analysis": "Word cloud, top words, n-grams, TF-IDF, LDA topic modeling",
        "Repair Time Prediction": "Regression model to predict repair duration",
        "Sentiment Analysis": "TextBlob, NLTK VADER, and Hugging Face based feedback analysis",
        "AI Assistant": "RAG-based chatbot using knowledge base, FAISS, and embeddings"
    }

    for module, description in modules.items():
        st.success(f"{module}: {description}")

    st.subheader("🧰 Technologies Used")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("Python")
        st.info("Pandas")
        st.info("NumPy")
        st.info("Scikit-Learn")

    with col2:
        st.info("Streamlit")
        st.info("Plotly")
        st.info("TextBlob")
        st.info("NLTK")

    with col3:
        st.info("LangChain")
        st.info("FAISS")
        st.info("Sentence Transformers")
        st.info("Hugging Face")

    st.subheader("🧠 ML Models Used")

    st.markdown(
        """
        ### Ticket Classification Models

        - Logistic Regression
        - Linear SVM
        - Naive Bayes
        - Random Forest

        ### Repair Time Prediction Models

        - Linear Regression
        - Decision Tree Regressor
        - Random Forest Regressor
        - Gradient Boosting Regressor
        """
    )

    st.subheader("📊 Dataset")

    st.write(
        """
        The dataset contains 2000 realistic service center records with customer details,
        product information, complaints, complaint type, warranty status, priority,
        repair time, repair cost, feedback, and sentiment.
        """
    )

    st.subheader("🚀 How to Run")

    st.code(
        """
pip install -r requirements.txt
python data.py
python train_models.py
streamlit run app.py
        """,
        language="bash"
    )

    st.subheader("🔮 Future Improvements")

    st.markdown(
        """
        - Add real-time database
        - Add customer login and admin login
        - Add email/SMS notifications
        - Add advanced analytics chatbot 
        - Deploy on Streamlit Cloud
        - Add multilingual support
        """
    )

    st.markdown("---")

    st.success("Project developed by Ranish A")


if __name__ == "__main__":
    main()