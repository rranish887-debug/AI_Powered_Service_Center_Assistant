import os
import sys

import streamlit as st


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chatbot.rag import ServiceCenterRAG, read_knowledge_base


st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource
def load_bot():
    """Load advanced chatbot."""
    bot = ServiceCenterRAG(use_llm=False)
    bot.build_or_load_index()
    return bot


def main():
    """Main AI Assistant page."""
    st.title("🤖 Advanced AI Service Center Assistant")

    st.write(
        "Ask questions about warranty, repair time, charges, battery issues, "
        "screen replacement, software issues, network issues, charging problems, "
        "and repair tracking."
    )

    bot = load_bot()

    if bot.ready:
        st.success("Advanced RAG chatbot is ready using FAISS + Sentence Transformers.")
    else:
        st.warning("FAISS/RAG could not load. Fallback chatbot is active.")

        if bot.error_message:
            with st.expander("Technical Error"):
                st.code(bot.error_message)

    if bot.generator is not None:
        st.info("Hugging Face FLAN-T5 answer generator is active.")
    else:
        st.info("Hugging Face generator is not active. Direct RAG answers are active.")

    tab1, tab2, tab3 = st.tabs(
        [
            "💬 Chat Assistant",
            "🧪 Sample Questions",
            "📚 Knowledge Base",
        ]
    )

    with tab1:
        if "advanced_chat_history" not in st.session_state:
            st.session_state.advanced_chat_history = []

        question = st.chat_input("Ask your service center question...")

        if question:
            result = bot.answer(question)

            st.session_state.advanced_chat_history.append(
                {
                    "question": question,
                    "answer": result.answer,
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "sources": result.sources,
                }
            )

        for chat in st.session_state.advanced_chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])

            with st.chat_message("assistant"):
                st.write(chat["answer"])

                col1, col2 = st.columns(2)

                with col1:
                    st.caption(f"Intent: {chat['intent']}")

                with col2:
                    st.caption(f"Confidence: {chat['confidence']}")

                if chat["sources"]:
                    with st.expander("Retrieved RAG Context"):
                        for index, source in enumerate(chat["sources"], start=1):
                            st.markdown(f"**Source {index}:**")
                            st.info(source)

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.advanced_chat_history = []
            st.rerun()

    with tab2:
        st.subheader("🧪 Test these questions")

        samples = [
            "What are your service center timings?",
            "Is physical damage covered under warranty?",
            "What is the battery replacement cost?",
            "How much does screen replacement cost?",
            "How long will motherboard replacement take?",
            "What is the diagnostic charge?",
            "Is software reset free under warranty?",
            "My phone is not charging. What could be the reason?",
            "WiFi is not connecting. What should I do?",
            "Speaker sound is low. What will technician check?",
            "Track repair status for CUST00001",
        ]

        for sample in samples:
            st.code(sample)

    with tab3:
        st.subheader("📚 Knowledge Base")

        kb_text = read_knowledge_base()

        if kb_text:
            st.text_area(
                "knowledge_base.txt",
                value=kb_text,
                height=500,
            )
        else:
            st.error("Knowledge base file not found.")


if __name__ == "__main__":
    main()