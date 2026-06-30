from chatbot.rag import ServiceCenterRAG


def get_chatbot_response(question: str):
    """Get chatbot response from RAG system."""
    rag = ServiceCenterRAG()
    rag.build_or_load_index()

    answer, sources = rag.answer(question)

    return answer, sources