import os

# Force Transformers to use PyTorch only.
# This avoids Keras 3 / TensorFlow compatibility errors.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


KB_PATH = "chatbot/knowledge_base.txt"
FAISS_INDEX_PATH = "chatbot/faiss_index"
DATA_PATH = "data/clean_service_center_data.csv"


@dataclass
class ChatResult:
    """Structured chatbot result."""

    answer: str
    sources: List[str]
    intent: str
    confidence: float


def read_knowledge_base() -> str:
    """Read service center knowledge base."""
    if not os.path.exists(KB_PATH):
        return ""

    with open(KB_PATH, "r", encoding="utf-8") as file:
        return file.read()


def load_service_data() -> Optional[pd.DataFrame]:
    """Load service center dataset."""
    if not os.path.exists(DATA_PATH):
        return None

    return pd.read_csv(DATA_PATH)


def split_knowledge_base_by_sections(text: str) -> List[str]:
    """Split knowledge base into section-wise chunks."""
    if not text.strip():
        return []

    sections = re.split(r"\n(?=[A-Za-z ]+:)", text.strip())

    clean_sections = []
    for section in sections:
        section = section.strip()
        if len(section) > 20:
            clean_sections.append(section)

    return clean_sections


def extract_customer_id(text: str) -> Optional[str]:
    """Extract Customer ID like CUST00001."""
    match = re.search(r"\bCUST\d{5}\b", text.upper())

    if match:
        return match.group(0)

    return None


def detect_intent(question: str) -> str:
    """Detect user question intent."""
    q = question.lower()

    if extract_customer_id(question):
        return "repair_status"

    if any(word in q for word in ["timing", "timings", "open", "close", "working", "hours"]):
        return "service_timings"

    if "warranty" in q or "guarantee" in q:
        return "warranty"

    if "battery" in q:
        return "battery"

    if any(word in q for word in ["screen", "display", "cracked"]):
        return "screen"

    if any(word in q for word in ["software", "reset", "os", "update", "slow", "logo"]):
        return "software"

    if any(word in q for word in ["diagnostic", "charge", "cost", "price", "fees"]):
        return "charges"

    if "motherboard" in q or "dead" in q:
        return "motherboard"

    if any(word in q for word in ["repair", "days", "long", "duration", "time take"]):
        return "repair_timeline"

    if any(word in q for word in ["track", "status"]):
        return "tracking"

    if "charging" in q or "charger" in q:
        return "charging"

    if "network" in q or "wifi" in q or "bluetooth" in q or "sim" in q:
        return "network"

    if "speaker" in q or "sound" in q or "microphone" in q:
        return "speaker"

    return "general"


def direct_answer(question: str) -> Tuple[str, str, float]:
    """Return direct service-center answer for common questions."""
    intent = detect_intent(question)

    answers: Dict[str, str] = {
        "service_timings": (
            "Our service center operates Monday to Saturday, "
            "9:00 AM to 6:00 PM. We are closed on Sundays and public holidays."
        ),
        "warranty": (
            "Devices under warranty within 1 year of purchase get free repairs "
            "for manufacturing defects. Physical damage, water damage, accidental damage, "
            "and cracked screens are not covered under warranty."
        ),
        "battery": (
            "For battery drain issues, we recommend a battery health check. "
            "Battery replacement costs around $50 and usually takes about 1 day."
        ),
        "screen": (
            "Cracked screens are replaced with OEM parts. "
            "Screen replacement costs range from $100 to $300 depending on the model "
            "and usually takes 1 to 2 days."
        ),
        "software": (
            "Software reset and OS reinstall are usually free under warranty. "
            "Out-of-warranty software service costs around $30."
        ),
        "charges": (
            "Out-of-warranty diagnostics cost $20. "
            "This diagnostic charge is waived if you proceed with the repair. "
            "Parts and labor charges are extra."
        ),
        "motherboard": (
            "Motherboard issues are high-priority hardware issues. "
            "Repair or replacement may take up to 7 days depending on part availability."
        ),
        "repair_timeline": (
            "Standard repairs usually take 1 to 3 days. "
            "Motherboard replacement or part unavailability may take up to 7 days."
        ),
        "tracking": (
            "You can track repair status using your Customer ID and Phone Number "
            "on the service portal."
        ),
        "charging": (
            "Charging problems may be caused by a damaged charging port, faulty adapter, "
            "damaged cable, battery problem, or charging IC failure."
        ),
        "network": (
            "Network issues may involve weak signal, WiFi problems, Bluetooth pairing issues, "
            "SIM detection failure, or network IC problems."
        ),
        "speaker": (
            "Speaker issues include low sound, no sound, crackling noise, microphone problems, "
            "or distorted audio. A technician will inspect the speaker, microphone, and audio IC."
        ),
    }

    if intent in answers:
        return answers[intent], intent, 0.95

    return (
        "I can help with service timings, warranty, repair charges, repair status, "
        "battery issues, screen replacement, software issues, charging problems, "
        "network issues, and repair timelines.",
        intent,
        0.50,
    )


def get_customer_status_answer(customer_id: str) -> str:
    """Return repair status for a Customer ID."""
    df = load_service_data()

    if df is None:
        return "Dataset not found. Please run `python data.py` first."

    result = df[df["Customer_ID"].astype(str).str.upper() == customer_id.upper()]

    if result.empty:
        return (
            f"No repair record found for Customer ID {customer_id}. "
            "Please check the Customer ID and try again."
        )

    row = result.iloc[0]

    return (
        f"Repair status for {customer_id}:\n\n"
        f"Customer Name: {row['Customer_Name']}\n"
        f"Product: {row['Product']} - {row['Brand']}\n"
        f"Complaint: {row['Complaint']}\n"
        f"Complaint Type: {row['Complaint_Type']}\n"
        f"Priority: {row['Priority']}\n"
        f"Technician: {row['Technician']}\n"
        f"Status: {row['Status']}\n"
        f"Estimated Repair Time: {row['Repair_Time']} days\n"
        f"Repair Cost: ${row['Repair_Cost']}\n"
        f"Warranty Status: {row['Warranty_Status']}"
    )


class ServiceCenterRAG:
    """
    Advanced AI Service Center Chatbot.

    Features:
    - Intent detection
    - Customer ID repair tracking
    - FAISS semantic search
    - Sentence Transformer embeddings
    - Optional Hugging Face FLAN-T5 answer generation
    - Rule-based fallback for stability
    """

    def __init__(self, use_llm: bool = True) -> None:
        self.vector_store = None
        self.ready = False
        self.error_message = ""
        self.generator = None
        self.use_llm = use_llm

    def build_or_load_index(self) -> bool:
        """Build or load FAISS index."""
        try:
            from langchain_core.documents import Document
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            if os.path.exists(FAISS_INDEX_PATH):
                self.vector_store = FAISS.load_local(
                    FAISS_INDEX_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                self.ready = True
                self.load_generator()
                return True

            kb_text = read_knowledge_base()
            sections = split_knowledge_base_by_sections(kb_text)

            if not sections:
                self.error_message = "Knowledge base is empty."
                self.ready = False
                return False

            documents = [
                Document(
                    page_content=section,
                    metadata={"source": "knowledge_base.txt"},
                )
                for section in sections
            ]

            self.vector_store = FAISS.from_documents(documents, embeddings)
            self.vector_store.save_local(FAISS_INDEX_PATH)

            self.ready = True
            self.load_generator()
            return True

        except Exception as error:
            self.ready = False
            self.error_message = str(error)
            return False

    def load_generator(self) -> None:
        """
        Load optional Hugging Face FLAN-T5 model using PyTorch only.

        If loading fails, chatbot still works with RAG and direct answers.
        """
        if not self.use_llm:
            return

        try:
            from transformers import pipeline

            self.generator = pipeline(
                task="text2text-generation",
                model="google/flan-t5-small",
                framework="pt",
                device=-1,
                max_new_tokens=120,
            )

        except Exception as error:
            self.generator = None
            self.error_message = str(error)

    def retrieve_context(self, question: str, k: int = 3) -> List[str]:
        """Retrieve relevant knowledge base sections."""
        if not self.ready or self.vector_store is None:
            return []

        try:
            docs = self.vector_store.similarity_search(question, k=k)
            return [doc.page_content for doc in docs]

        except Exception:
            return []

    def generate_llm_answer(self, question: str, context: str) -> Optional[str]:
        """Generate answer using optional Hugging Face model."""
        if self.generator is None:
            return None

        prompt = f"""
You are an AI assistant for a service center.
Answer the customer question using only the context.
Give a short, clear, helpful answer.

Context:
{context}

Question:
{question}

Answer:
"""

        try:
            result = self.generator(prompt)
            answer = result[0]["generated_text"].strip()

            if len(answer) < 5:
                return None

            return answer

        except Exception:
            return None

    def answer(self, question: str) -> ChatResult:
        """Return chatbot answer."""
        if not question.strip():
            return ChatResult(
                answer="Please enter a question.",
                sources=[],
                intent="empty",
                confidence=0.0,
            )

        customer_id = extract_customer_id(question)

        if customer_id:
            status_answer = get_customer_status_answer(customer_id)

            return ChatResult(
                answer=status_answer,
                sources=[],
                intent="repair_status",
                confidence=0.98,
            )

        base_answer, intent, confidence = direct_answer(question)

        context_chunks = self.retrieve_context(question, k=3)
        context = "\n\n".join(context_chunks)

        if context and self.generator is not None and intent == "general":
            llm_answer = self.generate_llm_answer(question, context)

            if llm_answer:
                return ChatResult(
                    answer=llm_answer,
                    sources=context_chunks,
                    intent=intent,
                    confidence=0.85,
                )

        return ChatResult(
            answer=base_answer,
            sources=context_chunks,
            intent=intent,
            confidence=confidence,
        )