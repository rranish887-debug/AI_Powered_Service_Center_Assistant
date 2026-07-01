import os

# Force Hugging Face Transformers to use PyTorch only.
# This avoids TensorFlow / Keras 3 compatibility issues.
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
    """
    Detect user question intent.

    Hardware, cost, and troubleshooting intents are checked before service timings.
    This prevents questions like:
    'speaker and display are not working'
    from being incorrectly detected as service timings.
    """
    q = question.lower()

    if extract_customer_id(question):
        return "repair_status"

    # Product-specific display/screen cost questions
    if (
        ("display" in q or "screen" in q)
        and ("cost" in q or "price" in q or "charge" in q or "charges" in q)
        and ("laptop" in q or "mobile" in q or "phone" in q)
    ):
        return "product_display_cost"

    # Charger / adapter cost questions
    if (
        ("charger" in q or "adapter" in q)
        and ("cost" in q or "price" in q or "charge" in q or "charges" in q)
    ):
        return "charger_cost"

    # Multi-issue hardware problems
    has_audio_issue = any(
        word in q for word in ["speaker", "sound", "microphone", "audio"]
    )

    has_display_issue = any(
        word in q for word in ["screen", "display", "touch", "cracked"]
    )

    if has_audio_issue and has_display_issue:
        return "multi_hardware_issue"

    # Hardware / troubleshooting intents
    if has_audio_issue:
        return "speaker"

    if has_display_issue:
        return "screen"

    if any(word in q for word in ["battery", "drain", "backup"]):
        return "battery"

    if any(word in q for word in ["charging", "charger", "adapter", "port"]):
        return "charging"

    if any(word in q for word in ["network", "wifi", "bluetooth", "sim", "signal"]):
        return "network"

    if any(
        word in q
        for word in ["software", "reset", "os", "update", "slow", "logo", "app crash"]
    ):
        return "software"

    if any(word in q for word in ["motherboard", "dead", "power issue", "no power"]):
        return "motherboard"

    # Analytics questions
    if any(
        word in q
        for word in [
            "most common",
            "highest",
            "average",
            "how many",
            "total",
            "analysis",
            "count",
            "top",
        ]
    ):
        return "analytics"

    # Service timing should trigger only for service-center timing questions
    if any(
        phrase in q
        for phrase in [
            "service timing",
            "service timings",
            "service center timing",
            "service center timings",
            "working hours",
            "opening time",
            "closing time",
            "are you open",
            "open on sunday",
            "open today",
        ]
    ):
        return "service_timings"

    if any(
        word in q
        for word in [
            "warranty",
            "guarantee",
            "physical damage",
            "water damage",
            "bill",
        ]
    ):
        return "warranty"

    if any(
        word in q
        for word in ["diagnostic", "charge", "cost", "price", "fees", "refund"]
    ):
        return "charges"

    if any(
        phrase in q
        for phrase in ["repair time", "how long", "how many days", "duration", "timeline"]
    ):
        return "repair_timeline"

    if any(word in q for word in ["track", "status"]):
        return "tracking"

    if any(word in q for word in ["backup", "data loss", "data"]):
        return "data_backup"

    if any(word in q for word in ["pickup", "delivery", "collect"]):
        return "pickup_delivery"

    if any(word in q for word in ["delay", "escalation", "manager", "late"]):
        return "escalation"

    return "general"


def direct_answer(question: str) -> Tuple[str, str, float]:
    """Return direct service-center answer for common questions."""
    intent = detect_intent(question)
    q = question.lower()

    # Product-wise display cost answers
    if intent == "product_display_cost":
        has_laptop = "laptop" in q
        has_mobile = "mobile" in q or "phone" in q

        if has_laptop and has_mobile:
            return (
                "For display replacement, the estimated cost depends on the device model. "
                "Mobile display replacement usually costs around $80 to $180. "
                "Laptop display replacement usually costs around $120 to $300. "
                "Final cost may vary based on brand, screen size, display type, touch support, "
                "and spare part availability.",
                "screen_cost",
                0.96,
            )

        if has_laptop:
            return (
                "Laptop display replacement usually costs around $120 to $300. "
                "The final cost depends on brand, screen size, resolution, touch support, "
                "and part availability.",
                "laptop_display_cost",
                0.96,
            )

        if has_mobile:
            return (
                "Mobile display replacement usually costs around $80 to $180. "
                "The final cost depends on brand, model, OLED/LCD type, touch panel, "
                "and part availability.",
                "mobile_display_cost",
                0.96,
            )

    # Product-wise charger / adapter cost answers
    if intent == "charger_cost":
        has_laptop = "laptop" in q
        has_mobile = "mobile" in q or "phone" in q

        if has_laptop and has_mobile:
            return (
                "For charger replacement, the estimated cost depends on the device model and adapter type. "
                "Laptop charger replacement usually costs around $25 to $80 depending on brand, wattage, "
                "connector type, and original adapter availability. Mobile charger replacement usually costs "
                "around $10 to $40 depending on fast-charging support, cable type, and brand. Final cost is "
                "confirmed after checking the device model and adapter specification.",
                "charger_cost",
                0.96,
            )

        if has_laptop:
            return (
                "Laptop charger replacement usually costs around $25 to $80. "
                "The final cost depends on brand, wattage, connector type, USB-C or pin type, "
                "and original adapter availability.",
                "laptop_charger_cost",
                0.96,
            )

        if has_mobile:
            return (
                "Mobile charger replacement usually costs around $10 to $40. "
                "The final cost depends on brand, fast-charging support, cable type, "
                "adapter wattage, and original charger availability.",
                "mobile_charger_cost",
                0.96,
            )

        return (
            "Charger replacement cost depends on the device type. Laptop chargers usually cost around "
            "$25 to $80, while mobile chargers usually cost around $10 to $40. Final cost is confirmed "
            "after checking the model and adapter specification.",
            "charger_cost",
            0.95,
        )

    answers: Dict[str, str] = {
        "service_timings": (
            "Our service center operates Monday to Saturday, 9:00 AM to 6:00 PM. "
            "We are closed on Sundays and public holidays."
        ),
        "warranty": (
            "Devices under warranty within 1 year of purchase get free repairs "
            "for manufacturing defects. Physical damage, water damage, accidental damage, "
            "cracked screens, burned components, and unauthorized repair are not covered."
        ),
        "battery": (
            "For battery drain issues, we recommend a battery health check. "
            "Battery replacement costs around $50 and usually takes about 1 day."
        ),
        "screen": (
            "Your display issue may be caused by a damaged display panel, loose display connector, "
            "touch screen problem, or screen hardware fault. A technician should inspect the display "
            "panel, touch sensor, and connector. Screen replacement may cost $100 to $300 depending "
            "on the model."
        ),
        "speaker": (
            "Your speaker issue may be caused by speaker damage, microphone fault, audio IC issue, "
            "dust blockage, or software audio settings. A technician will inspect the speaker, "
            "microphone, and audio IC."
        ),
        "multi_hardware_issue": (
            "Your mobile has both speaker/audio and display-related symptoms. "
            "For the speaker issue, the technician will check the speaker, microphone, audio IC, "
            "and software audio settings. For the display issue, the technician will inspect the "
            "display panel, touch sensor, and display connector. Please submit the device for "
            "hardware diagnosis."
        ),
        "software": (
            "Software reset and OS reinstall are usually free under warranty. "
            "Out-of-warranty software service costs around $30. Please back up important data "
            "before software repair."
        ),
        "charges": (
            "Out-of-warranty diagnostics cost $20. This diagnostic charge is waived if you proceed "
            "with the repair. Parts and labor charges are extra."
        ),
        "motherboard": (
            "Motherboard issues are high-priority hardware issues. Symptoms include device dead, "
            "automatic restart, no power, overheating, and power IC failure. Repair or replacement "
            "may take up to 7 days depending on part availability."
        ),
        "repair_timeline": (
            "Standard repairs usually take 1 to 3 days. Battery replacement usually takes 1 day, "
            "screen replacement takes 1 to 2 days, and motherboard repair may take up to 7 days."
        ),
        "tracking": (
            "You can track repair status using your Customer ID and registered phone number "
            "on the service portal. You can also ask me like: Track repair status for CUST00001."
        ),
        "charging": (
            "Charging problems may be caused by a damaged charging port, faulty adapter, "
            "damaged cable, battery issue, or charging IC failure. A technician will inspect "
            "the charging port, adapter, cable, battery health, and charging IC."
        ),
        "network": (
            "Network issues may involve weak signal, WiFi not connecting, Bluetooth pairing issues, "
            "SIM detection failure, internet disconnection, or network IC problems."
        ),
        "data_backup": (
            "Customers are responsible for backing up personal data before repair. The service center "
            "is not responsible for data loss during software reset, OS reinstall, motherboard repair, "
            "or storage replacement."
        ),
        "pickup_delivery": (
            "Customers must bring the service receipt or Customer ID while collecting the device. "
            "Home pickup and delivery may be available for selected locations, and delivery charges "
            "may apply depending on location."
        ),
        "escalation": (
            "If repair is delayed beyond the estimated timeline, customers can contact the service "
            "manager. For urgent cases, provide Customer ID, device model, complaint details, and "
            "phone number."
        ),
    }

    if intent in answers:
        return answers[intent], intent, 0.95

    return (
        "Sorry, I can only help with service center related questions such as "
        "service timings, warranty, repair charges, repair status, battery issues, "
        "screen replacement, speaker issues, software issues, charging problems, "
        "network issues, and repair timelines.",
        intent,
        0.50,
    )


def get_customer_status_answer(customer_id: str) -> str:
    """Return repair status for a Customer ID."""
    df = load_service_data()

    if df is None:
        return "Dataset not found. Please upload `data/clean_service_center_data.csv` and redeploy."

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


def get_analytics_answer(question: str) -> Optional[str]:
    """Answer dataset analysis questions."""
    df = load_service_data()

    if df is None:
        return None

    q = question.lower()

    if "total" in q and ("ticket" in q or "request" in q):
        return f"There are {len(df)} total service tickets in the dataset."

    if "most common" in q and "complaint" in q:
        top = df["Complaint_Type"].value_counts().idxmax()
        count = df["Complaint_Type"].value_counts().max()
        return f"The most common complaint type is {top} with {count} tickets."

    if "brand" in q and ("highest" in q or "most" in q or "top" in q):
        top = df["Brand"].value_counts().idxmax()
        count = df["Brand"].value_counts().max()
        return f"The brand with the highest number of complaints is {top} with {count} complaints."

    if "average repair time" in q or "avg repair time" in q:
        avg_time = round(df["Repair_Time"].mean(), 2)
        return f"The average repair time is {avg_time} days."

    if "average repair cost" in q or "avg repair cost" in q:
        avg_cost = round(df["Repair_Cost"].mean(), 2)
        return f"The average repair cost is ${avg_cost}."

    if "under warranty" in q:
        count = df[df["Warranty_Status"] == "Under Warranty"].shape[0]
        return f"There are {count} tickets under warranty."

    if "out of warranty" in q:
        count = df[df["Warranty_Status"] == "Out of Warranty"].shape[0]
        return f"There are {count} out-of-warranty tickets."

    if "technician" in q and ("most" in q or "highest" in q or "handled" in q):
        top = df["Technician"].value_counts().idxmax()
        count = df["Technician"].value_counts().max()
        return f"The technician who handled the most tickets is {top} with {count} tickets."

    if "positive" in q and "feedback" in q:
        count = df[df["Sentiment"] == "Positive"].shape[0]
        return f"There are {count} positive feedback records."

    if "negative" in q and "feedback" in q:
        count = df[df["Sentiment"] == "Negative"].shape[0]
        return f"There are {count} negative feedback records."

    if "high priority" in q:
        count = df[df["Priority"] == "High"].shape[0]
        return f"There are {count} high-priority tickets."

    return None


class ServiceCenterRAG:
    """
    Advanced AI Service Center Chatbot.

    Features:
    - Intent detection
    - Product-wise display cost answers
    - Charger / adapter cost answers
    - Multi-issue hardware detection
    - Customer ID repair tracking
    - Dataset analytics answers
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
        """Load optional Hugging Face FLAN-T5 model using PyTorch only."""
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
Answer the customer question using only the given context.
Give a short and helpful answer.

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

        intent = detect_intent(question)

        if intent == "analytics":
            analytics_answer = get_analytics_answer(question)

            if analytics_answer:
                return ChatResult(
                    answer=analytics_answer,
                    sources=[],
                    intent="analytics",
                    confidence=0.92,
                )

        base_answer, detected_intent, confidence = direct_answer(question)

        context_chunks = self.retrieve_context(question, k=3)
        context = "\n\n".join(context_chunks)

        # Use LLM only for general questions.
        # For service-specific questions, direct answers are more reliable.
        if context and self.generator is not None and detected_intent == "general":
            llm_answer = self.generate_llm_answer(question, context)

            if llm_answer:
                return ChatResult(
                    answer=llm_answer,
                    sources=context_chunks,
                    intent=detected_intent,
                    confidence=0.85,
                )

        return ChatResult(
            answer=base_answer,
            sources=context_chunks,
            intent=detected_intent,
            confidence=confidence,
        )