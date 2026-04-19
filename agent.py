import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
import chromadb
from sentence_transformers import SentenceTransformer
from importlib.metadata import version

# Load environment variables
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY", "")
if not groq_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env")

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Knowledge Base Documents
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "OPD Timings",
        "text": "Q: When is OPD open? A: OPD runs from Monday to Saturday, 9 AM to 5 PM. Some departments may stay longer depending on doctor availability. On Sundays, only emergency cases are handled. It’s better to come 30 minutes early for registration. If you have an online booking, you’ll get priority over walk-ins. Any schedule changes during holidays are shared via the hospital website or helpline."
    },
    {
        "id": "doc_002",
        "topic": "Doctor Consultation and Departments",
        "text": "Q: Which doctor should I visit? A: It depends on your problem. Heart issues go to cardiology, bone or joint pain to orthopedics, skin problems to dermatology, and kids to pediatrics. For general issues like fever, visit a general physician. If unsure, just ask at the reception—they’ll guide you."
    },
    {
        "id": "doc_003",
        "topic": "Consultation Fees",
        "text": "Q: How much is the consultation fee? A: General doctors usually charge between ₹300–₹500, while specialists may charge ₹600–₹1200. Follow-ups within a week can be discounted or sometimes free. Emergency consultations cost more. You can pay via cash, card, or digital methods."
    },
    {
        "id": "doc_004",
        "topic": "Insurance Coverage",
        "text": "Q: Does the hospital accept insurance? A: Yes, insurance providers like Star Health, ICICI Lombard, and HDFC ERGO are accepted. Cashless treatment is available for admitted patients if approved. OPD coverage depends on your plan. Carry your ID, insurance card, and necessary documents. The insurance desk will help you with claims."
    },
    {
        "id": "doc_005",
        "topic": "Appointment Booking Process",
        "text": "Q: How can I book an appointment? A: You can book through the website, mobile app, or helpline. Just enter basic details like name, age, and preferred doctor or department. You’ll get a booking ID and SMS confirmation. Walk-ins are allowed but may take longer. You can also reschedule or cancel anytime."
    },
    {
        "id": "doc_006",
        "topic": "Emergency Services",
        "text": "Q: What about emergencies? A: Emergency services are available 24/7 for serious conditions like accidents, heart attacks, or severe injuries. Just call the helpline or come directly—no appointment needed. Ambulance services are also available, and emergency cases are always given priority."
    },
    {
        "id": "doc_007",
        "topic": "Laboratory and Diagnostic Services",
        "text": "Q: What tests are available? A: The hospital offers blood tests, urine tests, X-rays, MRI, CT scans, and more. Routine lab timings are 7 AM to 8 PM, but emergency tests are available anytime. Reports usually come within 24–48 hours and can be collected online or offline."
    },
    {
        "id": "doc_008",
        "topic": "Pharmacy Services",
        "text": "Q: Is there a pharmacy inside? A: Yes, the hospital has a 24/7 pharmacy. You can get prescribed medicines and basic medical supplies there. Digital payments and insurance billing are supported. Pharmacists can guide you on how to take medicines properly."
    },
    {
        "id": "doc_009",
        "topic": "Health Packages",
        "text": "Q: Are health checkup packages available? A: Yes, there are packages for full-body checkups, heart screening, diabetes, and more. These include tests and doctor consultations at discounted prices. Booking in advance is recommended, and fasting may be required for some tests."
    },
    {
        "id": "doc_010",
        "topic": "Hospital Contact and Helpline",
        "text": "Q: How can I contact the hospital? A: You can call the 24/7 helpline at 040-12345678 for any queries or emergencies. You can also use email or website chat for non-urgent questions."
    }
]

# Build ChromaDB
embedder = SentenceTransformer("all-MiniLM-L6-v2")
import streamlit as st
from chromadb.config import Settings

@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection("capstone_kb")

    # Only insert once (IMPORTANT)
    if collection.count() == 0:
        texts = [d["text"] for d in DOCUMENTS]
        ids   = [d["id"] for d in DOCUMENTS]
        embeddings = embedder.encode(texts).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
        )

    return collection
# State Definition
class CapstoneState(TypedDict):
    question:      str
    messages:      List[dict]
    route:         str
    retrieved:     str
    sources:       List[str]
    tool_result:   str
    answer:        str
    faithfulness:  float
    eval_retries:  int
    intent:        str
    department:    str
    urgency:       str

# Node Functions
def memory_node(state: CapstoneState) -> dict:
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > 7:
        msgs = msgs[-7:]
    return {"messages": msgs}

def router_node(state: CapstoneState) -> dict:
    question = state["question"]
    messages = state.get("messages", [])
    recent   = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"

    prompt = f"""
      You are a routing system for a hospital assistant chatbot (MediCare General Hospital).

      Your job is to classify the user query into ONE of:

      1. retrieve → medical info, hospital policies, fees, doctors, timings
      2. memory_only → follow-up like "what did you just say?", "repeat that"
      3. tool → external actions like booking, scheduling, forms

      Conversation context:
      {recent}

      User question:
      {question}

      Rules:
      - If question needs hospital knowledge → retrieve
      - If it refers to past conversation → memory_only
      - If it requires action → tool

      Return ONLY one word: retrieve, memory_only, tool
      """

    response = llm.invoke(prompt)
    decision = response.content.strip().lower()

    if "memory" in decision:       decision = "memory_only"
    elif "tool" in decision:       decision = "tool"
    else:                          decision = "retrieve"

    return {"route": decision}

def retrieval_node(state: CapstoneState) -> dict:
    collection = get_chroma_collection()
    q_emb   = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    chunks  = results["documents"][0]
    topics  = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
    return {"retrieved": context, "sources": topics}

def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}

def tool_node(state: CapstoneState) -> dict:
    question = state["question"]

    if any(word in question.lower() for word in ["emergency", "accident", "heart attack", "unconscious"]):
      tool_result = (
          "🚨 EMERGENCY DETECTED\n"
            "Call immediately: 040-12345678\n"
            "MediCare Emergency Department is available 24/7.\n"
            "Ambulance service is also available."
      )
    elif any(word in question for word in ["appointment", "book", "schedule", "doctor"]):
        tool_result = (
            "📅 APPOINTMENT REQUEST RECEIVED\n"
            "You can book via:\n"
            "- Hospital website\n"
            "- Mobile app\n"
            "- Helpline: 040-12345678\n\n"
            "Please provide: name, department, preferred time."
        )
    elif any(word in question for word in ["insurance", "claim", "billing", "cashless"]):
        tool_result = (
            "💳 INSURANCE SUPPORT\n"
            "Accepted: Star Health, ICICI Lombard, HDFC ERGO, Arogya\n"
            "Cashless available for admitted patients (subject to approval)\n"
            "Visit insurance desk with ID + policy card."
        )
    else:
      tool_result = (
            "ℹ️ TOOL INFO\n"
            "I can help with appointments, emergency guidance, and insurance queries.\n"
            "Please rephrase your request."
        )
    return {"tool_result": tool_result}

def answer_node(state: CapstoneState) -> dict:
    question    = state["question"]
    retrieved   = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    messages    = state.get("messages", [])
    eval_retries= state.get("eval_retries", 0)

    context_parts = []
    if retrieved:
        context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result:
        context_parts.append(f"TOOL RESULT:\n{tool_result}")
    context = "\n\n".join(context_parts)

    if context:
        system_content = f"""You are "MediCare Assistant", a helpful AI chatbot for MediCare General Hospital, Hyderabad.\n\nYour responsibilities:\n- Answer patient queries about hospital services, doctors, timings, fees, insurance, and appointments.\n- Use ONLY the information provided in the context below.\n- Do NOT assume or generate external medical advice.\n- If the answer is not in the context, clearly say:\n  "I don't have that information in my hospital knowledge base."\n\nSTRICT RULES:\n- Do not hallucinate information.\n- Do not use external knowledge.\n- Be concise, clear, and patient-friendly.\n\nCONTEXT:\n{context}
"""
    else:
        system_content = """\nYou are "MediCare Assistant", a helpful AI chatbot for MediCare General Hospital, Hyderabad.\n\nAnswer based only on conversation history.\nIf unsure, say you don't have enough hospital information."""

    if eval_retries > 0:
        system_content += "\n\nIMPORTANT: Your previous answer did not meet quality standards. Answer using ONLY information explicitly stated in the context above."

    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        lc_msgs.append(HumanMessage(content=msg["content"]) if msg["role"] == "user"
                       else AIMessage(content=msg["content"]))
    lc_msgs.append(HumanMessage(content=question))

    response = llm.invoke(lc_msgs)
    return {"answer": response.content}

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2

def eval_node(state: CapstoneState) -> dict:
    answer   = state.get("answer", "")
    context  = state.get("retrieved", "")[:500]
    retries  = state.get("eval_retries", 0)

    if not context:
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?\nReply with ONLY a number between 0.0 and 1.0.\n1.0 = fully faithful. 0.5 = some hallucination. 0.0 = mostly hallucinated.\n\nContext: {context}\nAnswer: {answer[:300]}"""

    result = llm.invoke(prompt).content.strip()
    try:
        score = float(result.split()[0].replace(",", "."))
        score = max(0.0, min(1.0, score))
    except:
        score = 0.5

    gate = "✅" if score >= FAITHFULNESS_THRESHOLD else "⚠️"
    print(f"  [eval] Faithfulness: {score:.2f} {gate}")
    return {"faithfulness": score, "eval_retries": retries + 1}

def save_node(state: CapstoneState) -> dict:
    messages = state.get("messages", [])
    messages = messages + [{"role": "assistant", "content": state["answer"]}]
    return {"messages": messages}

# Graph Assembly
def route_decision(state: CapstoneState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool":        return "tool"
    if route == "memory_only": return "skip"
    return "retrieve"

def eval_decision(state: CapstoneState) -> str:
    score   = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "retry"

graph = StateGraph(CapstoneState)

graph.add_node("memory",    memory_node)
graph.add_node("router",    router_node)
graph.add_node("retrieve",  retrieval_node)
graph.add_node("skip",      skip_retrieval_node)
graph.add_node("tool",      tool_node)
graph.add_node("generate", answer_node)
graph.add_node("eval",      eval_node)
graph.add_node("save",      save_node)

graph.set_entry_point("memory")
graph.add_edge("memory",   "router")

graph.add_conditional_edges(
    "router", route_decision,
    {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
)

graph.add_edge("retrieve", "generate")
graph.add_edge("skip",     "generate")
graph.add_edge("tool",     "generate")

graph.add_edge("generate", "eval")

graph.add_conditional_edges(
    "eval",
    eval_decision,
    {
        "retry": "generate",   # ✅ clean
        "save":  "save"
    }
)
graph.add_edge("save", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)