import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional, Any
import json
import fitz  # PyMuPDF
import pandas as pd


# ---------- STREAMLIT SETUP ----------
st.set_page_config(page_title="Drug–AE Extractor", layout="wide")


# ---------- LOAD USERS ----------
USERS = st.secrets["users"]


# ---------- LOGIN UTILS ----------
def check_login(username, password):
    return username in USERS and USERS[username] == password


def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("💊 Drug–AE Extractor")
        st.write("Biomedical Adverse Event Analysis Platform")

        username = st.text_input("👤 Username")
        password = st.text_input("🔐 Password", type="password")

        if st.button("Login"):
            if check_login(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")


# ---------- AGENT STATE ----------
class AgentState(BaseModel):
    input_text: str
    relation_flag: Optional[bool] = None
    drugs: Optional[str] = None
    result: Optional[Any] = None


# ---------- LLM ----------
def initialize_llm():
    return ChatGroq(
        model=st.secrets["model"],
        api_key=st.secrets["api"],
        temperature=0
    )


# ---------- NODE FUNCTIONS ----------
def classify_relation(state: AgentState):
    llm = initialize_llm()
    template = PromptTemplate(input_variables=["text"], template=st.secrets["prompt1"])
    response = llm.invoke(template.format(text=state.input_text))
    state.relation_flag = "YES" in response.content.upper()
    return state


def extract_drugs(state: AgentState):
    if not state.relation_flag:
        state.result = {"relation": False, "message": "No relation found"}
        return state

    llm = initialize_llm()
    template = PromptTemplate(input_variables=["text"], template=st.secrets["prompt2"])
    response = llm.invoke(template.format(text=state.input_text))
    state.drugs = response.content
    return state


def extract_ae_for_drugs(state: AgentState):
    if not state.drugs:
        state.result = {"relation": False, "message": "No drugs found"}
        return state

    llm = initialize_llm()
    template = PromptTemplate(
        input_variables=["text", "drugs"],
        template=st.secrets["prompt3"]
    )
    response = llm.invoke(template.format(text=state.input_text, drugs=state.drugs))

    content = response.content.strip()

    try:
        state.result = json.loads(content)
    except Exception:
        state.result = {"error": "Invalid JSON from LLM", "raw": content}
    
    return state


# ---------- GRAPH ----------
def create_agent():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_relation)
    graph.add_node("extract_drugs", extract_drugs)
    graph.add_node("extract_ae", extract_ae_for_drugs)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "extract_drugs")
    graph.add_edge("extract_drugs", "extract_ae")
    graph.add_edge("extract_ae", END)

    return graph.compile()


# ---------- MAIN APP ----------
def main_app():
    st.title("💊 Biomedical Drug–Adverse Event Extractor")
    st.write(f"👋 Welcome, **{st.session_state.username}**!")

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    uploaded_pdf = st.file_uploader("📄 Upload a biomedical PDF", type=["pdf"])

    if uploaded_pdf:
        pdf_text = ""
        with fitz.open(stream=uploaded_pdf.read(), filetype="pdf") as doc:
            for p in doc:
                pdf_text += p.get_text()

        st.text_area("📜 Text Extract Preview", pdf_text[:2000])

        if st.button("🔍 Analyze"):
            with st.spinner("Processing..."):
                agent = create_agent()
                output = agent.invoke({"input_text": pdf_text})

            result = output.get("result", {})

            if isinstance(result, list):
                rows = []
                for drug_obj in result:
                    drug = drug_obj.get("drug", "")
                    for ae in drug_obj.get("adverse_events", []):
                        rows.append({
                            "Drug": drug,
                            "Adverse Event": ae.get("event", ""),
                            "Sentence": ae.get("reference_sentence", "")
                        })

                df = pd.DataFrame(rows)
                st.success("Extraction Complete!")
                st.dataframe(df)

                st.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "drug_ae_results.csv"
                )
            else:
                st.error("⚠️ Parsing failed — showing raw response")
                st.code(result, language="json")


# ---------- ENTRY ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    main_app()
else:
    login_page()
