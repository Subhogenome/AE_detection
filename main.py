




import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional, Any
import json
import fitz  # PyMuPDF
import pandas as pd


# ---------- Authentication Configuration ----------
# In production, use a proper database and hashed passwords
# ---------- USERS FROM SECRETS ----------
USERS = st.secrets["users"]

# ---------- Login Function ----------
def check_login(username, password):
    """Verify user credentials"""
    return username in USERS and USERS[username] == password


def login_page():
    """Display login page"""
    st.set_page_config(page_title="Login - Drug–AE Extractor", layout="centered")
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1>💊 Drug–AE Extractor</h1>
            <p style='color: #666; font-size: 16px;'>Biomedical Adverse Event Analysis Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.container():
            st.subheader("🔐 Login")
            
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                login_button = st.button("Login", use_container_width=True, type="primary")
            
            if login_button:
                if username and password:
                    if check_login(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please enter both username and password")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
     
        
        st.markdown("""
        <div style='text-align: center; padding: 20px; color: #888; font-size: 12px;'>
            <p>Secure biomedical text analysis system</p>
        </div>
        """, unsafe_allow_html=True)


# ---------- Define Agent State ----------
class AgentState(BaseModel):
    input_text: str
    relation_flag: Optional[bool] = None
    drugs: Optional[Any] = None
    result: Optional[Any] = None


# ---------- LLM Setup ----------
def initialize_llm():
    api_key = st.secrets["api"]
    return ChatGroq(model=st.secrets["model"], temperature=0, api_key=api_key)


# ---------- Step 1: Classify Relation ----------
def classify_relation(state: AgentState):
    llm = initialize_llm()
    template = PromptTemplate(
        input_variables=["text"],
        template=st.secrets["prompt1"]
    )
    prompt = template.format(text=state.input_text.strip())
    response = llm.invoke(prompt)
    state.relation_flag = "YES" in response.content.upper()
    return state


# ---------- Step 2: Extract Drugs ----------
def extract_drugs(state: AgentState):
    if not state.relation_flag:
        state.result = {"relation": False, "message": "No drug-AE relation found"}
        return state

    llm = initialize_llm()
    template = PromptTemplate(
        input_variables=["text"],
        template=st.secrets["prompt2"]
    )
    prompt = template.format(text=state.input_text.strip())
    response = llm.invoke(prompt)
    state.drugs = response.content
    return state


# ---------- Step 3: Extract Adverse Events ----------
def extract_ae_for_drugs(state: AgentState):
    if not state.drugs:
        state.result = {"relation": False, "message": "Drugs not found"}
        return state

    llm = initialize_llm()
    template = PromptTemplate(
        input_variables=["text", "drugs"],
        template=st.secrets["prompt3"]
    )
    prompt = template.format(text=state.input_text.strip(), drugs=state.drugs)
    response = llm.invoke(prompt)
    try:
        state.result = json.loads(response.content)
    except:
        state.result = {"error": "Invalid JSON returned", "raw": response.content}
    return state


# ---------- LangGraph Assembly ----------
def create_agent():
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_relation)
    graph.add_node("extract_drugs", extract_drugs)
    graph.add_node("extract_aes", extract_ae_for_drugs)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "extract_drugs")
    graph.add_edge("extract_drugs", "extract_aes")
    graph.add_edge("extract_aes", END)

    return graph.compile()


# ---------- Main Application ----------
def main_app():
    st.set_page_config(page_title="Drug–AE Extractor", layout="wide")
    
    # Header with logout
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("💊 Biomedical Drug–Adverse Event Extractor")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
    
    st.markdown(f"**Welcome, {st.session_state.username}!**")
    st.divider()

    uploaded_pdf = st.file_uploader("📄 Upload a biomedical case report (PDF)", type=["pdf"])

    if uploaded_pdf:
        st.info("Extracting text from PDF...")
        pdf_text = ""
        with fitz.open(stream=uploaded_pdf.read(), filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text("text") + "\n"

        st.text_area("📜 Extracted Text Preview", pdf_text[:2000], height=200)

        if st.button("🔍 Analyze PDF"):
            with st.spinner("Running biomedical AE extraction..."):
                agent = create_agent()
                output = agent.invoke({"input_text": pdf_text})

            result = output["result"]

            if isinstance(result, list):
                st.success("✅ Extraction complete!")

                rows = []
                for drug_obj in result:
                    drug = drug_obj.get("drug", "")
                    for ae in drug_obj.get("adverse_events", []):
                        rows.append({
                            "Drug": drug,
                            "Adverse Event": ae.get("event", ""),
                            "Reference Sentence": ae.get("reference_sentence", "")
                        })

                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download Results as CSV",
                        df.to_csv(index=False),
                        "drug_ae_results.csv"
                    )
                else:
                    st.warning("No adverse events found.")
            else:
                st.error("⚠️ Could not parse output. Check raw response below:")
                st.code(result, language="json")


# ---------- App Entry Point ----------
if __name__ == "__main__":
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None

    # Route to appropriate page
    if st.session_state.authenticated:
        main_app()
    else:
        login_page()
