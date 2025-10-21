import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional, Any
import json
import fitz  # PyMuPDF
import pandas as pd


# ---------- Define Agent State ----------
class AgentState(BaseModel):
    input_text: str
    relation_flag: Optional[bool] = None
    drugs: Optional[Any] = None
    result: Optional[Any] = None


# ---------- LLM Setup ----------
api_key = st.secrets["api"]
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, api_key=api_key)


# ---------- Step 1: Classify Relation ----------
def classify_relation(state: AgentState):
    template = PromptTemplate(
        input_variables=["text"],
        template="""
You are a biomedical classifier. Respond with only YES or NO.
Does the text mention or imply a relationship between any drug and any adverse event?

Text: {text}
"""
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

    template = PromptTemplate(
        input_variables=["text"],
        template="""
Extract all drug names from the text.
Output must be a JSON array of strings only. No explanation, no additional keys.
Text: {text}
"""
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

    template = PromptTemplate(
        input_variables=["text", "drugs"],
        template="""
You are an adverse event extraction agent.
For each drug listed below, extract only the adverse events mentioned in the text.
Include the exact sentence where the AE is mentioned.

Output must be STRICT JSON using:
[
  {{
    "drug": "DrugName",
    "adverse_events": [
      {{
        "event": "EventName",
        "reference_sentence": "Sentence from text"
      }}
    ]
  }}
]

Drugs: {drugs}
Text: {text}
"""
    )
    prompt = template.format(text=state.input_text.strip(), drugs=state.drugs)
    response = llm.invoke(prompt)
    try:
        state.result = json.loads(response.content)
    except:
        state.result = {"error": "Invalid JSON returned", "raw": response.content}
    return state


# ---------- LangGraph Assembly ----------
graph = StateGraph(AgentState)
graph.add_node("classify", classify_relation)
graph.add_node("extract_drugs", extract_drugs)
graph.add_node("extract_aes", extract_ae_for_drugs)

graph.set_entry_point("classify")
graph.add_edge("classify", "extract_drugs")
graph.add_edge("extract_drugs", "extract_aes")
graph.add_edge("extract_aes", END)

agent = graph.compile()


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Drug–AE Extractor", layout="wide")
st.title("💊 Biomedical Drug–Adverse Event Extractor")

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
