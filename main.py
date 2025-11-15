import streamlit as st
import fitz  # PyMuPDF
import json
import pandas as pd

# ---- Import your existing pipeline and objects ----
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import Optional, Any
from pronto import Ontology
from rdflib import Graph, URIRef
from difflib import SequenceMatcher
import requests



# =========================================
# 🔹 Load Ontologies
# =========================================
@st.cache_resource
def load_ontologies():
    st.write("Loading ontologies... Please wait 30–60 seconds on first run.")

    hpo = Ontology("http://purl.obolibrary.org/obo/hp.obo")

    OAE_URL = "https://raw.githubusercontent.com/OAE-ontology/OAE/master/src/oae_merged.owl"
    OAE_FILE = "oae_merged.owl"
    r = requests.get(OAE_URL)
    with open(OAE_FILE, "wb") as f:
        f.write(r.content)
    oae = Graph()
    oae.parse(OAE_FILE, format="xml")

    MONDO_URL = "http://purl.obolibrary.org/obo/mondo.obo"
    MONDO_FILE = "mondo.obo"
    r = requests.get(MONDO_URL)
    with open(MONDO_FILE, "wb") as f:
        f.write(r.content)
    mondo = Ontology(MONDO_FILE)

    return hpo, oae, mondo


hpo, oae, mondo = load_ontologies()



# =========================================
# 🔹 Utility Functions
# =========================================
def normalize_term(term: str):
    return term.strip().lower().replace("-", " ").replace("_", " ")


def find_in_all_ontologies(term, top_n=5):
    """Find top matches across HPO, OAE, and MONDO"""
    term_norm = normalize_term(term)
    matches = []

    for t in hpo.terms():
        if t.name:
            ratio = SequenceMatcher(None, term_norm, normalize_term(t.name)).ratio()
            if ratio > 0.65:
                matches.append({"ontology": "HPO", "id": t.id, "name": t.name, "similarity": round(ratio, 3)})

    for s, p, o in oae.triples((None, URIRef("http://www.w3.org/2000/01/rdf-schema#label"), None)):
        ratio = SequenceMatcher(None, term_norm, normalize_term(str(o))).ratio()
        if ratio > 0.65:
            matches.append({"ontology": "OAE", "id": str(s), "name": str(o), "similarity": round(ratio, 3)})

    for t in mondo.terms():
        if t.name:
            ratio = SequenceMatcher(None, term_norm, normalize_term(t.name)).ratio()
            if ratio > 0.65:
                matches.append({"ontology": "MONDO", "id": t.id, "name": t.name, "similarity": round(ratio, 3)})

    matches.sort(key=lambda x: -x["similarity"])
    return matches[:top_n] or [{"ontology": None, "id": None, "name": "Not found", "similarity": 0}]



# =========================================
# 🔹 State Schema
# =========================================
class AgentState(BaseModel):
    input_text: str
    relation_flag: Optional[bool] = None
    drugs: Optional[Any] = None
    ae_raw: Optional[str] = None
    result: Optional[Any] = None



# =========================================
# 🔹 LLM Setup
# =========================================
api = st.secrets["api"]
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, api_key=api)



# =========================================
# 🔹 Step 1 — Relation Check
# =========================================
def classify_relation(state: AgentState):
    text = state.input_text
    prompt = f"""
You are a biomedical classifier. Respond ONLY YES or NO.

Determine if the text describes a causal relationship between a drug and an adverse event.

Text: {text}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state.relation_flag = "YES" in response.content.upper()
    return state



# =========================================
# 🔹 Step 2 — Extract Drugs
# =========================================
def extract_drugs(state: AgentState):
    text = state.input_text
    prompt = f"""
Extract all drug names mentioned in the text.
Output ONLY a JSON array of strings.

Text: {text}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state.drugs = response.content
    return state



# =========================================
# 🔹 Step 3 — AE Extraction (⚠ Ensures reference_sentence always exists)
# =========================================
def identify_adverse_events(state: AgentState):
    text = state.input_text

    causality_prompt = f"""
You are an adverse event extraction agent.

Rules:
- Extract ONLY adverse events for each drug.
- You MUST include "reference_sentence" (not "reference sentence").
- If no AE exists for a drug, return "Nan" for event and reference_sentence.

Output MUST be strictly:

[
  {{
    "drug": "DrugName",
    "adverse_events": [
      {{
        "event": "EventName or Nan",
        "reference_sentence": "Sentence from text or Nan"
      }}
    ]
  }}
]

Drugs: {state.drugs}
Text: {text}
"""

    response = llm.invoke([HumanMessage(content=causality_prompt)])
    state.ae_raw = response.content.strip()
    return state



# =========================================
# 🔹 Step 4 — Structure JSON
# =========================================
def structure_json_output(state: AgentState):
    raw = state.ae_raw
    prompt = f"""
Convert the extracted data into STRICT JSON format.

Required Structure:
- "drug": string
- "adverse_events": list of {{ "event": string, "reference_sentence": string }}

Extracted:
{raw}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state.result = response.content
    return state



# =========================================
# 🔹 Step 5 — Map Ontologies
# =========================================
def map_ontology_terms(state: AgentState):
    try:
        data_str = state.result
        if isinstance(data_str, str):
            data = json.loads(data_str.replace("```json","").replace("```","").strip())
        else:
            data = data_str

        for d in data:
            for ae in d.get("adverse_events", []):
                ae["ontology_mapping"] = find_in_all_ontologies(ae.get("event"))

        state.result = data
        return state

    except Exception as e:
        state.result = {"error": f"Ontology mapping failed: {e}"}
        return state



# =========================================
# 🔹 Step 6 — Validate & Pick Best Ontology (Ensures ref sentence stays)
# =========================================
def validate_and_select_best_ontology(state: AgentState):
    data = state.result
    if isinstance(data, str):
        data = json.loads(data)

    validated_output = []

    for d in data:
        drug_name = d.get("drug")
        validated_events = []

        for ae in d.get("adverse_events", []):
            ref = ae.get("reference_sentence") or ae.get("reference sentence") or "Nan"
            ontology_mappings = ae.get("ontology_mapping", [])

            reasoning_prompt = f"""
Return strictly formatted JSON:

{{
  "event": "{ae.get('event')}",
  "is_true_ae": "YES" or "NO",
  "reference_sentence": "{ref}",
  "best_ontology": {json.dumps(ontology_mappings[0])},
  "alternate_ontologies": {json.dumps(ontology_mappings[1:])},
  "reasoning_summary": "Short justification"
}}
"""

            response = llm.invoke([HumanMessage(content=reasoning_prompt)])
            validated_events.append(json.loads(response.content))

        validated_output.append({
            "drug": drug_name,
            "validated_adverse_events": validated_events
        })

    state.result = validated_output
    return state





# =========================================
# 🔹 Build Pipeline
# =========================================
graph = StateGraph(AgentState)

graph.add_node("classify", classify_relation)
graph.add_node("extract_drugs", extract_drugs)
graph.add_node("identify_aes", identify_adverse_events)
graph.add_node("structure_json", structure_json_output)
graph.add_node("map_ontology", map_ontology_terms)
graph.add_node("validate_best", validate_and_select_best_ontology)

graph.set_entry_point("classify")
graph.add_edge("classify","extract_drugs")
graph.add_edge("extract_drugs","identify_aes")
graph.add_edge("identify_aes","structure_json")
graph.add_edge("structure_json","map_ontology")
graph.add_edge("map_ontology","validate_best")
graph.add_edge("validate_best", END)

pipeline = graph.compile()



# =========================================
# 🔹 STREAMLIT UI
# =========================================
st.title("📌 Automated Pharmacovigilance Adverse Event Extractor")

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
text_input = st.text_area("Or paste text here")
run_btn = st.button("Run Extraction")

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


if run_btn:

    if uploaded_pdf:
        final_text = extract_text_from_pdf(uploaded_pdf)
        st.success("📄 PDF processed.")
    elif text_input.strip():
        final_text = text_input
    else:
        st.error("❗ Provide PDF or text.")
        st.stop()

    with st.spinner("Extracting AE relationships... ⏳"):
        output = pipeline.invoke({"input_text": final_text})

    # ---- Convert to Table ----
    rows = []

    for item in output["result"]:
        for evt in item["validated_adverse_events"]:
            best = evt.get("best_ontology") or {}

            rows.append({
                "Drug": item.get("drug"),
                "Adverse Event": evt.get("event"),
                "Ontology": best.get("ontology", "N/A"),
                "Ontology ID": best.get("id", "N/A"),
                "Ontology Term": best.get("name", "N/A"),
                "Is True AE": evt.get("is_true_ae"),
                "Source Sentence": evt.get("reference_sentence") or "Nan"
            })

    df = pd.DataFrame(rows)

    st.subheader("📊 Extracted Adverse Events")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download CSV",
        data=csv,
        file_name="AE_extraction_output.csv",
        mime="text/csv"
    )
