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
import whisper


# =========================================
# 🔹 Load Ontologies (HPO, OAE, MONDO)
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
# 🔹 Define Pipeline State Schema
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



# ---- Pipeline Steps (Use your existing functions unchanged) ----
#   (I am omitting for brevity — you already have them)

# Just paste your classify_relation, extract_drugs, identify_adverse_events,
# structure_json_output, map_ontology_terms, validate_and_select_best_ontology functions BELOW:

# ---- paste your functions here (unchanged) ----
def classify_relation(state: AgentState):
    text = state.input_text
    prompt = f"""
You are a biomedical classifier. Analyze the text and respond with only YES or NO.

Determine if the text explicitly or implicitly describes a *causal relationship* between a drug and an adverse event.

Guidelines:
- Answer YES only if the text clearly states or implies that the drug *caused, led to, resulted in, induced, or was associated with* an adverse event.
- If the drug and symptom merely co-occur (e.g., observational or disease-related), answer NO.
- When uncertain, choose NO.

Text: {text}

Respond strictly with YES or NO.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state.relation_flag = "YES" in response.content.upper()
    return state

# =========================================
# 🔹 Step 2 — Drug Extraction
# =========================================
def extract_drugs(state: AgentState):
    text = state.input_text
    prompt = f"""
Extract all drug names mentioned in the text.
Output must be a JSON array of strings only.

Text: {text}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state.drugs = response.content
    return state


# =========================================
# 🔹 Step 3 — Adverse Event Identification
# =========================================
def identify_adverse_events(state: AgentState):
    text = state.input_text

    causality_prompt = f"""
You are an adverse event extraction agent.
For each drug listed below, extract only the adverse events mentioned in the text.
Include the exact sentence where the AE is mentioned.
If a drug has no adverse events mentioned, use "Nan" for both the event and reference_sentence.

Output must be STRICT JSON using:
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

Do not include any extra text, explanation, or comments outside the JSON.

Drugs:  {state.drugs}
Text: {text}
"""



    response = llm.invoke([HumanMessage(content=causality_prompt)])
    state.ae_raw = response.content.strip()
    return state




# =========================================
# 🔹 Step 4 — Structure JSON
# =========================================
def structure_json_output(state: AgentState):
    if not state.ae_raw or "None" in state.ae_raw:
        state.result = {"relation": True, "message": "No adverse events identified"}
        return state

    raw = state.ae_raw
    prompt = f"""
Convert the following extracted information into STRICT JSON.

Rules:
- JSON must be a list of objects.
- Each object must have:
  "drug": string,
  "adverse_events": list of objects with keys "event" and "reference_sentence".
 
- If the drug has no AEs (None), return an empty list for "adverse_events".
- Do not include any explanation or text outside JSON.

Extracted text:
{raw}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state.result = response.content
    return state

# =========================================
# 🔹 Step 5 — Ontology Mapping
# =========================================
def map_ontology_terms(state: AgentState):
    try:
        data_str = state.result
        if isinstance(data_str, str):
            data_str = data_str.replace("```json", "").replace("```", "").strip()
            data = json.loads(data_str)
        else:
            data = data_str

        filtered_data = []

        for d in data:
            # skip drugs with no AE
            if not d.get("adverse_events") or len(d["adverse_events"]) == 0:
                continue

            for ae in d.get("adverse_events", []):
                term = ae.get("event")
                ae["ontology_mapping"] = find_in_all_ontologies(term, top_n=10)

            filtered_data.append(d)

        state.result = filtered_data
        return state

    except Exception as e:
        state.result = {"error": f"Ontology mapping failed: {e}"}
        return state



# =========================================
# 🔹 Step 6 — LLM Validation + Ontology Selection
# =========================================
def validate_and_select_best_ontology(state: AgentState):

    try:
        data = state.result

        if isinstance(data, str):
            data = json.loads(data.replace("```json", "").replace("```", "").strip())

        validated_output = []

        for d in data:
            drug_name = d.get("drug")
            validated_events = []

            for ae in d.get("adverse_events", []):
                ontology_mappings = ae.get("ontology_mapping", [])

                # skip if unmapped (similarity 0 or ontology None)
                if ontology_mappings[0]["ontology"] is None or ontology_mappings[0]["similarity"] == 0:
                    continue

                event_name = ae.get("event")

                # ---- Validation prompt unchanged ----
                reasoning_prompt = f"""
You are a biomedical ontology expert agent.

Task:
1. Determine if the given event is a TRUE adverse event 
2. Identify which ontology (HPO, OAE, or MONDO) provides the most contextually relevant definition.
3. Return the best ontology record, and include all others as alternates.
4. for each adverse event have the excat  refrence sentence also for that 
Drug: {drug_name}
Event: {event_name}
Ontology Mappings:
{json.dumps(ontology_mappings, indent=2)}

Return STRICT JSON (no markdown):

{{
  "event": "{event_name}",
  "is_true_ae": "YES" or "NO",
  "reference sentence": "'reference_sentence'",

  "best_ontology": {{
      "ontology": "HPO" or "OAE" or "MONDO",
      "id": "<ontology_id>",
      "name": "<ontology_label>",
      "similarity": <float between 0 and 1>
  }},
  "alternate_ontologies": [list of the other ontology mappings as shown above],
  "reasoning_summary": "short biomedical justification"
}}
"""
                response = llm.invoke([HumanMessage(content=reasoning_prompt)])
                content = response.content.strip()

                try:
                    content_json = json.loads(content[content.find("{"):content.rfind("}")+1])
                except:
                    content_json = {
                        "event": event_name,
                        "is_true_ae": "UNKNOWN",
                        "best_ontology": None,
                        "alternate_ontologies": ontology_mappings,
                        "reasoning_summary": "Model returned invalid JSON; validation skipped."
                    }

                validated_events.append(content_json)

            # only append drugs that still have validated AE after filtering
            if validated_events:
                validated_output.append({"drug": drug_name, "validated_adverse_events": validated_events})

        state.result = validated_output
        return state

    except Exception as e:
        state.result = {"error": f"Validation agent failed: {e}"}
        return state


# =========================================
# 🔹 Build LangGraph Pipeline
# =========================================
graph = StateGraph(AgentState)
graph.add_node("classify", classify_relation)
graph.add_node("extract_drugs", extract_drugs)
graph.add_node("identify_aes", identify_adverse_events)
graph.add_node("structure_json", structure_json_output)
graph.add_node("map_ontology", map_ontology_terms)
graph.add_node("validate_best", validate_and_select_best_ontology)

graph.set_entry_point("classify")
graph.add_edge("classify", "extract_drugs")
graph.add_edge("extract_drugs", "identify_aes")
graph.add_edge("identify_aes", "structure_json")
graph.add_edge("structure_json", "map_ontology")
graph.add_edge("map_ontology","validate_best")
graph.add_edge("validate_best", END)

pipeline = graph.compile()



# =========================================
# 🔹 STREAMLIT UI
# =========================================

USERS = st.secrets["users"]

# -------------------------
# OPTIONAL: Install if needed
# pip install openai-whisper torch
# -------------------------
def login_page():
    st.title("🔐 Login")

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state["logged_in"] = True
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")


# ======================= 🔧 PDF TEXT EXTRACTOR =======================
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# ======================= 🧠 LLM PIPELINE CALL =======================
# Make sure your pipeline object exists.
# Example call:
def run_pipeline(input_text):
    return pipeline.invoke({"input_text": input_text})


# ======================= 🧪 MAIN APPLICATION =======================
def main_interface():
    st.title("📌 Automated Pharmacovigilance Adverse Event Extractor")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    text_input = st.text_area("Or paste text here")

    run_btn = st.button("Run Extraction")

    if run_btn:

        if uploaded_pdf:
            final_text = extract_text_from_pdf(uploaded_pdf)
            st.success("📄 PDF successfully processed.")
        elif text_input.strip():
            final_text = text_input
        else:
            st.error("❗ Please provide a PDF or text.")
            st.stop()

        with st.spinner("⏳ Extracting Adverse Events..."):
            output = run_pipeline(final_text)

        # Convert to dataframe:
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
                    "Source Sentence": evt.get("reference sentence")
                })

        df = pd.DataFrame(rows)

        st.subheader("📊 Extracted Adverse Events Table")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download CSV",
            data=csv,
            file_name="AE_extraction_output.csv",
            mime="text/csv"
        )

    # Logout button
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()


# ======================= 🚀 STREAMLIT APP LOGIC =======================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
else:
    main_interface() why didn't I get amlodipine reference sentence
