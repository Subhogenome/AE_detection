import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import json
import requests
from pronto import Ontology
from rdflib import Graph, URIRef
from difflib import SequenceMatcher
from pydantic import BaseModel
from typing import Optional, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


# ======================================================
# 🔹 LOAD ONTOLOGIES (Unchanged)
# ======================================================
print("Loading ontologies...")

hpo = Ontology("http://purl.obolibrary.org/obo/hp.obo")

OAE_URL = "https://raw.githubusercontent.com/OAE-ontology/OAE/master/src/oae_merged.owl"
OAE_FILE = "oae.owl"
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


# ======================================================
# Helper Functions (unchanged)
# ======================================================
def normalize_term(term):
    return term.strip().lower().replace("-", " ").replace("_", " ")


def find_in_all_ontologies(term, top_n=10):
    term_norm = normalize_term(term)
    matches = []

    # HPO
    for t in hpo.terms():
        if t.name:
            score = SequenceMatcher(None, term_norm, normalize_term(t.name)).ratio()
            if score > 0.65:
                matches.append({
                    "ontology": "HPO",
                    "id": t.id,
                    "name": t.name,
                    "similarity": round(score, 3)
                })

    # OAE
    for s, p, o_term in oae.triples((None, URIRef("http://www.w3.org/2000/01/rdf-schema#label"), None)):
        score = SequenceMatcher(None, term_norm, normalize_term(str(o_term))).ratio()
        if score > 0.65:
            matches.append({
                "ontology": "OAE",
                "id": str(s),
                "name": str(o_term),
                "similarity": round(score, 3)
            })

    # MONDO
    for t in mondo.terms():
        if t.name:
            score = SequenceMatcher(None, term_norm, normalize_term(t.name)).ratio()
            if score > 0.65:
                matches.append({
                    "ontology": "MONDO",
                    "id": t.id,
                    "name": t.name,
                    "similarity": round(score, 3)
                })

    matches.sort(key=lambda x: -x["similarity"])
    return matches[:top_n] if matches else [{"ontology": None, "id": None, "name": "Not found", "similarity": 0}]


# ======================================================
# State Schema (unchanged)
# ======================================================
class AgentState(BaseModel):
    input_text: str
    relation_flag: Optional[bool] = None
    drugs: Optional[Any] = None
    ae_raw: Optional[str] = None
    result: Optional[Any] = None


# ======================================================
# LLM (Loaded from secrets!)
# ======================================================
api = st.secrets["api"]
model_name = st.secrets["model"]
llm = ChatGroq(model=model_name, temperature=0, api_key=api)


# ======================================================
# Pipeline Steps (ALL PROMPTS UNCHANGED)
# ======================================================
def classify_relation(state):
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
    r = llm.invoke([HumanMessage(content=prompt)])
    state.relation_flag = "YES" in r.content.upper()
    return state


def extract_drugs(state):
    prompt = f"""
Extract all drug names mentioned in the text.
Output must be a JSON array of strings only.

Text: {state.input_text}
"""
    r = llm.invoke([HumanMessage(content=prompt)])
    state.drugs = r.content
    return state


def identify_adverse_events(state):
    prompt = f"""
You are a biomedical relation identifier.

Your task is to identify **only true adverse events (AEs)** that are **causally related** to each drug mentioned in the text.

Guidelines:
- Identify an AE **only if the text explicitly states or implies a cause–effect relationship** (e.g., "caused", "led to", "resulted in", "induced", "associated with").
- If a drug and a symptom are both mentioned but not causally linked, output **None (None)** for that drug.
- Do **not** infer relationships based on co-occurrence or background disease symptoms.
- If uncertain, always output None (None). Err on the side of not linking.

Output format:
Drug: <drug_name> -> <adverse_event_1> (sentence_1), ...

Now analyze the following:
Drugs: {state.drugs}
Text: {state.input_text}
"""
    r = llm.invoke([HumanMessage(content=prompt)])
    state.ae_raw = r.content
    return state


def structure_json_output(state):
    prompt = f"""
Convert the following extracted information into STRICT JSON.

Rules:
- JSON must be a list of objects.
- Each object must have:
  "drug": string,
  "adverse_events": list of objects with keys "event" and "reference_sentence".

Extracted text:
{state.ae_raw}
"""
    r = llm.invoke([HumanMessage(content=prompt)])
    state.result = r.content
    return state


def map_ontology_terms(state):
    try:
        data_str = state.result
        data_str = data_str.replace("```json", "").replace("```", "").strip()
        data = json.loads(data_str)

        for d in data:
            for ae in d.get("adverse_events", []):
                ae["ontology_mapping"] = find_in_all_ontologies(ae.get("event"))

        state.result = data
        return state
    except Exception as e:
        state.result = {"error": f"Ontology mapping failed: {e}"}
        return state


# 🔥 FIXED VERSION: Only adds reference sentence & JSON resilience
def validate_and_select_best_ontology(state):

    data = state.result

    # Fix multi-AE failure — make sure we always have a list
    if isinstance(data, str):
        cleaned = data.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except:
            if "{" in cleaned:
                cleaned = cleaned[cleaned.find("{"):cleaned.rfind("}") + 1]
                data = json.loads(cleaned)
            else:
                state.result = {"error": "Invalid JSON returned before validation."}
                return state

    if isinstance(data, dict):
        data = [data]

    final = []

    for d in data:
        validated = []
        for ae in d.get("adverse_events", []):
            event = ae.get("event")
            mappings = ae.get("ontology_mapping", [])
            ref = ae.get("reference_sentence")

            prompt = f"""
You are a biomedical ontology expert agent.

Task:
1. Determine if the given event is a TRUE adverse event ie mentioned in the ontolgies
2. Identify which ontology provides the most contextually relevant definition.

Drug: {d.get("drug")}
Event: {event}
Ontology Mappings:
{json.dumps(mappings, indent=2)}
"""
            try:
                resp = llm.invoke([HumanMessage(content=prompt)])
                parsed = json.loads(resp.content)
            except:
                parsed = {"event": event, "is_true_ae": "UNKNOWN", "best_ontology": None}

            parsed["reference_sentence"] = ref  # <-- REQUIRED FIX
            parsed["alternate_ontologies"] = mappings

            validated.append(parsed)

        final.append({"drug": d.get("drug"), "validated_adverse_events": validated})

    state.result = final
    return state


# ======================================================
# Compile Pipeline
# ======================================================
graph = StateGraph(AgentState)
graph.add_node("classify", classify_relation)
graph.add_node("extract", extract_drugs)
graph.add_node("identify", identify_adverse_events)
graph.add_node("structure", structure_json_output)
graph.add_node("map", map_ontology_terms)
graph.add_node("validate", validate_and_select_best_ontology)
graph.set_entry_point("classify")
graph.add_edge("classify", "extract")
graph.add_edge("extract", "identify")
graph.add_edge("identify", "structure")
graph.add_edge("structure", "map")
graph.add_edge("map", "validate")
graph.add_edge("validate", END)
pipeline = graph.compile()


# ======================================================
# Streamlit UI
# ======================================================

def login():
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user in st.secrets["users"] and st.secrets["users"][user] == pwd:
            st.session_state.logged = True
            st.rerun()
        else:
            st.error("❌ Invalid Credentials")


def app():
    st.title("💊 Drug–AE Ontology Pipeline")

    text = ""
    mode = st.radio("Input", ["📝 Text", "📄 PDF"])

    if mode == "📝 Text":
        text = st.text_area("Paste biomedical text:", height=180)

    else:
        file = st.file_uploader("Upload PDF", type=["pdf"])
        if file:
            with fitz.open(stream=file.read(), filetype="pdf") as pdf:
                text = "\n".join(page.get_text("text") for page in pdf)
            st.text_area("Extracted text:", text[:1500])

    if st.button("🚀 Run"):
        with st.spinner("Processing..."):
            res = pipeline.invoke({"input_text": text})["result"]

        st.json(res)

        rows = []
        for entry in res:
            drug = entry.get("drug")
            for ae in entry.get("validated_adverse_events", []):
                best = ae.get("best_ontology") or {}
                rows.append({
                    "Drug": drug,
                    "Event": ae.get("event"),
                    "Is True AE": ae.get("is_true_ae"),
                    "Reference Sentence": ae.get("reference_sentence"),
                    "Best Ontology": best.get("ontology"),
                    "Ontology ID": best.get("id")
                })

        st.write("📊 Results Table")
        df = pd.DataFrame(rows)
        st.dataframe(df)

        st.download_button("⬇ Download CSV", df.to_csv(index=False), "results.csv")


if "logged" not in st.session_state:
    login()
else:
    app()
