import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import json
import requests

# ---- Your Original Pipeline Code (UNCHANGED) ----

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import Optional, Any
from pronto import Ontology
from rdflib import Graph, URIRef
from difflib import SequenceMatcher


# =========================================
# 🔹 Load Ontologies (HPO, OAE, MONDO)
# =========================================
print("Loading ontologies...")

# --- HPO ---
hpo = Ontology("http://purl.obolibrary.org/obo/hp.obo")

# --- OAE ---
OAE_URL = "https://raw.githubusercontent.com/OAE-ontology/OAE/master/src/oae_merged.owl"
OAE_FILE = "oae_merged.owl"
r = requests.get(OAE_URL)
with open(OAE_FILE, "wb") as f:
    f.write(r.content)
oae = Graph()
oae.parse(OAE_FILE, format="xml")

# --- MONDO ---
MONDO_URL = "http://purl.obolibrary.org/obo/mondo.obo"
MONDO_FILE = "mondo.obo"
r = requests.get(MONDO_URL)
with open(MONDO_FILE, "wb") as f:
    f.write(r.content)
mondo = Ontology(MONDO_FILE)

print("Ontologies loaded.\n")


# =========================================
# 🔹 Utility Functions
# =========================================
def normalize_term(term: str):
    return term.strip().lower().replace("-", " ").replace("_", " ")


def find_in_all_ontologies(term, top_n=10):
    term_norm = normalize_term(term)
    matches = []

    # HPO
    for t in hpo.terms():
        if not t.name:
            continue
        ratio = SequenceMatcher(None, term_norm, normalize_term(t.name)).ratio()
        if ratio > 0.65:
            matches.append({"ontology": "HPO","id": t.id,"name": t.name,"similarity": round(ratio,3)})

    # OAE
    for s,p,o in oae.triples((None, URIRef("http://www.w3.org/2000/01/rdf-schema#label"), None)):
        name=str(o)
        ratio=SequenceMatcher(None, term_norm, normalize_term(name)).ratio()
        if ratio>0.65:
            matches.append({"ontology":"OAE","id":str(s),"name":name,"similarity":round(ratio,3)})

    # MONDO
    for t in mondo.terms():
        if not t.name: continue
        ratio = SequenceMatcher(None, term_norm, normalize_term(t.name)).ratio()
        if ratio>0.65:
            matches.append({"ontology":"MONDO","id":t.id,"name":t.name,"similarity":round(ratio,3)})

    matches.sort(key=lambda x: -x["similarity"])
    return matches if matches else [{"ontology": None,"id": None,"name": "Not found","similarity":0}]


# =========================================
# 🔹 Define State Schema
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
# 🔹 Step 1 — Relation Classification
# =========================================
def classify_relation(state: AgentState):
    text=state.input_text
    prompt=f"""
You are a biomedical classifier. Analyze the text and respond with only YES or NO.

Determine if the text explicitly or implicitly describes a *causal relationship* between a drug and an adverse event.

Guidelines:
- Answer YES only if the text clearly states or implies that the drug *caused, led to, resulted in, induced, or was associated with* an adverse event.
- If the drug and symptom merely co-occur (e.g., observational or disease-related), answer NO.
- When uncertain, choose NO.

Text: {text}

Respond strictly with YES or NO.
"""
    response=llm.invoke([HumanMessage(content=prompt)])
    state.relation_flag=response.content.strip().upper()=="YES"
    return state


# =========================================
# 🔹 Step 2 — Drug Extraction
# =========================================
def extract_drugs(state: AgentState):
    text=state.input_text
    prompt=f"""
Extract all drug names mentioned in the text.
Output must be STRICT JSON array, ONLY this format:

["drug1", "drug2"]

Text: {text}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        state.drugs=json.loads(response.content.strip())
    except:
        state.drugs=[d.strip() for d in response.content.replace("[","").replace("]","").split(",")]
    
    return state


# =========================================
# 🔹 Step 3 — AE Extraction
# =========================================
def identify_adverse_events(state: AgentState):
    text=state.input_text
    causality_prompt=f"""
You are an adverse event extraction agent.

Your task is to identify adverse events that are **causally linked** to the drug(s) listed below.

A relationship counts as causal if ANY of the following appear:
- Direct causal words: "caused", "resulted in", "led to", "induced", "triggered", "produced"
- Clinical reasoning phrases: "linked to", "attributed to", "associated with", "likely due to", "suspected to be caused by"
- Pharmacovigilance evidence: "temporal association", "rechallenge reproduced", "dechallenge improved"
- Case report certainty language: "confirmed", "determined", "consistent with", "known side effect"

If **no causal relationship exists** for a drug, return an empty list.

Return STRICT JSON.

Drugs:  {state.drugs}
Text: {text}
"""
    response=llm.invoke([HumanMessage(content=causality_prompt)])
    state.ae_raw=response.content.strip()
    return state


# =========================================
# 🔹 Step 4 — Structure JSON
# =========================================
def structure_json_output(state: AgentState):
    raw=state.ae_raw.replace("```json","").replace("```","").strip()
    try:
        state.result=json.loads(raw)
    except:
        state.result=[]
    return state


# =========================================
# 🔹 Step 5 — Ontology Mapping
# =========================================
def map_ontology_terms(state: AgentState):
    for d in state.result:
        for ae in d.get("adverse_events",[]):
            ae["ontology_mapping"] = find_in_all_ontologies(ae["event"], top_n=10)
    return state


# =========================================
# 🔹 Step 6 — Validation Logic (No Prompt Change)
# =========================================
def validate_and_select_best_ontology(state: AgentState):

    validated_output = []

    for d in state.result:
        drug_name=d.get("drug")
        validated_events=[]

        for ae in d.get("adverse_events",[]):
            mapping=ae.get("ontology_mapping",[])
            no_mapping = (not mapping or mapping[0]["ontology"] is None)

            if no_mapping:  # <-- Your rule
                validated_events.append({
                    "event": ae["event"],
                    "reference_sentence": ae["reference_sentence"],
                    "is_true_ae": "PARTIAL",
                    "best_ontology": None,
                    "alternate_ontologies": [],
                    "reasoning_summary": "No ontology match found."
                })
                continue
        
            # → ONLY case where LLM is called
            prompt=f"""
You are a biomedical ontology expert agent.

Task:
1. Determine if the given event is a TRUE adverse event
2. Identify which ontology (HPO, OAE, or MONDO) provides the most contextually relevant definition. if no ontology given then give None
3. Return the best ontology record, and include all others as alternates. else None

Drug: {drug_name}
Event: {ae["event"]}
Ontology Mappings:
{json.dumps(mapping, indent=2)}

Return STRICT JSON (no markdown):

{{
  "event": "{ae['event']}",
  "is_true_ae": "YES" or "NO",
  "reference sentence": "{ae['reference_sentence']}",

  "best_ontology": {{
      "ontology": "HPO" or "OAE" or "MONDO",
      "id": "<ontology_id>",
      "name": "<ontology_label>",
      "similarity": <float between 0 and 1>
  }},
  "alternate_ontologies": {json.dumps(mapping)},
  "reasoning_summary": "short biomedical justification"
}}
"""
            response=llm.invoke([HumanMessage(content=prompt)])
            try:
                validated=json.loads(response.content[response.content.find("{"):response.content.rfind("}")+1])
            except:
                validated={"event":ae["event"],"is_true_ae":"UNKNOWN"}

            validated_events.append(validated)

        if validated_events:
            validated_output.append({"drug":drug_name,"validated_adverse_events":validated_events})

    state.result=validated_output
    return state


# ---------------------------------------------------------
# BUILD PIPELINE
# ---------------------------------------------------------
graph=StateGraph(AgentState)
graph.add_node("classify", classify_relation)
graph.add_node("extract", extract_drugs)
graph.add_node("identify", identify_adverse_events)
graph.add_node("structure", structure_json_output)
graph.add_node("map", map_ontology_terms)
graph.add_node("validate", validate_and_select_best_ontology)

graph.set_entry_point("classify")
graph.add_edge("classify","extract")
graph.add_edge("extract","identify")
graph.add_edge("identify","structure")
graph.add_edge("structure","map")
graph.add_edge("map","validate")
graph.add_edge("validate",END)

pipeline=graph.compile()


# =============================================================
# UI + Result Table
# =============================================================
def extract_text_from_pdf(pdf_file):
    doc=fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])


def output_to_dataframe(result):
    rows=[]
    for item in result:
        for ae in item.get("validated_adverse_events",[]):
            rows.append({
                "Drug": item["drug"],
                "Adverse Event": ae.get("event"),
                "Classification": ae.get("is_true_ae"),
                "Reference Sentence": ae.get("reference_sentence"),
                "Ontology": ae.get("best_ontology",{}).get("ontology") if ae.get("best_ontology") else None
            })
    return pd.DataFrame(rows)


# =============================================================
# STREAMLIT UI
# =============================================================
st.title("🧪 Pharmacovigilance AE Extraction Assistant")

uploaded_pdf = st.file_uploader("📄 Upload PDF", type=["pdf"])
text_input = st.text_area("✍ Or paste text manually:", height=200)

if st.button("🚀 Run"):

    if uploaded_pdf:
        text=extract_text_from_pdf(uploaded_pdf)
    elif text_input.strip():
        text=text_input
    else:
        st.warning("⚠ Please upload a PDF or enter text.")
        st.stop()

    with st.spinner("Processing..."):
        output=pipeline.invoke({"input_text":text})
        result=output["result"]

    st.subheader("📦 JSON Output")
    st.json(result)

    df=output_to_dataframe(result)
    st.subheader("📊 Table Output")
    st.dataframe(df)

    st.download_button("⬇ Download CSV", df.to_csv(index=False).encode("utf-8"), "output.csv")
