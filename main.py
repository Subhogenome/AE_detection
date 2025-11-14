import streamlit as st
import json
import pandas as pd
import fitz  # PyMuPDF
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
# 🔐 LOGIN
# ======================================================
USERS = st.secrets["users"]


def check_login(username, password):
    return username in USERS and USERS[username] == password


def login_page():
    st.set_page_config(page_title="Login - AE Ontology", layout="centered")
    st.markdown("<h2 style='text-align:center;'>🔐 Login</h2>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_login(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid credentials")


# ======================================================
# 🤖 GLOBAL LLM INSTANCE
# ======================================================
llm = ChatGroq(
    model=st.secrets["model"],
    api_key=st.secrets["api"],
    temperature=0
)


# ======================================================
# 🧬 Load Ontologies
# ======================================================
@st.cache_resource(show_spinner=True)
def load_ontologies():
    hpo = Ontology("http://purl.obolibrary.org/obo/hp.obo")

    oae_url = "https://raw.githubusercontent.com/OAE-ontology/OAE/master/src/oae_merged.owl"
    r = requests.get(oae_url)
    with open("oae.owl", "wb") as f:
        f.write(r.content)
    oae = Graph()
    oae.parse("oae.owl", format="xml")

    mondo_url = "http://purl.obolibrary.org/obo/mondo.obo"
    r = requests.get(mondo_url)
    with open("mondo.obo", "wb") as f:
        f.write(r.content)
    mondo = Ontology("mondo.obo")

    return hpo, oae, mondo


hpo, oae, mondo = load_ontologies()


# ======================================================
# 🧠 Helpers
# ======================================================
def normalize_term(term: str):
    return term.strip().lower().replace("-", " ").replace("_", " ")


def find_in_all_ontologies(term, top_n=10):
    term_norm = normalize_term(term)
    matches = []

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

    for s, p, o_term in oae.triples((None, URIRef("http://www.w3.org/2000/01/rdf-schema#label"), None)):
        score = SequenceMatcher(None, term_norm, normalize_term(str(o_term))).ratio()
        if score > 0.65:
            matches.append({
                "ontology": "OAE",
                "id": str(s),
                "name": str(o_term),
                "similarity": round(score, 3)
            })

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
    return matches[:top_n] if matches else [
        {"ontology": None, "id": None, "name": "Not found", "similarity": 0}
    ]


# ======================================================
# 📌 LangGraph State
# ======================================================
class AgentState(BaseModel):
    input_text: str
    relation_flag: Optional[bool] = None
    drugs: Optional[Any] = None
    ae_raw: Optional[str] = None
    result: Optional[Any] = None


# ======================================================
# 🔥 Your Agent Steps (prompts unchanged)
# ======================================================

def classify_relation(state: AgentState):
    prompt = f"""
You are a biomedical classifier. Analyze the text and respond with only YES or NO.

Determine if the text explicitly or implicitly describes a *causal relationship* between a drug and an adverse event.

Guidelines:
- Answer YES only if the text clearly states or implies that the drug *caused, led to, resulted in, induced, or was associated with* an adverse event.
- If the drug and symptom merely co-occur (e.g., observational or disease-related), answer NO.
- When uncertain, choose NO.

Text: {state.input_text}

Respond strictly with YES or NO.
"""
    r = llm.invoke([HumanMessage(content=prompt)])
    state.relation_flag = "YES" in r.content.upper()
    return state


def extract_drugs(state: AgentState):
    prompt = f"""
Extract all drug names mentioned in the text.
Output must be a JSON array of strings only.

Text: {state.input_text}
"""
    r = llm.invoke([HumanMessage(content=prompt)])
    state.drugs = r.content
    return state


def identify_adverse_events(state: AgentState):
    prompt = f"""
You are a biomedical relation identifier.

Your task is to identify **only true adverse events (AEs)** that are **causally related** to each drug mentioned in the text.

Guidelines:
- Identify an AE **only if the text explicitly states or implies a cause–effect relationship** (e.g., "caused", "led to", "resulted in", "induced", "associated with").
- If a drug and a symptom are both mentioned but not causally linked, output **None (None)** for that drug.
- Do **not** infer relationships based on co-occurrence or background disease symptoms.
- If uncertain, always output None (None). Err on the side of not linking.

Output format:
Drug: <drug_name> -> <adverse_event_1> (sentence_1), <adverse_event_2> (sentence_2), ...
If no AE is linked to a drug, output:
Drug: <drug_name> -> None (None)

Now analyze the following:
Drugs: {state.drugs}
Text: {state.input_text}
"""
    r = llm.invoke([HumanMessage(content=prompt)])
    state.ae_raw = r.content
    return state


def structure_json_output(state: AgentState):
    if not state.ae_raw:
        state.result = []
        return state

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
{state.ae_raw}
"""
    r = llm.invoke([HumanMessage(content=prompt)])
    cleaned = r.content.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            state.result = parsed
        else:
            state.result = []
    except Exception:
        state.result = []

    return state


def map_ontology_terms(state: AgentState):
    if not isinstance(state.result, list):
        return state

    for d in state.result:
        if not isinstance(d, dict):
            continue
        for ae in d.get("adverse_events", []):
            if not isinstance(ae, dict):
                continue
            term = ae.get("event")
            if term:
                ae["ontology_mapping"] = find_in_all_ontologies(term)
            else:
                ae["ontology_mapping"] = []
    return state


def validate_and_select_best_ontology(state: AgentState):
    """
    More defensive validation stage:
    - Handles bad JSON
    - Handles unexpected shapes
    - Preserves reference_sentence
    """

    if not isinstance(state.result, list):
        # if it's still a string, last-ditch parse attempt
        if isinstance(state.result, str):
            try:
                possible = json.loads(state.result)
                if isinstance(possible, list):
                    state.result = possible
                else:
                    return state
            except Exception:
                return state
        else:
            return state

    final = []

    for d in state.result:
        # Ensure each element is a dict
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except Exception:
                continue
        if not isinstance(d, dict):
            continue

        drug = d.get("drug")
        ae_list = d.get("adverse_events", [])
        if not isinstance(ae_list, list):
            continue

        validated_events = []

        for ae in ae_list:
            if not isinstance(ae, dict):
                continue

            event = ae.get("event")
            if not event:
                continue
            mappings = ae.get("ontology_mapping", [])
            reference_sentence = ae.get("reference_sentence")

            reasoning_prompt = f"""
You are a biomedical ontology expert agent.

Task:
1. Determine if the given event is a TRUE adverse event ie mentioned in the ontolgies
2. Identify which ontology (HPO, OAE, or MONDO) provides the most contextually relevant definition.
3. Return the best ontology record, and include all others as alternates.

Drug: {drug}
Event: {event}
Ontology Mappings:
{json.dumps(mappings, indent=2)}

Return STRICT JSON (no markdown):

{{
  "event": "{event}",
  "is_true_ae": "YES" or "NO",
  "best_ontology": {{
      "ontology": "HPO" or "OAE" or "MONDO",
      "id": "<ontology_id>",
      "name": "<ontology_label>",
      "similarity": <float between 0 and 1>
  }},
  "alternate_ontologies": [list],
  "reasoning_summary": "short biomedical justification"
}}
"""
            try:
                resp = llm.invoke([HumanMessage(content=reasoning_prompt)])
                content = resp.content.strip()

                # First attempt: direct JSON parse
                try:
                    parsed = json.loads(content)
                except Exception:
                    # Second attempt: ask LLM to repair formatting only
                    repair_prompt = f"""
The following output was intended to be strict JSON but is malformed.

Fix ONLY the formatting. Do NOT change any values or field names.

Malformed JSON:
{content}

Return VALID JSON only.
"""
                    repair_resp = llm.invoke([HumanMessage(content=repair_prompt)])
                    repaired = repair_resp.content.strip()
                    try:
                        parsed = json.loads(repaired)
                    except Exception:
                        parsed = {
                            "event": event,
                            "is_true_ae": "UNKNOWN",
                            "best_ontology": None,
                            "alternate_ontologies": mappings,
                            "reasoning_summary": "LLM could not produce valid JSON"
                        }

                # Ensure parsed is a dict
                if not isinstance(parsed, dict):
                    parsed = {
                        "event": event,
                        "is_true_ae": "UNKNOWN",
                        "best_ontology": None,
                        "alternate_ontologies": mappings,
                        "reasoning_summary": "LLM did not return an object"
                    }

            except Exception:
                parsed = {
                    "event": event,
                    "is_true_ae": "UNKNOWN",
                    "best_ontology": None,
                    "alternate_ontologies": mappings,
                    "reasoning_summary": "Validation step failed"
                }

            # Preserve the original reference sentence
            parsed["reference_sentence"] = reference_sentence
            validated_events.append(parsed)

        final.append({
            "drug": drug,
            "validated_adverse_events": validated_events
        })

    state.result = final
    return state


# ======================================================
# 🔗 Build Graph
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
# 🎨 UI
# ======================================================
def main_app():
    st.set_page_config(page_title="Drug–AE Ontology Extractor", layout="wide")
    st.title("💊 Drug–AE Ontology Extraction Pipeline")

    st.write(f"Welcome **{st.session_state.username}**")

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    st.divider()

    mode = st.radio("Input Type", ["Paste Text", "Upload PDF"])
    text = ""

    if mode == "Paste Text":
        text = st.text_area("Paste biomedical text here:", height=200)
    else:
        pdf = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf:
            with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
                text = "\n".join(page.get_text("text") for page in doc)
            st.text_area("Extracted Text Preview:", text[:2000])

    if st.button("Run Extraction"):
        if not text.strip():
            st.warning("Please enter or upload text.")
        else:
            with st.spinner("Running LLM + Ontology pipeline..."):
                out = pipeline.invoke({"input_text": text})
                result = out["result"]

            st.success("✔ Completed")
            st.subheader("📦 Final JSON Output")
            st.json(result)

            # Flatten table if result looks like list of dicts
            rows = []
            if isinstance(result, list):
                for entry in result:
                    if not isinstance(entry, dict):
                        continue
                    drug = entry.get("drug")
                    for ae in entry.get("validated_adverse_events", []):
                        if not isinstance(ae, dict):
                            continue
                        best = ae.get("best_ontology") or {}
                        rows.append({
                            "Drug": drug,
                            "Event": ae.get("event"),
                            "Valid AE?": ae.get("is_true_ae"),
                            "Reference Sentence": ae.get("reference_sentence"),
                            "Ontology": best.get("ontology"),
                            "Ontology ID": best.get("id"),
                            "Ontology Name": best.get("name"),
                            "Similarity": best.get("similarity"),
                            "Reasoning": ae.get("reasoning_summary"),
                        })

            if rows:
                df = pd.DataFrame(rows)
                st.subheader("📊 Structured Table")
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False),
                    "drug_ae_ontology_output.csv"
                )
            else:
                st.info("No structured validated AE events to display.")


# ======================================================
# 🚀 ENTRY
# ======================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    main_app()
else:
    login_page()
