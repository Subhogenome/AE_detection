import streamlit as st
import json
import pandas as pd
import fitz
import requests
from pronto import Ontology
from rdflib import Graph, URIRef
from difflib import SequenceMatcher
from pydantic import BaseModel
from typing import Optional, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import curies


# ======================================================
# 🔐 AUTH + LLM
# ======================================================
USERS = st.secrets["users"]

llm = ChatGroq(
    model=st.secrets["model"],
    api_key=st.secrets["api"],
    temperature=0
)


def check_login(username, password):
    return username in USERS and USERS[username] == password


def login_page():
    st.set_page_config(page_title="Login - Drug–AE Extractor", layout="centered")
    st.header("🔐 Login to Drug–AE Ontology Extractor")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_login(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")


# ======================================================
# 🧬 ONTOLOGY LOADING
# ======================================================
@st.cache_resource(show_spinner=True)
def load_ontologies():

    converter = curies.load_prefix_map({
        "OAE": "http://purl.obolibrary.org/obo/OAE_"
    })

    # HPO
    hpo = Ontology("http://purl.obolibrary.org/obo/hp.obo")

    # OAE
    OAE_URL = "https://raw.githubusercontent.com/OAE-ontology/OAE/master/src/oae_merged.owl"
    r = requests.get(OAE_URL)
    with open("oae_merged.owl", "wb") as f:
        f.write(r.content)
    oae = Graph()
    oae.parse("oae_merged.owl", format="xml")

    # MONDO
    MONDO_URL = "http://purl.obolibrary.org/obo/mondo.obo"
    r = requests.get(MONDO_URL)
    with open("mondo.obo", "wb") as f:
        f.write(r.content)
    mondo = Ontology("mondo.obo")

    return hpo, oae, mondo, converter


hpo, oae, mondo, converter = load_ontologies()


# ======================================================
# 🧬 ONTOLOGY MATCH FUNCTION
# ======================================================
def normalize_term(term: str):
    return term.strip().lower().replace("-", " ").replace("_", " ")


def find_in_all_ontologies(term, top_n=10):
    term_norm = normalize_term(term)
    matches = []

    # HPO
    for t in hpo.terms():
        if not t.name: continue
        score = SequenceMatcher(None, term_norm, normalize_term(t.name)).ratio()
        if score > 0.65:
            matches.append({
                "ontology": "HPO",
                "id": t.id,
                "name": t.name,
                "similarity": round(score, 3)
            })

    # OAE
    for s,p,o_term in oae.triples((None, URIRef("http://www.w3.org/2000/01/rdf-schema#label"), None)):
        score = SequenceMatcher(None, term_norm, normalize_term(str(o_term))).ratio()
        if score > 0.65:
            matches.append({
                "ontology": "OAE",
                "id": converter.compress(str(s)),
                "name": str(o_term),
                "similarity": round(score, 3)
            })

    # MONDO
    for t in mondo.terms():
        if not t.name: continue
        score = SequenceMatcher(None, term_norm, normalize_term(t.name)).ratio()
        if score > 0.65:
            matches.append({
                "ontology": "MONDO",
                "id": t.id,
                "name": t.name,
                "similarity": round(score, 3)
            })

    matches.sort(key=lambda x: -x["similarity"])
    return matches[:top_n] or [{"ontology":None,"id":None,"name":"Not found","similarity":0}]


# ======================================================
# 🧠 AGENT STATE
# ======================================================
class AgentState(BaseModel):
    input_text: str
    relation_flag: Optional[bool] = None
    drugs: Optional[Any] = None
    ae_raw: Optional[str] = None
    result: Optional[Any] = None


# ======================================================
# 🧠 AGENT STEPS — YOUR PROMPTS (UNTOUCHED)
# ======================================================

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


def extract_drugs(state: AgentState):
    text = state.input_text
    prompt = f"""
Extract all drug names mentioned in the text.
Output must be a JSON array of strings only.
Do not include any explanation or commentary—just the JSON output.
No Markdown, no code blocks, no json backticks, no comments.
only return the plain json string

Text: {text}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state.drugs = response.content
    return state


def identify_adverse_events(state: AgentState):
    if not state.drugs:
        state.result = {"relation": True, "message": "Drugs not found"}
        return state

    text = state.input_text
    prompt = f"""
You are a biomedical relation identifier.

Your task is to identify **only true adverse events (AEs)** that are **causally related** to each drug mentioned in the text.

Guidelines:
- Identify an AE **only if the text explicitly states or implies a cause–effect relationship** (e.g., "caused", "led to", "resulted in", "induced", "associated with").
- If a drug and a symptom are both mentioned but not causally linked, output **None (None)** for that drug.
- Do **not** infer relationships based on co-occurrence or background disease symptoms.
- If uncertain, always output None (None). Err on the side of not linking.

OUTPUT FORMAT:
[
  {{
    "drug": "<drug_name>",
    "adverse_events": [
      {{
        "event": "<adverse_event_name>",
        "reference_sentence": "<sentence>"
      }}
    ]
  }}
]

Rules for output:
- If no drug is identified, return an empty list.
- If a drug has no AEs (None), return an empty list for "adverse_events".
- Do not include any explanation or commentary—just the JSON output.
- No Markdown, no code blocks, no json backticks, no comments.
- only return the plain json string


Now analyze the following:
Drugs: {state.drugs}
Text: {text}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state.result = response.content
    return state


def map_ontology_terms(state: AgentState):
    try:
        data = json.loads(state.result)
        for d in data:
            for ae in d.get("adverse_events", []):
                ae["ontology_mapping"] = find_in_all_ontologies(ae["event"], top_n=10)
        state.result = data
    except:
        pass
    return state


def validate_and_select_best_ontology(state: AgentState):
    try:
        data = state.result
        if isinstance(data, str):
            data = json.loads(data)

        output = []

        for d in data:
            drug_name = d.get("drug")
            validated_events = []

            for ae in d.get("adverse_events", []):
                event_name = ae.get("event")
                ontology_mappings = ae.get("ontology_mapping", [])

                reasoning_prompt = f"""
              You are a biomedical ontology expert agent.

              Tasks:
              1. Determine if the given event is a TRUE adverse event ie mentioned in the ontolgies
              2. Ontology selection (HPO, OAE, MONDO)

                  Determine whether the text describes a true adverse event (a harmful outcome occurring after and plausibly due to a medical intervention).

                  Select the single best ontology concept from OAE, MONDO, or HPO that most faithfully models the clinical meaning and usage of the event.

                  Provide alternates with clear, concise justifications.

                  Never pick concepts based on surface or embedding similarity; reason from semantics, not scores.

                  Elicit your own reasoning:
                    - First, before analyzing the text, articulate your own understanding: What do you understand each ontology (OAE, MONDO, HPO) to primarily model? What are the key differences between them in terms of what relationships and concepts they capture?

                    - Design your own reasoning structure for this task: list 3–6 short steps you choose to evaluate which ontology fits best.

                    - Then apply your structure to the instance.

                  Forbidden heuristics:

                    - Do not cite or use string/lexical match, embedding similarity, or ‘closest label’ arguments.

                    - Do not justify by “matching score”, “semantic similarity”, or keyword overlap.

                    - Do not select multiple primaries; choose one best primary and put the rest as alternates.


              3. Return the best ontology record.

              INPUTS:

                Clinical trial text : {state.input_text}
                Drug: {drug_name}
                Event: {event_name}
                Ontology Mappings:
                {json.dumps(ontology_mappings, indent=2)}

              OUTPUT FORMAT :

              {{
                "event": "{event_name}",
                "is_true_ae": <bool>,
                "best_ontology": {{
                    "ontology": "HPO" or "OAE" or "MONDO",
                    "id": "<ontology_id>",
                    "name": "<ontology_label>",
                    "similarity": <float between 0 and 1>
                }},
                "ae_classification_summary": "short biomedical justification for classification",
                "ontology_reasoning_summary": "short biomedical justification for choosing the ontology"
              }}

              - Do not include any explanation or commentary—just the JSON output.
              - No Markdown, no code blocks, no json backticks, no comments.
              - only return the plain json string
              """

                resp = llm.invoke([HumanMessage(content=reasoning_prompt)])

                try:
                    reasoning_json = json.loads(resp.content.strip())
                except:
                    reasoning_json = {
                        "event": event_name,
                        "is_true_ae": "UNKNOWN",
                        "best_ontology": None,
                        "alternate_ontologies": ontology_mappings,
                        "ae_classification_summary": "LLM returned non-JSON response",
                        "ontology_reasoning_summary": "LLM returned non-JSON response"
                    }

                if reasoning_json.get("best_ontology"):
                    reasoning_json["alternate_ontologies"] = [
                        alt for alt in ontology_mappings
                        if alt["id"] != reasoning_json["best_ontology"]["id"]
                    ]

                validated_events.append(reasoning_json)

            output.append({
                "drug": drug_name,
                "validated_adverse_events": validated_events
            })

        state.result = output
        return state

    except Exception as e:
        state.result = {"error": f"Validation failed: {e}"}
        return state


# ======================================================
# 🔗 PIPELINE CONSTRUCTION
# ======================================================
graph = StateGraph(AgentState)

graph.add_node("classify", classify_relation)
graph.add_node("extract_drugs", extract_drugs)
graph.add_node("identify_aes", identify_adverse_events)
graph.add_node("map_ontology", map_ontology_terms)
graph.add_node("validate_best", validate_and_select_best_ontology)

graph.set_entry_point("classify")
graph.add_edge("classify", "extract_drugs")
graph.add_edge("extract_drugs", "identify_aes")
graph.add_edge("identify_aes", "map_ontology")
graph.add_edge("map_ontology", "validate_best")
graph.add_edge("validate_best", END)

pipeline = graph.compile()


# ======================================================
# 🎨 STREAMLIT UI
# ======================================================
def main_app():

    st.set_page_config(page_title="Ontology Drug–AE Extractor", layout="wide")
    st.title("💊 Drug–AE Ontology Extraction Agent")

    col1, col2 = st.columns([7,1])
    with col2:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()

    st.write(f"👋 Welcome, **{st.session_state.username}**")
    st.divider()

    input_type = st.radio("Input Type", ["Paste Text", "Upload PDF"])
    user_text = ""

    if input_type == "Paste Text":
        user_text = st.text_area("Paste biomedical case report text:", height=200)

    else:
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_pdf:
            with st.spinner("Extracting text from PDF..."):
                text = ""
                with fitz.open(stream=uploaded_pdf.read(), filetype="pdf") as doc:
                    for page in doc:
                        text += page.get_text("text")
            st.success("PDF text extracted.")
            user_text = text
            st.text_area("Extracted Text Preview", text[:2000])

    if st.button("🚀 Run Extraction"):
        if not user_text.strip():
            st.warning("Please provide text before running.")
        else:
            with st.spinner("Running multi-stage ontology reasoning..."):
                result = pipeline.invoke({"input_text": user_text})["result"]

            st.success("Extraction completed!")
            st.subheader("📦 Raw JSON Output")
            st.json(result)

            rows = []
            for item in result:
                drug = item["drug"]
                for ae in item["validated_adverse_events"]:
                    best = ae.get("best_ontology", {})
                    rows.append({
                        "Drug": drug,
                        "Event": ae.get("event"),
                        "Valid AE?": ae.get("is_true_ae"),
                        "Ontology": best.get("ontology") if best else None,
                        "Ontology ID": best.get("id") if best else None,
                        "Label": best.get("name") if best else None,
                        "Reasoning": ae.get("ae_classification_summary")
                    })

            if rows:
                df = pd.DataFrame(rows)
                st.subheader("📊 Structured Table")
                st.dataframe(df, use_container_width=True)

                st.download_button(
                    "📥 Download CSV",
                    df.to_csv(index=False),
                    file_name="drug_ae_ontology_results.csv"
                )
            else:
                st.warning("No valid AE relations detected.")


# ======================================================
# 🚀 ENTRY POINT
# ======================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    main_app()
else:
    login_page()
