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
# 🔐 AUTH CONFIG
# ======================================================
# Expect in .streamlit/secrets.toml:
# [users]
# user1 = "password1"
# user2 = "password2"
USERS = st.secrets["users"]


def check_login(username, password):
    return username in USERS and USERS[username] == password


def login_page():
    st.set_page_config(page_title="Login - Drug–AE Ontology Extractor", layout="centered")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style='text-align:center;'>
                <h2>🔐 Drug–AE Ontology Extractor</h2>
                <p>Secure access for authorized users only</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if check_login(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")


# ======================================================
# 🤖 GLOBAL LLM (as you requested)
# ======================================================
# Expect in secrets:
# api = "YOUR_GROQ_KEY"
# model = "meta-llama/llama-4-scout-17b-16e-instruct"  (or whatever you want)
llm = ChatGroq(
    model=st.secrets["model"],
    api_key=st.secrets["api"],
    temperature=0,
)


# ======================================================
# 🧬 ONTOLOGY LOADING (HPO, OAE, MONDO)
# ======================================================
@st.cache_resource(show_spinner=True)
def load_ontologies():
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

    print(f"HPO terms: {len(list(hpo.terms()))}")
    print(f"OAE triples: {len(oae)}")
    print(f"MONDO terms: {len(list(mondo.terms()))}")
    print("Ontologies loaded successfully.\n")

    return hpo, oae, mondo


hpo, oae, mondo = load_ontologies()


# ======================================================
# 🔧 Utility Functions
# ======================================================
def normalize_term(term: str):
    return term.strip().lower().replace("-", " ").replace("_", " ")


def find_in_all_ontologies(term, top_n=10):
    """Find top N close matches across HPO, OAE, and MONDO ontologies."""
    term_norm = normalize_term(term)
    matches = []

    # --- HPO ---
    for t in hpo.terms():
        if not t.name:
            continue
        name_norm = normalize_term(t.name)
        ratio = SequenceMatcher(None, term_norm, name_norm).ratio()
        if ratio > 0.65:
            matches.append({
                "ontology": "HPO",
                "id": t.id,
                "name": t.name,
                "similarity": round(ratio, 3)
            })

    # --- OAE ---
    for s, p, o in oae.triples((None, URIRef("http://www.w3.org/2000/01/rdf-schema#label"), None)):
        name = str(o)
        name_norm = normalize_term(name)
        ratio = SequenceMatcher(None, term_norm, name_norm).ratio()
        if ratio > 0.65:
            matches.append({
                "ontology": "OAE",
                "id": str(s),
                "name": name,
                "similarity": round(ratio, 3)
            })

    # --- MONDO ---
    for t in mondo.terms():
        if not t.name:
            continue
        name_norm = normalize_term(t.name)
        ratio = SequenceMatcher(None, term_norm, name_norm).ratio()
        if ratio > 0.65:
            matches.append({
                "ontology": "MONDO",
                "id": t.id,
                "name": t.name,
                "similarity": round(ratio, 3)
            })

    matches.sort(key=lambda x: -x["similarity"])
    return matches[:top_n] if matches else [
        {"ontology": None, "id": None, "name": "Not found", "similarity": 0}
    ]


# ======================================================
# 🧠 Agent State
# ======================================================
class AgentState(BaseModel):
    input_text: str
    relation_flag: Optional[bool] = None
    drugs: Optional[Any] = None
    ae_raw: Optional[str] = None
    result: Optional[Any] = None


# ======================================================
# 🧠 Step 1 — Relation Classification (PROMPT UNCHANGED)
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


# ======================================================
# 🧠 Step 2 — Drug Extraction (PROMPT UNCHANGED)
# ======================================================
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


# ======================================================
# 🧠 Step 3 — Adverse Event Identification (PROMPT UNCHANGED)
# ======================================================
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

Output format:
Drug: <drug_name> -> <adverse_event_1> (sentence_1), <adverse_event_2> (sentence_2), ...
If no AE is linked to a drug, output:
Drug: <drug_name> -> None (None)

Now analyze the following:
Drugs: {state.drugs}
Text: {text}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state.ae_raw = response.content
    return state


# ======================================================
# 🧠 Step 4 — Structure JSON (PROMPT UNCHANGED)
# ======================================================
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

    # 🔥 NEW: normalize JSON immediately here
    try:
        cleaned = response.content.replace("```json", "").replace("```", "").strip()
        state.result = json.loads(cleaned)
    except Exception:
        # fallback: store raw so later logic handles it
        state.result = response.content  

    return state



# ======================================================
# 🧠 Step 5 — Ontology Mapping (UNCHANGED LOGIC)
# ======================================================
def map_ontology_terms(state: AgentState):
    try:
        data_str = state.result
        if isinstance(data_str, str):
            data_str = data_str.replace("```json", "").replace("```", "").strip()
            data = json.loads(data_str)
        else:
            data = data_str

        for d in data:
            for ae in d.get("adverse_events", []):
                term = ae.get("event")
                ae["ontology_mapping"] = find_in_all_ontologies(term, top_n=10)

        state.result = data
        return state

    except Exception as e:
        state.result = {"error": f"Ontology mapping failed: {e}"}
        return state


# ======================================================
# 🧠 Step 6 — LLM Validation + Ontology Selection (PROMPT UNCHANGED)
# ======================================================
def validate_and_select_best_ontology(state: AgentState):
    """
    LLM-based validation + ontology prioritization:
    1. Determines if each AE is a true biomedical AE.
    2. Selects best ontology entry contextually.
    3. Returns other ontologies as alternates.
    """
    try:
        data = state.result

        # --- Robust JSON normalization ---
        if isinstance(data, str):
            try:
                data = json.loads(data.replace("```json", "").replace("```", "").strip())
            except Exception as err:
                raise ValueError(f"Invalid JSON input: {err}")

        if not isinstance(data, list):
            raise TypeError("Expected list of dicts after ontology mapping.")

        output = []

        for d in data:
            if isinstance(d, str):
                d = json.loads(d)
            drug_name = d.get("drug")
            validated_events = []

            for ae in d.get("adverse_events", []):
                event_name = ae.get("event")
                ontology_mappings = ae.get("ontology_mapping", [])

                # --- Prepare prompt ---
                reasoning_prompt = f"""
You are a biomedical ontology expert agent.

Task:
1. Determine if the given event is a TRUE adverse event ie mentioned in the ontolgies
2. Identify which ontology (HPO, OAE, or MONDO) provides the most contextually relevant definition.
3. Return the best ontology record, and include all others as alternates.

Drug: {drug_name}
Event: {event_name}
Ontology Mappings:
{json.dumps(ontology_mappings, indent=2)}

Return STRICT JSON (no markdown):

{{
  "event": "{event_name}",
  "is_true_ae": "YES" or "NO",
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
                # --- Run LLM ---
                response = llm.invoke([HumanMessage(content=reasoning_prompt)])
                content = response.content.strip()

                # --- Extract JSON safely ---
                if "{" not in content:
                    reasoning_json = {
                        "event": event_name,
                        "is_true_ae": "UNKNOWN",
                        "best_ontology": None,
                        "alternate_ontologies": ontology_mappings,
                        "reasoning_summary": "LLM did not return JSON."
                    }
                else:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    json_block = content[start:end]
                    try:
                        reasoning_json = json.loads(json_block)
                    except Exception:
                        reasoning_json = {
                            "event": event_name,
                            "is_true_ae": "UNKNOWN",
                            "best_ontology": None,
                            "alternate_ontologies": ontology_mappings,
                            "reasoning_summary": "Malformed JSON from model."
                        }

                # --- Fallback: ensure alternate ontologies are preserved ---
                if "alternate_ontologies" not in reasoning_json or not reasoning_json["alternate_ontologies"]:
                    reasoning_json["alternate_ontologies"] = ontology_mappings

                validated_events.append(reasoning_json)

            output.append({
                "drug": drug_name,
                "validated_adverse_events": validated_events
            })

        state.result = output
        return state

    except Exception as e:
        state.result = {"error": f"Validation agent failed: {e}"}
        return state


# ======================================================
# 🔗 LangGraph Pipeline Assembly
# ======================================================
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
graph.add_edge("map_ontology", "validate_best")
graph.add_edge("validate_best", END)

pipeline = graph.compile()


# ======================================================
# 🎨 Streamlit Main App (with login)
# ======================================================
def main_app():
    st.set_page_config(page_title="Drug–AE Ontology Extractor", layout="wide")

    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("💊 Drug–AE Ontology Extraction Agent")
        st.caption(f"Welcome, **{st.session_state.username}**")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()

    st.divider()

    input_mode = st.radio("Input Type", ["Paste Text", "Upload PDF"])
    user_text = ""

    if input_mode == "Paste Text":
        user_text = st.text_area("Paste biomedical text here:", height=200)
    else:
        uploaded_pdf = st.file_uploader("Upload a biomedical PDF", type=["pdf"])
        if uploaded_pdf:
            with st.spinner("Extracting text from PDF..."):
                text = ""
                with fitz.open(stream=uploaded_pdf.read(), filetype="pdf") as doc:
                    for page in doc:
                        text += page.get_text("text") + "\n"
            st.success("Text extracted from PDF.")
            st.text_area("Extracted Text Preview:", text[:2000], height=200)
            user_text = text

    if st.button("🚀 Run Pipeline"):
        if not user_text.strip():
            st.warning("Please provide some input text or upload a PDF.")
        else:
            with st.spinner("Running ontology-aware AE extraction pipeline..."):
                out = pipeline.invoke({"input_text": user_text})
                result = out["result"]

            st.success("✅ Pipeline run completed!")

            st.subheader("📦 Raw JSON Output")
            st.json(result)

            # Try to flatten into a table if result is list-like
            if isinstance(result, list):
                rows = []
                for item in result:
                    drug = item.get("drug")
                    for ae in item.get("validated_adverse_events", []):
                        best = ae.get("best_ontology") or {}
                        rows.append({
                            "Drug": drug,
                            "Event": ae.get("event"),
                            "Is True AE": ae.get("is_true_ae"),
                            "Best Ontology": best.get("ontology"),
                            "Ontology ID": best.get("id"),
                            "Ontology Label": best.get("name"),
                            "Similarity": best.get("similarity"),
                            "Reasoning": ae.get("reasoning_summary"),
                        })

                if rows:
                    df = pd.DataFrame(rows)
                    st.subheader("📊 Structured Table View")
                    st.dataframe(df, use_container_width=True)

                    st.download_button(
                        "⬇️ Download as CSV",
                        df.to_csv(index=False),
                        file_name="drug_ae_ontology_results.csv",
                    )
                else:
                    st.info("No validated adverse events found in the output.")
            else:
                st.info("Result is not a list (may be a message or error).")


# ======================================================
# 🚀 Entry Point
# ======================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None

if st.session_state.authenticated:
    main_app()
else:
    login_page()
