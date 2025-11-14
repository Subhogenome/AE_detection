import streamlit as st
import pandas as pd
import fitz  # PyMuPDF

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import Optional, Any
from pronto import Ontology
from rdflib import Graph, URIRef
from difflib import SequenceMatcher
import json
import requests


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

print(f"HPO terms: {len(list(hpo.terms()))}")
print(f"OAE triples: {len(oae)}")
print(f"MONDO terms: {len(list(mondo.terms()))}")
print("Ontologies loaded successfully.\n")


# =========================================
# 🔹 Utility Functions
# =========================================
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
# 👇 Replace with your real key or use st.secrets if you want
api = st.secrets["api"]
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, api_key=api)


# =========================================
# 🔹 Step 1 — Relation Classification
# =========================================
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

        for d in data:
            for ae in d.get("adverse_events", []):
                term = ae.get("event")
                ae["ontology_mapping"] = find_in_all_ontologies(term, top_n=10)

        state.result = data
        return state

    except Exception as e:
        state.result = {"error": f"Ontology mapping failed: {e}"}
        return state


# =========================================
# 🔹 Step 6 — LLM Validation + Ontology Selection
# =========================================
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

        # 🔧 Minimal fix: if a dict comes instead of list, wrap it
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
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

                # --- Prepare prompt (UNCHANGED) ---
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

                # ✅ NEW: preserve the reference sentence from Step 4
                reasoning_json["reference_sentence"] = ae.get("reference_sentence")

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


# =========================================
# 🔹 Build Unified Pipeline
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
graph.add_edge("map_ontology", "validate_best")
graph.add_edge("validate_best", END)

pipeline = graph.compile()


# =========================================
# 🔹 (Optional) CLI Example – still works
# =========================================
if __name__ == "__main__":
    user_text = """cough was observed following administration of Losartan."""

    output = pipeline.invoke({"input_text": user_text})
    print("\n=== FINAL OUTPUT (CLI) ===")
    print(output["result"])


# =========================================
# 🎨 Streamlit UI (single file)
# =========================================
st.title("💊 Drug–AE Ontology Extraction & Validation")

mode = st.radio("Input Type", ["📝 Paste Text", "📄 Upload PDF"])
input_text = ""

if mode == "📝 Paste Text":
    input_text = st.text_area("Paste biomedical text:", height=200)
else:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            input_text = "\n".join(page.get_text("text") for page in doc)
        st.text_area("Extracted text (preview):", input_text[:2000])

if st.button("🚀 Run Pipeline"):
    if not input_text.strip():
        st.warning("Please provide text or upload a PDF.")
    else:
        with st.spinner("Running pipeline..."):
            out = pipeline.invoke({"input_text": input_text})
            result = out["result"]

        st.subheader("📦 Raw Final Output")
        st.json(result)

        # Handle error case
        if isinstance(result, dict) and "error" in result:
            st.error(result["error"])
        else:
            # Build table: Drug | Event | Is True AE | Reference Sentence | Best Ontology | Ontology ID
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
                            "Is True AE": ae.get("is_true_ae"),
                            "Reference Sentence": ae.get("reference_sentence"),
                            "Best Ontology": best.get("ontology"),
                            "Ontology ID": best.get("id"),
                        })

            if rows:
                df = pd.DataFrame(rows)
                st.subheader("📊 Structured AE Table")
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "⬇ Download CSV",
                    df.to_csv(index=False),
                    "drug_ae_ontology_output.csv"
                )
            else:
                st.info("No validated adverse events to display.")
