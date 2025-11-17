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
    state.relation_flag = response.content.strip().upper() == "YES"
    return state

# =========================================
# 🔹 Step 2 — Drug Extraction
# =========================================
def extract_drugs(state: AgentState):
    text = state.input_text
    prompt = f"""
Extract all drug names mentioned in the text.
Output must be STRICT JSON array, ONLY this format:

["drug1", "drug2"]

Text: {text}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        state.drugs = json.loads(response.content.strip())
    except:
        # fallback recovery
        state.drugs = [d.strip() for d in response.content.replace("[","").replace("]","").split(",")]
    
    return state



# =========================================
# 🔹 Step 3 — Adverse Event Identification
# =========================================
def identify_adverse_events(state: AgentState):
    text = state.input_text

    causality_prompt = f"""
You are an adverse event extraction agent.

Your task is to identify adverse events that are **causally linked** to the drug(s) listed below.

A relationship counts as causal if ANY of the following appear:
- Direct causal words: "caused", "resulted in", "led to", "induced", "triggered", "produced"
- Clinical reasoning phrases: "linked to", "attributed to", "associated with", "likely due to", "suspected to be caused by"
- Pharmacovigilance evidence: "temporal association", "rechallenge reproduced", "dechallenge improved"
- Case report certainty language: "confirmed", "determined", "consistent with", "known side effect"

If **no causal relationship exists** for a drug, return an empty list for that drug.

DO NOT extract:
- Disease symptoms not attributed to the drug
- Observational statements without causality
- Expected symptoms of pre-existing illness

---

### Output Format (STRICT):

Return ONLY valid JSON in the format:

[
  {{
    "drug": "DrugName",
    "adverse_events": [
      {{
        "event": "EventName",
        "reference_sentence": "Exact sentence from the text supporting causality"
      }}
    ]
  }}
]

Rules:
- `adverse_events` must be an empty list if no causal AE exists.
- Do not invent or infer missing events.
- Use exact wording from the text (do not paraphrase events).
- No additional text, markdown, comments, or explanation.

---

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

            # --- Case: Drug with NO AE -> skip entirely ---
            if not d.get("adverse_events"):
                continue

            for ae in d.get("adverse_events", []):
                ontology_mappings = ae.get("ontology_mapping", [])
                event_name = ae["event"]
                ref_sentence = ae.get("reference_sentence")

                # ====================================================
                # CASE 1: AE has NO usable ontology → mark PARTIAL
                # ====================================================
                no_mapping = (
                    not ontology_mappings or
                    ontology_mappings[0].get("ontology") is None or
                    ontology_mappings[0].get("similarity", 0) == 0
                )

                if no_mapping:
                    validated_events.append({
                        "event": event_name,
                        "reference_sentence": ref_sentence,
                        "is_true_ae": "PARTIAL",
                        "best_ontology": None,
                        "alternate_ontologies": [],
                        "reasoning_summary": (
                            "Causally linked adverse event detected, "
                            "but no ontology match was found."
                        )
                    })
                    continue  # <-- prevent LLM processing

                # ====================================================
                # CASE 2: Ontology exists → send to LLM
                # ====================================================
                reasoning_prompt = f"""
You are a biomedical ontology expert agent.

Validate whether the event is a true adverse event and select the best ontology.

Return STRICT JSON ONLY:

{{
  "event": "{event_name}",
  "is_true_ae": "YES" or "NO",
  "reference_sentence": "{ref_sentence}",
  "best_ontology": {{
      "ontology": "HPO" or "OAE" or "MONDO",
      "id": "<ontology_id>",
      "name": "<ontology_label>",
      "similarity": <float>
  }},
  "alternate_ontologies": {json.dumps(ontology_mappings)},
  "reasoning_summary": "Short justification."
}}
"""

                response = llm.invoke([HumanMessage(content=reasoning_prompt)])
                content = response.content.strip()

                try:
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    validated = json.loads(content[json_start:json_end])
                except:
                    # fallback
                    validated = {
                        "event": event_name,
                        "is_true_ae": "UNKNOWN",
                        "reference_sentence": ref_sentence,
                        "best_ontology": ontology_mappings[0],
                        "alternate_ontologies": ontology_mappings,
                        "reasoning_summary": "LLM returned invalid JSON format."
                    }

                validated_events.append(validated)

            if validated_events:
                validated_output.append({
                    "drug": drug_name,
                    "validated_adverse_events": validated_events
                })

        state.result = validated_output
        return state

    except Exception as e:
        state.result = {"error": f"Validation agent failed: {e}"}
        return state



# =========================================
# 🔹 Build Unified Pipeline
# =========================================




