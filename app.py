"""
AI-Powered Drug Discovery Agent

A LangChain agent backed by an open-weight LLM (GLM-4.5) with tool access
to free public chemistry/biology APIs: PubChem (compounds), ChEMBL
(molecules), ClinicalTrials.gov (trials), and RCSB PDB (protein structures).

Note: GLM-4.5 is a large model — running it locally via
transformers.AutoModelForCausalLM requires a GPU with substantial VRAM.
For lighter-weight deployment, swap `model_name` for a smaller open model
or point HuggingFacePipeline at a hosted inference endpoint instead.
"""
import gradio as gr
import requests
from langchain import HuggingFacePipeline
from langchain.agents import Tool, initialize_agent
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ================== LLM ==================
MODEL_NAME = "THUDM/glm-4.5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
llm = HuggingFacePipeline(pipeline=pipe)


# ================== TOOLS ==================
def search_pubchem(compound_name: str) -> str:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound_name}/JSON"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        cid = data["PC_Compounds"][0]["id"]["id"]["cid"]
        formula = data["PC_Compounds"][0]["props"][0]["value"]["sval"]
        return f"🔬 PubChem\nCompound: {compound_name}\nCID: {cid}\nFormula: {formula}"
    except requests.RequestException:
        return "❌ Error from PubChem"
    except (KeyError, IndexError):
        return "⚠️ No compound info found."


def search_chembl(compound_id: str) -> str:
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{compound_id}"
    try:
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        r.raise_for_status()
        return f"🧬 ChEMBL Data for {compound_id}\n{r.json()}"
    except requests.RequestException:
        return "❌ Error from ChEMBL"


def search_trials(keyword: str) -> str:
    url = f"https://clinicaltrials.gov/api/query/full_studies?expr={keyword}&min_rnk=1&max_rnk=3&fmt=json"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        studies = [
            s["Study"]["ProtocolSection"]["IdentificationModule"]["BriefTitle"]
            for s in data.get("FullStudiesResponse", {}).get("FullStudies", [])
        ]
        if not studies:
            return "🏥 Clinical Trials:\nNo studies found."
        return "🏥 Clinical Trials:\n" + "\n".join(studies)
    except requests.RequestException:
        return "❌ Error from ClinicalTrials.gov"
    except (KeyError, TypeError):
        return "⚠️ Unexpected response format from ClinicalTrials.gov"


def search_pdb(protein: str) -> str:
    url = f"https://data.rcsb.org/rest/v1/core/entry/{protein}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return f"🧩 PDB Data for {protein}\n{r.json()}"
    except requests.RequestException:
        return "❌ Error from PDB"


# ================== AGENT ==================
tools = [
    Tool(name="PubChemSearch", func=search_pubchem, description="Search compounds in PubChem by name."),
    Tool(name="ChEMBLSearch", func=search_chembl, description="Search molecules in ChEMBL by ChEMBL ID."),
    Tool(name="ClinicalTrialsSearch", func=search_trials, description="Search clinical trials by keyword."),
    Tool(name="PDBSearch", func=search_pdb, description="Search proteins/structures in Protein Data Bank."),
]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


# ================== GRADIO APP ==================
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 💊 AI-Powered Drug Discovery Agent")
    gr.Markdown(
        "Ask about compounds, targets, clinical trials, or proteins — "
        "powered by free APIs (PubChem, ChEMBL, ClinicalTrials.gov, PDB)."
    )
    chat = gr.ChatInterface(
        fn=lambda message, history: agent.run(message),
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Ask about Aspirin, CHEMBL IDs, clinical trials...", container=False),
        title="Drug Discovery Agent",
        description="Agent powered by GLM-4.5 + free chemistry APIs",
        examples=[
            "Find details about Aspirin",
            "Get ChEMBL info for CHEMBL25",
            "List clinical trials for Alzheimer's",
            "Find protein structure PDB entry 6LU7",
        ],
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch()
