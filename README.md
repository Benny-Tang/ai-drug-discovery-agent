# AI-Powered Drug Discovery Agent

A LangChain agent (GLM-4.5) with tool access to four free public
chemistry/biology databases:

| Tool | Source | Purpose |
|---|---|---|
| PubChemSearch | PubChem | Compound lookup by name (CID, formula) |
| ChEMBLSearch | ChEMBL | Molecule data by ChEMBL ID |
| ClinicalTrialsSearch | ClinicalTrials.gov | Trial search by keyword |
| PDBSearch | RCSB Protein Data Bank | Protein structure lookup |

Ask things like "Find details about Aspirin" or "List clinical trials for
Alzheimer's" and the agent picks the right tool(s) automatically.

## Hardware note

GLM-4.5 is a large open-weight model loaded locally via
`transformers.AutoModelForCausalLM` — this needs a GPU with substantial
VRAM. For lighter deployment, either swap in a smaller open model or
point `HuggingFacePipeline` at a hosted inference endpoint instead of
loading weights locally.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

## License

MIT
