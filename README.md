# MedMind

Clinical decision support system built from scratch. Fine-tuned OPT-1.3B on 10k+ USMLE questions, wired it up with a RAG pipeline (ChromaDB + medical sentence embeddings), and wrapped it in a FastAPI + Streamlit stack.

**Model:** [huggingface.co/Yakhilesh/medmind-opt-medical](https://huggingface.co/Yakhilesh/medmind-opt-medical)

## What it does

You give it a clinical case. It retrieves relevant medical knowledge from a vector database, builds MCQ-style options from the retrieved facts, and runs them through the fine-tuned model to pick the most likely diagnosis. The whole thing runs as a local web app.

## Architecture

```
User Input (Streamlit)
    │
    ▼
FastAPI Backend
    │
    ├── Embed query (S-PubMedBert-MS-MARCO)
    ├── Search ChromaDB for similar medical facts
    ├── Build MCQ prompt from retrieved options
    │
    ▼
Fine-tuned OPT-1.3B (LoRA)
    │
    ▼
Diagnosis + Confidence Score
```

## Results

| Metric | Value |
|--------|-------|
| Test accuracy (200 unseen questions) | 31.0% |
| Random baseline (4-option MCQ) | 25.0% |
| Training samples | 10,174 |
| Training loss | 1.16 → 0.94 over 3 epochs |
| Compute | ~1.5 hrs on free Colab T4 |

## Tech stack

- **Model:** facebook/opt-1.3b + LoRA (PEFT)
- **Dataset:** MedQA USMLE 4-options
- **Embeddings:** pritamdeka/S-PubMedBert-MS-MARCO
- **Vector DB:** ChromaDB
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit

## Project structure

```
medmind/
├── api/
│   └── main.py              # REST API
├── data/
│   ├── download_data.py     # fetch MedQA dataset
│   ├── prepare_instructions.py  # convert to instruction format
│   ├── clean_final.py       # dedup + length filter
│   └── validate_data.py     # sanity checks
├── eval/
│   └── benchmark.py         # run evaluation on test set
├── frontend/
│   └── app.py               # Streamlit UI
├── rag/
│   ├── embedder.py          # embed chunks into ChromaDB
│   ├── retriever.py         # vector search
│   ├── pipeline.py          # RAG pipeline (retriever + model)
│   └── rebuild_knowledge.py # rebuild vector DB with clean data
├── app_spaces.py            # single-file version for HF Spaces
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/YadavAkhileshh/medmind.git
cd medmind
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Run the API** (terminal 1):
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Run the UI** (terminal 2):
```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501`.

## Limitations

Keeping it real — this is a 1.3B parameter model trained on a single free GPU. It's not going to match GPT-4 on clinical reasoning.

What works well:
- The full pipeline runs end-to-end (data → training → RAG → API → UI)
- MCQ format accuracy is measurably above random baseline
- Architecture is modular — swap in a bigger model and everything else stays the same

What doesn't:
- Open-ended clinical reasoning is weak at this model size
- RAG retrieval sometimes pulls surface-similar but clinically irrelevant cases
- Inference is slow on CPU (~15-30 seconds per query)

The point of this project was building the full system, not achieving SOTA accuracy. A 7B+ model would slot right in without changing a single line of the retrieval or API code.

## Disclaimer

This is an educational project. Not for actual clinical use — always see a real doctor.