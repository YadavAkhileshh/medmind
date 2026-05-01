import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from rag.retriever import retrieve

app = FastAPI(
    title="MedMind API",
    description="Clinical decision support — fine-tuned OPT-1.3B + RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class ClinicalQuery(BaseModel):
    question: str
    patient_age: Optional[int] = None
    symptoms: Optional[list] = []


class MedMindResponse(BaseModel):
    answer: str
    sources: list
    confidence: str
    model_used: str


# load model at startup
print("Loading model...")

base_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-1.3b",
    dtype=torch.float32
)
model = PeftModel.from_pretrained(base_model, "Yakhilesh/medmind-opt-medical")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.3,
    do_sample=True
)

print("Model loaded, API ready")


@app.get("/")
def root():
    return {
        "name": "MedMind API",
        "version": "1.0.0",
        "status": "running",
        "model": "Yakhilesh/medmind-opt-medical",
        "endpoints": ["/diagnose", "/health", "/docs"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "gpu_available": torch.cuda.is_available()
    }


@app.post("/diagnose", response_model=MedMindResponse)
def diagnose(query: ClinicalQuery):
    if not query.question or len(query.question) < 10:
        raise HTTPException(status_code=400, detail="Question too short")

    full_question = query.question
    if query.symptoms:
        full_question += f"\nSymptoms: {', '.join(query.symptoms)}"
    if query.patient_age:
        full_question += f"\nPatient age: {query.patient_age}"

    retrieved_docs = retrieve(full_question, top_k=4)

    # pull diagnoses from RAG results to build MCQ options
    unique_diags = []
    for doc in retrieved_docs:
        content = doc['content']
        if "Correct answer:" in content:
            ans = content.split("Correct answer:")[-1].strip()
        else:
            ans = content.split("Correct diagnosis/treatment:")[-1].strip()
        if ans and ans not in unique_diags:
            unique_diags.append(ans)

    # pad if we don't have 4 options
    fallbacks = ["Further clinical evaluation required", "Viral infection",
                 "Bacterial infection", "Undetermined"]
    for fb in fallbacks:
        if len(unique_diags) >= 4:
            break
        if fb not in unique_diags:
            unique_diags.append(fb)

    letters = ['A', 'B', 'C', 'D']
    options_text = ""
    for i, diag in enumerate(unique_diags[:4]):
        options_text += f"{letters[i]}: {diag}\n"

    # prompt uses the exact format from training
    system_prompt = (
        "You are MedMind, an expert clinical decision support AI trained on medical knowledge.\n"
        "You help medical professionals with diagnosis, treatment planning, and drug selection.\n"
        "Always reason step by step before giving your final answer.\n"
        "Be accurate, cite your reasoning, and flag any uncertainty."
    )

    prompt = (
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"Clinical Question:\n{full_question}\n\n"
        f"Options:\n{options_text}\n"
        f"What is the best answer and why? [/INST] "
        f"Let me analyze this clinical scenario step by step.\n\n"
        f"Looking at the patient presentation and the available options, the correct answer is:"
    )

    output = pipe(prompt)[0]['generated_text']

    # parse out the selected diagnosis
    generated_tail = output.split("the correct answer is:")[-1].strip()
    diagnosis_raw = generated_tail.split("Clinical Reasoning:")[0].strip()
    diagnosis = re.sub(r"^[A-D]:\s*", "", diagnosis_raw).strip()

    if not diagnosis or len(diagnosis) < 3:
        diagnosis = "Further clinical evaluation required"

    symptoms_text = ', '.join(query.symptoms) if query.symptoms else "the described symptoms"

    # build response HTML — gets rendered directly in the Streamlit frontend
    html_answer = (
        f'<div style="font-size:1.15rem;font-weight:600;color:#fff;margin-bottom:1rem;">'
        f'Most likely: <span style="color:#4ade80;">{diagnosis}</span></div>'
        f'<div style="font-size:0.9rem;line-height:1.7;color:#ccc;padding:1rem 1.2rem;'
        f'background:#161616;border-left:3px solid #3b82f6;border-radius:0 6px 6px 0;margin-bottom:1rem;">'
        f'Patient presenting with <b>{symptoms_text}</b> — the retrieved clinical knowledge '
        f'points toward <b>{diagnosis}</b>. Confirm with physical exam, labs, and imaging as needed.</div>'
    )

    avg_rel = sum(d['relevance'] for d in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
    confidence = "high" if avg_rel > 0.8 else "medium" if avg_rel > 0.6 else "low"

    return MedMindResponse(
        answer=html_answer,
        sources=retrieved_docs,
        confidence=confidence,
        model_used="Yakhilesh/medmind-opt-medical (OPT-1.3B + LoRA)"
    )