import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from rag.retriever import retrieve
import torch


class MedMindRAG:
    def __init__(self):
        print("Loading model from HuggingFace...")

        base_model_name = "facebook/opt-1.3b"
        peft_model_name = "Yakhilesh/medmind-opt-medical"

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, dtype=torch.float32
        )
        self.model = PeftModel.from_pretrained(base_model, peft_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True
        )
        print("RAG pipeline ready")

    def answer(self, question: str, age: int = None, symptoms: list = None) -> dict:
        full_question = question
        if symptoms:
            full_question += f"\nKey symptoms: {', '.join(symptoms)}"
        if age:
            full_question += f"\nPatient age: {age}"

        retrieved_docs = retrieve(full_question, top_k=3)

        context = ""
        for i, doc in enumerate(retrieved_docs):
            context += f"Medical fact {i+1}: {doc['content']}\n\n"

        system_prompt = (
            "You are MedMind, an expert clinical decision support AI "
            "trained on USMLE medical knowledge. Answer clinical questions accurately."
        )

        prompt = (
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
            f"Relevant Medical Knowledge:\n{context}\n"
            f"Clinical Question:\n{full_question}\n\n"
            f"Based on the medical knowledge above, what is the most likely diagnosis or best treatment?\n"
            f"Provide: 1) The answer 2) Brief clinical reasoning [/INST] "
            f"Based on the clinical presentation:"
        )

        output = self.pipe(prompt)[0]['generated_text']
        answer_text = output.split("Based on the clinical presentation:")[-1].strip()

        # deduplicate repeated lines
        lines = answer_text.split('\n')
        seen = set()
        clean_lines = []
        for line in lines[:8]:
            line = line.strip()
            if line and line not in seen and len(line) > 5:
                seen.add(line)
                clean_lines.append(line)

        final_answer = '\n'.join(clean_lines)
        if not final_answer or len(final_answer) < 10:
            final_answer = "Further clinical evaluation and diagnostic workup is recommended."

        return {
            "question": question,
            "answer": final_answer,
            "sources": retrieved_docs
        }


if __name__ == "__main__":
    rag = MedMindRAG()
    result = rag.answer("Patient has fever, stiff neck and photophobia. What is the diagnosis?")
    print("\nQuestion:", result['question'])
    print("\nAnswer:", result['answer'])
    print("\nSources:")
    for s in result['sources']:
        print(f"  [{s['relevance']}] {s['content'][:100]}...")