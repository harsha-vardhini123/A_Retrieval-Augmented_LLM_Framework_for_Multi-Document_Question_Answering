import sys
import os
import json
from dotenv import load_dotenv

# Add app to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.models.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.config import config
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load env variables
load_dotenv()

def run_evaluation():
    # Initialize components
    vector_store = VectorStore(config.VECTOR_DB_PATH)
    llm_service = LLMService(vector_store)

    # 1. Ingest Sample Document
    print("Ingesting sample fictional knowledge dataset to Vector DB...")
    loader = TextLoader("sample_knowledge.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)
    
    try:
        vector_store.add_documents(text_chunks)
        print("Ingestion complete. Document added to vector store.")
    except Exception as e:
        print(f"Skipping ingestion error (might already exist): {e}")

    # 2. Fictional Evaluation Dataset
    # We use a purely fictional subject so the baseline LLM (which is trained on real internet data) 
    # cannot possibly know the answer without RAG context.
    eval_dataset = [
        {"question": "Who founded Quantum Nexus Technology?", "truth": "Dr. Elara Vance."},
        {"question": "What is the flagship product of Quantum Nexus Technology?", "truth": "The Chronos Neural Core, a revolutionary quantum AI processor."},
        {"question": "How many exaflops can the Chronos Neural Core handle?", "truth": "500 exaflops."},
        {"question": "Where is the headquarters of Quantum Nexus Technology located?", "truth": "Neo-Kyoto."},
        {"question": "How much power does the Chronos Neural Core require?", "truth": "Precisely 50 Megawatts."},
        {"question": "How much did Quantum Nexus acquire CyberDyne Systems for?", "truth": "$45 billion."}
    ]

    # Baseline LLM (No context/knowledge)
    baseline_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

    print("\n--- Starting LLM vs RAG Accuracy Evaluation ---")
    
    results = []
    
    for i, item in enumerate(eval_dataset):
        q = item['question']
        truth = item['truth']
        
        print(f"\n[Question {i+1}]: {q}")
        
        # 1. Baseline Standard LLM Answer (without RAG context)
        baseline_response = baseline_llm.invoke(f"Answer concisely. If you don't know or if it references fictional info, just attempt to guess or state you don't know: {q}")
        baseline_answer = baseline_response.content.strip()
        
        # 2. Built RAG System Answer (with Vector DB Context)
        rag_answer = llm_service.get_response(q, "eval_session_123")
        
        # 3. Use LLM-as-a-judge to score accuracy numerically
        judge_prompt = PromptTemplate(
            template='''Evaluate the answer against the Ground Truth.
Ground Truth: {truth}
Answer: {answer}

If the Answer contains the correct facts from the Ground Truth, output exactly 1.0
If the Answer says "I don't know" or is completely wrong, output exactly 0.0

Respond ONLY with the number 1.0 or 0.0, with NO other text, NO words, and NO punctuation.''',
            input_variables=["truth", "answer"]
        )
        
        import re
        
        # Score Baseline
        baseline_eval = baseline_llm.invoke(judge_prompt.format(truth=truth, answer=baseline_answer))
        try:
            match = re.search(r'(0\.[0-9]+|1\.0)', baseline_eval.content)
            baseline_score = float(match.group(1)) if match else 0.0
        except:
            baseline_score = 0.0
            
        # Score RAG
        rag_eval = baseline_llm.invoke(judge_prompt.format(truth=truth, answer=rag_answer))
        try:
            match = re.search(r'(0\.[0-9]+|1\.0)', rag_eval.content)
            rag_score = float(match.group(1)) if match else 0.0
        except:
            rag_score = 0.0
            
        print(f" > Ground Truth: {truth}")
        print(f" > LLM Answer (No RAG): {baseline_answer}")
        print(f"   LLM Score: {baseline_score}")
        print(f" > System Answer (With RAG): {rag_answer}")
        print(f"   RAG Score: {rag_score}")
        
        results.append({
            "question": q,
            "baseline_answer": baseline_answer,
            "rag_answer": rag_answer,
            "baseline_score": baseline_score,
            "rag_score": rag_score
        })

    # Calculate final averages
    avg_baseline = sum(r['baseline_score'] for r in results) / len(results)
    avg_rag = sum(r['rag_score'] for r in results) / len(results)

    print("\n================ FINAL ACCURACY RESULTS ================")
    print(f"Average Baseline LLM Accuracy: {avg_baseline * 100:.1f}%")
    print(f"Average LBRCE RAG System Accuracy:   {avg_rag * 100:.1f}%")
    print("========================================================\n")

    # Ensure evaluation directory exists
    os.makedirs('evaluation_results', exist_ok=True)
    
    with open('evaluation_results/rag_llm_comparison_fictional.json', 'w') as f:
        json.dump({"results": results, "avg_baseline_accuracy": avg_baseline, "avg_rag_accuracy": avg_rag}, f, indent=4)

if __name__ == "__main__":
    run_evaluation()
