import sys
import os
import json
import numpy as np
import scipy.spatial.distance as dist
from dotenv import load_dotenv

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

def compute_distances(query_vector, document_vectors, test_docs):
    scores = {}
    
    # 1. Cosine Distance
    cosine_dists = [dist.cosine(query_vector, d) for d in document_vectors]
    best_idx = int(np.argmin(cosine_dists))
    best_doc_text = test_docs[best_idx].page_content
    scores["Cosine Similarity"] = 1 - cosine_dists[best_idx]
    
    # 2. Euclidean Distance (L2)
    euclidean_dists = [dist.euclidean(query_vector, d) for d in document_vectors]
    scores["Euclidean Distance"] = min(euclidean_dists)
    
    
    # 3. Dot Product
    dot_products = [np.dot(query_vector, d) for d in document_vectors]
    scores["Dot Product"] = max(dot_products)
    
    
    
    return scores, best_doc_text

def run_metric_comparisons():
    # Initialize components
    vector_store = VectorStore(config.VECTOR_DB_PATH)
    llm_service = LLMService(vector_store)
    embedding_model = vector_store.embedding
    
    # Fictional Evaluation Dataset
    eval_dataset = [
        {"question": "Who founded Quantum Nexus Technology?", "truth": "Dr. Elara Vance."},
        {"question": "What is the flagship product of Quantum Nexus Technology?", "truth": "The Chronos Neural Core, a revolutionary quantum AI processor."},
        {"question": "How many exaflops can the Chronos Neural Core handle?", "truth": "500 exaflops."},
        {"question": "Where is the headquarters of Quantum Nexus Technology located?", "truth": "Neo-Kyoto."},
        {"question": "How much power does the Chronos Neural Core require?", "truth": "Precisely 50 Megawatts."},
        {"question": "How much did Quantum Nexus acquire CyberDyne Systems for?", "truth": "$45 billion."}
    ]
    
    # Pull raw chunks out of the vector db to test raw mathematical distances
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20}) 
    test_docs = retriever.invoke("Quantum Nexus Technology")
    document_vectors = [embedding_model.embed_query(doc.page_content) for doc in test_docs]

    print("\nSTARTING METRIC COMPARISONS \n")
    
    overall_metrics = {
        "Cosine Similarity": [],
        "Euclidean Distance": [],
        
        "Dot Product": []
       
    }
    
    for i, item in enumerate(eval_dataset):
        q = item['question']
        
        # Embed the query
        query_vector = embedding_model.embed_query(q)
        
        # Calculate raw mathematical distances against the chunks, and get the top chunk
        math_scores, best_chunk = compute_distances(query_vector, document_vectors, test_docs)
        
        for metric, val in math_scores.items():
            overall_metrics[metric].append(val)
            
        print(f"\n--- [Q{i+1}]: {q} ---")
        print("\n  [STEP 1]: Vector Database Search Metrics (Distance from Query -> Document Chunk)")
        for metric, val in math_scores.items():
            print(f"    > {metric}: {val:.4f}")
            
        print("\n  [STEP 2]: Top Retrieved Document Chunk (Highest Cosine Similarity)")
        print(f"    \"{best_chunk.strip()}\"")
        
        # Step 3: LLM Generation
        prompt = f"Answer the question concisely and accurately using ONLY the provided Context.\n\nContext: {best_chunk}\n\nQuestion: {q}\n\nAnswer:"
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
        llm_answer = llm.invoke(prompt).content.strip()
        
        print("\n  [STEP 3]: Final LLM Generation (RAG Output)")
        print(f"    {llm_answer}\n")

    print("\n AVERAGE METRIC SCORES ACROSS DATASET ")
    summary = {}
    for metric, values in overall_metrics.items():
        avg = sum(values) / len(values)
        summary[metric] = avg
        print(f"Average {metric}: {avg:.4f}")
    
    
    # Save to file
    os.makedirs('evaluation_results', exist_ok=True)
    with open('evaluation_results/mathematical_similarity_metrics.json', 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    run_metric_comparisons()
