import os
import requests
import numpy as np
import chromadb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Connection to the Gemini API Client using Google Cloud Vertex AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp-key.json"

# FastAPI and Google Cloud Vertex AI Initialization
app = FastAPI(
    title="INFO-H512 RAG Benchmark Server",
    description="API to compare Hybrid Search (1) , Re-ranking (2) , and HyDE (3) on SQuAD data",
    version="1.0"
)

# Initialize client using Google Cloud parameters instead of API keys
ai_client = genai.Client(
    vertexai=True,
    project="info-h512-project-rag-project",
    location="europe-west1" 
)

# Connect to local ChromaDB and build the BM25 keyword matching engine
CHROMA_PATH = r"chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="squad_benchmark")

all_db_data = collection.get(include=['documents', 'embeddings'])
all_documents = all_db_data['documents']

tokenized_corpus = [doc.lower().split(" ") for doc in all_documents]
bm25 = BM25Okapi(tokenized_corpus)

# API endpoint used by evaluate_method
API_URL = "http://localhost:8000/api/ask"

# Load the Cross-Encoder re-ranking model into memory at startup
print("Loading Cross-Encoder re-ranker model into memory...")
reranker = CrossEncoder("BAAI/bge-reranker-base")

# ==========================================
# DATA STRUCTURES
# ==========================================
class QueryRequest(BaseModel):
    question: str
    method: int  # 1 = Hybrid, 2 = Re-ranking, 3 = HyDE


class QueryResponse(BaseModel):
    question: str
    method_used: str
    retrieved_context: list[str]
    answer: str


# ==========================================
# RAG PIPELINE CORE LOGIC FUNCTIONS
# ==========================================

def execute_dense_search(query_text, n_results=10):
    # Convert query to embedding vector and find similar documents in ChromaDB
    query_emb = ai_client.models.embed_content(
        model="text-embedding-004",
        contents=query_text
    ).embeddings[0].values
    results = collection.query(query_embeddings=[query_emb], n_results=n_results, include=['documents'])
    return results['documents'][0] if results['documents'] else []

def execute_hybrid_search(query_text, top_k=3):
    # Combine vector search (dense) with keyword search (BM25 sparse) using RRF scoring
    dense_results = execute_dense_search(query_text, n_results=10)
    
    # Keyword-based search: tokenize query and score documents using BM25
    tokenized_query = query_text.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:10]
    sparse_results = [all_documents[idx] for idx in top_bm25_indices]
    
    # Reciprocal Rank Fusion (RRF): combine both scores to rank documents
    rrf_scores = {}
    for rank, doc in enumerate(dense_results):
        rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (60 + (rank + 1))
    for rank, doc in enumerate(sparse_results):
        rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (60 + (rank + 1))
        
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc for doc, score in sorted_docs[:top_k]]

def execute_reranking(query_text, top_k=3):
    # Get candidate documents from hybrid search, then re-rank using cross-encoder model
    candidate_docs = execute_hybrid_search(query_text, top_k=15)
    if not candidate_docs: return []

    # Cross-encoder scores relevance between query and document pairs
    pairs = [[query_text, doc] for doc in candidate_docs]
    scores = reranker.predict(pairs)
    sorted_indices = np.argsort(scores)[::-1]
    return [candidate_docs[idx] for idx in sorted_indices[:top_k]]

def execute_hyde_search(query_text, top_k=3):
    # HyDE: Generate a hypothetical document matching the query, then search for similar docs
    hyde_prompt = f"""Write a factual paragraph that would likely appear in a document answering the following question.
    
    Question: '{query_text}'
    
    Paragraph:
    """
    hyde_response = ai_client.models.generate_content(model="gemini-2.5-flash", contents=hyde_prompt)
    hypothetical_doc = hyde_response.text
    print("Original Query:", query_text)
    print("Generated HyDE Document:", hypothetical_doc)
    # Search for documents similar to the generated hypothesis
    return execute_hybrid_search(hypothetical_doc, top_k=top_k)

# ==========================================
# FASTAPI ROUTER ENDPOINTS
# ==========================================
@app.post("/api/ask", response_model=QueryResponse)
def ask_rag(payload: QueryRequest):
    # Routing execution pattern according to evaluation configuration
    if payload.method == 1:
        method_name = "Baseline Hybrid Search"
        contexts = execute_hybrid_search(payload.question)
    elif payload.method == 2:
        method_name = "Hybrid Search + Cross-Encoder Re-ranking"
        contexts = execute_reranking(payload.question)
    elif payload.method == 3:
        method_name = "HyDE Expansion + Hybrid Search"
        contexts = execute_hyde_search(payload.question)
    else:
        raise HTTPException(status_code=400, detail="Invalid method. Choose 1, 2, or 3.")

    # Packing retrieved data fragments into target system instruction
    context_str = "\n---\n".join(contexts)
    system_prompt = f"""
You are a precise academic assistant. You answer questions based strictly on the knowledge 
I am providing you. You do not use your internal knowledge, and you do not make things up.
If the answer cannot be found in the data, simply say: I don't know
--------------------
The data:
{context_str}
"""
    # Forward request to Vertex AI Gemini text-generation engine
    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"System Constraints: {system_prompt}\n\nUser Question: {payload.question}"
        )
        answer_text = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

    # Return unified JSON object for score computations
    return QueryResponse(
        question=payload.question,
        method_used=method_name,
        retrieved_context=contexts,
        answer=answer_text
    )
