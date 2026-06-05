# RAG Benchmark Server - INFO-H512

A comprehensive Retrieval Augmented Generation (RAG) system that compares three different retrieval strategies on the SQuAD dataset using Google Cloud Vertex AI and Gemini.

## Project Structure

```
main.py                              # FastAPI server with RAG logic
RAG_Notebook.ipynb                   # Evaluation notebook with RAGAS metrics
requirements.txt                     # Python dependencies
gcp-key.json                         # Google Cloud credentials (keep secret!)
chroma_db/                           # Vector database storage
squad_eval_pairs.csv                 # 500 test Q&A pairs from SQuAD
scores_ragas_methode_*.csv          # Evaluation results for each method
README.md                            # This file
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- Google Cloud account with Vertex AI access
- SQuAD dataset access (auto-downloaded from Hugging Face)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Google Cloud Credentials

Place your Google Cloud service account key in the project directory:
```bash
# Save as gcp-key.json
export GOOGLE_APPLICATION_CREDENTIALS="gcp-key.json"
```

### 3. Initialize Vector Database (Optional)

If you need to re-index the SQuAD data:

```python
# In RAG_Notebook.ipynb, remove the raise ValueError() block in cell 2
# This downloads and indexes 500 unique SQuAD passages into ChromaDB
```

## Running the Server

Start the FastAPI server:

```bash
python main.py
# Or with Uvicorn:
uvicorn main:app --reload 
```

The server will:
- Load ChromaDB with 500 indexed SQuAD passages
- Initialize the Gemini API client
- Load the Cross-Encoder re-ranking model
- Start the FastAPI server at `http://localhost:8000`

Access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### POST `/api/ask`

Query the RAG system and get answers with retrieved context.

**Request:**
```json
{
  "question": "What is the capital of France?",
  "method": 1
}
```

**Parameters:**
- `question` (string): The question to answer
- `method` (integer): Retrieval method (1, 2, or 3)

**Response:**
```json
{
  "question": "What is the capital of France?",
  "method_used": "Baseline Hybrid Search",
  "retrieved_context": ["...", "...", "..."],
  "answer": "Paris is the capital of France."
}
```

## Evaluation & Metrics

Run the evaluation notebook to assess performance:

```python
# RAG_Notebook.ipynb
# Runs all 3 methods against 500 test questions
# Computes RAGAS metrics: Faithfulness, Answer Relevancy, Context Precision
```

### Output Files
- `scores_ragas_methode_1.csv` - Method 1 results
- `scores_ragas_methode_2.csv` - Method 2 results  
- `scores_ragas_methode_3.csv` - Method 3 results
- `rag_comparison_manual.csv` - Manual comparison results
- `ragas_evaluation_results.csv` - Aggregated RAGAS scores

## Dependencies

Key libraries:
- `fastapi` - Web API framework
- `chromadb` - Vector database
- `google-genai` - Gemini API client
- `sentence-transformers` - Cross-encoder model
- `rank-bm25` - BM25 keyword search
- `ragas` - RAG evaluation metrics
- `pandas` - Data manipulation

See `requirements.txt` for full list.

## Security Notes

**Never commit `gcp-key.json` to version control!**

Add to `.gitignore`:
```
gcp-key.json
.env
*.env
```

## References

- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [RAGAS Metrics](https://github.com/explodinggradients/ragas)
- [Google Vertex AI](https://cloud.google.com/vertex-ai)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## Author

BLANQUEZ-YESTE Nicolas, FRANÇOIS Bryan, YILDIRIM Emirhan

