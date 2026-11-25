import json
import numpy as np
import faiss

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastembed import TextEmbedding

# 1. Initialize FastAPI
app = FastAPI(title="RAG Colosseum API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load Knowledge Base
with open("passages.json", "r", encoding="utf-8") as file:
    passages = json.load(file)

if len(passages) != 50:
    raise ValueError("Knowledge base must contain exactly 50 passages")

# 3. Load Embedding Model
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 4. Generate Embeddings
embeddings = np.array(
    list(model.embed(passages, batch_size=8)),
    dtype="float32"
)

# 5. Create FAISS Index
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# 6. Request Schema
class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)


# 7. Query Endpoint
@app.post("/query")
def query_rag(request: QueryRequest):

    question = request.question.strip()

    if not question:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    # Embed Question
    query_vector = np.array([list(model.query_embed(question))[0]], dtype="float32")

    # FAISS Search
    faiss.normalize_L2(query_vector)
    scores, indices = index.search(
        query_vector,
        min(request.top_k, len(passages))
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append((int(idx), float(score)))

    if not results:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    best_score = results[0][1]

    # Abstention
    if best_score < 0.30:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.1
        }

    # 8. Improved Retrieval
    source_indices = []
    source_scores = []

    threshold = max(0.30, best_score - 0.20)

    for idx, score in results:

        if score >= threshold:
            source_indices.append(idx)

            source_scores.append(
                round(max(0.0, min(1.0, score)), 4)
            )

    # Keep only top 2 relevant passages
    source_indices = source_indices[:2]
    source_scores = source_scores[:2]

    # 9. Extractive Answer
    answer = "\n\n".join(
        passages[idx] for idx in source_indices
    )

    confidence = round(
        max(0.0, min(1.0, best_score)), 4
    )

    return {
        "answer": answer,
        "sources": source_indices,
        "scores": source_scores,
        "confidence": confidence
    }


# 10. Health Check
@app.get("/")
def home():
    return {
        "status": "RAG system running",
        "passages": len(passages),
        "model": "all-MiniLM-L6-v2"
    }
