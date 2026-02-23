import os
import psycopg2
from google import genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL   = "sentence-transformers/all-mpnet-base-v2"
RERANKER_PATH = os.environ.get("RERANKER_PATH", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY", "")
TOP_K_FETCH   = 20
TOP_K_RETURN  = 5

app = FastAPI(title="RAG System", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

embed_model   = None
reranker      = None
gemini_client = None

@app.on_event("startup")
async def load_models():
    global embed_model, reranker, gemini_client
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    print("Loading reranker...")
    reranker = CrossEncoder(RERANKER_PATH)
    print("Configuring Gemini...")
    gemini_client = genai.Client(api_key=GEMINI_KEY)
    print("Ready!")

def get_db():
    conn_name = os.environ.get("DB_CONN_NAME", "project-20c39e5c-cfef-4c32-976:us-central1:rag-postgres")
    db_name = os.environ.get("DB_NAME", "ragdb")
    user = os.environ.get("DB_USER", "raguser")
    password = os.environ.get("DB_PASSWORD", "RagPass2024!")
    use_connector = os.environ.get("USE_CLOUD_SQL_CONNECTOR", "false").lower() == "true"
    if use_connector:
        from google.cloud.sql.connector import Connector
        connector = Connector()
        return connector.connect(conn_name, "pg8000", user=user, password=password, db=db_name)
    return psycopg2.connect(
        host="127.0.0.1", port=5432,
        dbname=db_name, user=user, password=password
    )

def retrieve(query: str):
    emb  = embed_model.encode(query).tolist()
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        SELECT c.id, d.title, d.url, c.chunk_index, c.content,
               (1-(c.embedding<=>%s::vector))::float AS score
        FROM chunks c JOIN documents d ON c.doc_id=d.id
        ORDER BY c.embedding<=>%s::vector
        LIMIT %s
    """, (str(emb), str(emb), TOP_K_FETCH))
    rows = cur.fetchall()
    cur.close(); conn.close()

    if not rows:
        return []

    pairs  = [(query, r[4]) for r in rows]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)

    return [
        {"title": r[1], "url": r[2], "chunk_index": r[3],
         "content": r[4], "relevance_score": round(float(s), 4)}
        for r, s in ranked[:TOP_K_RETURN]
    ]

def generate(query: str, chunks: list):
    if not chunks:
        return "I don't know based on the available documents."

    context = "\n\n---\n\n".join(
        f"[Source {i}: {c['title']}]\n{c['content']}"
        for i, c in enumerate(chunks, 1)
    )
    prompt = f"""Answer the question using ONLY the sources below.
Rules:
1. Only use information from the provided sources.
2. Cite sources inline as [Source N].
3. If the answer is not in the sources, say "I don't know based on the provided documents."
4. Be clear and concise.

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

class QueryRequest(BaseModel):
    question: str

class Citation(BaseModel):
    title: str
    url: str
    chunk_index: int
    content: str
    relevance_score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    chunks = retrieve(req.question)
    answer = generate(req.question, chunks)
    return QueryResponse(
        question=req.question,
        answer=answer,
        citations=[Citation(**c) for c in chunks]
    )

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": embed_model is not None}

@app.get("/")
async def root():
    return {"message": "RAG API running", "docs": "/docs"}