import json
import sys
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import psycopg2

DATA_DIR   = Path("data/raw")
CHUNK_SIZE = 500
OVERLAP    = 60
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 32

def get_connection():
    return psycopg2.connect(
        host="127.0.0.1",
        port=5432,
        dbname="ragdb",
        user="raguser",
        password="RagPass2024!"
    )

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if len(chunk) > 80:
            chunks.append(chunk)
        start += CHUNK_SIZE - OVERLAP
    return chunks

def ingest():
    print(f"Loading embedding model: {MODEL_NAME}")
    print("(First time takes 2-3 minutes to download the model...)")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded!\n")

    conn = get_connection()
    cur  = conn.cursor()

    doc_files = sorted(DATA_DIR.glob("doc_*.json"))
    print(f"Found {len(doc_files)} documents\n")

    total_chunks = 0

    for doc_file in tqdm(doc_files, desc="Ingesting documents"):
        with open(doc_file, encoding="utf-8") as f:
            doc = json.load(f)

        cur.execute(
            "INSERT INTO documents (id, title, url, topic) "
            "VALUES (%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
            (doc["id"], doc["title"], doc.get("url",""), doc.get("topic",""))
        )

        chunks = chunk_text(doc["content"])

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch      = chunks[batch_start:batch_start + BATCH_SIZE]
            embeddings = model.encode(batch, show_progress_bar=False)

            for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
                cur.execute(
                    "INSERT INTO chunks (doc_id, chunk_index, content, content_length, embedding) "
                    "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (doc_id, chunk_index) DO NOTHING",
                    (doc["id"], batch_start + j, chunk, len(chunk), str(emb.tolist()))
                )

        total_chunks += len(chunks)
        conn.commit()

    cur.close()
    conn.close()
    print(f"\nDone! {len(doc_files)} documents → {total_chunks} chunks stored in database")

if __name__ == "__main__":
    ingest()