import json
import random
import psycopg2
from pathlib import Path
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader

BASE_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_OUTPUT = Path("models/reranker")
random.seed(42)

def get_connection():
    return psycopg2.connect(
        host="127.0.0.1", port=5432,
        dbname="ragdb", user="raguser", password="RagPass2024!"
    )

def load_chunks():
    print("Loading chunks from database...")
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT c.doc_id, d.title, c.content FROM chunks c JOIN documents d ON c.doc_id=d.id")
    rows = cur.fetchall()
    cur.close(); conn.close()
    print(f"Loaded {len(rows)} chunks\n")
    return [{"doc_id": r[0], "title": r[1], "content": r[2]} for r in rows]

def make_query(text):
    first = text.split(".")[0].strip()
    if len(first) < 25:
        return None
    return f"What is {first.lower()}?"

def build_data(chunks):
    train, eval_samples = [], []

    for i, chunk in enumerate(chunks[:600]):
        query = make_query(chunk["content"])
        if not query:
            continue

        # Positive pair
        train.append(InputExample(texts=[query, chunk["content"]], label=1.0))

        # Negative pair (different document)
        neg = random.choice(chunks)
        tries = 0
        while neg["doc_id"] == chunk["doc_id"] and tries < 10:
            neg = random.choice(chunks)
            tries += 1
        train.append(InputExample(texts=[query, neg["content"]], label=0.0))

        if i < 80:
            eval_samples.append({
                "query":    query,
                "positive": [chunk["content"]],
                "negative": [neg["content"]]
            })

    random.shuffle(train)
    print(f"Training pairs: {len(train)}")
    print(f"Eval samples:   {len(eval_samples)}\n")
    return train, eval_samples

def train():
    chunks = load_chunks()
    train_samples, eval_samples = build_data(chunks)

    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

    print(f"Fine-tuning reranker: {BASE_MODEL}")
    print("This takes about 20-30 minutes...\n")

    model     = CrossEncoder(BASE_MODEL, num_labels=1, max_length=512)
    evaluator = CERerankingEvaluator(eval_samples, name="wiki-eval")

    model.fit(
        train_dataloader=DataLoader(train_samples, shuffle=True, batch_size=16),
        epochs=3,
        warmup_steps=50,
        evaluation_steps=200,
        evaluator=evaluator,
        output_path=str(MODEL_OUTPUT),
        use_amp=False,
        save_best_model=True,
    )
    model.save(str(MODEL_OUTPUT))
    print(f"\nDone! Fine-tuned reranker saved to: {MODEL_OUTPUT}/")

if __name__ == "__main__":
    train()