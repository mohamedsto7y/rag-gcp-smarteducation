import json
import numpy as np
import psycopg2
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL     = "sentence-transformers/all-mpnet-base-v2"
FINETUNED_PATH  = "models/reranker"
PRETRAINED_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
K_VALUES        = [1, 3, 5, 10]

TEST_SET = [
    ("What is gradient descent optimization?",        "Gradient descent"),
    ("How does backpropagation update weights?",      "Backpropagation"),
    ("What is a convolutional neural network?",       "Convolutional neural network"),
    ("How does transfer learning work?",              "Transfer learning"),
    ("What causes overfitting in models?",            "Overfitting"),
    ("How does k-means clustering group data?",       "K-means clustering"),
    ("What is principal component analysis?",         "Principal component analysis"),
    ("How does the attention mechanism work?",        "Attention"),
    ("What is a random forest classifier?",           "Random forest"),
    ("How does reinforcement learning train agents?", "Reinforcement learning"),
    ("What is batch normalization?",                  "Batch normalization"),
    ("What is dropout regularization?",               "Dropout"),
    ("How does word2vec learn embeddings?",           "Word2vec"),
    ("What is sentiment analysis?",                   "Sentiment analysis"),
    ("How does a decision tree split data?",          "Decision tree"),
    ("What is the F1 score metric?",                  "F-score"),
    ("How does federated learning preserve privacy?", "Federated learning"),
    ("What is named entity recognition?",             "Named-entity recognition"),
    ("What is TF-IDF used for?",                      "Tf–idf"),
    ("How does logistic regression classify data?",   "Logistic regression"),
]

def get_connection():
    return psycopg2.connect(
        host="127.0.0.1", port=5432,
        dbname="ragdb", user="raguser", password="RagPass2024!"
    )

def vector_search(cur, emb, k=20):
    cur.execute("""
        SELECT c.id, d.title, c.content,
               (1-(c.embedding<=>%s::vector))::float AS score
        FROM chunks c JOIN documents d ON c.doc_id=d.id
        ORDER BY c.embedding<=>%s::vector
        LIMIT %s
    """, (str(emb), str(emb), k))
    return cur.fetchall()

def rerank(model, query, results):
    pairs  = [(query, r[2]) for r in results]
    scores = model.predict(pairs)
    return [r for r, _ in sorted(zip(results, scores), key=lambda x: x[1], reverse=True)]

def recall_at_k(results, title, k):
    return float(any(title.lower() in r[1].lower() for r in results[:k]))

def mrr_at_k(results, title, k):
    for i, r in enumerate(results[:k], 1):
        if title.lower() in r[1].lower():
            return 1.0 / i
    return 0.0

def ndcg_at_k(results, title, k):
    rels = [1.0 if title.lower() in r[1].lower() else 0.0 for r in results[:k]]
    dcg  = sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))
    return dcg

def evaluate(label, get_results_fn):
    metrics = {k: {"recall": [], "mrr": [], "ndcg": []} for k in K_VALUES}
    for query, title in TEST_SET:
        results = get_results_fn(query)
        for k in K_VALUES:
            metrics[k]["recall"].append(recall_at_k(results, title, k))
            metrics[k]["mrr"].append(mrr_at_k(results, title, k))
            metrics[k]["ndcg"].append(ndcg_at_k(results, title, k))

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    agg = {}
    for k in K_VALUES:
        agg[k] = {
            f"Recall@{k}": round(np.mean(metrics[k]["recall"]), 4),
            f"MRR@{k}":    round(np.mean(metrics[k]["mrr"]),    4),
            f"nDCG@{k}":   round(np.mean(metrics[k]["ndcg"]),   4),
        }
        for name, val in agg[k].items():
            print(f"  {name}: {val:.4f}")
    return agg

def main():
    print("Loading models...")
    embed      = SentenceTransformer(EMBED_MODEL)
    pretrained = CrossEncoder(PRETRAINED_NAME)
    finetuned  = CrossEncoder(FINETUNED_PATH)

    conn = get_connection()
    cur  = conn.cursor()

    def baseline(q):
        return vector_search(cur, embed.encode(q).tolist())

    def pretrained_fn(q):
        return rerank(pretrained, q, vector_search(cur, embed.encode(q).tolist()))

    def finetuned_fn(q):
        return rerank(finetuned, q, vector_search(cur, embed.encode(q).tolist()))

    print(f"\nEvaluating on {len(TEST_SET)} test queries...\n")

    r1 = evaluate("Baseline  (vector search only)",         baseline)
    r2 = evaluate("Pretrained Reranker (no fine-tuning)",   pretrained_fn)
    r3 = evaluate("Fine-tuned Reranker",                    finetuned_fn)

    report = {
        "baseline":             r1,
        "pretrained_reranker":  r2,
        "finetuned_reranker":   r3,
        "num_queries":          len(TEST_SET)
    }

    Path("evaluation").mkdir(exist_ok=True)
    with open("evaluation/metrics_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*55}")
    print("Report saved to evaluation/metrics_report.json")
    print("="*55)

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()