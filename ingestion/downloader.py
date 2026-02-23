import wikipedia
import json
from pathlib import Path

OUTPUT_DIR = Path("data/raw")

ARTICLES = [
    "Machine learning", "Deep learning", "Neural network",
    "Natural language processing", "Computer vision",
    "Reinforcement learning", "Supervised learning",
    "Unsupervised learning", "Gradient descent",
    "Backpropagation", "Convolutional neural network",
    "Transformer (machine learning model)", "BERT (language model)",
    "Attention (machine learning)", "Support-vector machine",
    "Random forest", "Decision tree", "Logistic regression",
    "Linear regression", "K-means clustering",
    "Principal component analysis", "Generative adversarial network",
    "Recurrent neural network", "Long short-term memory",
    "Transfer learning", "Data augmentation",
    "Overfitting", "Regularization (mathematics)",
    "Hyperparameter (machine learning)", "Cross-validation (statistics)",
    "Confusion matrix", "Precision and recall",
    "F-score", "Receiver operating characteristic",
    "Feature engineering", "Dimensionality reduction",
    "Word2vec", "Tf–idf",
    "Named-entity recognition", "Sentiment analysis",
    "Text classification", "Question answering",
    "Information retrieval", "Semantic search",
    "Knowledge graph", "Federated learning",
    "Batch normalization", "Dropout (neural networks)",
    "Autoencoder", "Ensemble learning"
]

def download():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved, failed = [], []

    print(f"Downloading {len(ARTICLES)} Wikipedia articles...\n")

    for i, title in enumerate(ARTICLES):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            doc = {
                "id": f"doc_{i:03d}",
                "title": page.title,
                "url": page.url,
                "topic": title,
                "content": page.content
            }
            with open(OUTPUT_DIR / f"doc_{i:03d}.json", "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            saved.append(title)
            print(f"  OK [{i+1}/50] {page.title}")

        except wikipedia.exceptions.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                doc = {"id": f"doc_{i:03d}", "title": page.title,
                       "url": page.url, "topic": title, "content": page.content}
                with open(OUTPUT_DIR / f"doc_{i:03d}.json", "w", encoding="utf-8") as f:
                    json.dump(doc, f, ensure_ascii=False, indent=2)
                saved.append(title)
                print(f"  OK [{i+1}/50] {page.title}")
            except Exception as e2:
                print(f"  FAILED [{i+1}/50] {title}: {e2}")
                failed.append(title)
        except Exception as e:
            print(f"  FAILED [{i+1}/50] {title}: {e}")
            failed.append(title)

    manifest = {"total": len(saved), "failed": failed, "articles": saved}
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! Saved: {len(saved)}  Failed: {len(failed)}")
    print(f"Files saved in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    download()