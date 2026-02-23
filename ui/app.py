import streamlit as st
import requests
import os

API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Wiki Assistant",
                   page_icon="📚", layout="wide")
st.title("📚 Wikipedia RAG Assistant")
st.caption("Ask about Machine Learning — answers grounded in Wikipedia articles")

with st.sidebar:
    st.header("💡 Example Questions")
    examples = [
        "What is gradient descent?",
        "How does backpropagation work?",
        "What is the difference between supervised and unsupervised learning?",
        "How do transformers use attention?",
        "What causes overfitting and how to prevent it?",
        "What is word2vec?",
        "How does a random forest make predictions?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.q = ex
    st.divider()
    try:
        h = requests.get(f"{API}/health", timeout=3).json()
        st.success(f"API: {h['status']}")
    except:
        st.error("API offline — start uvicorn first")

st.divider()
question = st.text_input("Your question:",
                          value=st.session_state.get("q", ""),
                          placeholder="e.g. What is gradient descent?")

if st.button("🔍 Ask", type="primary") and question:
    with st.spinner("Searching and generating answer..."):
        try:
            res = requests.post(f"{API}/query",
                                json={"question": question}, timeout=30)
            res.raise_for_status()
            data = res.json()

            st.subheader("💬 Answer")
            st.markdown(data["answer"])
            st.divider()

            st.subheader(f"📖 Sources ({len(data['citations'])} retrieved)")
            for i, c in enumerate(data["citations"], 1):
                with st.expander(
                    f"Source {i}: **{c['title']}** — relevance: `{c['relevance_score']:.3f}`"
                ):
                    st.markdown(f"🔗 [{c['url']}]({c['url']})")
                    st.info(c["content"])

        except requests.ConnectionError:
            st.error("Cannot reach API. Is uvicorn running?")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption("Built with FastAPI · Streamlit · pgvector · Gemini · Cloud Run ☁️")