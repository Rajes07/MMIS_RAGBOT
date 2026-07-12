"""
query.py

Shared retrieval + generation logic, used by both a CLI (when run directly)
and app.py (Streamlit UI). This is the only file that talks to an external
API (Groq) - and only with the final small prompt (top-k retrieved chunks +
the user's question), never raw documents.

Functions:
    load_index()                       - loads vectors.npz + chunks.json
    retrieve(question, top_k=3)        - returns top-k relevant chunks
    generate_answer(question, chunks)  - calls Groq to produce the final answer
"""

import json
import os

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VECTORS_PATH = os.path.join(DATA_DIR, "vectors.npz")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# System prompt guardrails: end users are EXTERNAL providers unsure of the
# portal UI, not internal staff. The assistant must stay grounded in the
# retrieved context and must not attempt case-specific determinations.
SYSTEM_PROMPT = """You are a help assistant for an MMIS (Medicaid Management \
Information System) provider portal. You help external providers understand \
policies and navigate processes like enrollment, revalidation, and claims.

Rules you must always follow:
1. Answer ONLY using the provided context below. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say so \
plainly and suggest the user contact Provider Support - do not guess.
3. Never make a case-specific determination (e.g. whether a specific claim \
will be approved, whether a specific enrollment is valid/eligible). For \
questions like that, explain that this requires review by Provider Support \
or a caseworker, and cannot be answered generally.
4. When you do answer from the context, mention which document (and page, \
if useful) the information came from, so the user can verify it themselves.
5. Keep answers clear and concise, written for someone unfamiliar with \
MMIS terminology.
"""

_model = None  # lazy-loaded singleton so the embedding model loads once


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def load_index():
    """
    Load vectors.npz and chunks.json into memory.
    Raises a clear, friendly error if ingest.py hasn't been run yet.
    """
    if not os.path.exists(VECTORS_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            "No index found. Run 'python ingest.py' first to build "
            "data/vectors.npz and data/chunks.json from your PDFs."
        )

    vectors_data = np.load(VECTORS_PATH)
    embeddings = vectors_data["embeddings"]

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Index mismatch: {len(embeddings)} vectors but {len(chunks)} "
            "chunk records. Try re-running ingest.py."
        )

    return embeddings, chunks


def _cosine_similarity(query_vec, all_vecs):
    """Compute cosine similarity between one query vector and a matrix of vectors."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    all_norms = all_vecs / (np.linalg.norm(all_vecs, axis=1, keepdims=True) + 1e-10)
    return all_norms @ query_norm


def retrieve(question, top_k=3):
    """
    Embed the question locally and return the top-k most similar chunks.
    Each result includes: text, filename, page, topic, score.
    """
    embeddings, chunks = load_index()
    model = _get_model()

    query_vec = model.encode([question], convert_to_numpy=True)[0]
    scores = _cosine_similarity(query_vec, embeddings)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append({
            "text": chunk["text"],
            "filename": chunk["filename"],
            "page": chunk["page"],
            "topic": chunk["topic"],
            "score": float(scores[idx]),
        })
    return results


def generate_answer(question, retrieved_chunks):
    """
    Build a context prompt from retrieved chunks and call Groq to generate
    the final answer. Only this small prompt (context + question) leaves
    the machine - not the original PDFs.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
        )

    client = Groq(api_key=api_key)

    context_blocks = []
    for chunk in retrieved_chunks:
        label = f"[Source: {chunk['filename']}, page {chunk['page']}, topic: {chunk['topic']}]"
        context_blocks.append(f"{label}\n{chunk['text']}")
    context_text = "\n\n---\n\n".join(context_blocks)

    user_prompt = f"""Context from help documents:

{context_text}

---

User question: {question}

Answer the question using only the context above, following your rules."""

    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def _cli_loop():
    """Simple terminal loop for testing retrieval + generation without the UI."""
    print("Help Content Assistant - CLI mode")
    print("Type a question, or 'quit' to exit.\n")

    try:
        load_index()
    except FileNotFoundError as e:
        print(str(e))
        return

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        chunks = retrieve(question, top_k=3)
        if not chunks:
            print("No relevant content found.\n")
            continue

        answer = generate_answer(question, chunks)
        print(f"\nAnswer:\n{answer}\n")
        print("Sources:")
        for c in chunks:
            print(f"  - {c['filename']} (page {c['page']}, topic: {c['topic']}, score: {c['score']:.3f})")
        print()


if __name__ == "__main__":
    _cli_loop()