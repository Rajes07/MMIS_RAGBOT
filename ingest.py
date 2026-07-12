"""
ingest.py

Reads topics.json to find which PDFs belong to which topic, extracts text
page-by-page from each PDF, splits it into overlapping chunks, embeds each
chunk locally (no external API calls here), and saves everything as flat
files: data/vectors.npz (embeddings) and data/chunks.json (chunk metadata).

Run this whenever PDFs are added/changed:
    python ingest.py

It does NOT talk to any LLM API - embedding happens locally via
sentence-transformers. Only query.py talks to an external API (Groq),
and only with the final small prompt, not raw documents.
"""

import json
import os
import sys

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PDFS_DIR = os.path.join(DATA_DIR, "pdfs")
TOPICS_PATH = os.path.join(DATA_DIR, "topics.json")
VECTORS_PATH = os.path.join(DATA_DIR, "vectors.npz")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def load_topics():
    """Load topic -> [pdf filenames] mapping from topics.json."""
    if not os.path.exists(TOPICS_PATH):
        print(f"ERROR: {TOPICS_PATH} not found. Create it first.")
        sys.exit(1)
    with open(TOPICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_pages(pdf_path):
    """Return a list of (page_number, page_text) tuples for a PDF, 1-indexed."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping character-based chunks.
    Simple sliding window: each chunk starts (chunk_size - overlap)
    characters after the previous one, so consecutive chunks share
    `overlap` characters of context.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def main():
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (local, one-time download)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    topics = load_topics()
    print(f"Found {len(topics)} topic(s) in topics.json.")

    all_chunks_meta = []
    all_texts_for_embedding = []

    for topic, filenames in topics.items():
        print(f"\n--- Topic: {topic} ---")
        for filename in filenames:
            pdf_path = os.path.join(PDFS_DIR, filename)
            if not os.path.exists(pdf_path):
                print(f"  WARNING: '{filename}' listed under '{topic}' but not found in {PDFS_DIR}. Skipping.")
                continue

            print(f"  Processing '{filename}'...")
            pages = extract_pages(pdf_path)
            print(f"    Extracted text from {len(pages)} page(s) with content.")

            file_chunk_count = 0
            for page_num, page_text in pages:
                page_chunks = chunk_text(page_text)
                for chunk in page_chunks:
                    all_chunks_meta.append({
                        "text": chunk,
                        "filename": filename,
                        "page": page_num,
                        "topic": topic,
                    })
                    all_texts_for_embedding.append(chunk)
                    file_chunk_count += 1

            print(f"    Created {file_chunk_count} chunk(s) from '{filename}'.")

    if not all_texts_for_embedding:
        print("\nNo chunks were created. Check that PDFs exist in data/pdfs/ and match topics.json filenames.")
        sys.exit(1)

    print(f"\nEmbedding {len(all_texts_for_embedding)} chunk(s) locally...")
    embeddings = model.encode(
        all_texts_for_embedding,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"\nSaving embeddings to {VECTORS_PATH} ...")
    np.savez_compressed(VECTORS_PATH, embeddings=embeddings)

    print(f"Saving chunk metadata to {CHUNKS_PATH} ...")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks_meta, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(all_chunks_meta)} chunks indexed from {len(topics)} topic(s).")
    print("You can now run: streamlit run app.py  (or python query.py for CLI testing)")


if __name__ == "__main__":
    main()