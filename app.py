"""
app.py

Streamlit two-pane UI for the Help Content Assistant.

Left pane  : browse/download the original help-content PDFs, grouped by topic.
Right pane : ask a question, get an AI-generated answer grounded in the PDFs,
             with sources listed underneath.

Run with:
    streamlit run app.py

Requires data/vectors.npz and data/chunks.json to exist (run ingest.py first)
for the right pane to work. The left pane works independently of the index.
"""

import json
import os

import streamlit as st

from query import retrieve, generate_answer, load_index

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PDFS_DIR = os.path.join(DATA_DIR, "pdfs")
TOPICS_PATH = os.path.join(DATA_DIR, "topics.json")

st.set_page_config(page_title="Help Content Assistant", layout="wide")
st.title("MMIS Help Content Assistant")

left_col, right_col = st.columns([0.65, 0.35])

# ---------------------------------------------------------------------------
# LEFT PANE - Help Content (browse / download original PDFs)
# ---------------------------------------------------------------------------
with left_col:
    st.header("Help Content")

    if not os.path.exists(TOPICS_PATH):
        st.error("data/topics.json not found.")
    else:
        with open(TOPICS_PATH, "r", encoding="utf-8") as f:
            topics = json.load(f)

        for topic, filenames in topics.items():
            with st.expander(topic):
                for filename in filenames:
                    pdf_path = os.path.join(PDFS_DIR, filename)
                    if os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label=f"Download: {filename}",
                                data=pdf_file.read(),
                                file_name=filename,
                                mime="application/pdf",
                                key=f"dl_{topic}_{filename}",
                            )
                    else:
                        st.caption(f"{filename} (not yet uploaded)")

# ---------------------------------------------------------------------------
# RIGHT PANE - RAG Chat
# ---------------------------------------------------------------------------
with right_col:
    st.header("Ask a question")

    # Check the index exists before allowing questions
    index_ready = True
    try:
        load_index()
    except FileNotFoundError as e:
        index_ready = False
        st.warning(str(e))

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of (question, answer, sources)

    question = st.text_input("Your question", disabled=not index_ready)
    ask_clicked = st.button("Ask", disabled=not index_ready)

    if ask_clicked and question.strip():
        with st.spinner("Searching help content..."):
            chunks = retrieve(question, top_k=3)
            answer = generate_answer(question, chunks)
        st.session_state.chat_history.insert(0, (question, answer, chunks))

    for q, a, sources in st.session_state.chat_history:
        st.markdown(f"**Q: {q}**")
        st.write(a)
        with st.expander("Sources"):
            for s in sources:
                st.caption(
                    f"{s['filename']} - page {s['page']} - topic: {s['topic']} "
                    f"- relevance: {s['score']:.2f}"
                )
        st.divider()

    st.sidebar.info(
        "All chunking and embedding runs locally. No database is used. "
        "Only the retrieved context and your question are sent to the LLM "
        "to generate the final answer."
    )