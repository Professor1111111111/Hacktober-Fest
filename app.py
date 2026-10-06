import streamlit as st
from pathlib import Path

from truthmesh.ingest import ingest_paths
from truthmesh.retriever import HybridRetriever
from truthmesh.claims import extract_claim
from truthmesh.contradictions import compare_claims


st.set_page_config(
    page_title="TruthMesh",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 TruthMesh")
st.caption("Offline Evidence & Contradiction Analysis")

question = st.text_input(
    "Ask a question",
    "What is required for examination eligibility?"
)

uploaded = st.file_uploader(
    "Upload evidence files",
    type=["txt", "pdf", "docx", "md"],
    accept_multiple_files=True
)

if st.button("Analyze Evidence"):

    if not uploaded:
        st.error("Upload at least one document.")
        st.stop()

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    for file in uploaded:
        (upload_dir / file.name).write_bytes(
            file.getbuffer()
        )

    paths = list(upload_dir.iterdir())

    chunks = ingest_paths(paths)

    if not chunks:
        st.error("No readable evidence found.")
        st.stop()

    with st.spinner("Searching evidence..."):
        retriever = HybridRetriever(
            [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "doc_name": c.doc_name,
                    "page": c.page,
                    "text": c.text,
                    "char_start": c.char_start,
                    "char_end": c.char_end
                }
                for c in chunks
            ]
        )

        results = retriever.search(
            question,
            top_k=5
        )

    st.subheader("📚 Retrieved Evidence")

    for i, result in enumerate(results, 1):
        with st.expander(
            f"Evidence {i} — {result['doc_name']}"
        ):
            st.write(result["text"])
            st.caption(
                f"Page: {result['page']} | "
                f"Chunk: {result['chunk_id']}"
            )

    evidence = "\n".join(
        result["text"]
        for result in results
    )

    with st.spinner("Analyzing with local Ollama model..."):
        claim = extract_claim(
            question,
            evidence
        )

    st.subheader("🧠 AI Claim Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Stance",
            claim["stance"]
        )

    with col2:
        st.metric(
            "Confidence",
            f"{claim['confidence']:.0%}"
        )

    st.write("**Claim:**")
    st.info(claim["claim"])

    st.subheader("🔍 Provenance")

    for result in results:
        st.write(
            f"**{result['doc_name']} — Page {result['page']}**"
        )
        st.write(result["text"])

    if len(results) >= 2:

        comparison = compare_claims(
            results[0]["text"],
            results[1]["text"]
        )

        st.subheader("⚖️ Cross-Document Analysis")

        st.write(
            f"**Relationship:** "
            f"{comparison['relationship']}"
        )

        st.write(
            f"**Confidence:** "
            f"{comparison['confidence']:.0%}"
        )

        st.write(
            f"**Reason:** "
            f"{comparison['reason']}"
        )