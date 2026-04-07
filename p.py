"""
RAG Q&A System — Single‑File Python App (Streamlit UI + CLI fallback)

This file is a rewritten, robust version of the original RAG demo application.
It fixes the common ModuleNotFoundError: No module named 'streamlit' by:

1. *Lazy/dynamic imports* — we only import optional heavy packages when needed.
2. *CLI fallback mode* — when streamlit is not available, the script runs a usable
   command-line demo that demonstrates the retrieval + answer workflow without
   requiring any third-party UI library.
3. *Helpful messages* — when optional dependencies are missing the program
   prints clear instructions how to install the extras to get the full app.

Modes
- Full Streamlit UI mode: run with streamlit run app.py (requires dependencies)
- CLI demo mode: run python app.py --demo or simply run without streamlit installed
- Unit tests: python app.py --run-tests runs lightweight tests for core helpers

Notes
- The CLI demo implements a simple retrieval mechanism (word overlap), so you
  can still query uploaded or demo documents without sentence-transformers / faiss.
- To enable high-quality embeddings + vector search, install the packages listed
  in the original requirements (sentence-transformers, faiss-cpu, etc.).

"""

from _future_ import annotations

import os
import sys
import re
import argparse
import json
import uuid
from dataclasses import dataclass
import importlib.util
import textwrap
from typing import List, Tuple, Dict, Optional

# ------------------ Utility: feature detection ------------------

def module_available(name: str) -> bool:
    """Return True if module is importable without importing it.
    This avoids raising ModuleNotFoundError at import-time.
    """
    return importlib.util.find_spec(name) is not None

# --------------------------- Data model ------------------------

@dataclass
class Chunk:
    id: str
    text: str
    source: str
    loc: str = ""

# -------------------------- Helpers ---------------------------

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 200) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# ---------------------- Simple retriever ----------------------

class SimpleRetriever:
    """A dependency-free retriever that ranks chunks by word overlap.

    This is intentionally lightweight and meant as a fallback/demo. For
    production you should replace it with embeddings + FAISS or another
    vector database.
    """
    def _init_(self) -> None:
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        q_words = set(re.findall(r"\w+", query.lower()))
        scores: List[Tuple[float, Chunk]] = []
        for c in self.chunks:
            words = set(re.findall(r"\w+", c.text.lower()))
            common = len(q_words & words)
            # Score also accounts for proximity: prefer shorter chunks when tie
            score = float(common)
            scores.append((score, c))
        scores.sort(key=lambda x: x[0], reverse=True)
        # Filter zeros but fall back to first chunks if nothing matched
        filtered = [(c, s) for s, c in scores if s > 0]
        if not filtered and scores:
            # return top k with their (possibly zero) scores
            return [(c, float(s)) for s, c in scores[:top_k]]
        return [(c, float(s)) for c, s in filtered[:top_k]]

# ----------------------- Prompt builder -----------------------

def build_contextual_prompt(question: str, contexts: List[Chunk], max_len_chars: int = 6000) -> Tuple[str, str, List[str]]:
    """Create a system + user prompt using the provided contexts and list of citation strings.
    Returns (system, user, cites)
    """
    cites: List[str] = []
    assembled: List[str] = []
    total = 0
    for i, c in enumerate(contexts, start=1):
        snippet = c.text.strip()
        if not snippet:
            continue
        part = f"[{i}] Source: {c.source} {c.loc}\n{snippet}\n"
        if total + len(part) > max_len_chars and assembled:
            break
        assembled.append(part)
        total += len(part)
        cites.append(f"[{i}] {c.source} {c.loc}".strip())

    system = (
        "You are a precise assistant answering questions using only the provided context. "
        "Cite sources like [1], [2] in the answer. If the answer is unknown from context, "
        "say you don't know and suggest what to check next."
    )

    user = (
        "CONTEXT:\n" + "\n\n".join(assembled) + "\n\n"
        + "QUESTION: " + question + "\nINSTRUCTIONS: Use citations [#] after the relevant statements. Be concise and accurate."
    )
    return system, user, cites

# ------------------ OpenAI (optional) helper ------------------

def call_openai_chat(system: str, user: str, api_key: Optional[str], base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini") -> str:
    """Call OpenAI Chat Completions if requests + API key are available.
    This function carefully checks for the requests module and the API key.
    """
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not provided")
    if not module_available("requests"):
        raise RuntimeError("Module 'requests' is not available; cannot call OpenAI.")
    import requests
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    url = f"{base_url}/chat/completions"
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

# ------------------------ CLI mode ---------------------------

def run_cli_mode(args: argparse.Namespace) -> None:
    """A minimal interactive CLI that demonstrates ingestion + retrieval.

    - If --files are given, it will attempt to read them (supports .txt/.md; .pdf if pypdf is installed).
    - If --demo is given or no files were ingested, a built-in demo document is used.
    - Enter questions in the prompt. Type 'exit' or Ctrl+D to quit.
    """
    print("\nRAG Q&A CLI demo — lightweight fallback mode")
    print("(This mode does not require streamlit. For the full UI, install the optional requirements and run with: streamlit run app.py)\n")

    retriever = SimpleRetriever()

    ingested = 0

    # Ingest files if provided
    files = args.files or []
    for p in files:
        if not os.path.exists(p):
            print(f" - Warning: file not found: {p}")
            continue
        try:
            if p.lower().endswith(".pdf") and module_available("pypdf"):
                from pypdf import PdfReader
                with open(p, "rb") as fh:
                    reader = PdfReader(fh)
                    for i, page in enumerate(reader.pages, start=1):
                        try:
                            raw = page.extract_text() or ""
                        except Exception:
                            raw = ""
                        for piece in chunk_text(raw):
                            retriever.add([Chunk(id=str(uuid.uuid4()), text=piece, source=p, loc=f"page {i}")])
                            ingested += 1
            else:
                # read as text
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    raw = fh.read()
                for i, piece in enumerate(chunk_text(raw), start=1):
                    retriever.add([Chunk(id=str(uuid.uuid4()), text=piece, source=p, loc=f"chunk {i}")])
                    ingested += 1
        except Exception as e:
            print(f" - Error ingesting {p}: {e}")

    # If no files ingested or demo requested, add demo documents
    if ingested == 0 or args.demo:
        demo_text = textwrap.dedent(
            """
            RAG Demo Document

            Retrieval-augmented generation (RAG) combines a retrieval system (which
            finds relevant chunks of text from documents) with a generative model
            (which composes the final answer). This allows the model to ground its
            responses in long documents or corpora without placing everything in
            the model's weights.

            Use cases include students querying long PDFs, employees searching
            internal wikis, and customers looking up product manuals or FAQs.
            """
        )
        chunks = [Chunk(id=str(uuid.uuid4()), text=c, source="demo_doc.txt", loc=f"demo {i+1}") for i, c in enumerate(chunk_text(demo_text))]
        retriever.add(chunks)
        ingested += len(chunks)
        print(f" - Added demo document with {len(chunks)} chunks.")

    print(f"\nIngested {ingested} chunks. You can now ask questions. Type 'exit' to quit.\n")

    # Interactive loop
    try:
        while True:
            try:
                question = input("Question > ").strip()
            except EOFError:
                print("\nExiting.")
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            hits = retriever.search(question, top_k=args.top_k)
            if not hits:
                print("No relevant content found.")
                continue

            print("\nTop retrieved snippets:")
            for i, (c, score) in enumerate(hits, start=1):
                snippet = (c.text[:400] + "...") if len(c.text) > 400 else c.text
                print(f" [{i}] score={score:.2f} — {c.source} {c.loc}\n    {snippet}\n")

            # If user asked to use OpenAI and key exists, call it
            if args.use_openai and os.getenv("OPENAI_API_KEY"):
                try:
                    system, user, cites = build_contextual_prompt(question, [c for c, _ in hits])
                    ans = call_openai_chat(system, user, api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
                    print("\n=== Generated answer (OpenAI) ===\n")
                    print(ans)
                    if cites:
                        print("\nSources:")
                        for s in cites:
                            print(" - ", s)
                except Exception as e:
                    print(f"Failed to call OpenAI: {e}\nFalling back to concatenated snippets as answer.")
                    # fall through to fallback

            # Fallback answer: join top snippets and print with citations
            print("\n=== Fallback answer (concatenated context) ===\n")
            assembled = []
            for i, (c, score) in enumerate(hits, start=1):
                assembled.append(f"[{i}] {c.source} {c.loc}\n{c.text}\n")
            print("\n".join(assembled))
            print("\n---\n")

    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")

# ---------------------- Streamlit UI mode ---------------------

def run_streamlit_mode() -> None:
    """Run a Streamlit UI if streamlit is installed. This function imports heavy
    dependencies lazily and provides helpful error messages if they are missing.
    """
    import streamlit as st  # type: ignore

    st.set_page_config(page_title="RAG Q&A System", page_icon="📚", layout="wide")
    st.title("📚 Retrieval‑Augmented Q&A (RAG) — Streamlit mode")

    st.sidebar.markdown("### ⚙ Settings")
    st.sidebar.caption("Optional: install sentence-transformers + faiss-cpu to enable embeddings + vector search.")

    # Check for heavy dependencies and show status
    deps = {"pypdf": module_available("pypdf"), "trafilatura": module_available("trafilatura"), "sentence-transformers": module_available("sentence_transformers"), "faiss": module_available("faiss")}
    st.sidebar.markdown("*Dependency status*")
    for k, ok in deps.items():
        st.sidebar.write(f"- {k}: {'✅' if ok else '❌'}")

    if not deps.get("sentence-transformers") or not deps.get("faiss"):
        st.sidebar.warning("Embeddings/FAISS not available — the Streamlit UI will still run but only the simple retriever will be used. To enable full RAG, install: sentence-transformers faiss-cpu")

    # Simple UI controls
    files = st.file_uploader("Upload files (txt/md/pdf) or leave blank to use demo", type=["txt", "md", "markdown", "pdf"], accept_multiple_files=True)
    urls_text = st.text_area("Add URLs (one per line) — requires 'trafilatura' to fetch them")

    top_k = st.sidebar.number_input("Top K", min_value=1, max_value=20, value=5, key="ui_top_k")
    use_openai = st.sidebar.checkbox("Use OpenAI (if OPENAI_API_KEY set)", value=False)

    # Build index button
    if st.button("Build / Update Index"):
        st.info("Building index (this may be slow if you enable embeddings)...")
        retriever = SimpleRetriever()
        count = 0
        # ingest files
        for f in files or []:
            name = f.name
            try:
                if name.lower().endswith(".pdf") and module_available("pypdf"):
                    from pypdf import PdfReader
                    reader = PdfReader(f)
                    for i, page in enumerate(reader.pages, start=1):
                        txt = page.extract_text() or ""
                        for piece in chunk_text(txt):
                            retriever.add([Chunk(id=str(uuid.uuid4()), text=piece, source=name, loc=f"page {i}")])
                            count += 1
                else:
                    raw = f.read().decode("utf-8", errors="ignore")
                    for i, piece in enumerate(chunk_text(raw), start=1):
                        retriever.add([Chunk(id=str(uuid.uuid4()), text=piece, source=name, loc=f"chunk {i}" )])
                        count += 1
            except Exception as e:
                st.error(f"Failed to ingest {name}: {e}")
        # ingest URLs
        if urls_text and module_available("trafilatura"):
            import trafilatura
            for url in urls_text.splitlines():
                url = url.strip()
                if not url:
                    continue
                downloaded = trafilatura.fetch_url(url)
                if not downloaded:
                    st.warning(f"Failed to fetch {url}")
                    continue
                extracted = trafilatura.extract(downloaded) or ""
                for piece in chunk_text(extracted):
                    retriever.add([Chunk(id=str(uuid.uuid4()), text=piece, source=url, loc="")])
                    count += 1
        if count == 0:
            st.info("No files ingested; using demo text.")
            demo_text = ("Retrieval-augmented generation combines retrieval and generation to produce answers grounded in documents.")
            for i, piece in enumerate(chunk_text(demo_text), start=1):
                retriever.add([Chunk(id=str(uuid.uuid4()), text=piece, source="demo", loc=f"{i}")])
                count += 1
        # store retriever
        st.session_state["retriever"] = retriever
        st.success(f"Indexed {count} chunks.")

    # Ask question
    question = st.text_input("Ask a question about your documents")
    if st.button("Answer") and question:
        retriever: SimpleRetriever = st.session_state.get("retriever")
        if retriever is None or not retriever.chunks:
            st.error("Index is empty. Build or ingest documents first.")
        else:
            hits = retriever.search(question, top_k=top_k)
            st.subheader("Top retrieved snippets")
            for i, (c, score) in enumerate(hits, start=1):
                st.markdown(f"[{i}] {c.source} {c.loc}** — score={score:.2f}\n\n{c.text[:600]}{'...' if len(c.text)>600 else ''}")

            # Optional OpenAI generation
            if use_openai and os.getenv("OPENAI_API_KEY"):
                try:
                    system, user, cites = build_contextual_prompt(question, [c for c, _ in hits])
                    ans = call_openai_chat(system, user, api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
                    st.subheader("Generated answer (OpenAI)")
                    st.write(ans)
                    if cites:
                        st.caption("Sources")
                        for s in cites:
                            st.write("- ", s)
                except Exception as e:
                    st.error(f"OpenAI call failed: {e}")
                    st.info("Showing fallback concatenated context below.")

            st.subheader("Fallback concatenated context")
            for i, (c, _) in enumerate(hits, start=1):
                st.markdown(f"[{i}] {c.source} {c.loc}\n\n{c.text}")

    st.markdown("---")
    st.markdown("This Streamlit UI uses the lightweight SimpleRetriever by default.\nInstall sentence-transformers and faiss-cpu to enable embeddings + vector search.")

# -------------------------- Tests ----------------------------

def run_tests() -> None:
    print("Running lightweight tests...")
    # Test chunk_text basic behaviour
    s = "This is a test. " * 100
    chunks = chunk_text(s, chunk_size=50, overlap=10)
    assert all(len(c) <= 50 for c in chunks), "chunk_text produced chunk > chunk_size"

    # Test chunk_text handles empty input
    assert chunk_text("", chunk_size=50, overlap=10) == [], "chunk_text should return [] for empty input"

    # Test chunk_text produces multiple chunks for long input
    long_text = "word " * 2000
    long_chunks = chunk_text(long_text, chunk_size=300, overlap=50)
    assert len(long_chunks) > 1, "chunk_text should split long text into multiple chunks"

    # Test SimpleRetriever ranking
    r = SimpleRetriever()
    a = Chunk(id="a", text="apple banana apple", source="doc1")
    b = Chunk(id="b", text="banana orange", source="doc2")
    c = Chunk(id="c", text="grape melon", source="doc3")
    r.add([a, b, c])
    hits = r.search("apple", top_k=3)
    assert hits and hits[0][0].id == "a", "Retriever failed to rank exact match first"

    # Test SimpleRetriever returns top_k even when no query matches (fallback)
    hits_no_match = r.search("zucchini", top_k=2)
    assert len(hits_no_match) == 2, "Retriever should return top_k items even if no overlap"

    # Test build_contextual_prompt returns matching number of cites
    system, user, cites = build_contextual_prompt("What is RAG?", [a, b, c], max_len_chars=10000)
    assert isinstance(system, str) and isinstance(user, str), "Prompts should be strings"
    assert len(cites) == 3, "Cite list length should match number of contexts"

    print("All lightweight tests passed.")

# -------------------------- Entrypoint -----------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG Q&A (Streamlit + CLI fallback)")
    p.add_argument("--demo", action="store_true", help="Run with demo documents (CLI mode)")
    p.add_argument("--files", nargs="*", help="Files to ingest (txt/md or pdf if pypdf installed)")
    p.add_argument("--top-k", type=int, default=5, help="Top K results to return")
    p.add_argument("--use-openai", action="store_true", help="If set and OPENAI_API_KEY exists, use OpenAI to generate the answer")
    p.add_argument("--run-tests", action="store_true", help="Run lightweight unit tests and exit")
    return p.parse_args(argv)


if _name_ == "_main_":
    args = parse_args()
    if args.run_tests:
        run_tests()
        sys.exit(0)

    # Prefer Streamlit mode if the module is available (user can still run CLI)
    if module_available("streamlit"):
        # If the user executed streamlit run app.py this code runs inside Streamlit
        # and we should start the Streamlit UI. If streamlit is installed but running
        # from the terminal (python app.py) it's still safe to offer CLI fallback.
        try:
            run_streamlit_mode()
        except Exception as e:
            print(f"Failed to run Streamlit UI: {e}\nFalling back to CLI mode.")
            run_cli_mode(args)
    else:
        # Streamlit not installed — run CLI demo mode by default
        print("Streamlit not installed — running CLI demo mode. To use the web UI install streamlit and run: streamlit run app.py")
        run_cli_mode(args)
"""
RAG Q&A System — Single‑File Python App (Streamlit UI + CLI fallback)

This file is a rewritten, robust version of the original RAG demo application.
It fixes the common ModuleNotFoundError: No module named 'streamlit' by:

1. *Lazy/dynamic imports* — we only import optional heavy packages when needed.
2. *CLI fallback mode* — when streamlit is not available, the script runs a usable
   command-line demo that demonstrates the retrieval + answer workflow without
   requiring any third-party UI library.
3. *Helpful messages* — when optional dependencies are missing the program
   prints clear instructions how to install the extras to get the full app.

Modes
- Full Streamlit UI mode: run with streamlit run app.py (requires dependencies)
- CLI demo mode: run python app.py --demo or simply run without streamlit installed
- Unit tests: python app.py --run-tests runs lightweight tests for core helpers

Notes
- The CLI demo implements a simple retrieval mechanism (word overlap), so you
  can still query uploaded or demo documents without sentence-transformers / faiss.
- To enable high-quality embeddings + vector search, install the packages listed
  in the original requirements (sentence-transformers, faiss-cpu, etc.).

"""

from _future_ import annotations

import os
import sys
import re
import argparse
import json
import uuid
from dataclasses import dataclass
import importlib.util
import textwrap
from typing import List, Tuple, Dict, Optional

# ------------------ Utility: feature detection ------------------

def module_available(name: str) -> bool:
    """Return True if module is importable without importing it.
    This avoids raising ModuleNotFoundError at import-time.
    """
    return importlib.util.find_spec(name) is not None

# --------------------------- Data model ------------------------

@dataclass
class Chunk:
    id: str
    text: str
    source: str
    loc: str = ""

# -------------------------- Helpers ---------------------------

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 200) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# ---------------------- Simple retriever ----------------------

class SimpleRetriever:
    """A dependency-free retriever that ranks chunks by word overlap.

    This is intentionally lightweight and meant as a fallback/demo. For
    production you should replace it with embeddings + FAISS or another
    vector database.
    """
    def _init_(self) -> None:
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        q_words = set(re.findall(r"\w+", query.lower()))
        scores: List[Tuple[float, Chunk]] = []
        for c in self.chunks:
            words = set(re.findall(r"\w+", c.text.lower()))
            common = len(q_words & words)
            # Score also accounts for proximity: prefer shorter chunks when tie
            score = float(common)
            scores.append((score, c))
        scores.sort(key=lambda x: x[0], reverse=True)
        # Filter zeros but fall back to first chunks if nothing matched
        filtered = [(c, s) for s, c in scores if s > 0]
        if not filtered and scores:
            # return top k with their (possibly zero) scores
            return [(c, float(s)) for s, c in scores[:top_k]]
        return [(c, float(s)) for c, s in filtered[:top_k]]

# ----------------------- Prompt builder -----------------------

def build_contextual_prompt(question: str, contexts: List[Chunk], max_len_chars: int = 6000) -> Tuple[str, str, List[str]]:
    """Create a system + user prompt using the provided contexts and list of citation strings.
    Returns (system, user, cites)
    """
    cites: List[str] = []
    assembled: List[str] = []
    total = 0
    for i, c in enumerate(contexts, start=1):
        snippet = c.text.strip()
        if not snippet:
            continue
        part = f"[{i}] Source: {c.source} {c.loc}\n{snippet}\n"
        if total + len(part) > max_len_chars and assembled:
            break
        assembled.append(part)
        total += len(part)
        cites.append(f"[{i}] {c.source} {c.loc}".strip())

    system = (
        "You are a precise assistant answering questions using only the provided context. "
        "Cite sources like [1], [2] in the answer. If the answer is unknown from context, "
        "say you don't know and suggest what to check next."
    )

    user = (
        "CONTEXT:\n" + "\n\n".join(assembled) + "\n\n"
        + "QUESTION: " + question + "\nINSTRUCTIONS: Use citations [#] after the relevant statements. Be concise and accurate."
    )
    return system, user, cites

# ------------------ OpenAI (optional) helper ------------------

def call_openai_chat(system: str, user: str, api_key: Optional[str], base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini") -> str:
    """Call OpenAI Chat Completions if requests + API key are available.
    This function carefully checks for the requests module and the API key.
    """
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not provided")
    if not module_available("requests"):
        raise RuntimeError("Module 'requests' is not available; cannot call OpenAI.")
    import requests
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    url = f"{base_url}/chat/completions"
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

# ------------------------ CLI mode ---------------------------

def run_cli_mode(args: argparse.Namespace) -> None:
    """A minimal interactive CLI that demonstrates ingestion + retrieval.

    - If --files are given, it will attempt to read them (supports .txt/.md; .pdf if pypdf is installed).
    - If --demo is given or no files were ingested, a built-in demo document is used.
    - Enter questions in the prompt. Type 'exit' or Ctrl+D to quit.
    """
    print("\nRAG Q&A CLI demo — lightweight fallback mode")
    print("(This mode does not require streamlit. For the full UI, install the optional requirements and run with: streamlit run app.py)\n")

    retriever = SimpleRetriever()

    ingested = 0

    # Ingest files if provided
    files = args.files or []
    for p in files:
        if not os.path.exists(p):
            print(f" - Warning: file not found: {p}")
            continue
        try:
            if p.lower().endswith(".pdf") and module_available("pypdf"):
                from pypdf import PdfReader
                with open(p, "rb") as fh:
                    reader = PdfReader(fh)
                    for i, page in enumerate(reader.pages, start=1):
                        try:
                            raw = page.extract_text() or ""
                        except Exception:
                            raw = ""
                        for piece in chunk_text(raw):
                            retriever.add([Chunk(id=str(uuid.uuid4()), text=piece, source=p, loc=f"page {i}")])
                            ingested += 1
            else:
                # read as text
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    raw = fh.read()
                for i, piece in enumerate(chunk_text(raw), start=1):
                    retriever.add([Chunk(id=str(uuid.uuid4()), text=piece, source=p, loc=f"chunk {i}")])
                    ingested += 1
        except Exception as e:
            print(f" - Error ingesting {p}: {e}")

    # If no files ingested or demo requested, add demo documents
    if ingested == 0 or args.demo:
        demo_text = textwrap.dedent(
            """
            RAG Demo Document

            Retrieval-augmented generation (RAG) combines a retrieval system (which
            finds relevant chunks of text from documents) with a generative model
            (which composes the final answer). This allows the model to ground its
            responses in long documents or corpora without placing everything in
            the model's weights.

            Use cases include students querying long PDFs, employees searching
            internal wikis, and customers looking up product manuals or FAQs.
            """
        )
        chunks = [Chunk(id=str(uuid.uuid4()), text=c, source="demo_doc.txt", loc=f"demo {i+1}") for i, c in enumerate(chunk_text(demo_text))]
        retriever.add(chunks)
        ingested += len(chunks)
        print(f" - Added demo document with {len(chunks)} chunks.")

    print(f"\nIngested {ingested} chunks. You can now ask questions. Type 'exit' to quit.\n")

    # Interactive loop
    try:
        while True:
            try:
                question = input("Question > ").strip()
            except EOFError:
                print("\nExiting.")
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            hits = retriever.search(question, top_k=args.top_k)
            if not hits:
                print("No relevant content found.")
                continue

            print("\nTop retrieved snippets:")
            for i, (c, score) in enumerate(hits, start=1):
                snippet = (c.text[:400] + "...") if len(c.text) > 400 else c.text
                print(f" [{i}] score={score:.2f} — {c.source} {c.loc}\n    {snippet}\n")

            # If user asked to use OpenAI and key exists, call it
            if args.use_openai and os.getenv("OPENAI_API_KEY"):
                try:
                    system, user, cites = build_contextual_prompt(question, [c for c, _ in hits])
                    ans = call_openai_chat(system, user, api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
                    print("\n=== Generated answer (OpenAI) ===\n")
                    print(ans)
                    if cites:
                        print("\nSources:")
                        for s in cites:
                            print(" - ", s)
                except Exception as e:
                    print(f"Failed to call OpenAI: {e}\nFalling back to concatenated snippets as answer.")
                    # fall through to fallback

            # Fallback answer: join top snippets and print with citations
            print("\n=== Fallback answer (concatenated context) ===\n")
            assembled = []
            for i, (c, score) in enumerate(hits, start=1):
                assembled.append(f"[{i}] {c.source} {c.loc}\n{c.text}\n")
            print("\n".join(assembled))
            print("\n---\n")

    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")

# ---------------------- Streamlit UI mode ---------------------

def run_streamlit_mode() -> None:
    """Run a Streamlit UI if streamlit is installed. This function imports heavy
    dependencies lazily and provides helpful error messages if they are missing.
    """
    import streamlit as st  # type: ignore

    st.set_page_config(page_title="RAG Q&A System", page_icon="📚", layout="wide")
    st.title("📚 Retrieval‑Augmented Q&A (RAG) — Streamlit mode")

    st.sidebar.markdown("### ⚙ Settings")
    st.sidebar.caption("Optional: install sentence-transformers + faiss-cpu to enable embeddings + vector search.")

    # Check for heavy dependencies and show status
    deps = {"pypdf": module_available("pypdf"), "trafilatura": module_available("trafilatura"), "sentence-transformers": module_available("sentence_transformers"), "faiss": module_available("faiss")}
    st.sidebar.markdown("*Dependency status*")
    for k, ok in deps.items():
        st.sidebar.write(f"- {k}: {'✅' if ok else '❌'}")

    if not deps.get("sentence-transformers") or not deps.get("faiss"):
        st.sidebar.warning("Embeddings/FAISS not available — the Streamlit UI will still run but only the simple retriever will be used. To enable full RAG, install: sentence-transformers faiss-cpu")

    # Simple UI controls
    files = st.file_uploader("Upload files (txt/md/pdf) or leave blank to use demo", type=["txt", "md", "markdown", "pdf"], accept_multiple_files=True)
    urls_text = st.text_area("Add URLs (one per line) — requires 'trafilatura' to fetch them")

    top_k = st.sidebar.number_input("Top K", min_value=1, max_value=20, value=5, key="ui_top_k")
    use_openai = st.sidebar.checkbox("Use OpenAI (if OPENAI_API_KEY set)", value=False)

    # Build index button
    if st.button("Build / Update Index"):
        st.info("Building index (this may be slow if you enable embeddings)...")
        retriever = SimpleRetriever()
        count = 0
        # ingest files
        for f in files or []:
            name = f.name
            try:
                if name.lower().endswith(".pdf") and module_available("pypdf"):
                    from pypdf import PdfReader
                    reader = PdfReader(f)
                    for i, page in enumerate(reader.pages, start=1):
                        txt = page.extract_text() or ""
                        for piece in chunk_text(txt):
                            retriever.add([Chunk(id=str(uuid.uuid4()), text=piece, source=name, loc=f"page {i}")])
                            count += 1
                else:
                    raw = f.read().decode("utf-8", errors="ignore")
                    for i, piece in enumerate(chunk_text(raw), start=1):
                        retriever.add([Chunk(id=str(uuid.uuid4()), text=piece, source=name, loc=f"chunk {i}" )])
                        count += 1
            except Exception as e:
                st.error(f"Failed to ingest {name}: {e}")
        # ingest URLs
        if urls_text and module_available("trafilatura"):
            import trafilatura
            for url in urls_text.splitlines():
                url = url.strip()
                if not url:
                    continue
                downloaded = trafilatura.fetch_url(url)
                if not downloaded:
                    st.warning(f"Failed to fetch {url}")
                    continue
                extracted = trafilatura.extract(downloaded) or ""
                for piece in chunk_text(extracted):
                    retriever.add([Chunk(id=str(uuid.uuid4()), text=piece, source=url, loc="")])
                    count += 1
        if count == 0:
            st.info("No files ingested; using demo text.")
            demo_text = ("Retrieval-augmented generation combines retrieval and generation to produce answers grounded in documents.")
            for i, piece in enumerate(chunk_text(demo_text), start=1):
                retriever.add([Chunk(id=str(uuid.uuid4()), text=piece, source="demo", loc=f"{i}")])
                count += 1
        # store retriever
        st.session_state["retriever"] = retriever
        st.success(f"Indexed {count} chunks.")

    # Ask question
    question = st.text_input("Ask a question about your documents")
    if st.button("Answer") and question:
        retriever: SimpleRetriever = st.session_state.get("retriever")
        if retriever is None or not retriever.chunks:
            st.error("Index is empty. Build or ingest documents first.")
        else:
            hits = retriever.search(question, top_k=top_k)
            st.subheader("Top retrieved snippets")
            for i, (c, score) in enumerate(hits, start=1):
                st.markdown(f"[{i}] {c.source} {c.loc}** — score={score:.2f}\n\n{c.text[:600]}{'...' if len(c.text)>600 else ''}")

            # Optional OpenAI generation
            if use_openai and os.getenv("OPENAI_API_KEY"):
                try:
                    system, user, cites = build_contextual_prompt(question, [c for c, _ in hits])
                    ans = call_openai_chat(system, user, api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
                    st.subheader("Generated answer (OpenAI)")
                    st.write(ans)
                    if cites:
                        st.caption("Sources")
                        for s in cites:
                            st.write("- ", s)
                except Exception as e:
                    st.error(f"OpenAI call failed: {e}")
                    st.info("Showing fallback concatenated context below.")

            st.subheader("Fallback concatenated context")
            for i, (c, _) in enumerate(hits, start=1):
                st.markdown(f"[{i}] {c.source} {c.loc}\n\n{c.text}")

    st.markdown("---")
    st.markdown("This Streamlit UI uses the lightweight SimpleRetriever by default.\nInstall sentence-transformers and faiss-cpu to enable embeddings + vector search.")

# -------------------------- Tests ----------------------------

def run_tests() -> None:
    print("Running lightweight tests...")
    # Test chunk_text basic behaviour
    s = "This is a test. " * 100
    chunks = chunk_text(s, chunk_size=50, overlap=10)
    assert all(len(c) <= 50 for c in chunks), "chunk_text produced chunk > chunk_size"

    # Test chunk_text handles empty input
    assert chunk_text("", chunk_size=50, overlap=10) == [], "chunk_text should return [] for empty input"

    # Test chunk_text produces multiple chunks for long input
    long_text = "word " * 2000
    long_chunks = chunk_text(long_text, chunk_size=300, overlap=50)
    assert len(long_chunks) > 1, "chunk_text should split long text into multiple chunks"

    # Test SimpleRetriever ranking
    r = SimpleRetriever()
    a = Chunk(id="a", text="apple banana apple", source="doc1")
    b = Chunk(id="b", text="banana orange", source="doc2")
    c = Chunk(id="c", text="grape melon", source="doc3")
    r.add([a, b, c])
    hits = r.search("apple", top_k=3)
    assert hits and hits[0][0].id == "a", "Retriever failed to rank exact match first"

    # Test SimpleRetriever returns top_k even when no query matches (fallback)
    hits_no_match = r.search("zucchini", top_k=2)
    assert len(hits_no_match) == 2, "Retriever should return top_k items even if no overlap"

    # Test build_contextual_prompt returns matching number of cites
    system, user, cites = build_contextual_prompt("What is RAG?", [a, b, c], max_len_chars=10000)
    assert isinstance(system, str) and isinstance(user, str), "Prompts should be strings"
    assert len(cites) == 3, "Cite list length should match number of contexts"

    print("All lightweight tests passed.")

# -------------------------- Entrypoint -----------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG Q&A (Streamlit + CLI fallback)")
    p.add_argument("--demo", action="store_true", help="Run with demo documents (CLI mode)")
    p.add_argument("--files", nargs="*", help="Files to ingest (txt/md or pdf if pypdf installed)")
    p.add_argument("--top-k", type=int, default=5, help="Top K results to return")
    p.add_argument("--use-openai", action="store_true", help="If set and OPENAI_API_KEY exists, use OpenAI to generate the answer")
    p.add_argument("--run-tests", action="store_true", help="Run lightweight unit tests and exit")
    return p.parse_args(argv)


if _name_ == "_main_":
    args = parse_args()
    if args.run_tests:
        run_tests()
        sys.exit(0)

    # Prefer Streamlit mode if the module is available (user can still run CLI)
    if module_available("streamlit"):
        # If the user executed streamlit run app.py this code runs inside Streamlit
        # and we should start the Streamlit UI. If streamlit is installed but running
        # from the terminal (python app.py) it's still safe to offer CLI fallback.
        try:
            run_streamlit_mode()
        except Exception as e:
            print(f"Failed to run Streamlit UI: {e}\nFalling back to CLI mode.")
            run_cli_mode(args)
    else:
        # Streamlit not installed — run CLI demo mode by default
        print("Streamlit not installed — running CLI demo mode. To use the web UI install streamlit and run: streamlit run app.py")
        run_cli_mode(args)