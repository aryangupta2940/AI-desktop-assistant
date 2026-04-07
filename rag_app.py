"""
RAG Q&A single-file app
- Save as rag_app.py
- Run CLI demo:   python rag_app.py --demo
- Run Streamlit:  pip install streamlit  then: streamlit run rag_app.py
- Run tests:      python rag_app.py --run-tests

Optional dependencies for full features:
  pip install sentence-transformers faiss-cpu pypdf trafilatura beautifulsoup4 requests scikit-learn openai

Environment variables (optional):
  OPENAI_API_KEY  - if set, OpenAI embeddings + chat are used
  OPENAI_BASE_URL - override base URL (defaults to https://api.openai.com/v1)
  OPENAI_EMBED_MODEL - embedding model (default: text-embedding-3-large or text-embedding-ada-002 if older)
  OPENAI_CHAT_MODEL - chat model (default: gpt-4o-mini or gpt-4)

This app uses lazy imports: missing optional libs won't crash import time.
"""

from __future__ import annotations
import os
import sys
import re
import json
import uuid
import argparse
import textwrap
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import importlib.util
import math

# Utility: check availability of optional modules without importing them
def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

# Basic chunk dataclass
@dataclass
class Chunk:
    id: str
    text: str
    source: str
    loc: str = ""

# -------------------- Text extraction helpers --------------------
def extract_text_from_pdf_bytes(b: bytes) -> str:
    """Extract text from PDF bytes. Uses pypdf if available, otherwise returns ''."""
    if not has_module("pypdf"):
        return ""
    from pypdf import PdfReader
    from io import BytesIO
    try:
        reader = PdfReader(BytesIO(b))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML using readability via trafilatura or BeautifulSoup fallback."""
    # Prefer trafilatura if present
    if has_module("trafilatura"):
        import trafilatura
        extracted = trafilatura.extract(html)
        if extracted:
            return extracted
    # Fallback to beautifulsoup stripping tags
    if has_module("bs4"):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        # remove scripts/styles
        for s in soup(["script", "style", "noscript"]):
            s.extract()
        text = soup.get_text(separator="\n")
        return text
    # Last resort: basic tag removal
    return re.sub(r"<[^>]+>", "", html)

# -------------------- Chunking logic --------------------
# We want approx 300-500 tokens per chunk. Token estimation: 1 token ~ 4 chars (approx)
# So target chars_per_chunk = tokens * 4. Use adjustable parameter.
DEFAULT_TOKENS_PER_CHUNK = 400
DEFAULT_CHUNK_OVERLAP_TOKENS = 80

def chars_per_token_estimate() -> float:
    # Conservative average
    return 4.0

def chunk_text(text: str, tokens_per_chunk: int = DEFAULT_TOKENS_PER_CHUNK,
               overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    cpt = chars_per_token_estimate()
    chunk_size = max(200, int(tokens_per_chunk * cpt))
    overlap = int(overlap_tokens * cpt)
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

# -------------------- Embeddings --------------------
class EmbeddingBackend:
    """Wrapper supporting OpenAI embeddings (if key present) or local sentence-transformers."""
    def __init__(self, openai_key: Optional[str] = None, local_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
        self.local_model_name = local_model_name
        self._local = None
        self._can_use_openai = bool(self.openai_key) and has_module("requests")
        self._can_use_local = has_module("sentence_transformers")
        if self._can_use_local:
            # lazy import wrapper
            from sentence_transformers import SentenceTransformer
            self._local = SentenceTransformer(self.local_model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._can_use_openai:
            # batch call to OpenAI embeddings
            return self._embed_openai(texts)
        if self._can_use_local and self._local is not None:
            return self._local.encode(texts, show_progress_bar=False).tolist()
        # fallback: simple TF-like vector from word counts (very weak but works offline)
        return [self._simple_count_vector(t) for t in texts]

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        import requests
        url = f"{self.openai_base}/embeddings"
        headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
        payload = {"model": self.openai_model, "input": texts}
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        embs = [item["embedding"] for item in data["data"]]
        return embs

    def _simple_count_vector(self, text: str) -> List[float]:
        words = re.findall(r"\w+", text.lower())
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        # deterministic ordering: sorted keys slice (not ideal but ok for fallback)
        keys = sorted(list(freq.keys()))[:256]
        vec = [float(freq.get(k, 0)) for k in keys]
        return vec

# -------------------- Vector store (FAISS preferred) --------------------
class VectorStore:
    """Vector store abstraction: attempts FAISS, otherwise sklearn-based, otherwise simple list with dot-product."""
    def __init__(self, embedder: EmbeddingBackend):
        self.embedder = embedder
        self.ids: List[str] = []
        self.metad: Dict[str, Chunk] = {}
        self.embeddings = None
        # detect faiss
        self.use_faiss = has_module("faiss")
        self.use_sklearn = has_module("sklearn")
        if self.use_faiss:
            import faiss
            self.faiss = faiss
            self.index = None
            self.dim = None
        else:
            self.index = None
            self.faiss = None

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embs = self.embedder.embed(texts)
        # normalize if using faiss IndexFlatIP
        if self.use_faiss:
            import numpy as np
            mat = np.array(embs, dtype="float32")
            # first-time index creation
            if self.index is None:
                self.dim = mat.shape[1]
                self.index = self.faiss.IndexFlatIP(self.dim)
            # l2norm -> inner product approximates cosine when normalized
            # normalize
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
            self.index.add(mat)
            # store ids and metadata
            for c in chunks:
                key = str(len(self.ids)) + "::" + c.id
                self.ids.append(key)
                self.metad[key] = c
        else:
            # fallback store: append to lists
            if self.embeddings is None:
                self.embeddings = []
            for c, emb in zip(chunks, embs):
                key = str(len(self.ids)) + "::" + c.id
                self.ids.append(key)
                self.metad[key] = c
                # normalize emb for cosine
                vec = emb
                norm = math.sqrt(sum([x*x for x in vec])) or 1.0
                self.embeddings.append([x / norm for x in vec])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if not self.ids:
            return []
        q_emb = self.embedder.embed([query])[0]
        # normalize q
        norm = math.sqrt(sum(x*x for x in q_emb)) or 1.0
        q_norm = [x / norm for x in q_emb]
        if self.use_faiss and self.index is not None:
            import numpy as np
            qv = np.array([q_emb], dtype="float32")
            # normalize
            qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
            D, I = self.index.search(qv, min(top_k, len(self.ids)))
            results = []
            for idx, score in zip(I[0], D[0]):
                if idx < 0 or idx >= len(self.ids):
                    continue
                key = self.ids[idx]
                results.append((self.metad[key], float(score)))
            return results
        else:
            # cosine similarity with stored embeddings
            sims = []
            for emb_vec, key in zip(self.embeddings, self.ids):
                # if lengths mismatch (fallback simple vectors) use dot with truncation/padding
                m = min(len(emb_vec), len(q_norm))
                dot = sum(emb_vec[i] * q_norm[i] for i in range(m))
                sims.append((dot, key))
            sims.sort(key=lambda x: x[0], reverse=True)
            out = []
            for sim, key in sims[:top_k]:
                out.append((self.metad[key], float(sim)))
            return out

# -------------------- RAG / LLM call --------------------
def build_prompt(question: str, contexts: List[Chunk], max_chars: int = 6000) -> Tuple[str, str, List[str]]:
    cites = []
    assembled = []
    total = 0
    for i, c in enumerate(contexts, start=1):
        part = f"[{i}] Source: {c.source} {c.loc}\n{c.text}\n"
        if total + len(part) > max_chars and assembled:
            break
        assembled.append(part)
        total += len(part)
        cites.append(f"[{i}] {c.source} {c.loc}")
    system = ("You are an assistant answering from the provided context. Use citations like [1] after factual statements.\n"
              "If the answer is not in context, say you don't know and suggest where to check.")
    user = "CONTEXT:\n" + "\n\n".join(assembled) + f"\n\nQUESTION: {question}\nINSTRUCTIONS: Be concise, include citations."
    return system, user, cites

def call_openai_chat(system: str, user: str) -> str:
    if not has_module("requests"):
        raise RuntimeError("requests module required for OpenAI API calls")
    import requests
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    url = f"{base}/chat/completions"
    payload = {"model": model, "messages": [{"role":"system","content":system},{"role":"user","content":user}], "temperature": 0.2}
    r = requests.post(url, headers={"Authorization":f"Bearer {key}", "Content-Type":"application/json"}, json=payload, timeout=120)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()

# -------------------- Ingest helpers --------------------
def ingest_file(path: str, chunk_tokens: int = DEFAULT_TOKENS_PER_CHUNK, overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS) -> List[Chunk]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    name = os.path.basename(path)
    ext = name.lower().split(".")[-1]
    chunks: List[Chunk] = []
    if ext == "pdf":
        with open(path, "rb") as fh:
            raw = extract_text_from_pdf_bytes(fh.read())
        if raw:
            for i, part in enumerate(chunk_text(raw, chunk_tokens, overlap_tokens), start=1):
                chunks.append(Chunk(id=str(uuid.uuid4()), text=part, source=name, loc=f"page-chunk {i}"))
        return chunks
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        if "<html" in content.lower() and has_module("trafilatura"):
            content = extract_text_from_html(content)
        for i, part in enumerate(chunk_text(content, chunk_tokens, overlap_tokens), start=1):
            chunks.append(Chunk(id=str(uuid.uuid4()), text=part, source=name, loc=f"chunk {i}"))
        return chunks

def ingest_url(url: str, chunk_tokens: int = DEFAULT_TOKENS_PER_CHUNK, overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS) -> List[Chunk]:
    if not has_module("trafilatura") and not has_module("requests"):
        raise RuntimeError("trafilatura or requests required to fetch URL content")
    if has_module("trafilatura"):
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        extracted = trafilatura.extract(downloaded) or ""
    else:
        import requests
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        extracted = extract_text_from_html(r.text)
    chunks = [Chunk(id=str(uuid.uuid4()), text=t, source=url, loc="") for t in chunk_text(extracted, chunk_tokens, overlap_tokens)]
    return chunks

# -------------------- Simple UI (CLI) --------------------
def cli_demo(args):
    print("\nRAG CLI demo - minimal. Press Ctrl+C or type 'exit' to quit.")
    embedder = EmbeddingBackend(openai_key=os.getenv("OPENAI_API_KEY"))
    vs = VectorStore(embedder)
    ingested = 0

    # ingest files if given
    for f in (args.files or []):
        try:
            chs = ingest_file(f)
            vs.add_chunks(chs)
            ingested += len(chs)
            print(f" - Ingested {len(chs)} chunks from {f}")
        except Exception as e:
            print(" - Failed to ingest", f, ":", e)

    # ingest demo if nothing
    if ingested == 0 or args.demo:
        demo = textwrap.dedent("""\
            Retrieval-augmented generation (RAG) combines a retrieval system that finds
            relevant document chunks with a generative model that composes answers.
            Use it to ask questions about long PDFs, wikis, or manuals.
            """)
        chs = [Chunk(id=str(uuid.uuid4()), text=t, source="demo", loc=f"part{i}") for i,t in enumerate(chunk_text(demo))]
        vs.add_chunks(chs)
        print(f" - Loaded demo doc with {len(chs)} chunks.")

    print(f"Ingested {len(vs.ids)} chunks in total.\n")

    try:
        while True:
            q = input("Question > ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break
            hits = vs.search(q, top_k=args.top_k)
            if not hits:
                print("No matches.")
                continue
            print("\nTop snippets:")
            for i,(c,score) in enumerate(hits, start=1):
                print(f"[{i}] score={score:.3f} -- {c.source} {c.loc}\n  {c.text[:300]}{'...' if len(c.text)>300 else ''}\n")
            # try OpenAI generation if available
            try:
                system,user,cites = build_prompt(q, [c for c,_ in hits])
                if os.getenv("OPENAI_API_KEY"):
                    ans = call_openai_chat(system, user)
                    print("\n=== Generated answer ===\n")
                    print(ans)
                    if cites:
                        print("\nSources:")
                        for s in cites:
                            print(" -", s)
                else:
                    print("\n(No OPENAI_API_KEY set) Fallback concatenated context:\n")
                    print("\n".join([f"[{i}] {c.text}" for i,(c,_) in enumerate(hits, start=1)]))
            except Exception as e:
                print("Error generating answer:", e)
                print("Fallback concatenated context:")
                print("\n".join([f"[{i}] {c.text}" for i,(c,_) in enumerate(hits, start=1)]))
    except KeyboardInterrupt:
        print("\nGoodbye.")

# -------------------- Streamlit UI (optional) --------------------
def run_streamlit_app():
    if not has_module("streamlit"):
        raise RuntimeError("streamlit not installed. Install and run: streamlit run rag_app.py")
    import streamlit as st
    st.set_page_config(page_title="RAG Q&A", layout="wide")
    st.title("RAG Q&A System")

    st.sidebar.header("Settings")
    top_k = st.sidebar.number_input("Top K", min_value=1, max_value=20, value=5)
    tokens_per_chunk = st.sidebar.number_input("Tokens per chunk (~300-500)", min_value=128, max_value=1024, value=DEFAULT_TOKENS_PER_CHUNK)
    overlap_tokens = st.sidebar.number_input("Overlap tokens", min_value=0, max_value=512, value=DEFAULT_CHUNK_OVERLAP_TOKENS)
    use_openai = st.sidebar.checkbox("Use OpenAI (if OPENAI_API_KEY set)", value=False)

    if "vs" not in st.session_state:
        st.session_state.vs = None

    st.header("Upload or paste content")
    files = st.file_uploader("Files (pdf, md, txt)", accept_multiple_files=True)
    url_input = st.text_area("Or paste URLs (one per line)")

    if st.button("Ingest / Build index"):
        embedder = EmbeddingBackend(openai_key=os.getenv("OPENAI_API_KEY"))
        vs = VectorStore(embedder)
        total = 0
        # files
        for f in files or []:
            name = f.name
            content_bytes = f.read()
            if name.lower().endswith(".pdf"):
                raw = extract_text_from_pdf_bytes(content_bytes)
                if raw:
                    for i, part in enumerate(chunk_text(raw, tokens_per_chunk, overlap_tokens), start=1):
                        vs.add_chunks([Chunk(id=str(uuid.uuid4()), text=part, source=name, loc=f"page-chunk:{i}")])
                        total += 1
            else:
                try:
                    raw = content_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    raw = ""
                for i, part in enumerate(chunk_text(raw, tokens_per_chunk, overlap_tokens), start=1):
                    vs.add_chunks([Chunk(id=str(uuid.uuid4()), text=part, source=name, loc=f"chunk:{i}")])
                    total += 1
        # urls
        for u in (url_input or "").splitlines():
            u = u.strip()
            if not u:
                continue
            try:
                chs = ingest_url(u, chunk_tokens=tokens_per_chunk, overlap_tokens=overlap_tokens)
                vs.add_chunks(chs)
                total += len(chs)
            except Exception as e:
                st.warning(f"Failed to ingest {u}: {e}")
        if total == 0:
            st.info("No content ingested. Loading demo content.")
            demo = "Retrieval-augmented generation (RAG) demo text."
            chs = [Chunk(id=str(uuid.uuid4()), text=t, source="demo", loc=f"{i}") for i,t in enumerate(chunk_text(demo, tokens_per_chunk, overlap_tokens))]
            vs.add_chunks(chs)
            total += len(chs)
        st.session_state.vs = vs
        st.success(f"Indexed {total} chunks.")

    st.header("Ask a question")
    q = st.text_input("Question")
    if st.button("Answer") and q:
        vs: VectorStore = st.session_state.get("vs")
        if not vs:
            st.error("No index. Ingest documents first.")
        else:
            hits = vs.search(q, top_k=top_k)
            st.subheader("Top Retrieved")
            for i, (c, score) in enumerate(hits, start=1):
                st.markdown(f"**[{i}] {c.source} {c.loc}** — score={score:.3f}\n\n{c.text[:800]}{'...' if len(c.text)>800 else ''}")
            # generate if desired
            if use_openai and os.getenv("OPENAI_API_KEY"):
                try:
                    system,user,cites = build_prompt(q, [c for c,_ in hits])
                    ans = call_openai_chat(system,user)
                    st.subheader("Generated Answer")
                    st.write(ans)
                    if cites:
                        st.caption("Sources:")
                        for s in cites:
                            st.write("- ", s)
                except Exception as e:
                    st.error(f"OpenAI call failed: {e}")
                    st.info("Showing concatenated context below.")
            st.subheader("Fallback concatenated context")
            for i,(c,_) in enumerate(hits, start=1):
                st.markdown(f"**[{i}] {c.source} {c.loc}**\n\n{c.text}")

# -------------------- Tests --------------------
def run_tests():
    print("Running tests...")
    # chunk_text sanity
    s = "This is a test. " * 100
    chunks = chunk_text(s, tokens_per_chunk=50, overlap_tokens=10)
    assert all(len(c) <= int(50 * chars_per_token_estimate()) for c in chunks), "chunk longer than expected"
    assert chunk_text("", tokens_per_chunk=50) == [], "empty input should give empty list"

    # Simple ingest + retrieval cycle using fallback (no faiss)
    embedder = EmbeddingBackend(openai_key=None)
    vs = VectorStore(embedder)
    a = Chunk(id="a", text="supervised learning uses labeled data and examples", source="doc1", loc="page 1")
    b = Chunk(id="b", text="unsupervised learning groups unlabeled data", source="doc2", loc="page 2")
    c = Chunk(id="c", text="reinforcement learning uses rewards", source="doc3", loc="page 3")
    vs.add_chunks([a, b, c])
    hits = vs.search("difference between supervised and unsupervised", top_k=2)
    assert hits, "search should return results"
    # ensure at least one of top results mentions supervised or unsupervised
    texts = " ".join([h[0].text.lower() for h in hits])
    assert "supervised" in texts or "unsupervised" in texts, "expected keyword in retrieved text"

    # prompt builder
    system,user,cites = build_prompt("What is supervised learning?", [a,b,c])
    assert isinstance(system,str) and isinstance(user,str) and isinstance(cites,list)

    print("All tests passed.")

# -------------------- CLI arg parsing & entrypoint --------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="Use demo doc")
    p.add_argument("--files", nargs="*", help="Files to ingest")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--run-tests", action="store_true")
    p.add_argument("--streamlit", action="store_true", help="Run streamlit app (if module available)")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args()
    if args.run_tests:
        run_tests()
        sys.exit(0)
    # prefer streamlit UI when requested and installed
    if args.streamlit or has_module("streamlit"):
        try:
            # when executed via `streamlit run rag_app.py`, Streamlit runs this module differently,
            # but running streamlit programmatically works for convenience.
            if has_module("streamlit"):
                run_streamlit_app()
                sys.exit(0)
        except Exception as e:
            print("Failed to start streamlit UI:", e)
    # fallback to CLI demo
    cli_demo(args)
