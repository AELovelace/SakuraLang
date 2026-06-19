# -*- coding: utf-8 -*-
"""
rag.py — SakuraLang's local Hybrid RAG agent  ✿

This module is the "research brain helper" for our third AI (Qwen). It indexes the
files in the agent's working directory and answers retrieval queries using a
*Hybrid RAG* pipeline. Hybrid means we blend two very different ways of searching:

    • DENSE  (vector / embeddings)  → great at *concepts* and fuzzy meaning
    • SPARSE (BM25 keyword search)   → great at *exact* things: model names,
                                       error codes, terminal IDs, PLUs, IPs, configs

Neither alone is enough for messy IT docs and logs, so we fuse them together.

The full pipeline (the part that lives here) is:

    ensure_indexed(cwd)                      # one-time: read + chunk + embed files
        │
    retrieve(query, cwd)
        │
        ├─ hybrid retrieval  (dense + BM25, fused with RRF) → top ~20 small chunks
        ├─ rerank            (cross-encoder) → keep the best 5
        ├─ small-to-big      (swap each tiny matched chunk for its parent section)
        └─ compress          (greedily pack parents into a token budget)
                              → returns clean context text for Qwen

The query-rewriting and final answer synthesis steps live in main.py's `rag` and
`respond` graph nodes, because those need the Qwen LLM client. This file is purely
the indexing + retrieval machinery so it stays easy to test on its own.

Design note: all the *heavy* imports (torch / sentence-transformers / qdrant) are
done lazily inside methods, NOT at module top level. That keeps `import rag` cheap
so the curses app starts instantly — we only pay the model-loading cost the first
time RAG is actually used.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading

# ---------------------------------------------------------------------------
# Configuration constants — tweak these to change models / budgets
# ---------------------------------------------------------------------------

# Dense embedding model (small + fast + good quality for English IT text).
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
# Cross-encoder used to rerank the fused candidates. Small & quick on CPU.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INDEX_DIRNAME = ".sakura_rag"   # per-workspace folder that holds the persisted index
COLLECTION    = "sakura_docs"   # Qdrant collection name

RETRIEVE_K           = 20       # "retrieve bigger than you need" — top 20 candidates
RERANK_TOP_N         = 5        # "...then shrink" — rerank down to the best 5
CONTEXT_TOKEN_BUDGET = 1500     # compress final context to <= ~1500 tokens
CHARS_PER_TOKEN      = 4        # rough char→token estimate (good enough for budgeting)

MAX_FILE_BYTES = 5 * 1024 * 1024  # skip files larger than 5 MiB to stay responsive

# Directories we never descend into (noise / binaries / our own index).
SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    "env", "dist", "build", ".idea", ".mypy_cache", ".pytest_cache",
    INDEX_DIRNAME,
}

# File extensions we treat as code, mapped to a splitter language where possible.
# (We fill the Language enum lazily in _code_language() to avoid an early import.)
CODE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cc", ".cs", ".go", ".rs", ".rb", ".php", ".kt", ".swift", ".scala",
    ".sh", ".ps1", ".lua", ".pl", ".sql",
}
CONFIG_EXTS = {
    ".json", ".yaml", ".yml", ".toml", ".ini", ".conf", ".cfg", ".env",
    ".xml", ".properties",
}

# Per-document-type chunking. Smaller "child" chunks are what we actually search
# (precise), while the bigger "parent" chunk is what we hand to the model
# (preserves surrounding context). Char sizes are ~4× the token target.
CHUNK_PARAMS = {
    # Vendor manuals / docs / markdown: 500–900 tokens → ~2000–3600 chars.
    "doc":    {"parent": 3600, "child": 900, "overlap": 150},
    # Code: roughly function/class sized chunks.
    "code":   {"parent": 2400, "child": 700, "overlap": 100},
    # Config files: one section/object-ish per chunk.
    "config": {"parent": 2000, "child": 600, "overlap": 80},
    # Logs: chunk by lines, not characters (50–200 lines per parent).
    "log":    {"parent_lines": 120, "child_lines": 30},
    # Inventory / CSV: one device/entity (row group) per chunk, header repeated.
    "csv":    {"parent_rows": 50, "child_rows": 10},
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _classify(path: str) -> str:
    """Decide how to chunk a file based on its name/extension."""
    ext  = os.path.splitext(path)[1].lower()
    name = os.path.basename(path).lower()
    if ext == ".log" or "log" in name:
        return "log"
    if ext in (".csv", ".tsv"):
        return "csv"
    if ext in CONFIG_EXTS:
        return "config"
    if ext in CODE_EXTS:
        return "code"
    return "doc"  # markdown, txt, rst, README, or anything else that's text


def _code_language(ext: str):
    """Map a code extension to a LangChain splitter Language, or None if unknown.

    Imported lazily so we don't pull in langchain_text_splitters until indexing."""
    from langchain_text_splitters import Language
    table = {
        ".py": Language.PYTHON, ".js": Language.JS, ".jsx": Language.JS,
        ".ts": Language.TS, ".tsx": Language.TS, ".java": Language.JAVA,
        ".c": Language.C, ".h": Language.C, ".cpp": Language.CPP,
        ".hpp": Language.CPP, ".cc": Language.CPP, ".cs": Language.CSHARP,
        ".go": Language.GO, ".rs": Language.RUST, ".rb": Language.RUBY,
        ".php": Language.PHP, ".kt": Language.KOTLIN, ".swift": Language.SWIFT,
        ".scala": Language.SCALA,
    }
    # Extensions without a dedicated Language (e.g. .sql, .sh) fall back to None
    # and use the plain recursive splitter instead.
    return table.get(ext)


def _read_text(path: str) -> str | None:
    """Read a file as UTF-8 text, or return None if it's binary / unreadable / huge."""
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            return None
        with open(path, "rb") as f:
            head = f.read(1024)
        if b"\x00" in head:          # null byte → almost certainly binary
            return None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return None


def _est_tokens(text: str) -> int:
    """Rough token estimate from character count."""
    return max(1, len(text) // CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Chunking — produce (parent_text, [child_texts]) pairs per file
# ---------------------------------------------------------------------------

def _chunk_text_type(text: str, doc_type: str, ext: str) -> list[tuple[str, list[str]]]:
    """Character-based small-to-big chunking for doc / code / config files."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    p = CHUNK_PARAMS[doc_type]

    # Build parent + child splitters. For code we use language-aware separators
    # so we cut on sensible boundaries (functions/classes) when we can.
    lang = _code_language(ext) if doc_type == "code" else None
    if lang is not None:
        parent_splitter = RecursiveCharacterTextSplitter.from_language(
            lang, chunk_size=p["parent"], chunk_overlap=0)
        child_splitter = RecursiveCharacterTextSplitter.from_language(
            lang, chunk_size=p["child"], chunk_overlap=p["overlap"])
    else:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=p["parent"], chunk_overlap=0)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=p["child"], chunk_overlap=p["overlap"])

    pairs = []
    for parent_text in parent_splitter.split_text(text):
        children = child_splitter.split_text(parent_text)
        if children:
            pairs.append((parent_text, children))
    return pairs


def _chunk_lines(text: str, parent_lines: int, child_lines: int,
                 header: str = "") -> list[tuple[str, list[str]]]:
    """Line-based small-to-big chunking, used for logs and CSV/inventory files.

    `header` (e.g. a CSV header row) is prepended to every chunk so each piece
    stays self-describing when it's retrieved in isolation."""
    lines = text.splitlines()
    pairs = []
    for i in range(0, len(lines), parent_lines):
        block = lines[i:i + parent_lines]
        if not block:
            continue
        parent_text = ((header + "\n") if header else "") + "\n".join(block)
        children = []
        for j in range(0, len(block), child_lines):
            sub = block[j:j + child_lines]
            child_text = ((header + "\n") if header else "") + "\n".join(sub)
            children.append(child_text)
        pairs.append((parent_text, children))
    return pairs


def _chunk_file(text: str, doc_type: str, ext: str) -> list[tuple[str, list[str]]]:
    """Dispatch to the right chunker for this document type."""
    if doc_type == "log":
        p = CHUNK_PARAMS["log"]
        return _chunk_lines(text, p["parent_lines"], p["child_lines"])
    if doc_type == "csv":
        p = CHUNK_PARAMS["csv"]
        rows = text.splitlines()
        header = rows[0] if rows else ""
        body = "\n".join(rows[1:]) if len(rows) > 1 else ""
        # Reuse the line chunker on the body, repeating the header into each chunk.
        return _chunk_lines(body, p["parent_rows"], p["child_rows"], header=header)
    return _chunk_text_type(text, doc_type, ext)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion — blend dense + sparse result lists
# ---------------------------------------------------------------------------

def _rrf_fuse(ranked_lists: list[list], k: int = 60, top: int = RETRIEVE_K) -> list:
    """Fuse several ranked Document lists using Reciprocal Rank Fusion (RRF).

    RRF score for a doc = sum over lists of 1 / (k + rank). It rewards docs that
    rank highly in *either* retriever, which is exactly what we want for hybrid
    dense+sparse search. We identify "the same doc" by its page_content."""
    scores: dict[str, float] = {}
    keep: dict[str, object] = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs):
            key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            keep.setdefault(key, doc)
    ordered = sorted(scores, key=lambda kk: scores[kk], reverse=True)
    return [keep[key] for key in ordered[:top]]


# ---------------------------------------------------------------------------
# The engine — one per working directory
# ---------------------------------------------------------------------------

class _Engine:
    """Holds the persisted hybrid index for a single working directory."""

    def __init__(self, cwd: str):
        self.cwd        = os.path.abspath(cwd)
        self.index_dir  = os.path.join(self.cwd, INDEX_DIRNAME)
        self.qdrant_dir = os.path.join(self.index_dir, "qdrant")
        self._lock      = threading.RLock()

        # Lazily-created heavy objects.
        self.embeddings  = None     # HuggingFaceEmbeddings
        self.reranker    = None     # CrossEncoderReranker
        self.client      = None     # QdrantClient (local/on-disk)
        self.vectorstore = None     # QdrantVectorStore
        self.bm25        = None     # BM25Retriever

        # Data.
        self.children: list = []            # list[Document] (small chunks)
        self.parents: dict[str, dict] = {}  # parent_id -> {"text":.., "source":..}
        self.signature: str = ""            # fingerprint of the corpus on disk
        self.built = False

    # -- model loading ---------------------------------------------------

    def _load_models(self):
        """Load the embedding + reranker models (first use only)."""
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        # CrossEncoderReranker moved to langchain_classic in LangChain 1.x; fall back
        # to the legacy location for older installs.
        try:
            from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        except ImportError:
            from langchain.retrievers.document_compressors import CrossEncoderReranker

        if self.embeddings is None:
            # normalize_embeddings=True pairs well with cosine distance.
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                encode_kwargs={"normalize_embeddings": True},
            )
        if self.reranker is None:
            ce = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
            self.reranker = CrossEncoderReranker(model=ce, top_n=RERANK_TOP_N)

    # -- corpus scanning -------------------------------------------------

    def _scan(self) -> tuple[list[str], str]:
        """Walk the working dir, returning text file paths + a corpus signature.

        The signature is a hash of (relpath, mtime, size) for every file, so we
        can cheaply detect when the corpus changed and needs reindexing."""
        files: list[str] = []
        sig_parts: list[str] = []
        for root, dirs, names in os.walk(self.cwd):
            # Prune skip-dirs and hidden dirs in place so os.walk won't descend.
            dirs[:] = [d for d in dirs
                       if d not in SKIP_DIRS and not d.startswith(".")]
            for name in names:
                path = os.path.join(root, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size > MAX_FILE_BYTES:
                    continue
                rel = os.path.relpath(path, self.cwd)
                files.append(path)
                sig_parts.append(f"{rel}|{int(st.st_mtime)}|{st.st_size}")
        signature = hashlib.md5("\n".join(sorted(sig_parts)).encode()).hexdigest()
        return files, signature

    # -- build / load ----------------------------------------------------

    def ensure(self) -> dict:
        """Make sure a fresh index exists for the current corpus. Returns stats."""
        with self._lock:
            files, signature = self._scan()

            # Already built this exact corpus in-memory → nothing to do.
            if self.built and signature == self.signature:
                return {"status": "cached", "files": len(files),
                        "chunks": len(self.children)}

            # Try to load a persisted index whose signature still matches.
            if self._try_load(signature):
                return {"status": "loaded", "files": len(files),
                        "chunks": len(self.children)}

            # Otherwise (re)build from scratch.
            return self._build(files, signature)

    def _meta_path(self) -> str:
        return os.path.join(self.index_dir, "meta.json")

    def _try_load(self, signature: str) -> bool:
        """Load a persisted index from disk if it matches the current signature."""
        meta_path = self._meta_path()
        if not (os.path.isfile(meta_path) and os.path.isdir(self.qdrant_dir)):
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("signature") != signature:
                return False  # corpus changed since last index → force rebuild

            from langchain_core.documents import Document
            self.parents = meta.get("parents", {})
            self.children = [
                Document(page_content=c["page_content"], metadata=c["metadata"])
                for c in meta.get("children", [])
            ]
            if not self.children:
                return False

            self._load_models()
            self._open_vectorstore(create=False)
            self._build_bm25()
            self.signature = signature
            self.built = True
            return True
        except Exception:
            return False

    def _build(self, files: list[str], signature: str) -> dict:
        """Read + chunk every file, embed children into Qdrant, persist to disk."""
        from langchain_core.documents import Document

        self._load_models()

        children: list = []
        parents: dict[str, dict] = {}

        for path in files:
            text = _read_text(path)
            if not text or not text.strip():
                continue
            rel       = os.path.relpath(path, self.cwd)
            doc_type  = _classify(path)
            ext       = os.path.splitext(path)[1].lower()
            try:
                pairs = _chunk_file(text, doc_type, ext)
            except Exception:
                continue

            for pi, (parent_text, child_texts) in enumerate(pairs):
                parent_id = f"{rel}#p{pi}"
                parents[parent_id] = {"text": parent_text, "source": rel}
                for ci, child_text in enumerate(child_texts):
                    if not child_text.strip():
                        continue
                    children.append(Document(
                        page_content=child_text,
                        metadata={
                            "source": rel,
                            "doc_type": doc_type,
                            "parent_id": parent_id,
                            "child": f"{pi}.{ci}",
                        },
                    ))

        self.children  = children
        self.parents   = parents
        self.signature = signature

        # Build the dense (Qdrant) index, then the sparse (BM25) index.
        os.makedirs(self.index_dir, exist_ok=True)
        self._open_vectorstore(create=True)
        if children:
            self.vectorstore.add_documents(children)
        self._build_bm25()

        self._persist(signature)
        self.built = True
        return {"status": "built", "files": len(files),
                "chunks": len(children), "parents": len(parents)}

    def _open_vectorstore(self, create: bool):
        """Open (or create) the on-disk Qdrant collection in local path mode."""
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        # Re-open the local client. Only one client may hold the path at a time,
        # so close any previous one before reopening (matters on rebuild).
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        os.makedirs(self.qdrant_dir, exist_ok=True)
        self.client = QdrantClient(path=self.qdrant_dir)

        if create:
            dim = len(self.embeddings.embed_query("dimension probe"))
            if self.client.collection_exists(COLLECTION):
                self.client.delete_collection(COLLECTION)
            self.client.create_collection(
                COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION,
            embedding=self.embeddings,
        )

    def _build_bm25(self):
        """(Re)build the in-memory BM25 keyword retriever from the child chunks."""
        from langchain_community.retrievers import BM25Retriever
        if self.children:
            self.bm25 = BM25Retriever.from_documents(self.children)
            self.bm25.k = RETRIEVE_K
        else:
            self.bm25 = None

    def _persist(self, signature: str):
        """Save children + parents + signature so we can reload without re-embedding."""
        meta = {
            "signature": signature,
            "parents": self.parents,
            "children": [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in self.children
            ],
        }
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f)

    # -- retrieval -------------------------------------------------------

    def retrieve(self, query: str) -> tuple[str, list[str]]:
        """Run the hybrid retrieve → rerank → small-to-big → compress pipeline.

        Returns (context_text, sources). context_text is empty when nothing
        useful was found (caller should then fall back to a normal answer)."""
        with self._lock:
            if not self.built or not self.children:
                return "", []
            self._load_models()

            # 1) HYBRID RETRIEVAL — dense + sparse, fused with RRF.
            dense_docs = self.vectorstore.similarity_search(query, k=RETRIEVE_K)
            sparse_docs = self.bm25.invoke(query) if self.bm25 else []
            fused = _rrf_fuse([dense_docs, sparse_docs], top=RETRIEVE_K)
            if not fused:
                return "", []

            # 2) RERANK — cross-encoder scores query↔chunk pairs, keep best N.
            try:
                reranked = self.reranker.compress_documents(fused, query)
            except Exception:
                reranked = fused[:RERANK_TOP_N]

            # 3) SMALL-TO-BIG — replace each tiny matched chunk with its parent
            #    section, preserving rank order and dropping duplicate parents.
            seen: set[str] = set()
            picked: list[tuple[str, str]] = []  # (parent_text, source)
            for doc in reranked:
                pid = doc.metadata.get("parent_id")
                parent = self.parents.get(pid)
                if parent is None:
                    parent = {"text": doc.page_content,
                              "source": doc.metadata.get("source", "?")}
                    pid = doc.page_content[:40]
                if pid in seen:
                    continue
                seen.add(pid)
                picked.append((parent["text"], parent["source"]))

            # 4) COMPRESS — greedily pack parents up to the token budget so we
            #    send Qwen only the most relevant context, nothing more.
            return self._compress(picked)

    def _compress(self, parents: list[tuple[str, str]]) -> tuple[str, list[str]]:
        """Pack ranked parent sections into the token budget; truncate the last."""
        blocks: list[str] = []
        sources: list[str] = []
        used = 0
        for text, source in parents:
            if used >= CONTEXT_TOKEN_BUDGET:
                break
            remaining = CONTEXT_TOKEN_BUDGET - used
            piece = text
            if _est_tokens(piece) > remaining:
                # Truncate to the remaining budget (keep the head of the section).
                piece = piece[: remaining * CHARS_PER_TOKEN].rstrip() + " …"
            blocks.append(f"[source: {source}]\n{piece}")
            used += _est_tokens(piece)
            if source not in sources:
                sources.append(source)
        return ("\n\n---\n\n".join(blocks), sources)


# ---------------------------------------------------------------------------
# Module-level singletons + public API
# ---------------------------------------------------------------------------

_engines: dict[str, _Engine] = {}
_engines_lock = threading.Lock()


def _close_all_engines():
    """Close local Qdrant clients cleanly at process exit.

    qdrant-client's local mode otherwise logs a harmless ImportError from its
    __del__ when the interpreter is already tearing down; closing explicitly here
    avoids that noise."""
    with _engines_lock:
        for eng in _engines.values():
            if eng.client is not None:
                try:
                    eng.client.close()
                except Exception:
                    pass


import atexit  # noqa: E402  (kept next to the handler it registers for clarity)
atexit.register(_close_all_engines)


def _get_engine(cwd: str) -> _Engine:
    key = os.path.abspath(cwd)
    with _engines_lock:
        eng = _engines.get(key)
        if eng is None:
            eng = _Engine(key)
            _engines[key] = eng
        return eng


def ensure_indexed(cwd: str) -> dict:
    """Build or load the hybrid index for `cwd`. Safe to call on every request."""
    return _get_engine(cwd).ensure()


def retrieve(query: str, cwd: str) -> tuple[str, list[str]]:
    """Return (context_text, sources) for `query` against the index of `cwd`."""
    return _get_engine(cwd).retrieve(query)
