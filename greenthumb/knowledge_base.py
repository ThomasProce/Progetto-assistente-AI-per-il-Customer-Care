"""Long-term semantic memory: loading, chunking, indexing and retrieval of the knowledge base.

Pipeline
--------
1. ``load_documents``: every Markdown file under ``data/knowledge_base`` becomes a
   LangChain ``Document``; the YAML front matter becomes metadata and the path
   relative to the KB root (e.g. ``guides/bulbi_autunnali.md``) is the ``source``
   id that the agent cites in ``sources``.
2. ``split_documents``: structure-aware chunking. Documents are first split on
   Markdown headings (``#``/``##``), so a chunk never mixes two sections, then
   long sections are split with a recursive character splitter
   (``chunk_size``/``chunk_overlap``). Each chunk is prefixed with the document
   title and section name, so that its embedding carries the context
   ("Quando piantare" alone is meaningless without "Tulipani").
3. ``build_index``: chunks are embedded (LiteLLM embeddings) and stored in a
   persistent Chroma collection using cosine distance.
4. ``KnowledgeBaseRetriever.search`` (hybrid, small-to-big):
   - **dense** ranking: cosine relevance of every chunk (optionally filtered by
     category) to the query embedding;
   - **lexical** ranking: BM25 over the same chunks, with a light Italian
     normalisation (lower-case, stop-words, 5-character prefix stemming). It
     rescues short keyword queries ("cesoie", product names, SKUs) on which
     dense embeddings are weak;
   - the two rankings are fused with **Reciprocal Rank Fusion** (RRF, k=60);
     a chunk is a candidate if its dense score is >= ``min_relevance_score`` or
     it is in the BM25 top ``fetch_k``;
   - **parent-document expansion** (``parent_document=True``, default): the best
     ``top_docs`` documents are returned *whole* (KB documents are short, 250-450
     words), so the model sees the full policy/guide instead of an isolated
     section. With ``parent_document=False`` the best ``top_k`` chunks are
     returned (at most ``max_chunks_per_doc`` per document), i.e. the v1 behaviour.
   The ``score`` of a result is always the best *dense* cosine relevance of its
   chunks, so it stays interpretable for the confidence policy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from greenthumb.config import Settings

logger = logging.getLogger(__name__)

MANIFEST_FILE = "index_manifest.json"
HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2")]


@dataclass(frozen=True)
class RetrievedChunk:
    """A knowledge-base chunk returned by a search, with its relevance score (0-1)."""

    source: str
    title: str
    category: str
    section: str
    content: str
    score: float

    def to_dict(self) -> dict:
        """Serialise the chunk for tool outputs and traces."""
        data = asdict(self)
        data["score"] = round(self.score, 3)
        return data


class Retriever(Protocol):
    """Interface of a knowledge-base retriever (lets tests plug in a fake one)."""

    def search(self, query: str, categories: list[str] | None = None) -> list[RetrievedChunk]:
        """Return the most relevant chunks for ``query``."""
        ...


def _parse_front_matter(text: str) -> tuple[dict, str]:
    """Split a Markdown file into its YAML front matter and its body."""
    if not text.startswith("---"):
        return {}, text
    _, raw_meta, body = text.split("---", 2)
    return yaml.safe_load(raw_meta) or {}, body.strip()


def load_documents(kb_dir: Path) -> list[Document]:
    """Load every Markdown document of the knowledge base with its metadata."""
    documents = []
    for path in sorted(kb_dir.rglob("*.md")):
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        source = path.relative_to(kb_dir).as_posix()
        metadata = {
            "source": source,
            "title": meta.get("title", path.stem),
            "category": meta.get("category", path.parent.name),
            "sku": meta.get("sku", ""),
            "tags": ", ".join(meta.get("tags", [])),
        }
        documents.append(Document(page_content=body, metadata=metadata))
    return documents


def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split documents into section-aware chunks enriched with title and section."""
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Document] = []
    for document in documents:
        sections = header_splitter.split_text(document.page_content)
        chunk_index = 0
        for section in sections:
            section_name = section.metadata.get("h2") or section.metadata.get("h1") or ""
            for piece in text_splitter.split_text(section.page_content):
                header = f"Documento: {document.metadata['title']}\n"
                if section_name:
                    header += f"Sezione: {section_name}\n"
                metadata = {
                    **document.metadata,
                    "section": section_name,
                    "chunk_id": f"{document.metadata['source']}#{chunk_index}",
                }
                chunks.append(Document(page_content=f"{header}\n{piece}", metadata=metadata))
                chunk_index += 1
    return chunks


def _manifest(settings: Settings) -> dict:
    """Fingerprint of everything that affects the index content."""
    digest = hashlib.sha256()
    for path in sorted(settings.kb_dir.rglob("*.md")):
        digest.update(path.relative_to(settings.kb_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return {
        "kb_hash": digest.hexdigest(),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }


def open_vector_store(settings: Settings, embeddings: Embeddings) -> Chroma:
    """Open (or create) the persistent Chroma collection, using cosine distance."""
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.vector_store_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index(settings: Settings, embeddings: Embeddings, force: bool = False) -> Chroma:
    """Index the knowledge base, rebuilding only if documents or chunking settings changed."""
    vector_store = open_vector_store(settings, embeddings)
    manifest_path = settings.vector_store_dir / MANIFEST_FILE
    manifest = _manifest(settings)
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if not force and previous == manifest and vector_store._collection.count() > 0:
        logger.info("Vector store up to date (%d chunks)", vector_store._collection.count())
        return vector_store

    vector_store.reset_collection()
    chunks = split_documents(load_documents(settings.kb_dir), settings.chunk_size, settings.chunk_overlap)
    vector_store.add_documents(chunks, ids=[chunk.metadata["chunk_id"] for chunk in chunks])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Indexed %d chunks", len(chunks))
    return vector_store


ITALIAN_STOPWORDS = frozenset(
    "il lo la i gli le un uno una di da in con su per tra fra a e o ma se che chi cui non piu "
    "del dello della dei degli delle al allo alla ai agli alle dal dallo dalla dai dagli dalle "
    "nel nello nella nei negli nelle sul sullo sulla sui sugli sulle mi ti si ci vi ne come "
    "quando dove cosa quale quali quanto quanti sono sei essere ho hai ha abbiamo hanno posso "
    "puoi puo mio mia miei mie tuo tua vostro vostra questo questa quello quella anche".split()
)
RRF_K = 60
STEM_LENGTH = 5


def tokenize(text: str) -> list[str]:
    """Lower-case, strip accents, drop stop-words and apply 5-character prefix stemming."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return [
        token[:STEM_LENGTH]
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 1 and token not in ITALIAN_STOPWORDS
    ]


class KnowledgeBaseRetriever:
    """Hybrid (dense + BM25) retrieval with RRF fusion and parent-document expansion."""

    def __init__(self, vector_store: Chroma, settings: Settings) -> None:
        """Load all chunks from the vector store and build the BM25 index and parent-document map."""
        self._vector_store = vector_store
        self._fetch_k = settings.retrieval_fetch_k
        self._top_k = settings.retrieval_top_k
        self._top_docs = settings.retrieval_top_docs
        self._max_chunks_per_doc = settings.max_chunks_per_doc
        self._min_score = settings.min_relevance_score
        self._hybrid = settings.hybrid_search
        self._parent_document = settings.parent_document

        stored = vector_store.get(include=["documents", "metadatas"])
        self._chunk_ids: list[str] = stored["ids"]
        self._chunks = {
            chunk_id: Document(page_content=text, metadata=metadata)
            for chunk_id, text, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"])
        }
        self._bm25 = BM25Okapi([tokenize(self._chunks[chunk_id].page_content) for chunk_id in self._chunk_ids])
        self._parents = {
            document.metadata["source"]: document for document in load_documents(settings.kb_dir)
        }

    def _dense_scores(self, query: str, categories: list[str] | None) -> dict[str, float]:
        """Cosine relevance of every chunk in the allowed categories (the KB is small enough)."""
        where = None
        if categories:
            where = {"category": categories[0]} if len(categories) == 1 else {"category": {"$in": categories}}
        results = self._vector_store.similarity_search_with_relevance_scores(
            query, k=len(self._chunk_ids), filter=where
        )
        return {document.metadata["chunk_id"]: float(score) for document, score in results}

    def _lexical_ranking(self, query: str, allowed: set[str]) -> list[str]:
        """Chunk ids ranked by BM25 score (only chunks with a positive score)."""
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            (pair for pair in zip(self._chunk_ids, scores) if pair[0] in allowed and pair[1] > 0),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [chunk_id for chunk_id, _ in ranked]

    def _fused_candidates(self, query: str, categories: list[str] | None) -> list[tuple[str, float]]:
        """Return ``(chunk_id, dense_score)`` candidates ordered by RRF of dense and BM25 ranks."""
        dense = self._dense_scores(query, categories)
        dense_ranking = sorted(dense, key=dense.get, reverse=True)
        fused = {chunk_id: 1.0 / (RRF_K + rank) for rank, chunk_id in enumerate(dense_ranking, start=1)}
        candidates = {chunk_id for chunk_id in dense_ranking if dense[chunk_id] >= self._min_score}
        if self._hybrid:
            lexical_ranking = self._lexical_ranking(query, set(dense))
            for rank, chunk_id in enumerate(lexical_ranking, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            candidates |= set(lexical_ranking[: self._fetch_k])
        ordered = sorted(candidates, key=fused.get, reverse=True)
        return [(chunk_id, dense.get(chunk_id, 0.0)) for chunk_id in ordered]

    def search(self, query: str, categories: list[str] | None = None) -> list[RetrievedChunk]:
        """Return the most relevant documents (or chunks), optionally restricted to some categories."""
        candidates = self._fused_candidates(query, categories)
        if self._parent_document:
            return self._expand_to_documents(candidates)
        return self._select_chunks(candidates)

    def _expand_to_documents(self, candidates: list[tuple[str, float]]) -> list[RetrievedChunk]:
        """Group candidate chunks by document and return the best ``top_docs`` documents whole."""
        best_score: dict[str, float] = {}
        sections: dict[str, list[str]] = {}
        order: list[str] = []
        for chunk_id, score in candidates:
            source = self._chunks[chunk_id].metadata["source"]
            if source not in best_score:
                order.append(source)
            best_score[source] = max(score, best_score.get(source, 0.0))
            section = self._chunks[chunk_id].metadata.get("section", "")
            if section and section not in sections.setdefault(source, []):
                sections[source].append(section)
        results = []
        for source in order[: self._top_docs]:
            parent = self._parents[source]
            results.append(
                RetrievedChunk(
                    source=source,
                    title=parent.metadata["title"],
                    category=parent.metadata["category"],
                    section=" | ".join(sections.get(source, [])),
                    content=f"Documento: {parent.metadata['title']}\n\n{parent.page_content}",
                    score=best_score[source],
                )
            )
        return results

    def _select_chunks(self, candidates: list[tuple[str, float]]) -> list[RetrievedChunk]:
        """Return the best ``top_k`` chunks, at most ``max_chunks_per_doc`` per document."""
        results: list[RetrievedChunk] = []
        chunks_per_doc: dict[str, int] = {}
        for chunk_id, score in candidates:
            chunk = self._chunks[chunk_id]
            source = chunk.metadata["source"]
            if chunks_per_doc.get(source, 0) >= self._max_chunks_per_doc:
                continue
            chunks_per_doc[source] = chunks_per_doc.get(source, 0) + 1
            results.append(
                RetrievedChunk(
                    source=source,
                    title=chunk.metadata.get("title", ""),
                    category=chunk.metadata.get("category", ""),
                    section=chunk.metadata.get("section", ""),
                    content=chunk.page_content,
                    score=score,
                )
            )
            if len(results) >= self._top_k:
                break
        return results
