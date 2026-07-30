"""
RAG search tool: hybrid retrieval (dense + sparse, server-side RRF) -> local
cross-encoder rerank -> hard score floor. Same non-negotiables as
rag-multimodal's step_c: below RERANK_SCORE_FLOOR, return nothing rather
than the least-bad chunk, so the agent can tell "no evidence" apart from
"weak evidence" instead of being handed a false positive.

Attaches to an EXISTING collection (query-only, no indexing here).
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_openai import OpenAIEmbeddings
from pydantic import ConfigDict

from agentic_rag import config


class _ScoringCrossEncoderReranker(BaseDocumentCompressor):
    model: object
    top_n: int = 5
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def compress_documents(self, documents, query, callbacks=None):
        documents = list(documents)
        scores = self.model.score([(query, d.page_content) for d in documents])
        for d, s in zip(documents, scores):
            d.metadata["relevance_score"] = float(s)
        ranked = sorted(documents, key=lambda d: d.metadata["relevance_score"], reverse=True)
        ranked = [d for d in ranked if d.metadata["relevance_score"] >= config.RERANK_SCORE_FLOOR]
        return ranked[: self.top_n]


@lru_cache(maxsize=None)
def _get_store() -> QdrantVectorStore:
    config.require_openai()
    return QdrantVectorStore.from_existing_collection(
        collection_name=config.QDRANT_COLLECTION,
        embedding=OpenAIEmbeddings(model=config.EMBED_MODEL, api_key=config.OPENAI_API_KEY),
        sparse_embedding=FastEmbedSparse(model_name=config.SPARSE_MODEL),
        retrieval_mode=RetrievalMode.HYBRID,
        url=config.QDRANT_URL,
        vector_name="dense",
        sparse_vector_name="sparse",
    )


@lru_cache(maxsize=None)
def _get_reranker() -> _ScoringCrossEncoderReranker:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    return _ScoringCrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name=config.LOCAL_RERANK_MODEL),
        top_n=config.FINAL_TOP_K,
    )


def search(query: str) -> str:
    """Search the indexed document corpus. Returns numbered sources with
    citations, or an explicit no-evidence message -- never a bare guess."""
    try:
        store = _get_store()
        candidates = store.similarity_search(query, k=config.FUSED_TOP_N)
        docs: list[Document] = _get_reranker().compress_documents(candidates, query)
    except Exception as e:                                       # noqa: BLE001
        return f"ERROR: rag_search failed ({e.__class__.__name__}: {e}). Do not treat this as NO_EVIDENCE -- it's a tool failure, not an absence of evidence."

    if not docs:
        return "NO_EVIDENCE: nothing in the document corpus scored above the relevance floor for this query."

    # R-prefixed so citations are unambiguous against web_search's W-prefixed
    # ones by construction, not just by the model's prose labeling them.
    parts = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        parts.append(
            f"[R{i}] {m.get('file_name')} p.{m.get('page')} "
            f"(score={m.get('relevance_score', 0):.3f})\n{d.page_content}"
        )
    return "\n\n".join(parts)
