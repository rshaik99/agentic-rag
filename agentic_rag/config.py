"""
Config for the agentic RAG layer.

NOTE: during local dev this points at the SAME Qdrant instance/collection
that rag-multimodal (C:\\rag-lab) already indexed -- so the RAG tool has real
data to query without re-running ingestion. When this becomes its own repo,
it should get its own ingest pipeline (or a documented pointer to run
rag-multimodal's step_a/b first) rather than silently depending on a
sibling project's local state.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

LLM_MODEL = os.getenv("AGENT_LLM_MODEL", "gpt-4.1")
TEMPERATURE = 0.0

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# Shares rag-multimodal's LangChain-track collection during local dev.
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_lab_lc")

EMBED_MODEL = "text-embedding-3-large"
SPARSE_MODEL = "Qdrant/bm25"
LOCAL_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

RETRIEVE_TOP_K = 20
FUSED_TOP_N = 40
FINAL_TOP_K = 5
RERANK_SCORE_FLOOR = 0.30


def require_openai() -> None:
    if not OPENAI_API_KEY:
        raise SystemExit(
            "OPENAI_API_KEY is not set.\n  cp .env.example .env  &&  edit .env"
        )


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")
