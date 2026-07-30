"""
Web search tool via Tavily -- for anything outside the indexed document
corpus (current events, general facts, anything rag_search comes back
NO_EVIDENCE on). Kept as a separate tool rather than a rag_search fallback
so the agent's tool CHOICE is visible and auditable: which questions needed
the live web versus the internal corpus is exactly the kind of thing you'd
want to log in a real deployment.
"""
from __future__ import annotations

from functools import lru_cache

from agentic_rag import config


@lru_cache(maxsize=None)
def _get_client():
    from tavily import TavilyClient
    if not config.TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Get a free key at https://tavily.com "
            "and add it to .env."
        )
    return TavilyClient(api_key=config.TAVILY_API_KEY)


def web_search(query: str, max_results: int = 5) -> str:
    """Search the live web. Returns numbered results with title/url/snippet."""
    try:
        resp = _get_client().search(query, max_results=max_results)
    except Exception as e:                                       # noqa: BLE001
        return f"ERROR: web search failed: {e}"

    results = resp.get("results", [])
    if not results:
        return "NO_RESULTS: web search returned nothing for this query."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.get('title')} ({r.get('url')})\n{r.get('content', '')[:500]}")
    return "\n\n".join(parts)
