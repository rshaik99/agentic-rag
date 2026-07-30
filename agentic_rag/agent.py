"""
Multi-step tool-using agent (ReAct: reason -> pick a tool -> observe -> repeat
-> answer). Three tools: RAG search over the indexed corpus, a calculator for
anything the retrieved numbers need computed on top of them (e.g. "what's the
percentage growth" when the corpus only has raw figures), and web search for
anything outside the corpus entirely.

    python -m agentic_rag.agent "What was the percentage growth in cloud revenue from Q1 to Q4?"
    python -m agentic_rag.agent "What was the percentage growth, and how does that compare to industry cloud growth this year?"
"""
from __future__ import annotations

import sys

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agentic_rag import config
from agentic_rag.tools.calculator import calculate
from agentic_rag.tools.rag_search import search
from agentic_rag.tools.web_search import web_search as _web_search

SYSTEM_PROMPT = """You are a research assistant with three tools:

- rag_search: searches an INTERNAL document corpus. Try this FIRST for any
  question that could plausibly be about the corpus's own documents/data.
  If it returns NO_EVIDENCE, do not fall back to your own general knowledge --
  either say the corpus doesn't cover it, or use web_search if the question
  is about something genuinely external (current events, general facts,
  anything not about the internal documents).
- web_search: searches the live web. Use for anything outside the internal
  corpus. Cite it distinctly from rag_search results so it's clear which
  claims came from the internal documents versus the open web.
- calculate: for arithmetic on numbers you already have. Don't do math in
  your head -- call the tool so the computation is auditable.

Always be explicit about which source (internal corpus vs. web) each claim
came from. Cite sources as [n] matching the numbers each tool returned.
Be concise."""


@tool
def rag_search(query: str) -> str:
    """Search the internal document corpus for information relevant to the query.
    Returns numbered sources, or NO_EVIDENCE if nothing is relevant enough."""
    return search(query)


@tool
def web_search(query: str) -> str:
    """Search the live web for information outside the internal document corpus."""
    return _web_search(query)


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(340 - 210) / 210 * 100'."""
    return calculate(expression)


def build_agent():
    config.require_openai()
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.TEMPERATURE,
                     api_key=config.OPENAI_API_KEY)
    return create_agent(llm, tools=[rag_search, web_search, calculator], system_prompt=SYSTEM_PROMPT)


def ask(question: str, verbose: bool = True) -> str:
    agent = build_agent()
    messages = [HumanMessage(content=question)]
    result = agent.invoke({"messages": messages})

    if verbose:
        for m in result["messages"]:
            role = m.__class__.__name__
            if role == "AIMessage" and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    print(f"  [tool call] {tc['name']}({tc['args']})")
            elif role == "ToolMessage":
                preview = m.content if len(m.content) < 300 else m.content[:300] + "..."
                print(f"  [tool result: {m.name}] {preview}")

    final = result["messages"][-1].content
    return final


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What caused incident ERR-4417 and how was it fixed?"
    config.banner(f"agentic RAG\n  query: {q!r}")
    answer = ask(q)
    print("\n" + "=" * 72)
    print(answer)
