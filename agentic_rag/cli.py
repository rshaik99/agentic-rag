"""
Interactive multi-turn CLI for the agentic RAG agent.

    python -m agentic_rag.cli

Unlike rag-multimodal's chatbot.py, this doesn't need an LLM-based "condense
follow-up into a standalone query" step -- the agent's own message history
(including past tool calls/results) is carried forward turn to turn, so
follow-ups like "what about Q4?" naturally have the prior context available
to the agent when it decides whether/how to call a tool again.

Commands: reset (clear history), exit / quit (leave).
"""
from __future__ import annotations

from agentic_rag import colors, config
from agentic_rag.agent import ask


def chat() -> None:
    config.require_openai()
    history: list = []

    print(colors.status("agentic-rag -- multi-tool RAG agent (rag_search, web_search, calculator)"))
    print(colors.status("Commands: reset, exit\n"))

    while True:
        try:
            question = input(colors.question("You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "/exit", "/quit"):
            break
        if question.lower() in ("reset", "/reset"):
            history = []
            print(colors.status("  (history cleared)"))
            continue

        history, final_answer = ask(question, history=history)
        print(colors.answer(f"\n{final_answer}\n"))


if __name__ == "__main__":
    chat()
