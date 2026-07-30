"""ANSI color helpers for CLI output. Uses colorama for Windows-safe rendering."""
from __future__ import annotations

import colorama

colorama.init(autoreset=True)


def question(text: str) -> str:
    return f"{colorama.Fore.CYAN}{text}{colorama.Style.RESET_ALL}"


def answer(text: str) -> str:
    return f"{colorama.Fore.GREEN}{text}{colorama.Style.RESET_ALL}"


def status(text: str) -> str:
    return f"{colorama.Style.DIM}{text}{colorama.Style.RESET_ALL}"


def warn(text: str) -> str:
    return f"{colorama.Fore.YELLOW}{text}{colorama.Style.RESET_ALL}"


def error(text: str) -> str:
    return f"{colorama.Fore.RED}{text}{colorama.Style.RESET_ALL}"
