from pathlib import Path
from typing import Any

from ddgs import DDGS
from pydantic import BaseModel, Field, ValidationError


# -------------------------
# Tool 1: Web Search
# -------------------------

class WebSearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=200)


def web_search(query: str) -> str:
    """Search the web and return a short list of results."""
    try:
        data = WebSearchInput(query=query)

        results = []
        with DDGS() as ddgs:
            for item in ddgs.text(data.query, max_results=5):
                results.append(
                    f"- {item.get('title', '')}: {item.get('body', '')}"
                )

        if not results:
            return "No search results found."

        return "\n".join(results)

    except ValidationError as e:
        return f"Validation error: {e}"
    except Exception as e:
        return f"Web search failed gracefully: {e}"


# -------------------------
# Tool 2: Calculator
# -------------------------

class CalculatorInput(BaseModel):
    expression: str = Field(min_length=1, max_length=100)


def calculator(expression: str) -> str:
    """Safely evaluate a basic mathematical expression."""
    try:
        data = CalculatorInput(expression=expression)

        allowed = set("0123456789+-*/().% ")

        if not set(data.expression) <= allowed:
            return "Validation error: only basic arithmetic characters are allowed."

        result = eval(data.expression, {"__builtins__": {}}, {})
        return f"Result: {result}"

    except ValidationError as e:
        return f"Validation error: {e}"
    except ZeroDivisionError:
        return "Calculator error: division by zero."
    except Exception as e:
        return f"Calculator failed gracefully: {e}"


# -------------------------
# Tool 3: Save Note
# -------------------------

class SaveNoteInput(BaseModel):
    note: str = Field(min_length=1, max_length=1000)


def save_note(note: str) -> str:
    """Save a note locally."""
    try:
        data = SaveNoteInput(note=note)

        notes_dir = Path("data")
        notes_dir.mkdir(exist_ok=True)

        notes_file = notes_dir / "notes.txt"

        with notes_file.open("a", encoding="utf-8") as f:
            f.write(data.note + "\n")

        return "Note saved successfully."

    except ValidationError as e:
        return f"Validation error: {e}"
    except Exception as e:
        return f"Could not save note gracefully: {e}"


# -------------------------
# Tool Registry
# -------------------------

TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
    "save_note": save_note,
}


def validate_and_run_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """Validate and execute a tool without crashing the agent."""

    if tool_name not in TOOLS:
        return f"Unknown tool: {tool_name}"

    try:
        tool = TOOLS[tool_name]
        return tool(**arguments)

    except ValidationError as e:
        return f"Validation error: {e}"

    except Exception as e:
        return f"Tool execution error: {e}"