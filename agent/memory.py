
import os

from dotenv import load_dotenv
from mem0 import Memory

load_dotenv()

# Mem0 Gemini authentication
if os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

USER_ID = "karim"

memory_config = {
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-3.6-flash",
            "temperature": 0.2,
        },
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/gemini-embedding-001",
            "embedding_dims": 768,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "karim_second_brain_768",
            "embedding_model_dims": 768,
        },
    },
}

memory = Memory.from_config(memory_config)


def search_memory(query: str, top_k: int = 5):
    try:
        return memory.search(
            query=query,
            filters={"user_id": USER_ID},
            limit=top_k,
        )
    except Exception as exc:
        return {
            "error": f"Memory search failed safely: {exc}"
        }
def classify_memory(content: str) -> str:
    text = content.lower()

    profile_words = {
        "prefer",
        "preference",
        "like",
        "love",
        "favorite",
        "tone",
        "i am",
        "my name",
    }

    task_words = {
        "project",
        "deadline",
        "task",
        "assignment",
        "todo",
        "due",
    }

    if any(word in text for word in profile_words):
        return "profile"

    if any(word in text for word in task_words):
        return "task"

    return "episodic"

def add_memory(
    content: str,
    category: str,
    consent_given: bool,
):
    if not consent_given:
        return {
            "stored": False,
            "reason": "Explicit consent was not provided.",
        }

    if category not in {"profile", "task", "episodic"}:
        return {
            "stored": False,
            "reason": "Invalid memory category.",
        }

    try:
        result = memory.add(
            [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            user_id=USER_ID,
            metadata={
                "category": category,
            },
        )

        return {
            "stored": True,
            "category": category,
            "result": result,
        }

    except Exception as exc:
        return {
            "stored": False,
            "reason": f"Memory write failed safely: {exc}",
        }


def forget_all():
    try:
        before = memory.get_all(
            filters={"user_id": USER_ID}
        )

        if isinstance(before, dict):
            count = len(before.get("results", []))
        else:
            count = len(before or [])

        memory.delete_all(user_id=USER_ID)

        return {
            "deleted": True,
            "count": count,
            "message": f"Deleted {count} memories.",
        }

    except Exception as exc:
        return {
            "deleted": False,
            "count": 0,
            "message": f"Memory deletion failed safely: {exc}",
        }


def reset():
    return forget_all()

