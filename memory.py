from datetime import datetime
from pymongo import MongoClient

from config import DB_NAME, MONGO_URI, CONVERSATION_COLLECTION

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
history_col = db[CONVERSATION_COLLECTION]
session_state_col = db["conversation_state"]


def save_message(session_id: str, role: str, content: str):
    history_col.insert_one({
        "session_id": session_id,
        "role": role, # "user" | "assistant"
        "content": content,
        "timestamp": datetime.utcnow()
    })


def get_history(session_id: str, limit: int = 10) -> list[dict]:
    messages = history_col.find(
        {"session_id": session_id},
        sort=[("timestamp", -1)],
        limit=limit
    )
    return list(reversed(list(messages)))


def format_history(session_id: str) -> str:
    msgs = get_history(session_id)
    return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs])


def get_scope_preference(session_id: str) -> str | None:
    doc = session_state_col.find_one({"session_id": session_id}, {"scope_preference": 1})
    if not doc:
        return None
    scope = doc.get("scope_preference")
    if scope in ("global", "context"):
        return scope
    return None


def set_scope_preference(session_id: str, scope: str | None):
    if scope not in ("global", "context", None):
        return
    session_state_col.update_one(
        {"session_id": session_id},
        {
            "$set": {
                "scope_preference": scope,
                "updated_at": datetime.utcnow(),
            }
        },
        upsert=True,
    )


def get_active_docs(session_id: str) -> list[str]:
    doc = session_state_col.find_one({"session_id": session_id}, {"active_docs": 1})
    if not doc:
        return []
    docs = doc.get("active_docs", [])
    if isinstance(docs, list):
        return [str(d) for d in docs[:15]]
    return []


def set_active_docs(session_id: str, docs: list[str]):
    safe_docs = [str(d) for d in (docs or [])][:15]
    session_state_col.update_one(
        {"session_id": session_id},
        {
            "$set": {
                "active_docs": safe_docs,
                "updated_at": datetime.utcnow(),
            }
        },
        upsert=True,
    )


def get_listing_state(session_id: str) -> dict:
    doc = session_state_col.find_one(
        {"session_id": session_id},
        {"listing_docs": 1, "listing_shown": 1},
    ) or {}
    docs = doc.get("listing_docs", [])
    if not isinstance(docs, list):
        docs = []
    try:
        shown = int(doc.get("listing_shown", 0))
    except (TypeError, ValueError):
        shown = 0
    return {
        "listing_docs": [str(d) for d in docs[:50]],
        "listing_shown": max(0, shown),
    }


def set_listing_state(session_id: str, docs: list[str], shown: int):
    safe_docs = [str(d) for d in (docs or [])][:50]
    try:
        shown_int = int(shown)
    except (TypeError, ValueError):
        shown_int = 0
    session_state_col.update_one(
        {"session_id": session_id},
        {
            "$set": {
                "listing_docs": safe_docs,
                "listing_shown": max(0, shown_int),
                "updated_at": datetime.utcnow(),
            }
        },
        upsert=True,
    )
