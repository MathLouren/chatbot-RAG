# config.py
MONGO_URI   = "mongodb://localhost:27017"
DB_NAME     = "eprodutos"

# ── ChromaDB local (pasta no disco) ──────────────────────────────────
# config.py corrigido
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000        # ← NOVO (era CHROMA_HOST + CHROMA_PORT)
COLLECTION_NAME   = "eprodutos"

EMBED_MODEL       = "qwen3-embedding:8b"
CHAT_MODEL        = "gemma3:4b"
OLLAMA_BASE_URL   = "http://localhost:11434"

CONVERSATION_COLLECTION = "conversation_history"