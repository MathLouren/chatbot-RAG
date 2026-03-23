from __future__ import annotations

import json
import os
import re
import signal
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import wraps
from typing import Any, Generator

import chromadb
import structlog
from cachetools import TTLCache
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_compress import Compress
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pymongo import MongoClient

from config import CHROMA_HOST, CHROMA_PORT, CONVERSATION_COLLECTION, DB_NAME, MONGO_URI, OLLAMA_BASE_URL
from graph import run_graph

load_dotenv()

# ---------------- Configuração ----------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))
API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", "")
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "").split(",") if x.strip()]
REDIS_URL = os.getenv("REDIS_URL", "").strip()

MAX_MESSAGE_LEN = int(os.getenv("MAX_MESSAGE_LEN", "1000"))
MAX_SESSION_ID_LEN = int(os.getenv("MAX_SESSION_ID_LEN", "64"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

MONGO_MAX_POOL = int(os.getenv("MONGO_MAX_POOL", "10"))
MONGO_MIN_POOL = int(os.getenv("MONGO_MIN_POOL", "2"))

SESSION_CACHE_SIZE = int(os.getenv("SESSION_CACHE_SIZE", "100"))
SESSION_CACHE_TTL = int(os.getenv("SESSION_CACHE_TTL_SECONDS", "600"))
COMPRESS_MIN_SIZE = int(os.getenv("COMPRESS_MIN_SIZE", "1024"))

SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9\-]{1,64}$")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PROMPT_INJECTION_BLOCKLIST = ["<script", "javascript:", "\n\nhuman:", "\n\nassistant:"]


# ---------------- Logging estruturado ----------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger("rag_api")


# ---------------- Singleton de conexões ----------------
mongo_client = MongoClient(MONGO_URI, maxPoolSize=MONGO_MAX_POOL, minPoolSize=MONGO_MIN_POOL)
mongo_db = mongo_client[DB_NAME]
history_col = mongo_db[CONVERSATION_COLLECTION]
state_col = mongo_db["conversation_state"]
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
executor = ThreadPoolExecutor(max_workers=4)


# ---------------- Cache de sessão ----------------
session_cache: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=SESSION_CACHE_SIZE, ttl=SESSION_CACHE_TTL)


# ---------------- Métricas thread-safe ----------------
metrics_lock = threading.Lock()
metrics = {
    "requests_total": 0,
    "errors_total": 0,
    "chat_requests_total": 0,
    "latency_total_ms": 0.0,
    "latency_avg_ms": 0.0,
}

health_cache_lock = threading.Lock()
health_cache: dict[str, Any] = {"ts": 0.0, "ok": True}


def _inc_metric(key: str, amount: float = 1.0) -> None:
    with metrics_lock:
        metrics[key] = metrics.get(key, 0) + amount
        total = max(1, metrics.get("requests_total", 1))
        metrics["latency_avg_ms"] = metrics.get("latency_total_ms", 0.0) / total


# ---------------- App e middlewares ----------------
app = Flask(__name__)

if ALLOWED_ORIGINS:
    CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=True)
else:
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)

app.config["COMPRESS_MIN_SIZE"] = COMPRESS_MIN_SIZE
Compress(app)

limiter_storage = REDIS_URL if REDIS_URL else "memory://"
limiter = Limiter(app=app, key_func=get_remote_address, storage_uri=limiter_storage, default_limits=[])


@app.before_request
def before_request() -> None:
    request._start_time = time.perf_counter()
    _inc_metric("requests_total", 1)


@app.after_request
def after_request(resp: Response) -> Response:
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-XSS-Protection"] = "1; mode=block"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Content-Security-Policy"] = "default-src 'self'"
    if hasattr(request, "_start_time"):
        elapsed = (time.perf_counter() - request._start_time) * 1000.0
        _inc_metric("latency_total_ms", elapsed)
    return resp


# ---------------- Decorators ----------------
def require_api_key(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not API_KEY:
            _inc_metric("errors_total", 1)
            return jsonify({"error": "API key não configurada no servidor."}), 503
        provided = request.headers.get("X-API-Key", "")
        if provided != API_KEY:
            _inc_metric("errors_total", 1)
            return jsonify({"error": "API Key ausente ou inválida."}), 401
        return fn(*args, **kwargs)

    return wrapper


def validate_chat_input(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        payload = request.get_json(silent=True)
        if payload is None:
            _inc_metric("errors_total", 1)
            return jsonify({"error": "JSON inválido ou ausente."}), 400

        session_id = str(payload.get("session_id", ""))
        message = payload.get("message")

        if not isinstance(message, str):
            _inc_metric("errors_total", 1)
            return jsonify({"error": "Schema inválido: message deve ser string."}), 422
        if not isinstance(session_id, str):
            _inc_metric("errors_total", 1)
            return jsonify({"error": "Schema inválido: session_id deve ser string."}), 422
        if len(session_id) > MAX_SESSION_ID_LEN or not SESSION_ID_RE.match(session_id):
            _inc_metric("errors_total", 1)
            return jsonify({"error": "session_id inválido. Use ^[a-zA-Z0-9\\-]{1,64}$."}), 422

        sanitized = CONTROL_RE.sub(" ", message.replace("\x00", "")).strip()
        if not sanitized:
            _inc_metric("errors_total", 1)
            return jsonify({"error": "message deve ter ao menos 1 caractere."}), 422
        if len(sanitized) > MAX_MESSAGE_LEN:
            _inc_metric("errors_total", 1)
            return jsonify({"error": f"message excede {MAX_MESSAGE_LEN} caracteres."}), 422

        lower = sanitized.lower()
        if any(p in lower for p in PROMPT_INJECTION_BLOCKLIST):
            _inc_metric("errors_total", 1)
            return jsonify({"error": "message bloqueada por política de segurança."}), 422

        request.validated_payload = {"session_id": session_id, "message": sanitized}
        return fn(*args, **kwargs)

    return wrapper


# ---------------- Health checks ----------------
def _check_ollama() -> tuple[bool, str]:
    try:
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if 200 <= resp.status < 300:
                return True, "ok"
            return False, f"http_{resp.status}"
    except urllib.error.URLError as e:
        return False, f"url_error:{e.reason}"
    except Exception as e: # noqa: BLE001
        return False, f"error:{type(e).__name__}"


def _check_chroma() -> tuple[bool, str]:
    try:
        if hasattr(chroma_client, "heartbeat"):
            chroma_client.heartbeat()
        return True, "ok"
    except Exception as e: # noqa: BLE001
        return False, f"error:{type(e).__name__}"


def _check_mongo() -> tuple[bool, str]:
    try:
        mongo_client.admin.command("ping")
        return True, "ok"
    except Exception as e: # noqa: BLE001
        return False, f"error:{type(e).__name__}"


def _chunk_words(text: str, chunk_size: int = 3) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks


def _sse_json(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.errorhandler(429)
def handle_429(_e):
    _inc_metric("errors_total", 1)
    resp = jsonify({"error": "Rate limit atingido."})
    resp.status_code = 429
    resp.headers["Retry-After"] = "60"
    return resp


def _session_key_func() -> str:
    payload = request.get_json(silent=True) or {}
    return str(payload.get("session_id", "unknown"))


def _dependencies_available() -> bool:
    # Cache curto para evitar executar 3 checks em toda requisição.
    now = time.time()
    with health_cache_lock:
        if now - float(health_cache.get("ts", 0.0)) <= 15.0:
            return bool(health_cache.get("ok", True))

    ollama_ok, _ = _check_ollama()
    chroma_ok, _ = _check_chroma()
    mongo_ok, _ = _check_mongo()
    overall_ok = ollama_ok and chroma_ok and mongo_ok

    with health_cache_lock:
        health_cache["ts"] = now
        health_cache["ok"] = overall_ok
    return overall_ok


# ---------------- Endpoints ----------------
@app.route("/api/chat", methods=["POST"])
@limiter.limit("30 per minute")
@limiter.limit("10 per minute", key_func=_session_key_func)
@validate_chat_input
def api_chat():
    payload = request.validated_payload
    session_id = payload["session_id"]
    message = payload["message"]

    session_cache[session_id] = {"last_seen": time.time()}
    _inc_metric("chat_requests_total", 1)

    if not _dependencies_available():
        _inc_metric("errors_total", 1)
        return jsonify({"error": "Serviço temporariamente indisponível."}), 503

    started = time.perf_counter()
    model_name = os.getenv("CHAT_MODEL_NAME", "graph_default")
    try:
        future = executor.submit(run_graph, session_id, message)
        answer = str(future.result(timeout=REQUEST_TIMEOUT_SECONDS))
    except FutureTimeoutError:
        _inc_metric("errors_total", 1)
        logger.warning("chat_timeout", session_id=session_id, status=504)
        return jsonify({"error": f"Timeout do LLM ({REQUEST_TIMEOUT_SECONDS}s)."}), 504
    except Exception as e: # noqa: BLE001
        _inc_metric("errors_total", 1)
        err_name = type(e).__name__.lower()
        if any(x in err_name for x in ["serverselectiontimeouterror", "connection", "timeout", "transport"]):
            logger.warning("chat_dependency_error", session_id=session_id, status=503, error=type(e).__name__)
            return jsonify({"error": "Serviço temporariamente indisponível."}), 503
        # Não expor detalhes internos ao cliente.
        logger.error("chat_internal_error", session_id=session_id, status=500, error=type(e).__name__)
        return jsonify({"error": "Erro interno ao processar a solicitação."}), 500

    def event_stream() -> Generator[str, None, None]:
        for piece in _chunk_words(answer, 3):
            yield _sse_json({"token": piece + " "})
            time.sleep(0.03)
        yield "data: [DONE]\n\n"

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "chat_ok",
        session_id=session_id,
        status=200,
        model=model_name,
        latency_ms=round(elapsed_ms, 2),
    )
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@app.route("/api/health", methods=["GET"])
@limiter.limit("30 per minute")
def api_health():
    ollama_ok, ollama_msg = _check_ollama()
    chroma_ok, chroma_msg = _check_chroma()
    mongo_ok, mongo_msg = _check_mongo()
    checks = {
        "ollama": {"ok": ollama_ok, "detail": ollama_msg},
        "chroma": {"ok": chroma_ok, "detail": chroma_msg},
        "mongo": {"ok": mongo_ok, "detail": mongo_msg},
    }
    status = "ok" if all(x["ok"] for x in checks.values()) else "degraded"
    code = 200 if status == "ok" else 503
    return jsonify({"status": status, "checks": checks}), code


@app.route("/api/session/<session_id>", methods=["DELETE"])
@limiter.limit("30 per minute")
@require_api_key
def api_delete_session(session_id: str):
    if not SESSION_ID_RE.match(session_id):
        return jsonify({"error": "session_id inválido."}), 422
    history_col.delete_many({"session_id": session_id})
    state_col.delete_many({"session_id": session_id})
    session_cache.pop(session_id, None)
    return jsonify({"status": "ok", "deleted_session_id": session_id})


@app.route("/api/session/<session_id>/history", methods=["GET"])
@limiter.limit("30 per minute")
@require_api_key
def api_get_history(session_id: str):
    if not SESSION_ID_RE.match(session_id):
        return jsonify({"error": "session_id inválido."}), 422
    try:
        limit = int(request.args.get("limit", "20"))
    except ValueError:
        return jsonify({"error": "limit inválido."}), 422
    limit = max(1, min(100, limit))

    cache_key = f"hist:{session_id}:{limit}"
    cached = session_cache.get(cache_key)
    if cached:
        return jsonify({"session_id": session_id, "history": cached["history"], "cached": True})

    docs = list(
        history_col.find(
            {"session_id": session_id},
            {"_id": 0, "role": 1, "content": 1, "timestamp": 1},
            sort=[("timestamp", -1)],
            limit=limit,
        )
    )
    docs.reverse()
    session_cache[cache_key] = {"history": docs}
    return jsonify({"session_id": session_id, "history": docs, "cached": False})


@app.route("/api/metrics", methods=["GET"])
@limiter.limit("30 per minute")
@require_api_key
def api_metrics():
    with metrics_lock:
        snapshot = dict(metrics)
    return jsonify({"status": "ok", "metrics": snapshot})


# ---------------- Graceful shutdown ----------------
def _shutdown(_signum, _frame) -> None:
    logger.info("shutdown_start")
    try:
        mongo_client.close()
    except Exception: # noqa: BLE001
        pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception: # noqa: BLE001
        pass
    logger.info("shutdown_done")


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


if __name__ == "__main__":
    logger.info(
        "api_start",
        host=API_HOST,
        port=API_PORT,
        debug=API_DEBUG,
        redis=bool(REDIS_URL),
        allowed_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    )
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG, threaded=True)