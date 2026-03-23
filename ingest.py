from dotenv import load_dotenv
import os
import uuid
import chromadb
import importlib
from pymongo import MongoClient

from config import (
    CHROMA_HOST,
    CHROMA_PORT,
    COLLECTION_NAME,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    MONGO_URI,
    DB_NAME,
)

load_dotenv()

if importlib.util.find_spec("langchain_ollama") is not None:
    OllamaEmbeddings = importlib.import_module("langchain_ollama").OllamaEmbeddings
else:
    from langchain_community.embeddings import OllamaEmbeddings # type: ignore

VALID_PROCESSES = ["Lista Diária Ingram Micro ES - Preço", "Estoque Geral Coletek"]


def build_product_text(stock: dict, product: dict) -> str:
    """
    Monta o texto que será vetorizado.
    Inclui APENAS campos descritivos — nunca preço, estoque ou processo.
    """
    pn = str(stock.get("pn", "") or "")
    descricao = str(stock.get("description", "") or "") # descrição vem de stocks
    fabricante = str(product.get("manufacturer", "") or "")
    categoria = str(product.get("category", "") or "")

    partes = []
    if fabricante: partes.append(f"Fabricante: {fabricante}")
    if categoria: partes.append(f"Categoria: {categoria}")
    if descricao: partes.append(f"Descrição: {descricao}")
    if pn: partes.append(f"PN: {pn}")

    return " | ".join(partes)


def ingest_from_mongodb(batch_size: int = 1000):
    # ── Conexões ──────────────────────────────────────────────────────────
    mongo = MongoClient(MONGO_URI)
    db = mongo[DB_NAME]
    chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    collection = chroma.get_or_create_collection(name=COLLECTION_NAME)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    # ── Lê stocks do MongoDB (somente processos desejados) ────────────────
    # A descrição vem de stocks; fabricante/categoria continuam vindo de products.
    cursor = db.stocks.aggregate([
        {"$match": {"process": {"$in": VALID_PROCESSES}}},
        {"$lookup": {
            "from": "products",
            "localField": "pn",
            "foreignField": "pn",
            "as": "prod_info",
        }},
        {"$addFields": {
            "manufacturer": {"$ifNull": [{"$arrayElemAt": ["$prod_info.manufacturer", 0]}, ""]},
            "category": {"$ifNull": [{"$arrayElemAt": ["$prod_info.category", 0]}, ""]},
        }},
        {"$project": {
            "pn": 1,
            "description": 1,
            "process": 1,
            "manufacturer": 1,
            "category": 1,
        }},
        # Consolida por PN para manter um único vetor por produto
        # e preservar todos os processos de origem em metadado.
        {"$group": {
            "_id": "$pn",
            "pn": {"$first": "$pn"},
            "description": {"$first": "$description"},
            "manufacturer": {"$first": "$manufacturer"},
            "category": {"$first": "$category"},
            "processes": {"$addToSet": "$process"},
        }},
    ], allowDiskUse=True)

    batch_docs, batch_metas, batch_ids = [], [], []
    total = 0

    for stock in cursor:
        pn = str(stock.get("pn", ""))
        if not pn:
            continue

        product = {
            "manufacturer": str(stock.get("manufacturer", "")),
            "category": str(stock.get("category", "")),
        }

        texto = build_product_text(stock, product)
        if not texto.strip():
            continue

        process_list = [
            str(p).strip() for p in (stock.get("processes", []) or [])
            if str(p).strip()
        ]
        process_text = " | ".join(process_list)

        batch_docs.append(texto)
        batch_metas.append({
            "pn": pn,
            "manufacturer": str(product.get("manufacturer", "")),
            "category": str(product.get("category", "")),
            "process": process_text,
        })
        # ID determinístico por PN — permite re-rodar sem duplicar
        batch_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, pn)))

        # Processa em lotes para não sobrecarregar o Ollama
        if len(batch_docs) >= batch_size:
            _upsert_batch(collection, embeddings, batch_docs, batch_metas, batch_ids)
            total += len(batch_docs)
            print(f" → {total} produtos indexados...")
            batch_docs, batch_metas, batch_ids = [], [], []

    # Último lote
    if batch_docs:
        _upsert_batch(collection, embeddings, batch_docs, batch_metas, batch_ids)
        total += len(batch_docs)

    print(f"\n✅ {total} produtos indexados no ChromaDB (coleção: {COLLECTION_NAME}).")
    mongo.close()


def _upsert_batch(collection, embeddings, docs, metas, ids):
    vectors = embeddings.embed_documents(docs)
    if hasattr(collection, "upsert"):
        collection.upsert(ids=ids, embeddings=vectors, documents=docs, metadatas=metas)
    else:
        collection.add(ids=ids, embeddings=vectors, documents=docs, metadatas=metas)


if __name__ == "__main__":
    ingest_from_mongodb()
