import chromadb
import importlib
import warnings
import re
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

if importlib.util.find_spec("langchain_ollama") is not None:
    OllamaEmbeddings = importlib.import_module("langchain_ollama").OllamaEmbeddings
else:
    try:
        # type: ignore[import-untyped]
        from langchain_community.embeddings import OllamaEmbeddings # type: ignore
    except ImportError:
        OllamaEmbeddings = None


_chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
_collection = _chroma_client.get_collection(name=COLLECTION_NAME)
_mongo = MongoClient(MONGO_URI)
_db = _mongo[DB_NAME]
_valid_processes = ["Lista Diária Ingram Micro ES - Preço", "Estoque Geral Coletek"]

# Adicione esta constante no topo do seu retrievers.py, abaixo dos imports
MAX_DISTANCE_THRESHOLD = 1.3  # Valores comuns para L2 no ChromaDB vão de 0.0 (idêntico) até ~2.0. Ajuste se necessário.

def retrieve_docs(query: str, n_results: int = 3) -> str:
    """
    Busca documentos no ChromaDB, mas ignora resultados que são muito 
    diferentes da pergunta original (distância alta).
    """
    results = _collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    # Se não retornou nada
    if not results['documents'] or not results['documents']:
        return ""
        
    valid_docs = []
    
    # Itera sobre os resultados e suas respectivas distâncias
    docs = results['documents']
    distances = results['distances']
    
    for doc, distance in zip(docs, distances):
        # Só adiciona ao contexto se a distância for MENOR que o limite máximo
        if distance <= MAX_DISTANCE_THRESHOLD:
            valid_docs.append(doc)
        else:
            # Documento descartado por ser muito distante semanticamente
            pass

    # Se nenhum documento passou na validação de distância, retorna vazio
    if not valid_docs:
        return ""
        
    return "\n\n".join(valid_docs)

def _normalize_pn_display(pn: str) -> str:
    """
    Normaliza PN apenas para exibição.
    Alguns PNs chegam com "." em posição onde usuários esperam "X" (ex.: MM.G3BZ/A).
    Mantemos o valor original no banco; esta função altera apenas a apresentação.
    """
    pn = str(pn or "")
    if len(pn) >= 4 and pn[2] == ".":
        return pn[:2] + "X" + pn[3:]
    return pn


def _format_brl(value: float) -> str:
    # Formata apenas o número para pt-BR, sem alterar texto descritivo.
    formatted = f"{float(value):,.2f}"
    return formatted.replace(",", "§").replace(".", ",").replace("§", ".")


def _format_stock_text(s: dict) -> str:
    pn_raw = str(s.get("pn", "N/A"))
    pn = _normalize_pn_display(pn_raw)
    desc = str(s.get("description", "") or "")
    if not desc.strip():
        return ""
    fabricante = str(s.get("manufacturer", "") or "")
    categoria = str(s.get("category", "") or "")
    price = float(s.get("sale_price", 0)) / 100.0
    qty = int(float(s.get("quantity", 0))) if s.get("quantity") else 0
    uf = str(s.get("uf", "") or "")
    return (
        f"{desc} | PN: {pn} | Fabricante: {fabricante} | Categoria: {categoria} | "
        f"Preço: R$ {_format_brl(price)} | Estoque: {qty} | UF: {uf}"
    )


def _format_stock_list(stocks: list[dict]) -> list[str]:
    out: list[str] = []
    for s in stocks:
        txt = _format_stock_text(s)
        if txt.strip():
            out.append(txt)
    return out


def _qty_to_int(value) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except Exception:
        return 0


def _get_embeddings():
    if OllamaEmbeddings is None:
        raise RuntimeError(
            "OllamaEmbeddings não disponível. Instale `langchain-community` "
            "ou garanta dependências corretas no ambiente."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def _is_global_price_query(query: str) -> tuple[bool, str | None]:
    q = query.lower()
    is_min = any(k in q for k in ["mais barato", "menor preço", "menor preco", "valor mais baixo", "preço mais baixo", "preco mais baixo"])
    is_max = any(k in q for k in ["mais caro", "maior preço", "maior preco", "valor mais alto", "preço mais alto", "preco mais alto"])
    if not (is_min or is_max):
        return False, None

    # Heurística de consulta global: pergunta sobre loja/estoque/catálogo sem tipo específico
    termos_globais = ["loja", "estoque", "catálogo", "catalogo", "vocês têm", "voces tem", "de tudo", "geral", "item", "produto"]
    tipos = [
        "monitor", "notebook", "smartphone", "iphone", "ssd", "teclado", "mouse",
        "impressora", "servidor", "celular", "tablet", "headset", "galaxy", "moto"
    ]
    tem_tipo = any(re.search(rf"\b{re.escape(t)}\b", q) for t in tipos)
    global_query = any(t in q for t in termos_globais) and not tem_tipo

    if global_query:
        return True, "MIN_PRICE" if is_min else "MAX_PRICE"
    return False, None


def _extract_price_intent(query: str) -> str | None:
    q = query.lower()
    is_min = any(k in q for k in ["mais barato", "menor preço", "menor preco", "valor mais baixo", "preço mais baixo", "preco mais baixo"])
    is_max = any(k in q for k in ["mais caro", "maior preço", "maior preco", "valor mais alto", "preço mais alto", "preco mais alto"])
    if is_min:
        return "MIN_PRICE"
    if is_max:
        return "MAX_PRICE"
    return None


def _extract_product_type(query: str) -> str | None:
    q = query.lower()
    type_map = {
        "monitor": ["monitor", "display", "tela"],
        "notebook": ["notebook", "laptop", "macbook"],
        "smartphone": ["smartphone", "celular", "iphone", "telefone"],
        "nobreak": ["nobreak", "ups", "smart-ups", "easy ups"],
        "impressora": ["impressora", "printer", "multifuncional"],
        "mouse": ["mouse"],
        "teclado": ["teclado", "keyboard"],
        "ssd": ["ssd"],
        "desktop": ["desktop", "pc", "workstation"],
    }
    for product_type, termos in type_map.items():
        if any(t in q for t in termos):
            return product_type
    return None


def _extract_brand(query: str) -> str | None:
    q = query.lower()
    brand_map = {
        "apple": ["apple", "iphone", "ios", "ipad", "macbook"],
        "samsung": ["samsung", "galaxy"],
        "motorola": ["motorola", "moto"],
        "xiaomi": ["xiaomi", "redmi", "poco"],
        "logitech": ["logitech", "logi"],
        "lg": ["lg"],
        "dell": ["dell"],
        "lenovo": ["lenovo"],
        "hp": ["hp"],
        "asus": ["asus"],
        "acer": ["acer"],
    }
    for brand, termos in brand_map.items():
        if any(t in q for t in termos):
            return brand
    return None


def _extract_stock_intent(query: str) -> str | None:
    q = (query or "").lower()
    is_min = any(k in q for k in ["menor estoque", "menos estoque", "estoque mais baixo"])
    is_max = any(k in q for k in ["maior estoque", "mais estoque", "mais unidades", "mais em estoque"])
    if is_min:
        return "MIN_STOCK"
    if is_max:
        return "MAX_STOCK"
    return None


def _extract_pn_from_query(query: str) -> str | None:
    q = str(query or "")
    candidates = re.findall(r"\b[A-Z0-9][A-Z0-9\.\-\/#]{4,}\b", q.upper())
    for c in candidates:
        has_letter = any(ch.isalpha() for ch in c)
        has_digit = any(ch.isdigit() for ch in c)
        if has_letter and has_digit:
            return c.strip()
    return None


def _extract_price_threshold(query: str) -> tuple[str, float] | None:
    q = (query or "").lower()

    max_terms = ["abaixo de", "até", "ate", "menor que", "no máximo", "no maximo"]
    min_terms = ["acima de", "a partir de", "maior que", "no mínimo", "no minimo"]
    has_max = any(t in q for t in max_terms)
    has_min = any(t in q for t in min_terms)
    if not (has_max or has_min):
        return None

    matches = re.findall(r"\d+(?:[.,]\d+)?", q)
    if not matches:
        return None
    raw = matches[-1].replace(".", "").replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return None

    if has_max:
        return ("MAX_PRICE", value)
    if has_min:
        return ("MIN_PRICE", value)
    return None


def _extract_price_between(query: str) -> tuple[float, float] | None:
    q = (query or "").lower()
    if "entre" not in q:
        return None
    matches = re.findall(r"\d+(?:[.,]\d+)?", q)
    if len(matches) < 2:
        return None
    try:
        a = float(matches[0].replace(".", "").replace(",", "."))
        b = float(matches[1].replace(".", "").replace(",", "."))
    except ValueError:
        return None
    low = min(a, b)
    high = max(a, b)
    return (low, high)


def _is_color_variation_followup(query: str) -> bool:
    q = (query or "").lower().strip()
    return any(
        t in q
        for t in [
            "outra cor",
            "outras cores",
            "tem em outra cor",
            "tem em outras cores",
            "há outra cor",
            "ha outra cor",
            "cor diferente",
        ]
    )


def _is_larger_version_followup(query: str) -> bool:
    q = (query or "").lower().strip()
    return any(
        t in q
        for t in [
            "versão maior",
            "versao maior",
            "versão mais potente",
            "versao mais potente",
            "mais potente",
            "mais completo",
            "modelo maior",
        ]
    )


def _extract_main_color(text: str) -> str | None:
    t = (text or "").lower()
    colors = [
        "preto", "branco", "azul", "cinza", "vermelho", "rosa", "prata", "dourado",
        "grafite", "starlight", "midnight", "black", "blue", "silver", "gray", "grey",
        "green", "roxo", "purple", "amarelo", "yellow",
    ]
    for c in colors:
        if re.search(rf"\b{re.escape(c)}\b", t):
            return c
    return None


def _model_keywords(description: str) -> list[str]:
    text = (description or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    terms = [w for w in text.split() if w]
    stop = {
        "smartphone", "celular", "iphone", "galaxy", "tablet", "tab", "5g", "4g",
        "gb", "ram", "preto", "branco", "azul", "cinza", "vermelho", "rosa", "prata",
        "dourado", "grafite", "black", "blue", "silver", "gray", "grey", "green",
        "starlight", "midnight",
    }
    keys = [w for w in terms if len(w) >= 3 and w not in stop]
    return keys[:4]


def _buscar_outras_cores_do_modelo(active_docs: list[str], k: int = 12) -> list[str]:
    if not active_docs:
        return []
    ref_doc = active_docs[0]
    ref_pn = _extract_pn_from_doc(ref_doc)
    if not ref_pn:
        return []

    candidates = _pn_lookup_candidates(ref_pn)
    ref_stock = _db.stocks.find_one(
        {"pn": {"$in": candidates}, "sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}},
        sort=[("sale_price", 1)],
    )
    if not ref_stock:
        return []
    ref_prod = _db.products.find_one({"pn": str(ref_stock.get("pn", ""))}) or {}

    ref_desc = str(ref_stock.get("description", "") or "")
    if not ref_desc.strip():
        return []
    ref_color = _extract_main_color(ref_desc) or ""
    ref_manufacturer = str(ref_prod.get("manufacturer", "") or "")
    ref_category = str(ref_prod.get("category", "") or "")
    keys = _model_keywords(ref_desc)

    pipeline = [
        {"$match": {"sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}}},
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
        {"$sort": {"quantity": -1, "sale_price": 1}},
    ]
    stocks = list(_db.stocks.aggregate(pipeline, allowDiskUse=True))

    out: list[str] = []
    ref_pn_real = str(ref_stock.get("pn", "") or "")
    for s in stocks:
        pn = str(s.get("pn", "") or "")
        if not pn or pn == ref_pn_real:
            continue
        desc = str(s.get("description", "") or "")
        if not desc.strip():
            continue
        manufacturer = str(s.get("manufacturer", "") or "")
        category = str(s.get("category", "") or "")
        if ref_manufacturer and manufacturer != ref_manufacturer:
            continue
        if ref_category and category != ref_category:
            continue

        desc_low = desc.lower()
        if keys and not all(kword in desc_low for kword in keys[:2]):
            continue
        cand_color = _extract_main_color(desc) or ""
        if ref_color and cand_color and cand_color == ref_color:
            continue
        if ref_color and not cand_color:
            continue

        txt = _format_stock_text(s)
        if txt.strip():
            out.append(txt)
        if len(out) >= max(1, k):
            break
    return out


def _is_between_prices_followup(query: str) -> bool:
    q = (query or "").lower()
    hints = [
        "entre esses dois preços",
        "entre estes dois preços",
        "entre os dois preços",
        "entre esses preços",
        "entre estes preços",
        "entre os preços",
    ]
    return any(h in q for h in hints)


def _parse_price_from_doc(doc: str) -> float | None:
    text = str(doc or "")
    match = re.search(r"pre[çc]o:\s*R\$\s*([0-9\.\,]+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).replace(".", "").replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def _retrieve_between_prices_from_active_context(active_docs: list[str]) -> list[str]:
    if len(active_docs) < 2:
        return []
    prices = [p for p in (_parse_price_from_doc(d) for d in active_docs) if p is not None]
    if len(prices) < 2:
        return []

    joined = " ".join(str(d) for d in active_docs[:3])
    product_type = _extract_product_type(joined)
    if not product_type:
        return []
    brand = _extract_brand(joined)

    low = min(float(p) for p in prices)
    high = max(float(p) for p in prices)
    return _buscar_por_tipo_intervalo_preco_texto(
        product_type=product_type,
        min_valor=low,
        max_valor=high,
        k=None,
        brand=brand,
    )


def _retrieve_larger_versions_from_active_context(active_docs: list[str]) -> list[str]:
    if not active_docs:
        return []
    ref_doc = str(active_docs[0] or "")
    ref_price = _parse_price_from_doc(ref_doc)
    if ref_price is None:
        return []
    product_type = _extract_product_type(ref_doc)
    if not product_type:
        return []
    brand = _extract_brand(ref_doc)
    ref_pn = _extract_pn_from_doc(ref_doc)

    docs = _buscar_por_tipo_faixa_preco_texto(
        product_type=product_type,
        intencao="MIN_PRICE",
        valor=float(ref_price),
        k=None,
        brand=brand,
    )
    # Mantém apenas versões realmente "maiores" (preço estritamente superior)
    # e remove o próprio item de referência.
    out: list[str] = []
    ref_key = str(ref_pn or "").strip().upper()
    for d in docs:
        p = _parse_price_from_doc(d)
        if p is None or p <= float(ref_price):
            continue
        pn = _extract_pn_from_doc(d).strip().upper()
        if ref_key and pn == ref_key:
            continue
        out.append(d)
    return out


def _buscar_por_pn_texto(pn_query: str) -> list[str]:
    if not pn_query:
        return []
    pn = pn_query.strip()
    candidates = [pn]
    if len(pn) >= 4 and pn[2] == "X":
        candidates.append(pn[:2] + "." + pn[3:])
    if len(pn) >= 4 and pn[2] == ".":
        candidates.append(pn[:2] + "X" + pn[3:])

    stock = _db.stocks.find_one(
        {
            "pn": {"$in": candidates},
            "sale_price": {"$gt": 0},
            "process": {"$in": _valid_processes},
        },
        sort=[("sale_price", 1)],
    )
    if not stock:
        return []
    prod = _db.products.find_one({"pn": str(stock.get("pn", "") or "")}) or {}
    stock["manufacturer"] = prod.get("manufacturer", "")
    stock["category"] = prod.get("category", "")
    txt = _format_stock_text(stock)
    return [txt] if txt.strip() else []


def _is_total_products_query(query: str) -> bool:
    q = (query or "").lower()
    has_quantity_intent = ("quantos" in q) or ("total" in q) or ("quantidade" in q)
    has_product_scope = ("produto" in q) or ("itens" in q) or ("item" in q)
    has_catalog_hint = any(
        t in q for t in ["estoque", "catálogo", "catalogo", "disponível", "disponivel", "vocês têm", "voces tem"]
    )
    return has_quantity_intent and has_product_scope and has_catalog_hint


def _is_total_manufacturers_query(query: str) -> bool:
    q = (query or "").lower()
    return ("quantos" in q or "total" in q) and "fabricante" in q


def _is_categories_query(query: str) -> bool:
    q = (query or "").lower()
    return "categoria" in q and any(t in q for t in ["quais", "lista", "listar", "tem", "existem"])


def _count_total_products() -> int:
    pipeline = [
        {"$match": {"sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}}},
        {"$group": {"_id": "$pn"}},
        {"$count": "total"},
    ]
    rows = list(_db.stocks.aggregate(pipeline, allowDiskUse=True))
    return int(rows[0]["total"]) if rows else 0


def _count_total_manufacturers() -> int:
    pipeline = [
        {"$match": {"sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}}},
        {"$lookup": {
            "from": "products",
            "localField": "pn",
            "foreignField": "pn",
            "as": "prod_info",
        }},
        {"$addFields": {
            "manufacturer": {"$ifNull": [{"$arrayElemAt": ["$prod_info.manufacturer", 0]}, ""]},
        }},
        {"$match": {"manufacturer": {"$ne": ""}}},
        {"$group": {"_id": "$manufacturer"}},
        {"$count": "total"},
    ]
    rows = list(_db.stocks.aggregate(pipeline, allowDiskUse=True))
    return int(rows[0]["total"]) if rows else 0


def _list_categories() -> list[str]:
    pipeline = [
        {"$match": {"sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}}},
        {"$lookup": {
            "from": "products",
            "localField": "pn",
            "foreignField": "pn",
            "as": "prod_info",
        }},
        {"$addFields": {
            "category": {"$ifNull": [{"$arrayElemAt": ["$prod_info.category", 0]}, ""]},
        }},
        {"$match": {"category": {"$ne": ""}}},
        {"$group": {"_id": "$category"}},
        {"$project": {"_id": 0, "category": "$_id"}},
        {"$sort": {"category": 1}},
    ]
    rows = list(_db.stocks.aggregate(pipeline, allowDiskUse=True))
    return [str(r.get("category", "")).strip() for r in rows if str(r.get("category", "")).strip()]


def _buscar_todos_por_preco_texto(intencao: str, k: int) -> list[str]:
    ordem = 1 if intencao == "MIN_PRICE" else -1
    pipeline = [
        {"$match": {"sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}}},
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
        {"$sort": {"sale_price": ordem}},
        {"$limit": k},
    ]
    stocks = list(_db.stocks.aggregate(pipeline, allowDiskUse=True))
    return _format_stock_list(stocks)


def _buscar_por_tipo_faixa_preco_texto(
    product_type: str,
    intencao: str,
    valor: float,
    k: int | None = None,
    brand: str | None = None,
) -> list[str]:
    termos_tipo = {
        "monitor": ["monitor"],
        "notebook": ["notebook", "laptop", "macbook"],
        "smartphone": ["smartphone", "celular", "iphone", "telefone", "galaxy"],
        "nobreak": ["nobreak", "ups", "smart-ups", "easy ups"],
        "impressora": ["impressora", "printer", "multifuncional", "plotter"],
        "mouse": ["mouse"],
        "teclado": ["teclado", "keyboard"],
        "ssd": ["ssd"],
        "desktop": ["desktop", "pc", "workstation"],
    }.get(product_type, [product_type])
    termos_exclusao_por_tipo = {
        "monitor": [
            "painel", "video wall", "stand alone", "all in one", "aio", "totem",
            "digital signage", "sinalizacao", "sinalização", "display profissional",
            "smart signage", "gaveta", "kvm", "rack",
        ],
        "smartphone": ["tablet", "tab ", "ipad"],
    }
    termos_excluir = termos_exclusao_por_tipo.get(product_type, [])

    pipeline = [
        {"$match": {"sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}}},
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
        {"$sort": {"sale_price": 1, "quantity": -1}},
    ]
    stocks = list(_db.stocks.aggregate(pipeline, allowDiskUse=True))

    textos: list[str] = []
    seen_pn = set()
    for s in stocks:
        desc = str(s.get("description", "")).lower()
        categoria = str(s.get("category", "")).lower()
        fabricante = str(s.get("manufacturer", "")).lower()
        text_match = f"{desc} {categoria} {fabricante}"
        if not any(t in text_match for t in termos_tipo):
            continue
        if any(t in text_match for t in termos_excluir):
            continue
        if brand and brand not in text_match:
            continue

        price = float(s.get("sale_price", 0)) / 100.0
        if intencao == "MAX_PRICE" and price > valor:
            continue
        if intencao == "MIN_PRICE" and price < valor:
            continue

        pn = str(s.get("pn", ""))
        if not pn or pn in seen_pn:
            continue
        seen_pn.add(pn)
        txt = _format_stock_text(s)
        if not txt.strip():
            continue
        textos.append(txt)
        if k is not None and len(textos) >= max(1, k):
            break

    return textos


def _buscar_por_tipo_intervalo_preco_texto(
    product_type: str,
    min_valor: float,
    max_valor: float,
    k: int | None = None,
    brand: str | None = None,
) -> list[str]:
    termos_tipo = {
        "monitor": ["monitor"],
        "notebook": ["notebook", "laptop", "macbook"],
        "smartphone": ["smartphone", "celular", "iphone", "telefone", "galaxy"],
        "nobreak": ["nobreak", "ups", "smart-ups", "easy ups"],
        "impressora": ["impressora", "printer", "multifuncional", "plotter"],
        "mouse": ["mouse"],
        "teclado": ["teclado", "keyboard"],
        "ssd": ["ssd"],
        "desktop": ["desktop", "pc", "workstation"],
    }.get(product_type, [product_type])
    termos_exclusao_por_tipo = {
        "monitor": [
            "painel", "video wall", "stand alone", "all in one", "aio", "totem",
            "digital signage", "sinalizacao", "sinalização", "display profissional",
            "smart signage", "gaveta", "kvm", "rack",
        ],
        "smartphone": ["tablet", "tab ", "ipad"],
    }
    termos_excluir = termos_exclusao_por_tipo.get(product_type, [])

    pipeline = [
        {"$match": {"sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}}},
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
        {"$sort": {"sale_price": 1, "quantity": -1}},
    ]
    stocks = list(_db.stocks.aggregate(pipeline, allowDiskUse=True))

    textos: list[str] = []
    seen_pn = set()
    for s in stocks:
        desc = str(s.get("description", "")).lower()
        categoria = str(s.get("category", "")).lower()
        fabricante = str(s.get("manufacturer", "")).lower()
        text_match = f"{desc} {categoria} {fabricante}"
        if not any(t in text_match for t in termos_tipo):
            continue
        if any(t in text_match for t in termos_excluir):
            continue
        if brand and brand not in text_match:
            continue

        price = float(s.get("sale_price", 0)) / 100.0
        if price < min_valor or price > max_valor:
            continue

        pn = str(s.get("pn", ""))
        if not pn or pn in seen_pn:
            continue
        seen_pn.add(pn)
        txt = _format_stock_text(s)
        if not txt.strip():
            continue
        textos.append(txt)
        if k is not None and len(textos) >= max(1, k):
            break

    return textos


def _buscar_mais_estoque_texto(k: int = 1) -> list[str]:
    pipeline = [
        {"$match": {"sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}}},
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
        {"$limit": 5000},
    ]
    stocks = list(_db.stocks.aggregate(pipeline, allowDiskUse=True))
    if not stocks:
        return []
    stocks.sort(key=lambda s: _qty_to_int(s.get("quantity")), reverse=True)
    filtered = [s for s in stocks if _qty_to_int(s.get("quantity")) > 0]
    return _format_stock_list(filtered[:max(1, k)])


def _buscar_estoque_igual_um_texto(k: int = 10) -> list[str]:
    pipeline = [
        {"$match": {"sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}}},
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
        {"$limit": 5000},
    ]
    stocks = list(_db.stocks.aggregate(pipeline, allowDiskUse=True))
    filtered = [s for s in stocks if _qty_to_int(s.get("quantity")) == 1]
    filtered.sort(key=lambda s: float(s.get("sale_price", 0)))
    return _format_stock_list(filtered[:max(1, k)])


def _buscar_por_fabricante_texto(fabricante_query: str, k: int | None = 6) -> list[str]:
    if not fabricante_query:
        return []
    pipeline = [
        {"$match": {"sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}}},
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
        {"$match": {"manufacturer": {"$regex": fabricante_query, "$options": "i"}}},
        {"$sort": {"sale_price": 1}},
    ]
    if k is not None:
        pipeline.append({"$limit": max(1, k)})
    stocks = list(_db.stocks.aggregate(pipeline, allowDiskUse=True))
    return _format_stock_list(stocks)


def _buscar_por_fabricante_estoque_texto(
    fabricante_query: str,
    intencao: str,
    k: int | None = None,
) -> list[str]:
    if not fabricante_query:
        return []
    ordem_qty = 1 if intencao == "MIN_STOCK" else -1
    pipeline = [
        {"$match": {"sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}}},
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
        {"$match": {"manufacturer": {"$regex": fabricante_query, "$options": "i"}}},
        {"$sort": {"quantity": ordem_qty, "sale_price": 1}},
    ]
    if k is not None:
        pipeline.append({"$limit": max(1, k)})
    stocks = list(_db.stocks.aggregate(pipeline, allowDiskUse=True))
    return _format_stock_list(stocks)


def _listar_fabricantes(k: int | None = None) -> list[str]:
    pipeline = [
        {"$match": {"sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}}},
        {"$lookup": {
            "from": "products",
            "localField": "pn",
            "foreignField": "pn",
            "as": "prod_info",
        }},
        {"$addFields": {
            "manufacturer": {"$ifNull": [{"$arrayElemAt": ["$prod_info.manufacturer", 0]}, ""]},
        }},
        {"$match": {"manufacturer": {"$ne": ""}}},
        {"$group": {"_id": "$manufacturer"}},
        {"$project": {"_id": 0, "manufacturer": "$_id"}},
        {"$sort": {"manufacturer": 1}},
    ]
    if k is not None:
        pipeline.append({"$limit": max(1, k)})
    rows = list(_db.stocks.aggregate(pipeline, allowDiskUse=True))
    return [str(r.get("manufacturer", "")).strip() for r in rows if str(r.get("manufacturer", "")).strip()]


def _is_manufacturer_list_query(query: str) -> bool:
    q = (query or "").lower()
    return (
        "fabricante" in q
        and any(t in q for t in ["liste", "listar", "quais", "todos", "todas", "aparecem", "aparece"])
    )


def _buscar_por_tipo_preco_texto(
    intencao: str,
    product_type: str,
    k: int,
    brand: str | None = None,
) -> list[str]:
    """
    Busca determinística por menor/maior preço para um tipo específico de produto.
    Não depende do Chroma; usa MongoDB com ordenação por sale_price.
    """
    ordem = 1 if intencao == "MIN_PRICE" else -1
    pipeline = [
        {"$match": {"sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}}},
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
        {"$sort": {"sale_price": ordem}},
        {"$limit": 3000},
    ]
    stocks = list(_db.stocks.aggregate(pipeline, allowDiskUse=True))

    termos_tipo = {
        "monitor": ["monitor"],
        "notebook": ["notebook", "laptop", "macbook"],
        "smartphone": ["smartphone", "celular", "iphone", "telefone", "galaxy"],
        "impressora": ["impressora", "printer", "multifuncional"],
        "mouse": ["mouse"],
        "teclado": ["teclado", "keyboard"],
        "ssd": ["ssd"],
        "desktop": ["desktop", "pc", "workstation"],
    }.get(product_type, [product_type])
    termos_exclusao_por_tipo = {
        "monitor": [
            "base", "suporte", "pedestal", "braço", "braco", "cabo", "adaptador",
            "painel", "video wall", "stand alone", "all in one", "aio", "totem",
            "digital signage", "sinalizacao", "sinalização", "display profissional",
            "smart signage", "gaveta", "kvm", "rack",
        ],
        "smartphone": ["tablet", "tab ", "ipad"],
    }
    termos_excluir = termos_exclusao_por_tipo.get(product_type, [])

    textos = []
    for s in stocks:
        pn = _normalize_pn_display(str(s.get("pn", "N/A")))
        desc = str(s.get("description", ""))
        categoria = str(s.get("category", ""))
        fabricante = str(s.get("manufacturer", ""))

        texto_match = f"{desc} {categoria} {fabricante}".lower()
        if not any(t in texto_match for t in termos_tipo):
            continue
        if brand and brand not in texto_match:
            continue
        if any(t in texto_match for t in termos_excluir):
            continue

        price = float(s.get("sale_price", 0)) / 100.0
        qty = int(float(s.get("quantity", 0))) if s.get("quantity") else 0
        uf = str(s.get("uf", ""))
        textos.append(
            f"{desc} | PN: {pn} | Fabricante: {fabricante} | Categoria: {categoria} | "
            f"Preço: R$ {_format_brl(price)} | Estoque: {qty} | UF: {uf}"
        )

        if len(textos) >= k:
            break

    return textos


def _buscar_por_tipo_catalogo_texto(
    product_type: str,
    k: int | None = 5,
    brand: str | None = None,
) -> list[str]:
    termos_tipo = {
        "monitor": ["monitor"],
        "notebook": ["notebook", "laptop", "macbook"],
        "smartphone": ["smartphone", "celular", "iphone", "telefone", "galaxy"],
        "nobreak": ["nobreak", "ups", "smart-ups", "easy ups"],
        "impressora": ["impressora", "printer", "multifuncional", "plotter"],
        "mouse": ["mouse"],
        "teclado": ["teclado", "keyboard"],
        "ssd": ["ssd"],
        "desktop": ["desktop", "pc", "workstation"],
    }.get(product_type, [product_type])
    termos_exclusao_por_tipo = {
        "monitor": [
            "painel", "video wall", "stand alone", "all in one", "aio", "totem",
            "digital signage", "sinalizacao", "sinalização", "display profissional",
            "smart signage", "gaveta", "kvm", "rack",
        ],
        "smartphone": ["tablet", "tab ", "ipad"],
    }
    termos_excluir = termos_exclusao_por_tipo.get(product_type, [])

    pipeline = [
        {"$match": {"sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}}},
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
        {"$sort": {"quantity": -1, "sale_price": 1}},
    ]
    stocks = list(_db.stocks.aggregate(pipeline, allowDiskUse=True))

    textos = []
    seen_pn = set()
    for s in stocks:
        desc = str(s.get("description", "")).lower()
        categoria = str(s.get("category", "")).lower()
        fabricante = str(s.get("manufacturer", "")).lower()
        text_match = f"{desc} {categoria} {fabricante}"
        if not any(t in text_match for t in termos_tipo):
            continue
        if any(t in text_match for t in termos_excluir):
            continue
        if brand and brand not in text_match:
            continue
        pn = str(s.get("pn", ""))
        if pn in seen_pn:
            continue
        seen_pn.add(pn)
        txt = _format_stock_text(s)
        if not txt.strip():
            continue
        textos.append(txt)
        if k is not None and len(textos) >= max(1, k):
            break
    return textos


def _buscar_semanticamente_ordenado_texto(
    query: str,
    k: int = 5,
    product_type: str | None = None,
    brand: str | None = None,
) -> list[str]:
    """
    Busca semântica no Chroma e mantém ordenação por relevância (distância).
    Depois enriquece cada PN com dados atuais do Mongo (preço/estoque).
    """
    embeddings = _get_embeddings()
    query_embedding = embeddings.embed_query(query)
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=max(40, k * 8),
    )

    ids = results.get("ids", [[]])[0] if results.get("ids") else []
    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []
    if not ids or not metadatas:
        return []

    type_terms = {
        "monitor": ["monitor"],
        "notebook": ["notebook", "laptop", "macbook"],
        "smartphone": ["smartphone", "celular", "iphone", "telefone", "galaxy"],
        "nobreak": ["nobreak", "ups", "smart-ups", "easy ups"],
        "impressora": ["impressora", "printer", "multifuncional", "plotter"],
        "mouse": ["mouse"],
        "teclado": ["teclado", "keyboard"],
        "ssd": ["ssd"],
        "desktop": ["desktop", "pc", "workstation"],
    }.get(product_type or "", [])

    ranked = []
    seen_pn = set()
    for idx, md in enumerate(metadatas):
        pn = str((md or {}).get("pn", "") or "")
        if not pn or pn in seen_pn:
            continue
        seen_pn.add(pn)

        # Tenta casar o PN em stocks atuais (preço > 0 e processos válidos)
        stock = _db.stocks.find_one(
            {"pn": pn, "sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}},
            sort=[("sale_price", 1)],
        )
        if not stock:
            continue
        prod = _db.products.find_one({"pn": pn}) or {}
        text_match = f"{stock.get('description','')} {prod.get('category','')} {prod.get('manufacturer','')}".lower()

        if type_terms and not any(t in text_match for t in type_terms):
            continue
        if brand and brand not in text_match:
            continue

        stock["manufacturer"] = prod.get("manufacturer", "")
        stock["category"] = prod.get("category", "")
        dist = distances[idx] if idx < len(distances) else 999.0
        txt = _format_stock_text(stock)
        if not txt.strip():
            continue
        ranked.append((float(dist), txt))

        if len(ranked) >= k:
            break

    ranked.sort(key=lambda x: x[0]) # menor distância = mais relevante
    return [txt for _, txt in ranked[:k]]


def _enriquecer_docs_com_preco(docs: list[str], metadatas: list[dict]) -> list[str]:
    saida = []
    for idx, doc in enumerate(docs):
        md = metadatas[idx] if idx < len(metadatas) else {}
        pn = md.get("pn")
        if not pn:
            saida.append(str(doc))
            continue
        pn_display = _normalize_pn_display(str(pn))

        stock = _db.stocks.find_one(
            {"pn": pn, "sale_price": {"$gt": 0}, "process": {"$in": _valid_processes}},
            sort=[("sale_price", 1)],
        )
        prod = _db.products.find_one({"pn": pn}) or {}

        if not stock:
            saida.append(str(doc))
            continue

        price = float(stock.get("sale_price", 0)) / 100.0
        qty = int(float(stock.get("quantity", 0))) if stock.get("quantity") else 0
        uf = str(stock.get("uf", ""))
        fabricante = str(prod.get("manufacturer", md.get("manufacturer", "")))
        categoria = str(prod.get("category", md.get("category", "")))

        enriquecido = (
            f"{doc} | PN: {pn_display} | Fabricante: {fabricante} | Categoria: {categoria} | "
            f"Preço: R$ {_format_brl(price)} | Estoque: {qty} | UF: {uf}"
        )
        saida.append(enriquecido)
    return saida


def _extract_pn_from_doc(doc: str) -> str:
    text = str(doc or "")
    if "| PN:" in text:
        return text.rsplit("| PN:", 1)[1].split("|", 1)[0].strip()
    match = re.search(r"\bPN[:\s]+([A-Z0-9\.\-\/#]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _pn_lookup_candidates(pn_display: str) -> list[str]:
    pn = str(pn_display or "").strip()
    if not pn:
        return []
    candidates = [pn]
    # Se o PN exibido tiver X na terceira posição (ex.: MMXG3BZ/A),
    # tenta também com "." para bater com o dado real do Mongo.
    if len(pn) >= 4 and pn[2] == "X":
        candidates.append(pn[:2] + "." + pn[3:])
    return list(dict.fromkeys(candidates))


def refresh_docs_live_from_mongodb(docs: list[str]) -> list[str]:
    """
    Revalida PREÇO e ESTOQUE no MongoDB em tempo de resposta.
    Não usa valores cacheados do contexto para esses campos.
    """
    refreshed: list[str] = []
    for doc in (docs or []):
        pn_from_doc = _extract_pn_from_doc(doc)
        if not pn_from_doc:
            refreshed.append(str(doc))
            continue

        candidates = _pn_lookup_candidates(pn_from_doc)
        stock = _db.stocks.find_one(
            {
                "pn": {"$in": candidates},
                "sale_price": {"$gt": 0},
                "process": {"$in": _valid_processes},
            },
            sort=[("sale_price", 1)],
        )
        if not stock:
            refreshed.append(str(doc))
            continue

        pn_real = str(stock.get("pn", "") or "")
        prod = _db.products.find_one({"pn": pn_real}) or {}
        stock["manufacturer"] = prod.get("manufacturer", stock.get("manufacturer", ""))
        stock["category"] = prod.get("category", stock.get("category", ""))
        txt = _format_stock_text(stock)
        if txt.strip():
            refreshed.append(txt)

    return refreshed


def retrieve_docs(query: str, k: int = 4) -> list[str]:
    """
    Recupera documentos diretamente no ChromaDB (HTTP), usando embeddings do Ollama.

    Mantém a mesma interface de antes: retorna list[str] com o texto (page_content).
    """
    raw_query = query
    forced_scope = None
    if query.startswith("[SCOPE:GLOBAL] "):
        forced_scope = "global"
        query = query.replace("[SCOPE:GLOBAL] ", "", 1)
    elif query.startswith("[SCOPE:CONTEXT] "):
        forced_scope = "context"
        query = query.replace("[SCOPE:CONTEXT] ", "", 1)

    is_global, intencao = _is_global_price_query(query)
    if forced_scope == "global":
        is_global = True if intencao else False
    elif forced_scope == "context":
        is_global = False
    if is_global and intencao:
        return _buscar_todos_por_preco_texto(intencao=intencao, k=max(1, k))

    # Consulta por preço com tipo específico (ex.: "monitor mais barato do estoque")
    intencao_preco = _extract_price_intent(query)
    product_type = _extract_product_type(query)
    brand = _extract_brand(query)
    if intencao_preco and product_type:
        docs_tipo = _buscar_por_tipo_preco_texto(
            intencao=intencao_preco,
            product_type=product_type,
            k=max(1, k),
            brand=brand,
        )
        if docs_tipo:
            return docs_tipo

    embeddings = _get_embeddings()
    query_embedding = embeddings.embed_query(query)

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    docs = results.get("documents", [[]])
    if not docs or not docs[0]:
        return []
    metadatas = results.get("metadatas", [[]])
    docs_texto = [str(d) for d in docs[0]]
    metadados = metadatas[0] if metadatas and metadatas[0] else []
    return _enriquecer_docs_com_preco(docs_texto, metadados)


def get_semantic_best_distance(query: str) -> float | None:
    """
    Retorna a menor distância semântica do Chroma para a query.
    Quanto menor, melhor. Retorna None se não houver resultado.
    """
    q = str(query or "")
    if q.startswith("[SCOPE:GLOBAL] "):
        q = q.replace("[SCOPE:GLOBAL] ", "", 1)
    elif q.startswith("[SCOPE:CONTEXT] "):
        q = q.replace("[SCOPE:CONTEXT] ", "", 1)
    try:
        embeddings = _get_embeddings()
        query_embedding = embeddings.embed_query(q)
        results = _collection.query(query_embeddings=[query_embedding], n_results=1)
        distances = results.get("distances", [[]])
        if not distances or not distances[0]:
            return None
        return float(distances[0][0])
    except Exception:
        return None


def retrieve_deterministic(query: str, active_docs: list[str] | None = None, k: int = 4) -> tuple[bool, list[str]]:
    """
    Rotas determinísticas para intents críticas.
    Retorna (handled, docs).
    """
    q = (query or "").lower().strip()
    active_docs = active_docs or []

    is_short_followup = q in {
        "tem mais unidades?", "tem mais unidades",
        "quantas unidades?", "quantas unidades",
        "quantas unidades tem?", "quantas unidades tem",
        "qual o pn dele?", "qual pn dele?",
    }
    is_followup_ref = is_short_followup or any(
        t in q for t in [" dele", " desse", " deste", "do item", "desse item", "deste item"]
    )

    # Follow-ups pontuais sobre o item atual
    if is_followup_ref and any(t in q for t in ["qual o pn", "qual pn", "part number", "qual partnumber"]):
        return (True, active_docs[:1] if active_docs else [])
    if is_followup_ref and any(t in q for t in ["tem mais unidades", "quantas unidades", "qual o estoque", "estoque dele"]):
        return (True, active_docs[:1] if active_docs else [])
    if is_followup_ref and any(t in q for t in ["diferença de preço", "diferenca de preco", "diferença entre os dois", "diferenca entre os dois"]):
        return (True, active_docs[:2] if len(active_docs) >= 2 else active_docs[:1])
    if _is_larger_version_followup(query):
        docs = _retrieve_larger_versions_from_active_context(active_docs)
        return (True, docs)
    if _is_color_variation_followup(query):
        docs = _buscar_outras_cores_do_modelo(active_docs, k=12)
        return (True, docs)
    if _is_between_prices_followup(query):
        docs = _retrieve_between_prices_from_active_context(active_docs)
        if docs:
            return (True, docs)

    # Estatísticas globais
    if _is_total_products_query(query):
        total = _count_total_products()
        return (True, [f"__STAT_TOTAL_PRODUCTS__:{total}"])
    if _is_total_manufacturers_query(query):
        total = _count_total_manufacturers()
        return (True, [f"__STAT_TOTAL_MANUFACTURERS__:{total}"])
    if _is_categories_query(query):
        categories = _list_categories()
        return (True, [f"__STAT_CATEGORIES__:{' || '.join(categories)}"])

    # Busca exata por PN (ex.: "Quanto custa o MMXG3BZ/A?")
    pn_query = _extract_pn_from_query(query)
    if pn_query:
        docs = _buscar_por_pn_texto(pn_query)
        if docs:
            return (True, docs)

    # Fabricante explícito
    if any(t in q for t in ["produtos da ", "tem produtos da ", "tem algum produto da "]):
        brand = _extract_brand(query)
        if brand:
            stock_intent = _extract_stock_intent(query)
            if stock_intent:
                docs = _buscar_por_fabricante_estoque_texto(brand, intencao=stock_intent, k=None)
                if docs:
                    return (True, docs)
            docs = _buscar_por_fabricante_texto(brand, k=None)
            if docs:
                return (True, docs)

    # Lista de fabricantes
    if _is_manufacturer_list_query(query):
        docs = _listar_fabricantes(k=None)
        if docs:
            return (True, docs)

    # Listagem por tipo explícito ("tem nobreak?", "tem outros nobreaks?", etc.)
    listing_terms = ["tem ", "quais", "listar", "lista", "disponível", "disponivel", "outros", "outras"]
    product_type = _extract_product_type(query)
    if product_type and any(t in q for t in listing_terms):
        brand = _extract_brand(query)
        between = _extract_price_between(query)
        threshold = _extract_price_threshold(query)

        # Ex.: "tem celular entre 1000 e 2000"
        if between:
            docs = _buscar_por_tipo_intervalo_preco_texto(
                product_type=product_type,
                min_valor=between[0],
                max_valor=between[1],
                k=None,
                brand=brand,
            )
            if docs:
                return (True, docs)

        # Ex.: "tem iphone com preço abaixo de 8000"
        if threshold:
            docs = _buscar_por_tipo_faixa_preco_texto(
                product_type=product_type,
                intencao=threshold[0],
                valor=threshold[1],
                k=None,
                brand=brand,
            )
            if docs:
                return (True, docs)

        # Busca determinística completa no Mongo para retornar total real.
        docs = _buscar_por_tipo_catalogo_texto(product_type, k=None, brand=brand)
        if docs:
            return (True, docs)

    # Estoque: maior quantidade
    if any(t in q for t in ["mais unidades", "maior estoque", "mais em estoque"]):
        docs = _buscar_mais_estoque_texto(k=1)
        return (True, docs)

    # Estoque igual a 1
    if any(t in q for t in ["1 unidade", "uma unidade", "apenas 1", "só 1", "somente 1"]):
        docs = _buscar_estoque_igual_um_texto(k=max(2, k))
        return (True, docs)

    return (False, [])