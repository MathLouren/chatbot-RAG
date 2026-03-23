from langgraph.graph import StateGraph, END
from typing import TypedDict
import re
import warnings
from retrievers import retrieve_docs, retrieve_deterministic, get_semantic_best_distance
from memory import (
    format_history,
    save_message,
    get_scope_preference,
    set_scope_preference,
    get_active_docs,
    set_active_docs,
    get_listing_state,
    set_listing_state,
)
from config import CHAT_MODEL, OLLAMA_BASE_URL
from system_prompt import SYSTEM_PROMPT

try:
    if hasattr(__import__("importlib"), "util") and __import__("importlib").util.find_spec("langchain_ollama") is not None:
        # type: ignore[import-untyped]
        from langchain_ollama import OllamaLLM # type: ignore
    else:
        raise ImportError
except Exception:
    # type: ignore[import-untyped]
    from langchain_community.llms import Ollama # type: ignore


# ── Estado compartilhado entre os nós ──────────────────────────────────────
class ChatState(TypedDict):
    session_id: str
    question: str
    rewritten_question: str
    chroma_context: list[str]
    chat_history: str
    active_docs: list[str]
    needs_clarification: bool
    scope_preference: str
    listing_docs: list[str]
    listing_shown: int
    handled: bool
    answer_override: str
    answer: str


# ── Nó 1: Carrega histórico do MongoDB ─────────────────────────────────────
def node_fetch_history(state: ChatState) -> dict:
    history = format_history(state["session_id"])
    scope_preference = get_scope_preference(state["session_id"]) or ""
    active_docs = get_active_docs(state["session_id"])
    listing_state = get_listing_state(state["session_id"])
    return {
        "chat_history": history,
        "scope_preference": scope_preference,
        "active_docs": active_docs,
        "listing_docs": listing_state.get("listing_docs", []),
        "listing_shown": listing_state.get("listing_shown", 0),
    }


def _build_llm(temp: float = 0.3):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "OllamaLLM" in globals():
            return OllamaLLM(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=temp)
        return Ollama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=temp)


def _infer_topic_from_history(history: str) -> str | None:
    h = history.lower()
    topic_map = [
        ("iphone", ["iphone"]),
        ("smartphone", ["smartphone", "celular", "galaxy", "moto"]),
        ("monitor", ["monitor", "display"]),
        ("notebook", ["notebook", "laptop", "macbook"]),
        ("impressora", ["impressora", "plotter", "printer"]),
        ("ssd", ["ssd"]),
        ("mouse", ["mouse"]),
        ("teclado", ["teclado", "keyboard"]),
    ]
    for topic, keys in topic_map:
        if any(k in h for k in keys):
            return topic
    return None


def _infer_topic_from_docs(docs: list[str]) -> str | None:
    if not docs:
        return None
    text = " ".join(docs[:3]).lower()
    topic_map = [
        ("nobreak", ["nobreak", "smart-ups", "easy ups", "ups"]),
        ("iphone", ["iphone"]),
        ("smartphone", ["smartphone", "celular", "galaxy", "moto"]),
        ("monitor", ["monitor", "display"]),
        ("notebook", ["notebook", "laptop", "macbook"]),
        ("impressora", ["impressora", "plotter", "printer"]),
        ("ssd", ["ssd"]),
        ("mouse", ["mouse"]),
        ("teclado", ["teclado", "keyboard"]),
    ]
    for topic, keys in topic_map:
        if any(k in text for k in keys):
            return topic
    return None


def _is_contextual_price_followup(question: str) -> bool:
    q = question.lower().strip()
    price_followups = {
        "qual o mais barato?",
        "qual o mais barato",
        "e o mais barato?",
        "e o mais barato",
        "qual o mais caro?",
        "qual o mais caro",
        "e o mais caro?",
        "e o mais caro",
    }
    if q in price_followups:
        return True
    if _is_price_query(q)[0] and any(
        ref in q for ref in ["desses", "desses", "dessas", "dessa lista", "dessa seleção", "dessa selecao", "destes", "destas"]
    ):
        return True
    return False


def _extract_category_slot(question: str) -> str | None:
    q = (question or "").lower()
    categories = [
        "iphone", "celular", "smartphone", "monitor", "notebook", "nobreak",
        "mouse", "teclado", "ssd", "impressora", "tablet",
    ]
    for c in categories:
        if re.search(rf"\b{re.escape(c)}\b", q):
            return c
    return None


def _is_all_scope_followup(question: str) -> bool:
    q = (question or "").lower().strip()
    return q in {"de todos", "de tudo", "geral", "todos", "tudo"}


def _is_category_plus_catalog_price_query(question: str) -> bool:
    q = (question or "").lower()
    has_price, intent = _is_price_query(q)
    has_category = _extract_category_slot(q) is not None
    # Regra restrita ao caso pedido: "categoria + mais barato + catálogo/geral".
    has_global_hint = any(t in q for t in ["catalogo", "catálogo", "de todos", "de tudo", "geral"])
    return has_price and intent == "MIN_PRICE" and has_category and has_global_hint


def _clarification_for_ambiguity(question: str, active_docs: list[str]) -> str | None:
    q = (question or "").strip().lower()
    if active_docs:
        return None
    if _is_global_price_anchor_query(q):
        return None
    # Se já veio PN explícito, deixa seguir para busca determinística por PN.
    if re.search(r"\b[A-Z0-9][A-Z0-9\.\-\/#]{4,}\b", (question or "").upper()):
        return None
    if not q:
        return "Pode especificar qual produto ou categoria você procura?"
    if q in {"tem", "tem?", "quero", "qual", "qual?"}:
        return "Pode especificar qual produto ou categoria você procura?"
    if q in {"o mais barato", "mais barato", "o mais caro", "mais caro", "quero o mais barato", "quero o mais caro"}:
        return "Mais barato/caro de qual categoria? Celular, monitor, nobreak, mouse...?"
    if re.search(r"\b(esse|esse aqui|aquele|isso)\b", q):
        return "Pode me dizer a qual produto você está se referindo?"
    if re.search(r"^\w{1,4}\??$", q):
        return "Pode detalhar um pouco mais o que você está buscando?"
    if _is_price_query(q)[0] and _extract_category_slot(q) is None:
        return "Mais barato/caro de qual categoria? Celular, monitor, nobreak, mouse...?"
    return None


def _clarification_for_low_confidence(question: str, active_docs: list[str], best_distance: float | None) -> str | None:
    if active_docs:
        return None
    if best_distance is None:
        return None
    # Distância alta => baixa confiança semântica.
    if best_distance > 1.05:
        return _clarification_for_ambiguity(question, active_docs) or (
            "Não entendi com segurança sua solicitação. Pode especificar o produto, marca ou categoria?"
        )
    return None


def _is_color_variation_followup(question: str) -> bool:
    q = (question or "").lower().strip()
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


def _is_price_query(question: str) -> tuple[bool, str]:
    q = question.lower()
    is_min = any(k in q for k in ["mais barato", "menor preço", "menor preco", "preço mais baixo", "preco mais baixo"])
    is_max = any(k in q for k in ["mais caro", "maior preço", "maior preco", "preço mais alto", "preco mais alto"])
    if is_min:
        return True, "MIN_PRICE"
    if is_max:
        return True, "MAX_PRICE"
    return False, ""


def _get_last_user_question(chat_history: str) -> str:
    history = str(chat_history or "")
    user_lines = [line for line in history.splitlines() if line.startswith("USER:")]
    if not user_lines:
        return ""
    return user_lines[-1].split("USER:", 1)[1].strip()


def _is_specific_type_query(text: str) -> bool:
    q = (text or "").lower()
    specific_terms = [
        "iphone", "celular", "smartphone", "monitor", "notebook", "ssd", "impressora",
        "mouse", "teclado", "tablet", "nobreak", "galaxy", "samsung", "apple",
        "motorola", "lenovo", "dell", "hp", "asus", "acer", "xiaomi",
    ]
    return any(t in q for t in specific_terms)


def _is_global_price_anchor_query(text: str) -> bool:
    q = (text or "").lower()
    if not _is_price_query(q)[0]:
        return False
    if _is_specific_type_query(q):
        return False
    return any(t in q for t in ["produto", "item", "loja", "estoque", "catálogo", "catalogo", "geral"])


def _build_price_query_from_category(category: str, intent: str) -> str:
    cat = (category or "").strip().lower()
    if not cat:
        return ""
    label = "mais barato" if intent == "MIN_PRICE" else "mais caro"
    return f"qual o {cat} {label}?"


def _parse_price_value_from_doc(doc: str) -> float:
    text = str(doc or "")
    token = ""
    if "Preço: " in text:
        token = text.split("Preço: ", 1)[1].split("|", 1)[0].strip()
    elif "Pre�o: " in text:
        token = text.split("Pre�o: ", 1)[1].split("|", 1)[0].strip()
    if not token:
        return float("inf")
    try:
        return float(token.replace("R$", "").replace(".", "").replace(",", ".").strip())
    except Exception:
        return float("inf")


def _extract_requested_count(question: str, default_value: int = 1, max_value: int = 10) -> int:
    q = question.lower()
    # Primeiro tenta número explícito
    digits = "".join(ch if ch.isdigit() else " " for ch in q).split()
    if digits:
        try:
            n = int(digits[0])
            return max(1, min(max_value, n))
        except ValueError:
            pass

    # Depois tenta palavras comuns
    words_map = {
        "um": 1, "uma": 1,
        "dois": 2, "duas": 2,
        "tres": 3, "três": 3,
        "quatro": 4, "cinco": 5,
    }
    for w, n in words_map.items():
        if w in q:
            return max(1, min(max_value, n))
    return default_value


def _build_price_answer_from_context(question: str, context_docs: list[str]) -> str | None:
    if not context_docs:
        return None
    is_price, intent = _is_price_query(question)
    if not is_price:
        return None

    # Não assumir ordem do contexto; sempre ordenar pelo preço parseado.
    n = _extract_requested_count(question, default_value=1, max_value=min(10, len(context_docs)))

    def parse_item(doc: str) -> tuple[str, str, str]:
        def normalize_pn_display(pn: str) -> str:
            pn = (pn or "").strip()
            if len(pn) >= 4 and pn[2] == ".":
                return pn[:2] + "X" + pn[3:]
            return pn

        # Se houver "Descrição:", priorizar esse campo para não retornar
        # prefixos como "Fabricante: ... | Categoria: ..."
        desc = doc.split("| PN:")[0].strip()
        if "Descrição:" in doc:
            desc = doc.split("Descrição:", 1)[1].split("|", 1)[0].strip()
        elif "Descricao:" in doc:
            desc = doc.split("Descricao:", 1)[1].split("|", 1)[0].strip()

        pn = "N/A"
        if "| PN:" in doc:
            # Usa o último PN do texto (o bloco enriquecido mais recente)
            pn = doc.rsplit("| PN:", 1)[1].split("|", 1)[0].strip()
            pn = normalize_pn_display(pn)
        preco = "N/A"
        if "Preço: " in doc:
            preco = doc.split("Preço: ", 1)[1].split("|", 1)[0].strip()
        elif "Pre�o: " in doc:
            preco = doc.split("Pre�o: ", 1)[1].split("|", 1)[0].strip()
        return desc, pn, preco

    def parse_preco_num(preco_txt: str) -> float:
        try:
            raw = preco_txt.replace("R$", "").replace(".", "").replace(",", ".").strip()
            return float(raw)
        except Exception:
            return float("inf")

    label = "mais barato" if intent == "MIN_PRICE" else "mais caro"
    parsed = [parse_item(d) for d in context_docs]
    if intent == "MIN_PRICE":
        parsed.sort(key=lambda x: parse_preco_num(x[2]))
    else:
        parsed.sort(key=lambda x: parse_preco_num(x[2]), reverse=True)

    chosen = parsed[:n]
    if n == 1 and chosen:
        desc, pn, preco = chosen[0]
        return f"O item {label} é {desc} (PN: {pn}), com preço de {preco}."

    linhas = []
    for idx, (desc, pn, preco) in enumerate(chosen, 1):
        linhas.append(f"{idx}. {desc} (PN: {pn}) - {preco}")
    return f"Os {n} itens {label}s são:\n" + "\n".join(linhas)


def _extract_last_assistant_pn(chat_history: str) -> str | None:
    history = str(chat_history or "")
    assistant_lines = [line for line in history.splitlines() if line.startswith("ASSISTANT:")]
    if not assistant_lines:
        return None
    last = assistant_lines[-1]
    matches = re.findall(r"PN:\s*([A-Z0-9\.\-\/#]+)", last, flags=re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].strip()


def _extract_last_assistant_pns(chat_history: str) -> list[str]:
    history = str(chat_history or "")
    assistant_lines = [line for line in history.splitlines() if line.startswith("ASSISTANT:")]
    if not assistant_lines:
        return []
    last = assistant_lines[-1]
    matches = re.findall(r"PN:\s*([A-Z0-9\.\-\/#]+)", last, flags=re.IGNORECASE)
    cleaned = [m.strip() for m in matches if m.strip()]
    return list(dict.fromkeys(cleaned))


def _extract_last_assistant_line(chat_history: str) -> str:
    history = str(chat_history or "")
    assistant_lines = [line for line in history.splitlines() if line.startswith("ASSISTANT:")]
    return assistant_lines[-1] if assistant_lines else ""


def _pn_key(pn: str) -> str:
    p = (pn or "").strip().upper()
    if len(p) >= 4 and p[2] == ".":
        p = p[:2] + "X" + p[3:]
    return p


def _build_followup_answer(question: str, active_docs: list[str], chat_history: str = "") -> str | None:
    q = question.lower()
    if not active_docs:
        return None
    doc = active_docs[0]
    parsed = _parse_context_doc(doc)

    is_short_followup_stock = q.strip() in {
        "tem mais unidades?", "tem mais unidades",
        "quantas unidades?", "quantas unidades",
        "quantas unidades tem?", "quantas unidades tem",
    }
    is_followup_ref = is_short_followup_stock or any(
        t in q for t in [" dele", " desse", " deste", "do item", "desse item", "deste item"]
    )

    last_assistant_pn = _extract_last_assistant_pn(chat_history)
    if is_followup_ref and any(t in q for t in ["qual o pn", "qual pn", "part number", "qual partnumber"]):
        if last_assistant_pn:
            return f"O PN do item é {last_assistant_pn}."
        return f"O PN do item é {parsed['pn']}."
    if ("pn do mais caro" in q or "pn do mais barato" in q) and active_docs:
        last_line = _extract_last_assistant_line(chat_history).lower()
        last_pns_raw = _extract_last_assistant_pns(chat_history)
        # Regra direta para comparação recém-respondida entre dois itens.
        if "diferença de preço entre" in last_line and len(last_pns_raw) >= 2:
            if "mais caro" in q:
                return f"O PN do item é {last_pns_raw[1]}."
            return f"O PN do item é {last_pns_raw[0]}."

        parsed_docs = [_parse_context_doc(d) for d in active_docs]
        by_pn = {}
        for p in parsed_docs:
            key = _pn_key(p["pn"])
            try:
                v = float(str(p["preco"]).replace("R$", "").replace(".", "").replace(",", ".").strip())
            except Exception:
                v = float("inf")
            by_pn[key] = (v, p)

        # Se a última resposta comparou dois itens, prioriza exatamente esses dois.
        last_pns = [_pn_key(pn) for pn in _extract_last_assistant_pns(chat_history)]
        comparable = [by_pn[pn] for pn in last_pns if pn in by_pn]
        candidates = comparable if len(comparable) >= 2 else list(by_pn.values())
        if not candidates:
            return None
        if "mais caro" in q:
            chosen = max(candidates, key=lambda x: x[0])[1]
        else:
            chosen = min(candidates, key=lambda x: x[0])[1]
        return f"O PN do item é {chosen['pn']}."
    if is_followup_ref and any(t in q for t in ["tem mais unidades", "quantas unidades", "qual o estoque", "estoque dele"]):
        if last_assistant_pn:
            target = _pn_key(last_assistant_pn)
            for d in active_docs:
                pd = _parse_context_doc(d)
                if _pn_key(pd["pn"]) == target:
                    return f"Temos {pd['estoque']} unidades em estoque desse item."
        return f"Temos {parsed['estoque']} unidades em estoque desse item."
    if any(t in q for t in ["diferença de preço", "diferenca de preco", "diferença entre os dois", "diferenca entre os dois", "qual a diferença", "diferença dos dois"]) and len(active_docs) >= 2:
        p1 = _parse_context_doc(active_docs[0])
        p2 = _parse_context_doc(active_docs[1])
        try:
            v1 = float(p1["preco"].replace("R$", "").replace(".", "").replace(",", ".").strip())
            v2 = float(p2["preco"].replace("R$", "").replace(".", "").replace(",", ".").strip())
            diff = abs(v1 - v2)
            diff_txt = f"R$ {diff:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            return (
                f"A diferença de preço entre {p1['produto']} (PN: {p1['pn']}) "
                f"e {p2['produto']} (PN: {p2['pn']}) é {diff_txt}."
            )
        except Exception:
            return (
                f"{p1['produto']} custa {p1['preco']} e {p2['produto']} custa {p2['preco']}."
            )
    return None


def _build_stock_answer_from_context(question: str, context_docs: list[str]) -> str | None:
    q = question.lower()
    if not context_docs:
        return None

    # Maior estoque (determinístico, primeiro doc do retriever determinístico)
    if any(t in q for t in ["mais unidades", "maior estoque", "mais em estoque"]) and any(
        t in q for t in ["qual produto", "quais produtos", "produto com", "item com"]
    ):
        top = _parse_context_doc(context_docs[0])
        return (
            f"O produto com maior estoque é {top['produto']} (PN: {top['pn']}), "
            f"com {top['estoque']} unidades disponíveis."
        )

    # Menor estoque (determinístico, primeiro doc já vem ordenado por quantidade ascendente)
    if any(t in q for t in ["menor estoque", "menos estoque", "estoque mais baixo"]) and any(
        t in q for t in ["qual produto", "quais produtos", "produto com", "item com", "produto da", "item da"]
    ):
        top = _parse_context_doc(context_docs[0])
        return (
            f"O produto com menor estoque é {top['produto']} (PN: {top['pn']}), "
            f"com {top['estoque']} unidades disponíveis."
        )

    # Lista de produtos com estoque = 1
    if any(t in q for t in ["1 unidade", "uma unidade", "apenas 1", "só 1", "somente 1"]) and any(
        t in q for t in ["qual produto", "quais produtos", "produto com", "item com"]
    ):
        linhas = []
        for i, d in enumerate(context_docs[:10], 1):
            p = _parse_context_doc(d)
            linhas.append(f"{i}. {p['produto']} (PN: {p['pn']}) - {p['preco']}")
        if linhas:
            return "Produtos com 1 unidade em estoque:\n" + "\n".join(linhas)
    return None


def _parse_context_doc(doc: str) -> dict:
    """
    Faz parse defensivo do texto retornado pelo retriever.
    Formato esperado:
    DESC | PN: X | Fabricante: Y | Categoria: Z | Preço: R$ ... | Estoque: N | UF: ES
    """
    data = {
        "produto": "",
        "pn": "Consultar",
        "fabricante": "Consultar",
        "categoria": "Consultar",
        "preco": "Consultar",
        "estoque": "Consultar",
    }
    if not doc:
        return data

    def _clean_text(value: str) -> str:
        return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()

    raw_text = str(doc or "")
    parts = [_clean_text(p) for p in raw_text.split("|")]
    if parts:
        data["produto"] = parts[0] or "Produto sem descrição"

    for p in parts[1:]:
        low = p.lower()
        if low.startswith("pn:"):
            data["pn"] = p.split(":", 1)[1].strip() or "Consultar"
        elif low.startswith("fabricante:"):
            data["fabricante"] = p.split(":", 1)[1].strip() or "Consultar"
        elif low.startswith("categoria:"):
            data["categoria"] = p.split(":", 1)[1].strip() or "Consultar"
        elif low.startswith("descrição:") or low.startswith("descricao:"):
            data["produto"] = _clean_text(p.split(":", 1)[1]) or data["produto"]
        elif low.startswith("preço:") or low.startswith("preco:"):
            data["preco"] = _clean_text(p.split(":", 1)[1]) or "Consultar"
        elif low.startswith("estoque:"):
            data["estoque"] = _clean_text(p.split(":", 1)[1]) or "Consultar"
    data["produto"] = _clean_text(data["produto"]) or "Produto sem descrição"
    data["pn"] = _clean_text(data["pn"]) or "Consultar"
    data["fabricante"] = _clean_text(data["fabricante"]) or "Consultar"
    data["categoria"] = _clean_text(data["categoria"]) or "Consultar"
    return data


def _is_catalog_listing_query(question: str) -> bool:
    """
    Listagem de catálogo (tabela). Nunca confundir com pergunta de preço min/max.
    Evita substring perigosa: 'tem' casava dentro de palavras como 'item', 'sistema'.
    """
    q = (question or "").lower().strip()
    if _is_price_query(q)[0]:
        return False

    if re.search(r"\b(tem|temos)\b", q):
        return True
    if re.search(r"\bquais\b", q):
        return True
    if any(
        t in q
        for t in [
            "disponível",
            "disponivel",
            "disponíveis",
            "disponiveis",
            "mostra",
            "me mostre",
            "listar",
            "opções",
            "opcoes",
            "opção",
            "opcao",
        ]
    ):
        return True
    if re.search(r"\blista\b", q) or "listar" in q:
        return True
    return False


def _is_more_options_followup(question: str) -> bool:
    q = question.lower().strip()
    direct = {
        "tem outros?",
        "tem outros",
        "outros?",
        "outros",
        "tem outras?",
        "tem outras",
        "tem mais opções?",
        "tem mais opcoes?",
        "tem mais opções",
        "tem mais opcoes",
    }
    if q in direct:
        return True
    if ("outras opções" in q) or ("outras opcoes" in q) or ("mais opções" in q) or ("mais opcoes" in q):
        return True

    # Também trata variações como "tem mais iphones?", "tem mais monitores?", etc.
    if q.startswith("tem mais "):
        continuation_terms = [
            "iphone", "iphones",
            "monitor", "monitores",
            "notebook", "notebooks", "laptop", "laptops", "macbook", "macbooks",
            "impressora", "impressoras", "plotter", "plotters",
            "nobreak", "nobreaks", "ups",
            "ssd", "ssds",
            "mouse", "mouses",
            "teclado", "teclados",
            "celular", "celulares", "smartphone", "smartphones",
            "opção", "opções", "opcao", "opcoes",
        ]
        if any(t in q for t in continuation_terms):
            return True
    return False


def _is_show_more_followup(question: str) -> bool:
    q = question.lower().strip()
    direct = {
        "mostre",
        "mostrar",
        "mostrar mais",
        "mostre mais",
        "quero ver mais",
        "mais",
        "mais opções",
        "mais opcoes",
        "mostrar mais opções",
        "mostrar mais opcoes",
    }
    if q in direct:
        return True
    return _is_more_options_followup(question)


def _is_show_all_followup(question: str) -> bool:
    q = question.lower().strip()
    direct = {
        "mostrar todos",
        "mostrar todas",
        "mostrar tudo",
        "mostre todos",
        "mostre todas",
        "quero ver todos",
        "quero ver todas",
        "ver todos",
        "ver todas",
    }
    return q in direct


def _build_catalog_table_answer(
    question: str,
    context_docs: list[str],
    total_relevant: int,
    shown_start: int,
    shown_end: int,
    showing_all: bool = False,
) -> str | None:
    if not context_docs:
        return None
    if not (
        _is_catalog_listing_query(question)
        or _is_show_more_followup(question)
        or _is_show_all_followup(question)
    ):
        return None

    def _is_valid_attr(value: str) -> bool:
        v = (value or "").strip().lower()
        return v not in {"", "consultar", "n/a", "none"}

    rows = []
    for doc in context_docs:
        item = _parse_context_doc(doc)
        # Evita quebrar formatação por pipes vindos de texto bruto
        produto = item["produto"].replace("|", "/").strip()
        if not produto or produto.lower() == "produto sem descrição":
            produto = f"Produto PN {item['pn']}"
        caracteristicas_parts = [f"PN {item['pn']}"]
        if _is_valid_attr(item["fabricante"]):
            caracteristicas_parts.append(item["fabricante"])
        if _is_valid_attr(item["categoria"]):
            caracteristicas_parts.append(item["categoria"])
        caracteristicas_parts.append(f"Estoque: {item['estoque']}")
        caracteristicas = " • ".join(caracteristicas_parts).replace("|", "/")
        preco = item["preco"].replace("|", "/")
        beneficios = "Boa opção para disponibilidade imediata no catálogo Microware."
        rows.append(
            "\n".join(
                [
                    f"- Produto: {produto}",
                    f" Características: {caracteristicas}",
                    f" Preço: {preco}",
                    f" Benefício: {beneficios}",
                ]
            )
        )

    if not rows:
        return None

    intro = f"Encontrei {total_relevant} produtos relevantes para sua busca."
    if showing_all:
        range_text = "Mostrando todos os resultados:"
    else:
        range_text = f"Mostrando {shown_start}-{shown_end} de {total_relevant}:"
    products_block = "\n\n".join(rows)
    has_more = shown_end < total_relevant
    if has_more and not showing_all:
        footer = "Para continuar, digite 'mostrar mais'. Se preferir, digite 'mostrar todos'."
    else:
        footer = "Se quiser, posso filtrar por marca, faixa de preço ou estoque."
    return f"{intro}\n{range_text}\n\n{products_block}\n\n{footer}"


def _is_manufacturer_list_query(question: str) -> bool:
    q = (question or "").lower()
    return (
        "fabricante" in q
        and any(t in q for t in ["liste", "listar", "quais", "todos", "todas", "aparecem", "aparece"])
    )


def _build_manufacturer_list_answer(
    question: str,
    manufacturers: list[str],
    total_relevant: int,
    shown_start: int,
    shown_end: int,
    showing_all: bool = False,
) -> str | None:
    if not manufacturers or not _is_manufacturer_list_query(question):
        return None

    intro = f"Encontrei {total_relevant} fabricantes no estoque."
    if showing_all:
        range_text = "Mostrando todos os resultados:"
    else:
        range_text = f"Mostrando {shown_start}-{shown_end} de {total_relevant}:"

    bloco = "\n\n".join([f"- Fabricante: {m}" for m in manufacturers if str(m).strip()])
    if not bloco:
        return None

    has_more = shown_end < total_relevant
    if has_more and not showing_all:
        footer = "Para continuar, digite 'mostrar mais'. Se preferir, digite 'mostrar todos'."
    else:
        footer = "Se quiser, posso listar produtos de um fabricante específico."
    return f"{intro}\n{range_text}\n\n{bloco}\n\n{footer}"


def _build_stats_answer(question: str, context_docs: list[str]) -> str | None:
    if not context_docs:
        return None
    for doc in context_docs:
        text = str(doc or "")
        if text.startswith("__STAT_TOTAL_PRODUCTS__:"):
            total = text.split(":", 1)[1].strip()
            return f"Atualmente temos {total} produtos disponíveis no estoque."
        if text.startswith("__STAT_TOTAL_MANUFACTURERS__:"):
            total = text.split(":", 1)[1].strip()
            return f"Atualmente trabalhamos com {total} fabricantes no estoque."
        if text.startswith("__STAT_CATEGORIES__:"):
            raw = text.split(":", 1)[1].strip()
            categories = [c.strip() for c in raw.split("||") if c.strip()]
            if not categories:
                return "No momento não encontrei categorias cadastradas para os produtos filtrados."
            preview = ", ".join(categories[:12])
            more = ""
            if len(categories) > 12:
                more = f" e mais {len(categories) - 12}"
            return f"Temos {len(categories)} categorias no estoque. Principais: {preview}{more}."
    return None


def _build_pn_price_answer(question: str, context_docs: list[str]) -> str | None:
    q = (question or "").lower()
    if not context_docs:
        return None
    if not any(t in q for t in ["quanto custa", "qual o preço", "qual o preco", "preço", "preco"]):
        return None
    if "pn" not in q and not re.search(r"\b[A-Z0-9][A-Z0-9\.\-\/#]{4,}\b", question.upper()):
        return None
    p = _parse_context_doc(context_docs[0])
    return f"O produto {p['produto']} (PN: {p['pn']}) está por {p['preco']}."


def _build_pn_identity_answer(question: str, context_docs: list[str]) -> str | None:
    if not context_docs:
        return None
    q = (question or "").lower()
    has_pn_token = re.search(r"\b[A-Z0-9][A-Z0-9\.\-\/#]{4,}\b", (question or "").upper()) is not None
    if not has_pn_token:
        return None
    if not any(t in q for t in ["qual pn", "pn ", "esse produto", "produto é qual", "produto e qual"]):
        return None
    p = _parse_context_doc(context_docs[0])
    return f"O PN {p['pn']} corresponde ao produto {p['produto']}."


def _build_input_guard_answer(question: str) -> str | None:
    q = (question or "").strip()
    q_low = q.lower()
    if not q:
        return "Pode me dizer qual produto você procura? Posso te ajudar com preço, estoque e disponibilidade."
    if len(set(q_low)) == 1 and len(q_low) >= 5:
        return "Não consegui entender sua mensagem. Pode reformular dizendo o produto ou marca que você procura?"
    if q_low in {"tem", "tem?", "quais", "listar", "lista"}:
        return "Perfeito! Tem o quê exatamente? Me diga o produto ou marca para eu buscar no catálogo."
    if q_low in {"quero o mais barato", "mais barato", "quero o mais caro", "mais caro"}:
        return "Você quer o mais barato/caro de qual categoria? Ex.: iPhone, monitor, nobreak, mouse."

    if any(t in q_low for t in ["capital da ", "quem ganhou a copa", "previsão do tempo", "previsao do tempo", "me faz um poema"]):
        return "Posso te ajudar com produtos e soluções de TI da Microware. Me diga o que você está procurando no catálogo."
    if any(t in q_low for t in ["cadeira gamer", "playstation", "xbox", "nintendo switch"]):
        return "Esse tipo de item não faz parte do nosso catálogo no momento. Posso te ajudar com produtos de tecnologia e TI da Microware."
    if re.search(r"\biphone\s+99\b", q_low):
        return "Não encontrei esse modelo de iPhone no catálogo disponível. Se quiser, posso te mostrar os iPhones disponíveis agora."
    if "melhor" in q_low and any(t in q_low for t in ["celular", "smartphone", "iphone", "monitor", "notebook", "nobreak"]):
        return "Posso te indicar a melhor opção com base no nosso catálogo. Me diga seu critério: menor preço, maior desempenho ou melhor custo-benefício."
    return None


def _default_out_of_context_answer() -> str:
    return (
        "Posso te ajudar com consultas de catálogo, preço e estoque. "
        "Por exemplo: 'tem monitores?', 'mostrar mais', 'mostrar todos' "
        "ou 'qual o mais barato?'."
    )


def _extract_pn_from_context_doc(doc: str) -> str:
    text = str(doc or "")
    if "| PN:" in text:
        return text.rsplit("| PN:", 1)[1].split("|", 1)[0].strip()
    return ""


def _is_context_sensitive_followup(question: str) -> bool:
    q = (question or "").lower()
    return any(
        t in q
        for t in [
            " dele", " desse", " deste", " desses", " dessas", "dessa lista", "entre os dois",
            "entre esses dois", "qual a diferença", "diferença", "pn do mais caro", "pn do mais barato",
            "versão maior", "versao maior", "mais potente", "mais completo", "tem em outra cor",
            "outra cor", "outras cores",
        ]
    )


def _is_context_stale(state: ChatState) -> bool:
    if not _is_context_sensitive_followup(state.get("question", "")):
        return False
    active = state.get("active_docs", []) or []
    current = state.get("chroma_context", []) or []
    if not active:
        return True
    if not current:
        return True
    active_topic = _infer_topic_from_docs(active)
    current_topic = _infer_topic_from_docs(current)
    if active_topic and current_topic and active_topic != current_topic:
        return True
    return False


def _safe_fallback(state: ChatState) -> str:
    q = (state.get("question", "") or "").lower()
    active_docs = state.get("active_docs", []) or []
    if any(t in q for t in ["versão maior", "versao maior", "mais potente", "mais completo"]) and active_docs:
        p = _parse_context_doc(active_docs[0])
        return (
            f"Posso te ajudar com versões mais potentes do modelo {p['produto']} (PN: {p['pn']}). "
            "Se quiser, eu já listo as opções mais robustas disponíveis."
        )
    if any(t in q for t in ["diferença", "diferenca", "entre os dois"]) and len(active_docs) >= 2:
        p1 = _parse_context_doc(active_docs[0])
        p2 = _parse_context_doc(active_docs[1])
        try:
            v1 = float(str(p1["preco"]).replace("R$", "").replace(".", "").replace(",", ".").strip())
            v2 = float(str(p2["preco"]).replace("R$", "").replace(".", "").replace(",", ".").strip())
            diff = abs(v1 - v2)
            diff_txt = f"R$ {diff:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            return (
                f"A diferença de preço entre {p1['produto']} ({p1['preco']}) e {p2['produto']} ({p2['preco']}) "
                f"é {diff_txt}."
            )
        except Exception:
            return "Pode me confirmar quais dois produtos você quer comparar?"
    if _is_color_variation_followup(q):
        return "No momento não encontrei esse modelo em outra cor no estoque disponível."
    return (
        "Não consegui confirmar essa informação com segurança no contexto atual. "
        "Posso refazer a busca de forma objetiva para te responder com precisão."
    )


def _llm_exposed_prompt(answer: str) -> bool:
    text = (answer or "").lower()
    red_flags = [
        "1) **abertura", "2) **resposta objetiva", "3) **detalhamento", "4) **próximo passo",
        "4) **proximo passo", "## formato", "## regras", "nunca adicione", "nunca invente",
    ]
    return any(f in text for f in red_flags)


def _looks_like_meta_llm_feedback(answer: str) -> bool:
    text = (answer or "").lower()
    suspicious_terms = [
        "a resposta está bem formatada",
        "resposta da bela",
        "pronta para a próxima interação",
        "fluxo da conversa",
    ]
    return any(t in text for t in suspicious_terms)


# ── Nó 2: Reescreve pergunta usando histórico (follow-up) ──────────────────
def node_rewrite_query(state: ChatState) -> dict:
    question = state["question"]
    history = state.get("chat_history", "").strip()
    question_lower = question.lower()
    if _is_all_scope_followup(question_lower):
        history_lower = history.lower()
        if any(t in history_lower for t in ["mais caro", "maior preço", "maior preco", "preço mais alto", "preco mais alto"]):
            return {"rewritten_question": "[SCOPE:GLOBAL] qual o produto mais caro da loja?", "needs_clarification": False, "answer_override": ""}
        if any(t in history_lower for t in ["mais barato", "menor preço", "menor preco", "preço mais baixo", "preco mais baixo"]):
            return {"rewritten_question": "[SCOPE:GLOBAL] qual o produto mais barato da loja?", "needs_clarification": False, "answer_override": ""}

    session_id = state["session_id"]
    scope_preference = state.get("scope_preference", "")

    is_scope_choice_global = question_lower in {"loja inteira", "geral", "estoque inteiro", "toda a loja", "toda loja"}
    is_scope_choice_context = question_lower in {"contexto atual", "contexto", "dentro do contexto", "apenas contexto"}
    if is_scope_choice_global:
        set_scope_preference(session_id, "global")
        return {
            "rewritten_question": question,
            "needs_clarification": False,
            "scope_preference": "global",
            "answer_override": "Entendido. Vou considerar loja inteira nas próximas perguntas ambíguas de preço.",
        }
    if is_scope_choice_context:
        set_scope_preference(session_id, "context")
        return {
            "rewritten_question": question,
            "needs_clarification": False,
            "scope_preference": "context",
            "answer_override": "Entendido. Vou considerar o contexto atual nas próximas perguntas ambíguas de preço.",
        }

    # Follow-up "tem outros?" deve preservar o tópico atual da conversa
    if question_lower.strip() in {"tem outros?", "tem outros", "outros?", "outros"}:
        topic = _infer_topic_from_docs(state.get("active_docs", [])) or _infer_topic_from_history(history)
        if topic:
            rewritten = f"tem outros {topic} disponíveis?"
            return {"rewritten_question": rewritten, "needs_clarification": False, "answer_override": ""}

    # Consulta de preço global pode ser ambígua após um contexto específico.
    # Ex.: "qual o produto mais barato da loja?" depois de falar sobre iPhone.
    is_explicit_global_price_question = (
        any(k in question_lower for k in ["mais barato", "menor preço", "menor preco", "mais caro", "maior preço", "maior preco"])
        and any(k in question_lower for k in ["produto", "item"])
        and any(k in question_lower for k in ["loja", "estoque", "catálogo", "catalogo", "geral", "de todos", "de tudo"])
    )
    if is_explicit_global_price_question:
        return {
            "rewritten_question": f"[SCOPE:GLOBAL] {question}",
            "needs_clarification": False,
            "answer_override": "",
        }

    is_global_price_question = (
        any(k in question_lower for k in ["mais barato", "menor preço", "menor preco", "mais caro", "maior preço", "maior preco"])
        and any(k in question_lower for k in ["produto", "item", "loja", "estoque", "catálogo", "catalogo", "geral"])
        and not any(k in question_lower for k in ["iphone", "celular", "smartphone", "monitor", "notebook", "ssd", "impressora", "mouse", "teclado"])
    )
    has_specific_history = any(
        k in history.lower()
        for k in ["iphone", "smartphone", "monitor", "notebook", "ssd", "impressora", "mouse", "teclado", "samsung", "apple", "lg", "dell", "lenovo"]
    )

    # Se há preferência salva, aplica sem perguntar de novo.
    if is_global_price_question and has_specific_history and scope_preference in ("global", "context"):
        if scope_preference == "global":
            return {
                "rewritten_question": f"[SCOPE:GLOBAL] {question}",
                "needs_clarification": False,
                "answer_override": "",
            }
        topic = _infer_topic_from_history(history)
        intent_word = "mais barato" if any(k in question_lower for k in ["mais barato", "menor preço", "menor preco"]) else "mais caro"
        contextual_q = f"qual {topic} {intent_word} no contexto atual?" if topic else question
        return {
            "rewritten_question": f"[SCOPE:CONTEXT] {contextual_q}",
            "needs_clarification": False,
            "answer_override": "",
        }

    if is_global_price_question and has_specific_history:
        return {
            "rewritten_question": question,
            "needs_clarification": True,
            "answer_override": "",
        }

    # Se já é uma pergunta específica por tipo/marca/modelo, não reescrever.
    # Isso evita o LLM "generalizar" para consulta global por engano.
    explicit_specific_terms = [
        "iphone", "celular", "smartphone", "monitor", "notebook", "ssd", "impressora",
        "mouse", "teclado", "tablet", "plotter", "galaxy", "samsung", "apple", "lg",
        "dell", "lenovo", "hp", "asus", "acer", "motorola", "xiaomi"
    ]
    if any(t in question_lower for t in explicit_specific_terms):
        return {"rewritten_question": question, "needs_clarification": False, "answer_override": ""}

    # Preço min/max: nunca passar pelo LLM de rewrite — o modelo costuma injetar
    # "mostre/liste/tem" e quebra a rota determinística de preço + detecção de intent.
    if _is_price_query(question)[0]:
        return {"rewritten_question": question, "needs_clarification": False, "answer_override": ""}

    if not history:
        return {"rewritten_question": question, "needs_clarification": False, "answer_override": ""}

    prompt_text = f"""
Reescreva a PERGUNTA FINAL para que ela fique autocontida usando o histórico.
Não responda a pergunta, apenas reescreva em uma única frase.

Histórico:
{history}

Pergunta final:
{question}
"""
    llm = _build_llm(temp=0.0)

    rewritten = llm.invoke(prompt_text) if hasattr(llm, "invoke") else llm(prompt_text)
    rewritten_text = rewritten.content if hasattr(rewritten, "content") else str(rewritten)
    rewritten_text = rewritten_text.strip() or question
    return {"rewritten_question": rewritten_text, "needs_clarification": False, "answer_override": ""}


# ── Nó 2.5: Detecta ambiguidade antes de buscar ─────────────────────────────
def node_check_ambiguity(state: ChatState) -> dict:
    if state.get("answer_override"):
        return {"handled": True}
    clarification = _clarification_for_ambiguity(
        state.get("question", ""),
        state.get("active_docs", []),
    )
    if clarification:
        return {"answer_override": clarification, "handled": True}
    return {"handled": False}


# ── Nó 3: Recupera documentos do ChromaDB ──────────────────────────────────
def node_retrieve(state: ChatState) -> dict:
    original_question = state["question"]
    query = state.get("rewritten_question") or original_question
    if _is_category_plus_catalog_price_query(original_question) and not state.get("active_docs"):
        _, intent = _is_price_query(original_question)
        if intent == "MAX_PRICE":
            docs = retrieve_docs("qual o produto mais caro da loja?", k=8)
        else:
            docs = retrieve_docs("qual o produto mais barato da loja?", k=8)
        return {
            "chroma_context": docs,
            "listing_docs": [],
            "listing_shown": 0,
        }

    if state.get("answer_override"):
        return {"chroma_context": [], "listing_docs": [], "listing_shown": 0}

    # Follow-up curto de preço deve manter explicitamente o mesmo contexto ativo.
    # Evita perder referência de item em perguntas como "e o mais caro?" -> "qual o PN dele?"
    if _is_contextual_price_followup(original_question) and state.get("active_docs"):
        # Se o turno anterior era preço global da loja, reconsulta global
        # para não limitar ao subconjunto de active_docs.
        last_user_q = _get_last_user_question(state.get("chat_history", ""))
        last_is_price, _ = _is_price_query(last_user_q)
        current_is_price, current_intent = _is_price_query(original_question)
        last_category = _extract_category_slot(last_user_q)
        # Se o turno anterior era preço por categoria (ex.: monitor), refaz
        # a busca na mesma categoria com a nova intenção.
        if last_is_price and current_is_price and last_category and not _is_global_price_anchor_query(last_user_q):
            rebuilt_q = _build_price_query_from_category(last_category, current_intent)
            if rebuilt_q:
                docs = retrieve_docs(rebuilt_q, k=8)
                return {
                    "chroma_context": docs,
                    "listing_docs": [],
                    "listing_shown": 0,
                }

        if _is_global_price_anchor_query(last_user_q):
            _, intent = _is_price_query(original_question)
            if intent == "MIN_PRICE":
                docs = retrieve_docs("qual o produto mais barato da loja?", k=5)
            else:
                docs = retrieve_docs("qual o produto mais caro da loja?", k=5)
            return {
                "chroma_context": docs,
                "listing_docs": [],
                "listing_shown": 0,
            }

        current = list(state.get("active_docs", []))
        _, intent = _is_price_query(original_question)
        if intent == "MIN_PRICE":
            current.sort(key=_parse_price_value_from_doc)
        elif intent == "MAX_PRICE":
            current.sort(key=_parse_price_value_from_doc, reverse=True)
        return {
            "chroma_context": current,
            "listing_docs": [],
            "listing_shown": 0,
        }
    # Paginação de catálogo.
    if _is_show_all_followup(original_question):
        listing_docs = state.get("listing_docs", [])
        return {
            "chroma_context": listing_docs,
            "listing_docs": listing_docs,
            "listing_shown": len(listing_docs),
        }
    if _is_show_more_followup(original_question):
        listing_docs = state.get("listing_docs", [])
        shown = state.get("listing_shown", 0)
        page_size = 5
        next_chunk = listing_docs[shown: shown + page_size]
        return {
            "chroma_context": next_chunk,
            "listing_docs": listing_docs,
            "listing_shown": min(shown + len(next_chunk), len(listing_docs)),
        }

    listing_query = _is_catalog_listing_query(original_question)
    retrieval_k = 8

    # Primeiro tenta intenção determinística na pergunta original (evita contaminação do rewrite).
    handled, docs = retrieve_deterministic(
        original_question,
        active_docs=state.get("active_docs", []),
        k=retrieval_k,
    )
    # Preço: sempre usar a pergunta do usuário na busca (rewrite não deve alterar intent).
    if not handled and _is_price_query(original_question)[0]:
        docs = retrieve_docs(original_question, k=retrieval_k)
        return {
            "chroma_context": docs,
            "listing_docs": [],
            "listing_shown": 0,
        }
    # Se não tratou, tenta com a versão reescrita.
    if not handled:
        handled, docs = retrieve_deterministic(
            query,
            active_docs=state.get("active_docs", []),
            k=retrieval_k,
        )
    if not handled:
        docs = retrieve_docs(query, k=retrieval_k)
        clarification = _clarification_for_low_confidence(
            original_question,
            state.get("active_docs", []),
            get_semantic_best_distance(query),
        )
        if clarification:
            return {
                "chroma_context": [],
                "listing_docs": [],
                "listing_shown": 0,
                "answer_override": clarification,
            }

    if listing_query:
        page_size = 5
        listing_docs = docs[:50]
        first_chunk = listing_docs[:page_size]
        return {
            "chroma_context": first_chunk,
            "listing_docs": listing_docs,
            "listing_shown": len(first_chunk),
        }

    return {
        "chroma_context": docs,
        "listing_docs": [],
        "listing_shown": 0,
    }


# ── Nó 4: Gera resposta com LLM ────────────────────────────────────────────
def node_generate(state: ChatState) -> dict:
    if state.get("answer_override"):
        return {"answer": state["answer_override"]}

    if state.get("needs_clarification"):
        clarificacao = (
            "Você quer saber o produto mais barato/caro considerando o contexto atual da conversa "
            "(por exemplo, iPhones) ou o produto mais barato/caro da loja inteira? "
            "Responda com 'contexto atual' ou 'loja inteira'."
        )
        return {"answer": clarificacao}

    guard_answer = _build_input_guard_answer(state["question"])
    if guard_answer:
        return {"answer": guard_answer}

    followup_answer = _build_followup_answer(
        state["question"],
        state.get("active_docs", []),
        state.get("chat_history", ""),
    )
    if followup_answer:
        return {"answer": followup_answer}

    stats_answer = _build_stats_answer(
        state["question"],
        state.get("chroma_context", []),
    )
    if stats_answer:
        return {"answer": stats_answer}

    pn_price_answer = _build_pn_price_answer(
        state["question"],
        state.get("chroma_context", []),
    )
    if pn_price_answer:
        return {"answer": pn_price_answer}

    pn_identity_answer = _build_pn_identity_answer(
        state["question"],
        state.get("chroma_context", []),
    )
    if pn_identity_answer:
        return {"answer": pn_identity_answer}

    stock_answer = _build_stock_answer_from_context(
        state["question"],
        state.get("chroma_context", []),
    )
    if stock_answer:
        return {"answer": stock_answer}

    original_question = state["question"]
    price_context_docs = state.get("chroma_context", [])
    if _is_all_scope_followup(original_question):
        last_user_q = _get_last_user_question(state.get("chat_history", ""))
        if any(t in last_user_q.lower() for t in ["mais caro", "maior preço", "maior preco", "preço mais alto", "preco mais alto"]):
            deterministic_all_scope = _build_price_answer_from_context("qual o produto mais caro da loja?", price_context_docs)
        else:
            deterministic_all_scope = _build_price_answer_from_context("qual o produto mais barato da loja?", price_context_docs)
        if deterministic_all_scope:
            return {"answer": deterministic_all_scope}

    if _is_contextual_price_followup(original_question) and state.get("active_docs"):
        last_user_q = _get_last_user_question(state.get("chat_history", ""))
        last_is_price, _ = _is_price_query(last_user_q)
        last_category = _extract_category_slot(last_user_q)
        # Em follow-up curto de preço:
        # - se veio de âncora global (produto da loja), usa chroma_context do retrieve atual;
        # - se veio de preço por categoria (ex.: monitor), usa chroma_context do retrieve atual;
        # - caso contrário, usa contexto ativo da conversa.
        if not _is_global_price_anchor_query(last_user_q) and not (last_is_price and bool(last_category)):
            price_context_docs = state.get("active_docs", [])

    # Intent de preço sempre pela pergunta original (rewrite pode perder "mais barato/caro").
    deterministic_price_answer = _build_price_answer_from_context(
        original_question,
        price_context_docs,
    )
    if deterministic_price_answer:
        if _is_category_plus_catalog_price_query(original_question):
            cat = _extract_category_slot(original_question) or "produto"
            return {
                "answer": (
                    f"{deterministic_price_answer} "
                    f"Se quiser, eu também posso te mostrar o {cat} mais barato em específico."
                )
            }
        return {"answer": deterministic_price_answer}

    deterministic_catalog_answer = _build_catalog_table_answer(
        original_question,
        state.get("chroma_context", []),
        total_relevant=len(state.get("listing_docs", [])) or len(state.get("chroma_context", [])),
        shown_start=max(1, state.get("listing_shown", 0) - len(state.get("chroma_context", [])) + 1),
        shown_end=state.get("listing_shown", 0),
        showing_all=_is_show_all_followup(original_question),
    )
    if deterministic_catalog_answer:
        return {"answer": deterministic_catalog_answer}

    deterministic_manufacturer_answer = _build_manufacturer_list_answer(
        original_question,
        state.get("chroma_context", []),
        total_relevant=len(state.get("listing_docs", [])) or len(state.get("chroma_context", [])),
        shown_start=max(1, state.get("listing_shown", 0) - len(state.get("chroma_context", [])) + 1),
        shown_end=state.get("listing_shown", 0),
        showing_all=_is_show_all_followup(original_question),
    )
    if deterministic_manufacturer_answer:
        return {"answer": deterministic_manufacturer_answer}

    if (_is_show_more_followup(state["question"]) or _is_show_all_followup(state["question"])) and not state.get("chroma_context"):
        return {"answer": "No momento não encontrei mais opções além das que já te mostrei para esse produto."}
    if _is_color_variation_followup(state["question"]) and not state.get("chroma_context"):
        return {"answer": "No momento não encontrei esse modelo em outra cor no estoque disponível."}

    # Sem contexto recuperado, evita respostas alucinatórias e usa fallback padrão.
    if not state.get("chroma_context"):
        return {"answer": _default_out_of_context_answer()}
    if _is_context_stale(state):
        return {"answer": _safe_fallback(state)}

    # Observação: mantemos a lógica do nó (monta prompt -> chama LLM -> devolve answer)
    # apenas trocando o backend para Ollama.
    llm = _build_llm(temp=0.3)

    context = "\n\n".join(state["chroma_context"])
    prompt_text = SYSTEM_PROMPT.format(
        context=context,
        chat_history=state["chat_history"],
        question=state.get("rewritten_question") or state["question"],
    )

    if hasattr(llm, "invoke"):
        response = llm.invoke(prompt_text)
    else:
        response = llm(prompt_text)

    # Normaliza para string
    if hasattr(response, "content"):
        answer_text = response.content
    else:
        answer_text = str(response)

    if _looks_like_meta_llm_feedback(answer_text) or _llm_exposed_prompt(answer_text):
        return {"answer": _safe_fallback(state)}

    return {"answer": answer_text}


# ── Nó 5: Persiste mensagens no MongoDB ────────────────────────────────────
def node_save_memory(state: ChatState) -> dict:
    save_message(state["session_id"], "user", state["question"])
    save_message(state["session_id"], "assistant", state["answer"])
    # Mantém contexto ativo para follow-ups objetivos (PN/preço/estoque/diferença)
    preserve_active = _is_context_sensitive_followup(state.get("question", "")) and _is_context_stale(state)
    if not preserve_active:
        set_active_docs(state["session_id"], state.get("chroma_context", []))
    set_listing_state(
        state["session_id"],
        state.get("listing_docs", []),
        state.get("listing_shown", 0),
    )
    return {}


# ── Montagem do grafo ──────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("fetch_history", node_fetch_history)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("check_ambiguity", node_check_ambiguity)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("generate", node_generate)
    graph.add_node("save_memory", node_save_memory)

    graph.set_entry_point("fetch_history")
    graph.add_edge("fetch_history", "rewrite_query")
    graph.add_edge("rewrite_query", "check_ambiguity")
    graph.add_edge("check_ambiguity", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()


chatbot = build_graph()


def run_graph(session_id: str, question: str) -> str:
    result = chatbot.invoke(
        {
            "session_id": session_id,
            "question": question,
            "rewritten_question": question,
            "chroma_context": [],
            "chat_history": "",
            "active_docs": [],
            "listing_docs": [],
            "listing_shown": 0,
            "handled": False,
            "needs_clarification": False,
            "scope_preference": "",
            "answer_override": "",
            "answer": "",
        }
    )
    return str(result.get("answer", ""))
