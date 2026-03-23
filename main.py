from graph import chatbot
import uuid

def chat(session_id: str, question: str) -> str:
    result = chatbot.invoke({
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
        "answer": ""
    })
    return result["answer"]


if __name__ == "__main__":
    session = str(uuid.uuid4())
    print("Chatbot RAG iniciado. Digite 'sair' para encerrar.\n")
    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() in ("sair", "exit"):
            break
        response = chat(session, user_input)
        print(f"\nAssistente: {response}\n")
