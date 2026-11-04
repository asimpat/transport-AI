from sqlalchemy.orm import Session

from app.services.context_service import build_rag_context
from app.services.generation_service import generate_answer
from app.services.retrieval_service import retrieve_relevant_chunks


INSUFFICIENT_INFORMATION_ANSWER = (
    "I can only assist with public transportation information for Lafia. "
    "I don't have sufficient relevant, verified information in the knowledge "
    "base to answer that question."
)


async def ask_transport_question(db: Session, question: str) -> dict:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty.")

    retrieved_chunks = await retrieve_relevant_chunks(db, question)
    if not retrieved_chunks:
        return {
            "question": question,
            "answer": INSUFFICIENT_INFORMATION_ANSWER,
            "sources": [],
            "retrieved_chunks_count": 0,
        }

    rag_context = build_rag_context(retrieved_chunks)
    answer = await generate_answer(question, rag_context["context"])

    return {
        "question": question,
        "answer": answer,
        "sources": rag_context["sources"],
        "retrieved_chunks_count": len(retrieved_chunks),
    }
