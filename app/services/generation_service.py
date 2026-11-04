import httpx

from app.core.config import settings

SYSTEM_PROMPT = """You are a public transportation information assistant for Lafia, Nasarawa State, Nigeria.
Answer only questions relating to public transportation.
Use only the supplied retrieved context as factual evidence.
Do not invent transport operators, routes, terminals, fares, schedules, or contact details.
If the context does not contain enough information, clearly say that the information is not available in the current knowledge base.
Do not pretend to have live transport information.
If a question is unrelated to public transportation, politely explain your scope."""


async def generate_answer(question: str, context: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty.")
    if not isinstance(context, str):
        raise ValueError("Context must be a string.")

    if context.strip():
        context_text = context
    else:
        context_text = (
            "No retrieved context is available. Explain that the information "
            "is not available in the current knowledge base, and do not infer "
            "or invent an answer."
        )

    url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT_SECONDS) as client:
        response = await client.post(
            url,
            json={
                "model": settings.OLLAMA_CHAT_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Retrieved context:\n{context_text}\n\nQuestion:\n{question}",
                    },
                ],
            },
        )
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError as error:
            raise ValueError("Ollama returned invalid JSON.") from error

    if not isinstance(data, dict):
        raise ValueError("Ollama returned an invalid response.")
    message = data.get("message")
    answer = message.get("content") if isinstance(message, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Ollama response did not contain a valid message.content.")

    return answer
