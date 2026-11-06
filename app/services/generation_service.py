import httpx

from app.core.config import settings

SYSTEM_PROMPT = """You are a public transportation information assistant for Lafia, Nasarawa State, Nigeria.
Answer only public transportation information for Lafia.
Use only the supplied retrieved context as factual evidence.
Do not invent transport operators, routes, terminals, fares, schedules, or contact details.
Use relevant details that are present in the supplied context. Do not claim that information is unavailable when the context provides it.
Keep every fare, departure time, schedule, and operating day tied only to the exact route and operator stated in the same evidence. Never transfer those details from one route or operator to another.
When reporting historical information, preserve its dates and clearly identify it as historical rather than current.
When using a record classified as synthetic, explicitly identify all related information as fictional academic demonstration data, not actual Lafia transport services.
If requested information is genuinely absent or ambiguous in the context, say so clearly instead of guessing or combining details from different records.
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
    async with httpx.AsyncClient(
        timeout=settings.OLLAMA_CHAT_TIMEOUT_SECONDS
    ) as client:
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
