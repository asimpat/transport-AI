from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be empty or whitespace-only.")
        return value


class ChatSource(BaseModel):
    source_label: str
    chunk_id: int
    document_id: int
    document_title: str
    source_type: str
    source_reference: str | None
    similarity_score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[ChatSource]
    retrieved_chunks_count: int
