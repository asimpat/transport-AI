import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import ask_transport_question


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat/ask", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    try:
        return await ask_transport_question(db, request.question)
    except Exception as error:
        logger.exception("Failed to process chat question")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process the question at this time.",
        ) from error
