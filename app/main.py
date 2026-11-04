from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.api.v1.endpoints.chat import router as chat_router


app = FastAPI(
    title=settings.APP_NAME,
    description="RAG-powered public transportation information assistant for Lafia.",
    version="1.0.0",
)

app.include_router(chat_router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "message": "Welcome to Transport AI API",
        "status": "running",
    }


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))

    return {
        "status": "healthy",
        "database": "connected",
    }
