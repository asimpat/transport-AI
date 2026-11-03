from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db


app = FastAPI(
    title=settings.APP_NAME,
    description="RAG-powered public transportation information assistant for Lafia.",
    version="1.0.0",
)


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
