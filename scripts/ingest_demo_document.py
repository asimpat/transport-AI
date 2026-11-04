import asyncio
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.db.session import SessionLocal
from app.models.knowledge_document import KnowledgeDocument
from app.services.document_service import ingest_document


TITLE = "SYNTHETIC DEMO - Lafia Transport Information"
CONTENT = (
    "This document contains synthetic demonstration information created exclusively "
    "for testing the Lafia Public Transportation Information Assistant. It does not "
    "describe verified real-world transport services.\n\n"
    "DEMO ROUTE A: For this test only, a fictional bus service operates between "
    "Demo Terminal North and Demo Terminal South. Its fictional operating hours "
    "are 7:00 AM to 6:00 PM.\n\n"
    "DEMO FARE A: For this test only, the fictional fare between Demo Terminal "
    "North and Demo Terminal South is 500 naira.\n\n"
    "DEMO CONTACT A: No real transport operator or contact number is associated "
    "with this demonstration."
)


def main() -> None:
    db = SessionLocal()
    try:
        existing_document = (
            db.query(KnowledgeDocument)
            .filter(KnowledgeDocument.title == TITLE)
            .first()
        )
        if existing_document is not None:
            print(f"Document already exists; not inserting another copy: {TITLE}")
            return

        document = asyncio.run(
            ingest_document(
                db,
                title=TITLE,
                content=CONTENT,
                source_type="manual",
                source_reference="synthetic-integration-test-only",
            )
        )
        print(f"Created document ID {document.id}: {document.title}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
