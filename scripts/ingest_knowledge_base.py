import argparse
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DOCUMENTS = (
    ("knowledge_base/verified/yahaya_sabo_terminal.md", "RESEARCHED", "verified"),
    ("knowledge_base/verified/intercity_transport.md", "RESEARCHED", "verified"),
    (
        "knowledge_base/historical/lafia_keke_routes_and_fares_2022.md",
        "HISTORICAL",
        "historical",
    ),
    (
        "knowledge_base/historical/transport_movement_restrictions.md",
        "HISTORICAL",
        "historical",
    ),
    (
        "knowledge_base/synthetic/synthetic_keke_routes.md",
        "SIMULATED DATA",
        "synthetic",
    ),
    (
        "knowledge_base/synthetic/synthetic_transport_services.md",
        "SIMULATED DATA",
        "synthetic",
    ),
)


def remove_yaml_frontmatter(markdown: str) -> str:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return markdown

    for closing_line in range(1, len(lines)):
        if lines[closing_line].strip() == "---":
            return "\n".join(lines[closing_line + 1 :]).lstrip()

    raise ValueError("Markdown starts with YAML frontmatter but has no closing '---'.")


def prepare_documents() -> list[dict[str, str]]:
    documents = []
    for relative_path, classification, source_type in DOCUMENTS:
        path = project_root / relative_path
        markdown = path.read_text(encoding="utf-8")
        body = remove_yaml_frontmatter(markdown)
        if not body.strip():
            raise ValueError(f"Markdown document has no content after frontmatter: {relative_path}")

        title = f"{classification} - {path.stem.replace('_', ' ').title()}"
        if source_type == "historical":
            notice = (
                "Historical information: the dates and reports below are preserved, "
                "but this information may no longer be current.\n\n"
            )
        elif source_type == "synthetic":
            notice = (
                "SYNTHETIC DATA FOR ACADEMIC DEMONSTRATION ONLY: All routes, fares, "
                "stops, operators, and schedules in this document are fictional. "
                "They must not be presented as real transport information or used "
                "for real passenger guidance.\n\n"
            )
        else:
            notice = (
                "Evidence classification: researched from the sources described "
                "below; do not infer facts beyond those sources.\n\n"
            )

        documents.append(
            {
                "title": title,
                "content": notice + body,
                "source_type": source_type,
                "source_reference": relative_path,
            }
        )
    return documents


async def ingest_documents(db, documents: list[dict[str, str]]) -> tuple[int, int]:
    from app.models.knowledge_document import KnowledgeDocument
    from app.services.document_service import ingest_document, split_text_into_chunks

    ingested_documents = 0
    ingested_chunks = 0
    for item in documents:
        existing_document = (
            db.query(KnowledgeDocument)
            .filter(KnowledgeDocument.title == item["title"])
            .first()
        )
        if existing_document is not None:
            print(f"Already exists; skipped: {item['title']}")
            continue

        chunks = split_text_into_chunks(item["content"])
        document = await ingest_document(
            db,
            title=item["title"],
            content=item["content"],
            source_type=item["source_type"],
            source_reference=item["source_reference"],
        )
        ingested_documents += 1
        ingested_chunks += len(chunks)
        print(f"Ingested document ID {document.id}: {document.title} ({len(chunks)} chunks)")

    return ingested_documents, ingested_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the selected Lafia knowledge-base Markdown documents."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview titles, source references, and content sizes without database access.",
    )
    args = parser.parse_args()
    documents = prepare_documents()

    if args.dry_run:
        for item in documents:
            print(
                f"{item['title']} | source_type={item['source_type']} | "
                f"source_reference={item['source_reference']} | "
                f"characters={len(item['content'])}"
            )
        print(f"Dry run complete: {len(documents)} documents; nothing saved.")
        return

    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        document_count, chunk_count = asyncio.run(ingest_documents(db, documents))
        print(
            f"Ingestion complete: {document_count} documents and "
            f"{chunk_count} chunks ingested."
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
