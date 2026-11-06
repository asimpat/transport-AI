# Lafia Public Transportation Information Assistant

A Retrieval-Augmented Generation (RAG) backend that answers questions about public transportation information in Lafia, Nasarawa State, using a prepared knowledge base.

## Problem

Public transportation information can be fragmented, outdated, or difficult to verify. A general-purpose language model may also produce plausible but unsupported routes, fares, or schedules. This project explores an assistant that retrieves information from a prepared knowledge base before generating an answer, and discloses when the available information is insufficient.

## Solution

The backend implements a RAG pipeline:
1. Receive a user's transportation question.
2. Generate a question embedding with Ollama `nomic-embed-text`.
3. Retrieve relevant document chunks from PostgreSQL using pgvector similarity search.
4. Assemble the passages with source metadata.
5. Generate a response with Ollama `qwen2.5:1.5b`.
6. Return the answer, source metadata, and retrieved chunk count.

The assistant is scoped to public transportation information for Lafia. It does not provide live vehicle tracking, live seat availability, or guaranteed current fares and schedules.

## Features

- FastAPI question-answer endpoint: `POST /api/v1/chat/ask`
- Markdown knowledge-base ingestion and overlapping text chunking
- Local embedding generation and vector similarity retrieval
- PostgreSQL + pgvector storage
- Source-aware context assembly
- Ollama-based answer generation
- Prompt rules against inventing transport details or mixing facts across operators/routes
- Exact transport-record identifier prioritisation for synthetic demo IDs
- Insufficient-context and out-of-scope fallback responses
- Automated tests for core services

## Technology stack

- Python, FastAPI
- PostgreSQL, pgvector
- SQLAlchemy, Alembic
- Ollama: `nomic-embed-text:latest` (768 dimensions), `qwen2.5:1.5b`
- HTTPX
- Pydantic / pydantic-settings
- pytest


## Setup

Assumptions: Ubuntu/Linux, Python 3.12, PostgreSQL with pgvector available, and Ollama installed.

### 1. Clone and enter the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd transport-ai
```

### 2. Create a virtual environment and install dependencies

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Set the PostgreSQL connection string and Ollama settings expected by `app/core/config.py`. Never commit `.env` or real credentials.

### 4. Prepare the database

Create the PostgreSQL database and user referenced by your connection string. Ensure the PostgreSQL server has the pgvector extension available.

### 5. Download the Ollama models

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
```

Ensure Ollama is running and reachable at the configured base URL.

### 6. Apply migrations and ingest the knowledge base

```bash
./venv/bin/alembic upgrade head
./venv/bin/python scripts/ingest_knowledge_base.py
```

The ingestion script reads Markdown documents, strips YAML frontmatter, adds classification text, chunks and embeds the content, and stores it in PostgreSQL.

### 7. Start the API

```bash
uvicorn app.main:app --reload
```

API documentation: `http://127.0.0.1:8000/docs`

## API example

`POST /api/v1/chat/ask`

Request:

```json
{
  "question": "Where is Yahaya Sabo Bus Terminal in Lafia?"
}
```

Response shape:

```json
{
  "question": "Where is Yahaya Sabo Bus Terminal in Lafia?",
  "answer": "Generated answer based on retrieved knowledge.",
  "sources": [],
  "retrieved_chunks_count": 3
}
```

Actual answers and source entries depend on the knowledge base and retrieval results.

## How AI is used

`nomic-embed-text` converts questions and knowledge chunks into vectors for semantic retrieval. `qwen2.5:1.5b` receives the question and retrieved passages and generates a natural-language response. The generation prompt instructs the model to use the supplied context, preserve dates, keep facts tied to the correct route/operator, identify synthetic content as fictional, and state when details are missing or ambiguous.

## Knowledge-base categories

- **Researched:** information gathered from available public sources; this does not automatically prove a service is currently operating.
- **Historical:** dated information, such as reported 2022 fares; retain the date when answering.
- **Unverified:** candidate information not sufficiently confirmed.
- **Synthetic:** fictional records used only for development and academic demonstration.

## Key technical decisions

- RAG was chosen to ground generated answers in retrieved project documents.
- PostgreSQL and pgvector keep document records and vector search in one database.
- Ollama provides local embedding and chat inference.
- Chunking with overlap makes longer documents easier to retrieve.
- Source-aware context and conservative fallback behaviour reduce unsupported answers.

## Challenges

- Limited public information about current Keke routes, fares, stops, and operating hours in Lafia.
- Available fare and movement-restriction reports may be historical.
- Semantic retrieval can return related passages without the exact record needed.
- A small language model can omit details or confuse information across routes/operators.

## Limitations and future work

The assistant is not connected to live operators or real-time transport feeds. Current Keke fares and schedules are not comprehensively verified. Retrieval and generation can still omit or misstate details; the current demonstration evaluation exposed omissions in one synthetic Keke answer and confusion between a boarding point and an arrival point in one synthetic intercity answer. The system is an academic prototype, not the sole basis for real travel decisions.

Future work includes validating information with local unions and operators, improving retrieval and answer evaluation, adding clearer citations and update workflows, collecting user feedback, and deploying a user-facing web/mobile interface.

## Testing

The project test suite was reported as passing **51 tests and 26 subtests** after the latest RAG updates, with one existing Starlette/HTTPX deprecation warning. This development result does not guarantee that every generated answer is factually correct.

## GitHub repository

**Repository URL:** [Insert GitHub repository URL]

Before submission, replace this placeholder with the tutor-accessible URL. Confirm that the repository includes the complete code, README, dependencies, setup instructions, tests, migrations, and knowledge-base documentation. Ensure `.env` is ignored and `.env.example` contains placeholders only.