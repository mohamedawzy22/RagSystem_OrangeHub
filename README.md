# RAG System OrangeHub

> A production-oriented Retrieval-Augmented Generation backend built with FastAPI, MongoDB, Qdrant, and pluggable LLM providers.

RAG System OrangeHub provides project-scoped document ingestion, processing, embedding, semantic retrieval, and answer generation through a modular FastAPI architecture.

## Key Features

* Project-scoped RAG using `project_id`.
* PDF and plain-text file ingestion.
* File validation, storage, processing, and chunking.
* Embedding generation with Qdrant semantic search.
* MongoDB persistence for projects, assets, and chunks.
* Pluggable Ollama and OpenRouter providers.
* Primary/fallback chat models.
* Startup health checks and model warm-up.
* Reusable application resources with graceful shutdown.
* Centralized application logging.
* Layered automated testing with Pytest.
* Ruff and pre-commit for code quality.

## Architecture

The application separates API routes, controllers, business services, storage, models, and infrastructure providers.

```mermaid
flowchart TD

    Client[API Client] --> Routes[FastAPI Routes]

    Routes --> Data[Data Services]
    Routes --> RAG[RAG Controller]

    Data --> Upload[Upload Service]
    Data --> Processing[Processing Service]

    Upload --> Storage[Project Storage]
    Upload --> Mongo[(MongoDB)]

    Processing --> Processor[Document Processor]
    Processing --> Mongo
    Processing --> Chunks[Chunks]

    Chunks --> Embedding[Embedding Model]
    Embedding --> Qdrant[(Qdrant)]

    RAG --> RAGService[RAG Service]
    RAGService --> Retrieval[Retrieval Service]
    RAGService --> Chat[Chat Model]

    Retrieval --> Embedding
    Retrieval --> Qdrant

    Chat --> Primary[Primary Model]
    Chat --> Fallback[Fallback Model]

    Primary --> Ollama[Ollama]
    Primary --> OpenRouter[OpenRouter]

    Fallback --> Ollama
    Fallback --> OpenRouter

    Startup[FastAPI Lifespan] --> Container[Application Container]
    Container --> Mongo
    Container --> Models[Model Manager]
    Container --> Qdrant
```

## Main Flows

### Document Ingestion

```text
Upload
  ↓
Upload Service
  ↓
File Storage + MongoDB
  ↓
Processing Service
  ↓
Document Processor
  ↓
Chunking
  ↓
MongoDB Chunks
  ↓
Indexing Service
  ↓
Embedding Model
  ↓
Qdrant
```

### RAG Query

```text
Question
  ↓
RAG Controller
  ↓
RAG Service
  ↓
Retrieval Service
  ↓
Query Embedding
  ↓
Qdrant Search
  ↓
Relevant Chunks
  ↓
Primary Chat Model
  ↓
Fallback Model on failure
  ↓
Answer
```

Long-lived resources are initialized during FastAPI startup and reused during the application lifetime. Shutdown closes MongoDB, model clients, and Qdrant.

## Tech Stack

| Category            | Technologies                      |
| ------------------- | --------------------------------- |
| Language            | Python 3.11+                      |
| API                 | FastAPI, Uvicorn                  |
| Configuration       | Pydantic Settings                 |
| LLM                 | Ollama, OpenRouter                |
| Database            | MongoDB                           |
| Vector Database     | Qdrant                            |
| MongoDB Client      | PyMongo Async                     |
| Document Processing | PyMuPDF, LangChain Text Splitters |
| Testing             | Pytest                            |
| Quality             | Ruff, pre-commit                  |
| Environment         | uv, Docker, WSL2                  |

## Project Structure

```text
RagSystem_OrangeHub/
├── src/
│   ├── controllers/
│   │   └── rag_controller.py
│   │
│   ├── core/
│   │   ├── container.py
│   │   ├── dependencies.py
│   │   ├── exceptions.py
│   │   ├── exception_handlers.py
│   │   └── retry.py
│   │
│   ├── helpers/
│   │   └── config.py
│   │
│   ├── models/
│   │   ├── asset_model.py
│   │   ├── chunk_model.py
│   │   ├── project_model.py
│   │   └── enums/
│   │
│   ├── routes/
│   │   ├── base.py
│   │   ├── data.py
│   │   ├── rag.py
│   │   └── schemes/
│   │
│   ├── services/
│   │   ├── llm/
│   │   ├── processing/
│   │   ├── rag/
│   │   ├── storage/
│   │   ├── prompts/
│   │   └── vectordb/
│   │
│   ├── utils/
│   │   └── logger.py
│   │
│   └── main.py
│
├── tests/
│   ├── unit/
│   ├── api/
│   ├── integration/
│   └── e2e/
│
├── .env
├── pyproject.toml
├── uv.lock
└── README.md
```

## Prerequisites

* Python 3.11+
* `uv`
* MongoDB
* Qdrant
* Ollama for local inference
* OpenRouter API key when using OpenRouter models

For the current local setup:

```text
qwen2.5:3b
qwen3:8b
bge-m3
```

## Installation

Clone and install dependencies:

```bash
git clone <your-repository-url>
cd RagSystem_OrangeHub
uv sync
```

Create and configure `.env` using your local environment values.

## Run the API

Make sure MongoDB, Qdrant, and Ollama are available.

Then:

```bash
PYTHONPATH=src uv run uvicorn main:app --host 0.0.0.0 --port 8001
```

API:

```text
http://localhost:8001
```

Swagger UI:

```text
http://localhost:8001/docs
```

ReDoc:

```text
http://localhost:8001/redoc
```

## API

### Health

```http
GET /api/v1/health
```

### Upload

```http
POST /api/v1/data/upload/{project_id}
```

Upload a file using `multipart/form-data`.

### Process

```http
POST /api/v1/data/process/{project_id}
```

Processes uploaded project files and creates document chunks.

### Index

```http
POST /api/v1/rag/index/{project_id}
```

Generates embeddings for project chunks and stores them in Qdrant.

### Search

```http
POST /api/v1/rag/search/{project_id}
```

Example:

```json
{
  "query": "What is this document about?",
  "limit": 5
}
```

### Generate

```http
POST /api/v1/rag/generate/{project_id}
```

Example:

```json
{
  "query": "What is this document about?",
  "limit": 5
}
```

The response contains the generated answer and the retrieved documents used as context.

## Environment

Configuration is loaded from `.env`.

Main configuration groups include:

```text
Application
Files
MongoDB
LLM
Qdrant
Retry
```

Example local values:

```env
APP_NAME="rag-system"
APP_VERSION="0.1.0"

MONGODB_URL="mongodb://admin:admin@localhost:27007"
MONGODB_DATABASE="rag-system"

OLLAMA_BASE_URL="http://172.31.128.1:11434"

QDRANT_URL="http://localhost:6333"
QDRANT_VECTOR_SIZE=1024
QDRANT_DISTANCE="cosine"
```

Do not commit real secrets to Git.

## Testing

Run the test suite:

```bash
PYTHONPATH=src uv run pytest
```

Run with coverage:

```bash
PYTHONPATH=src uv run pytest --cov=src --cov-report=term-missing
```

## Code Quality

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pre-commit run --all-files
```

## Development

The project keeps provider and infrastructure details behind interfaces, factories, managers, and application services so they can be replaced independently.

Current development infrastructure is intentionally separated from future operational work such as:

```text
Observability
Redis
Prometheus / Grafana
Load Testing
LLMOps / Evaluation
vLLM
Deployment
```

These concerns can be added independently without changing the core application flow.

## License

No license is currently declared.
