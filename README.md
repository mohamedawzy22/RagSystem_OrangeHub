# RagSystem OrangeHub

A Retrieval-Augmented Generation (RAG) system built with FastAPI.

## Requirements

* Python 3.11+
* uv

## Installation

```bash
uv sync
```

Create a `.env` file in the project root and add the required environment variables.

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

The application will be available at:

```text
http://localhost:8001
```

API documentation:

```text
http://localhost:8001/docs
```

## CI

GitHub Actions automatically runs code quality checks and tests on pushes and pull requests.
