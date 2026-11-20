
# Modular RAG Chatbot

A production-oriented, multi-user Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask context-aware questions through an interactive chat interface.

## Architecture

```text
                  User
                   |
                   v
            Streamlit Frontend
                Port 8501
                   |
                   v
              FastAPI Backend
                Port 8000
                   |
          +--------+--------+
          |        |        |
          v        v        v
       Qdrant    Redis    RAG Pipeline
          |                 |
          |          +------+------+
          |          |             |
          v          v             v
       Retrieval  MultiQuery   FlashRank
                              Reranking
                                  |
                                  v
                             LLM Generation
                                  |
                                  v
                               Bifrost
                                  |
                                  v
                              LLM Provider

       Ollama: Local embedding generation
```

## Features

- PDF document upload and ingestion
- Recursive text chunking
- Local embeddings using Ollama `embeddinggemma`
- Vector storage and similarity search using Qdrant
- Multi-query retrieval
- FlashRank document reranking
- Context-grounded answer generation
- Source references with page numbers
- Redis-based retrieval caching
- Multi-user document isolation using metadata filters
- JWT-based API authentication
- FastAPI backend and Streamlit frontend
- Dockerized application services
- Health and readiness endpoints
- Request latency and LLM token-usage logging
- Unit and integration test coverage


## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| API Authentication | JWT |
| Language | Python 3.12 |
| LLM Integration | Bifrost |
| LLM | Gemini 2.5 Flash (configured model) |
| Embeddings | Ollama — embeddinggemma |
| Vector Database | Qdrant |
| Retrieval | LangChain MultiQueryRetriever |
| Reranking | FlashRank |
| Caching | Redis |
| PDF Processing | PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |
| Testing | Pytest |
| Evaluation | RAGAS / DeepEval |
| Observability | Python Logging |
| Containerization | Docker and Docker Compose |
| Workflow Automation | n8n |

## Project Structure

```text
Rag_chatbot/
│
├── app/
│   ├── api/
│   │   └── routes/
│   ├── core/
│   ├── infrastructure/
│   ├── monitoring/
│   ├── rag/
│   └── services/
│
├── frontend/
│   ├── streamlit_app.py
│   └── Dockerfile
│
├── tests/
├── uploads/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── benchmark_rag.py
├── benchmark_retrieval.py
├── benchmark_api.py
└── README.md
```

## Installation and Setup

### Prerequisites

- Python 3.12
- Docker Desktop
- Ollama
- Qdrant
- Redis
- Bifrost

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd Rag_chatbot
```

### 2. Configure Environment Variables

Create a `.env` file in the project root.

Required configuration:

```env
APP_NAME=Modular RAG Chatbot
ENVIRONMENT=development
LOG_LEVEL=INFO

OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=embeddinggemma

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_documents

BIFROST_BASE_URL=http://localhost:8080/v1
BIFROST_VIRTUAL_KEY=your_bifrost_virtual_key
LLM_MODEL=gemini-2.5-flash

REDIS_URL=redis://localhost:6379/0
REDIS_TTL=3600

JWT_SECRET_KEY=your_secure_random_secret
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

Use your actual service configuration and keep `.env` out of Git.

### 3. Start the External Services

Ensure Redis, Qdrant, and Bifrost are running and connected to the Docker network:

```bash
docker network create rag-network
```

Run this only if the network does not already exist.

Ollama runs on the host machine and must have the configured embedding model available:

```bash
ollama pull embeddinggemma
```

### 4. Build and Start the Application

```bash
docker compose up -d --build
```

This starts the FastAPI backend and Streamlit frontend.

### 5. Access the Application

| Component | URL |
|---|---|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI API | http://localhost:8000 |
| Swagger Documentation | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |
| Readiness Check | http://localhost:8000/ready |

### 6. Stop the Application

```bash
docker compose down
```

## RAG Workflow

The application follows a modular RAG pipeline.

### Document Ingestion

1. User authenticates and uploads a PDF.
2. FastAPI validates and processes the document.
3. PyPDFLoader extracts document content and metadata.
4. RecursiveCharacterTextSplitter divides content into chunks.
5. Ollama generates embeddings using `embeddinggemma`.
6. Qdrant stores vectors with metadata, including:
   - `user_id`
   - `document_id`
   - `file_name`
   - `page_number`
   - `chunk_id`
   - `file_hash`

### Question Answering

1. User submits a question through Streamlit.
2. FastAPI validates the JWT and identifies the user.
3. Redis checks for cached retrieval results.
4. MultiQueryRetriever generates alternative search queries.
5. Qdrant retrieves relevant document chunks using user-scoped filters.
6. FlashRank reranks the retrieved chunks.
7. The LLM generates an answer using the retrieved context.
8. The API returns the answer and source references.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/demo-login` | Generate a demo JWT |
| POST | `/documents/upload` | Upload and ingest a PDF |
| GET | `/documents` | List the authenticated user's documents |
| DELETE | `/documents/{document_id}` | Delete a user's document |
| POST | `/chat` | Ask questions about documents |
| GET | `/health` | Check API health |
| GET | `/ready` | Check Redis and Qdrant readiness |

Protected endpoints require a Bearer JWT.

### Authentication and Security

The application uses JWT-based authentication to protect API endpoints.

Document isolation is enforced through Qdrant metadata filters using `user_id`. This ensures that retrieval and document operations are scoped to the authenticated user.

**Demo authentication limitation:** `/auth/demo-login` accepts a user ID supplied by the caller. It is intended for demonstration purposes, not production identity verification. A production deployment should use proper user registration/login or an identity provider.


## Performance Benchmarks

The application includes retrieval-cache and end-to-end API benchmarks.

### Retrieval Cache Benchmark

| Metric | Result |
|---|---:|
| Cache misses measured | 5 |
| Cache hits measured | 30 |
| Average cache miss latency | 2042.39 ms |
| Average cache hit latency | 1.41 ms |
| Retrieval speedup | 1449× |
| Retrieval latency reduction | 99.93% |

A cache hit bypasses MultiQueryRetriever, Qdrant searches, and FlashRank reranking.

### End-to-End API Benchmark

Latest measured results:

| Metric | Cache Miss | Cache Hit |
|---|---:|---:|
| Average request latency | 3507.21 ms | 957.78 ms |

- Measured speedup: 3.66×
- Request errors: 0

The API benchmark includes retrieval and LLM generation. Cached retrieval still requires answer generation, which explains why total API latency remains higher than retrieval-cache latency.

*These are development benchmark results from a limited sample, not production load-test guarantees.*

## Testing

The project test suite was executed using:

```bash
pytest -v
```

Results:

- 77 tests passed
- 70 unit tests
- 7 integration tests

Coverage includes:

- Authentication and JWT validation
- Document ingestion and duplicate detection
- Multi-user document isolation
- RAG pipeline execution
- Redis retrieval caching
- API security and access control

## Monitoring

The application includes:

- HTTP request method, path, status, and latency logging
- LLM generation latency logging
- LLM input and output token usage logging
- Health endpoint
- Redis and Qdrant readiness checks
- Error logging for failed requests and generation


## Current Limitations

- Authentication currently uses demo login and does not verify user identity.
- PDF is the currently supported document format.
- Performance benchmarks were conducted in a local development environment.
- Readiness checks currently cover Redis and Qdrant, not every external dependency.
- The application has not yet been deployed to a production cloud environment.

## Future Improvements

- Implement production-grade authentication and authorization.
- Support additional document formats such as DOCX, PPTX, and CSV.
- Add advanced retrieval strategies such as hybrid search.
- Integrate automated RAG evaluation into the development workflow.
- Improve monitoring with production metrics and dashboards.
- Deploy the application to AWS with scalable infrastructure.

## Author

**AI / Generative AI Engineer**

GitHub: https://github.com/Sahil799717
