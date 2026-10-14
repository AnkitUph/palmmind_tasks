# Palm Mind RAG Backend

A FastAPI backend with two REST APIs:

1. **Document Ingestion API** (`POST /ingest`): upload a `.pdf` or `.txt` file, pick a chunking strategy, store the embeddings in **Qdrant**, and save the metadata in **SQL** (PostgreSQL, or SQLite for local runs).
2. **Conversational RAG API** (`POST /chat`): a custom RAG pipeline (no `RetrievalQAChain`) with **Redis** chat memory, multi-turn questions, and **interview booking** driven by an LLM.

Everything runs locally and needs no API keys. Embeddings come from `sentence-transformers`, and the LLM runs in **Ollama**.

## Architecture

```
            POST /ingest                              POST /chat
                 |                                        |
         IngestionService                            RagService
  extract -> chunk -> embed -> store            history (Redis)
        |          |        |                         |
   pypdf   2 strategies  Qdrant + SQL        BookingService  --(booking turn)--> SQL + Redis
                                                      |
                                              (not a booking turn)
                                                      |
                              condense follow-up -> embed -> Qdrant search -> LLM answer
```

| Layer | Folder | Job |
|---|---|---|
| API | `app/api/` | HTTP only: read input, call a service, return output, map errors to JSON |
| Services | `app/services/` | All business logic, one job per file |
| Models | `app/models/` | SQLAlchemy tables and Pydantic schemas |
| Core | `app/core/` | Settings, errors, dependency injection (`Depends`) |

### Design choices

- **Two chunking strategies**, chosen with the `strategy` form field:
  - `fixed`: an 800-character sliding window with 100 characters of overlap.
  - `sentence`: packs whole sentences up to 800 characters and repeats the last sentence as overlap.
- **Custom RAG**: the prompt is built by hand (`build_answer_messages` in `rag.py`). A follow-up such as "what about Fridays?" is rewritten into a standalone question before searching, which is how multi-turn questions work.
- **Retrieval quality**: chunks below a similarity score (`MIN_SCORE`) are dropped, and identical chunks (for example the same file uploaded twice) are shown only once. If nothing matches, the assistant says so instead of guessing.
- **Redis memory**: the last N turns per `session_id` are kept in a Redis list (`LTRIM` plus a TTL). A half-filled booking form is also kept there.
- **Interview booking**:
  - Messages that mention booking words go to the LLM, which returns JSON with `name`, `email`, `date` and `time`. Ordinary questions skip that call, which saves time.
  - A keyword check backs up the LLM in case a small model misses an obvious request.
  - Python validates the values (`EmailStr`, ISO date not in the past, `HH:MM` time), asks for whatever is missing, and saves a row in `bookings` when all four are valid.
  - The user can say "cancel" to drop the draft. A normal question during a booking is answered, and the draft is kept.
- **Consistency**: if saving the metadata fails after the vectors were stored, the vectors are deleted again.
- **Clean errors**: Redis, Qdrant, database and LLM outages return a `503` with a clear message, and any other unexpected error returns a JSON `500` and is logged.
- **Typing**: full type annotations, and `Protocol` classes for the embedder and the LLM so they are easy to swap.

## Quick start (Docker)

### Prerequisites

- Docker with Docker Compose
- **Ollama installed on the host machine.** The LLM runs outside Docker, so no large GPU image is needed.

### 1. Install Ollama and download the model

```bash
# Linux: https://ollama.com/download   (Arch: sudo pacman -S ollama)
sudo systemctl enable --now ollama      # or just run: ollama serve
ollama pull llama3.2
```

The API runs in a container, so Ollama must listen on all network interfaces, not only on localhost:

```bash
sudo systemctl edit ollama
# add these two lines, save, then restart:
#   [Service]
#   Environment="OLLAMA_HOST=0.0.0.0"
sudo systemctl restart ollama
```

If your firewall is on (for example `ufw`), allow Docker's network to reach Ollama:

```bash
sudo ufw allow from 172.16.0.0/12 to any port 11434 proto tcp
```

### 2. Start the project

```bash
cp .env.example .env        # hidden file: use `ls -a` to see it. Optional for Docker.
docker compose up -d --build
```

The first start downloads the embedding model (about 90 MB), so give it a minute. Follow progress with `docker compose logs -f api` and wait for `Application startup complete`.

Then open:

- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 3. Check that everything works

```bash
./scripts/smoke_test.sh
```

The script waits for the API to be ready, then uploads a sample document with both strategies, asks questions, tries a follow-up, and books an interview. It prints each answer and how long it took. Read the output and check that the answers make sense.

### Port clashes

If a port is already used on your machine (for example Redis on 6379), set another host port in `.env` and run `docker compose up -d` again:

```
REDIS_HOST_PORT=6381
POSTGRES_HOST_PORT=5433
QDRANT_HOST_PORT=6334
API_HOST_PORT=8000
```

### Start fresh

To delete all uploaded documents, bookings and the cached model:

```bash
docker compose down -v
```

## Run the API without Docker

Run only the databases in Docker, and the API on your machine (Ollama is reached at `localhost`):

```bash
docker compose up -d qdrant redis postgres
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` if you changed host ports, for example `REDIS_URL=redis://localhost:6381/0`. SQLite is the default database, so Postgres is optional here. Then start the server:

```bash
uvicorn app.main:app --reload
```

## API examples

**Ingest a document**

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@handbook.pdf" \
  -F "strategy=sentence"
```

```json
{"document_id": "6f1c...", "filename": "handbook.pdf", "strategy": "sentence", "chunk_count": 42}
```

**Ask a question**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-1", "message": "How many leave days do employees get?"}'
```

```json
{
  "answer": "Employees get 20 paid leave days per year.",
  "sources": [{"document_id": "6f1c...", "filename": "handbook.pdf", "chunk_index": 7, "score": 0.71, "snippet": "..."}],
  "booking": null
}
```

The assistant answers only from uploaded documents. If nothing matches, it says it could not find the answer.

**Follow-up question (same `session_id`)**

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id": "user-1", "message": "And does it carry over to next year?"}'
```

**Book an interview (several messages)**

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id": "user-1", "message": "I want to book an interview, my name is Ankit Sharma"}'
# -> asks for whatever is missing (email, date, time)

curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"session_id": "user-1", "message": "ankit@example.com, 2026-10-20 at 15:30"}'
# -> "Your interview is booked ... (booking ID 1)"
```

The `booking` field in the response shows the `state` (`collecting`, `confirmed` or `cancelled`), the collected fields, and what is still missing.

Saved bookings can be checked with:

```bash
docker compose exec postgres psql -U rag -d rag -c "select * from bookings;"
```

### Error codes

| Code | Meaning |
|---|---|
| `413` | File is too large (limit is `MAX_UPLOAD_MB`) |
| `415` | File type is not `.pdf` or `.txt` |
| `422` | No readable text found, or the request body is invalid |
| `503` | The LLM, Redis, Qdrant or the database is not reachable |
| `500` | Unexpected error (details are in the server log) |

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `503` with "LLM request failed: timed out" | The container cannot reach Ollama. Check that Ollama listens on `0.0.0.0` (`ss -tlnp \| grep 11434`) and that the firewall rule above is set. Test: `docker compose exec api python -c "import httpx; print(httpx.get('http://host.docker.internal:11434/api/tags', timeout=5).text)"` |
| `503` after a long wait | The model is slow on CPU. Raise `LLM_TIMEOUT_SECONDS` in `.env`, or use a smaller model: `ollama pull llama3.2:1b` and set `OLLAMA_MODEL=llama3.2:1b`. |
| `503` "Chat memory (Redis) is unavailable" | The Redis container is not running. Run `docker compose ps -a`, then `docker compose up -d`. |
| `Bind for 0.0.0.0:6379 failed: port is already allocated` | Another program uses that port. Set `REDIS_HOST_PORT` (see Port clashes). |
| `curl: Connection reset by peer` right after start | The API is still loading the embedding model. Wait for `Application startup complete` in `docker compose logs api`. |
| "I could not find anything..." | No uploaded document matches the question. Upload one with `/ingest` first, or lower `MIN_SCORE` in `.env`. |
| `no space left on device` while pulling images | Free disk space with `docker system prune`. Ollama is not run in Docker here, so no large Ollama image is needed. |

## Configuration

All settings are environment variables (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./app.db` | Metadata and bookings database (Docker uses Postgres) |
| `REDIS_URL` | `redis://localhost:6379/0` | Chat memory and booking drafts |
| `QDRANT_URL` | `http://localhost:6333` | Vector database |
| `QDRANT_COLLECTION` | `documents` | Collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama address |
| `OLLAMA_MODEL` | `llama3.2` | LLM used for answers and booking |
| `LLM_TIMEOUT_SECONDS` | `120` (`300` in Docker) | Time allowed per LLM call |
| `TOP_K` | `4` | Number of chunks used as context |
| `MIN_SCORE` | `0.3` | Minimum similarity for a chunk to be used |
| `CHAT_HISTORY_TURNS` | `6` | Past turns remembered per session |
| `CHAT_TTL_SECONDS` | `3600` | How long a session is remembered |
| `MAX_UPLOAD_MB` | `10` | Largest accepted upload |

About `MIN_SCORE`: with the default embedding model, matching text scores about 0.5 to 0.7 and unrelated text about 0.2. Lower the value if good answers are rejected, and raise it if unrelated text slips through.

## Code checks (optional)

```bash
pip install -r requirements-dev.txt   # also installs the app dependencies (PyTorch is large)
mypy                                  # strict type check
ruff check .                          # lint
```

## Project layout

```
app/
  main.py                  app, lifespan (creates tables and the collection)
  api/
    ingest.py              POST /ingest
    chat.py                POST /chat
    errors.py              maps errors to clean JSON responses
  core/                    config.py, deps.py, exceptions.py
  models/                  db.py (tables), schemas.py (Pydantic)
  services/
    chunking.py            fixed and sentence strategies
    document_loader.py     pdf/txt text extraction
    embeddings.py          local sentence-transformers embedder
    vector_store.py        Qdrant wrapper
    repositories.py        SQL access for documents and bookings
    llm.py                 Ollama client
    memory.py              Redis chat memory and booking draft
    booking.py             LLM extraction, validation, saving
    rag.py                 custom conversational RAG
    ingestion.py           ingestion pipeline
scripts/smoke_test.sh      end-to-end check against a running server
docker-compose.yml         Qdrant, Redis, Postgres and the API
Dockerfile                 API image
```