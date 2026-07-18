# ZapTheTrick AI — Thin-Slice Backend

A minimal but **fully working** FastAPI backend: one SQLite workspace, Ollama
chat integration, a Sense→Plan skeleton, and streaming responses over SSE.

This is the foundation layer from the architecture document — not the whole
system. It deliberately omits the multi-agent harness, RAG, the response-shape
layer, live audio, and the driver plugin system. Those build *on top of* this.

## What it does

- Creates and persists conversations and messages in a local SQLite file.
- Classifies each incoming question (behavioral / coding / concept / general)
  with a fast heuristic — the seam where a real Phase-1 LLM call goes later.
- Picks a system prompt based on that intent.
- Streams the model's reply token-by-token from Ollama to the client.
- Persists the completed assistant message.

## Prerequisites

1. **Python 3.11+**
2. **Ollama**, running locally — https://ollama.com
   After installing, pull a model:
   ```
   ollama pull llama3.2
   ```
   (Any chat model works. If you use a different one, see Configuration below.)

## Setup

```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
# from the backend/ directory, with the venv active:
uvicorn app.main:app --reload --port 8000
```

You should see Uvicorn start on `http://127.0.0.1:8000`.

Check it works:

```bash
curl http://127.0.0.1:8000/api/health
```

Expected response (with Ollama running):

```json
{"backend":"ok","database":"ok","ollama":"ok","model":"llama3.2"}
```

If `"ollama":"unreachable"`, start Ollama and pull a model. The backend still
runs without Ollama — only the chat endpoint needs it.

Interactive API docs are at `http://127.0.0.1:8000/docs`.

## Configuration

All configuration is environment variables, read in `app/config.py`:

| Variable               | Default                  | Purpose                          |
|------------------------|--------------------------|----------------------------------|
| `OLLAMA_MODEL`         | `llama3.2`               | Which Ollama model to use        |
| `OLLAMA_BASE_URL`      | `http://localhost:11434` | Where Ollama is listening        |
| `OLLAMA_TIMEOUT`       | `120`                    | Stream timeout, seconds          |
| `ZAPTHETRICK_DATA_DIR`     | `backend/data`           | Where `workspace.db` is created  |
| `ZAPTHETRICK_PORT`         | `8000`                   | Server port                      |
| `ZAPTHETRICK_CORS_ORIGINS` | `*`                      | Comma-separated allowed origins  |

Example — use a different model:

```bash
OLLAMA_MODEL=qwen2.5:7b uvicorn app.main:app --reload --port 8000
```

## API surface

| Method | Path                          | Purpose                              |
|--------|-------------------------------|--------------------------------------|
| GET    | `/api/health`                 | Backend / DB / Ollama status         |
| GET    | `/api/conversations`          | List conversation summaries          |
| GET    | `/api/conversations/{id}`     | One conversation with all messages   |
| POST   | `/api/chat/stream`            | Send a message, stream reply (SSE)   |

### The streaming endpoint

`POST /api/chat/stream` with body `{"message": "...", "conversation_id": "..."}`
(omit `conversation_id` to start a new conversation).

It returns `text/event-stream` with these events, in order:

```
event: meta    data: {"conversation_id": "...", "intent": "coding"}
event: token   data: {"text": "..."}      ← repeated, many times
event: done    data: {"message_id": "..."}
```

or, on failure:

```
event: error   data: {"detail": "..."}
```

## Project layout

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py            FastAPI app, CORS, lifespan
│   ├── config.py          all configuration
│   ├── database.py        SQLAlchemy models + async session
│   ├── schemas.py         Pydantic request/response shapes
│   ├── pipeline.py        the Sense→Plan skeleton
│   ├── ollama_client.py   async streaming Ollama client
│   └── routes.py          all API endpoints
├── requirements.txt
└── README.md
```

## Where to go next

The code is commented with the seams where the full architecture plugs in.
The highest-value next steps, in order:

1. **Replace `pipeline.classify_intent`** with a real Phase-1 LLM call that
   returns a structured intent object.
2. **Add a retrieval step** before the Ollama call — the RAG layer.
3. **Add the response-shape enforcement** — pick a structural template per
   intent and validate the model output against it.
4. **Split `database.py`** into the repository + driver abstraction so other
   databases (Postgres, etc.) become possible.
