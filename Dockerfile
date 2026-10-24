# One image: FastAPI serves the API and the built React app from the same origin.

# 1. Build the React app (also copies public/eval/results.json into dist/eval/).
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json frontend/.npmrc ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# 2. Python runtime with the search indexes baked in.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/backend/.venv/bin:$PATH"
WORKDIR /app/backend

COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Indexes (~400 MB) come from the public Hugging Face dataset at build time, so the layer is
# cached across code changes and a restart never re-downloads.
ARG HF_REPO=baxri/fiqa-hybrid-indexes
COPY backend/scripts/download_indexes.py scripts/
RUN python scripts/download_indexes.py --repo "$HF_REPO" --dest data

COPY backend/app app/
COPY --from=frontend /app/frontend/dist /app/frontend/dist

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
# Railway sets PORT; locally it defaults to 8000.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
