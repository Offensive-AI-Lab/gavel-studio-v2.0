# Python 3.12 — same as the backend + CI. The central server's deps are all
# pure-wheel installs (fastapi, psycopg2-binary, passlib/argon2, python-jose,
# huggingface-hub), so no compilers are needed — the slim base is enough.
FROM python:3.12-slim

# curl is only here so the HEALTHCHECK below can hit /health.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so the (slow) pip layer is cached across rebuilds
# that only change source.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest. In dev this is shadowed by the compose volume mount; the COPY
# keeps the image self-contained for `docker run` / prod use.
COPY . .

EXPOSE 8001

# /health lets `depends_on: condition: service_healthy` wait for a real serve.
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
  CMD curl --fail http://localhost:8001/health || exit 1

# compose overrides this to create the DB first (python _create_db.py). This
# CMD is the production-style fallback for running the image without compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
