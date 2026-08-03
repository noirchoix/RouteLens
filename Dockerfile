FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REACTS_PROJECT_ROOT=/app \
    REACTS_ARTIFACT_CACHE_DIR=/var/lib/reacts/artifacts \
    REACTS_ARTIFACT_REQUIRED=true \
    REACTS_ARTIFACT_VERIFY_SHA256=true \
    REACTS_ARTIFACT_WARMUP=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends libxrender1 libxext6 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 reacts \
    && mkdir -p /app /var/lib/reacts/artifacts /var/log/reacts \
    && chown -R reacts:reacts /app /var/lib/reacts /var/log/reacts

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
COPY .env.example ./

USER reacts
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/ready || exit 1
CMD ["reacts", "--project-root", "/app", "serve", "--host", "0.0.0.0", "--port", "8000", "--require-artifacts"]
