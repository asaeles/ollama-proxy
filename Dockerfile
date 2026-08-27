FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd -m appuser
WORKDIR /app
RUN mkdir output && chown -R appuser:appuser /app

USER appuser

COPY --chown=appuser:appuser pyproject.toml .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir .

COPY --chown=appuser:appuser . .

CMD ["ollama-proxy"]