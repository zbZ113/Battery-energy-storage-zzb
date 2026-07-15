FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY deploy ./deploy

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[api]"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "deploy.foundation_api:app", "--host", "0.0.0.0", "--port", "8000"]
