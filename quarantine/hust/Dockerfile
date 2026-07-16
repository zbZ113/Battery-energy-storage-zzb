FROM python:3.11.13-slim-bookworm

WORKDIR /app
COPY requirements.lock /app/requirements.lock
COPY discover.py /app/discover.py

USER 65532:65532
ENTRYPOINT ["python", "/app/discover.py"]
