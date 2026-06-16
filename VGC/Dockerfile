FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY space_app.py .
COPY static/ ./static/
COPY battle_logs/ ./battle_logs/
COPY requirements_space.txt .

RUN pip install --no-cache-dir -r requirements_space.txt

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

EXPOSE 7860

CMD ["uvicorn", "space_app:app", "--host", "0.0.0.0", "--port", "7860"]
