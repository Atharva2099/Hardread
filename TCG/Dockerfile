FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir kaggle_environments pyyaml numpy

COPY . /app/

CMD ["python", "scripts/local_eval.py"]
