FROM python:3.11-slim

WORKDIR /app

# Install system dependencies: git, curl for healthcheck, and Node.js for Pokemon Showdown
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    nodejs \
    npm \
  && rm -rf /var/lib/apt/lists/*

# Clone Pokemon Showdown so poke-env can connect to a local server
RUN git clone https://github.com/smogon/pokemon-showdown.git /opt/pokemon-showdown

# Copy the entire repo into the image (includes smogon_rl and env server)
COPY . /app

# Install Python dependencies, including OpenEnv core 0.2.1
RUN pip install --no-cache-dir \
    "openenv-core==0.2.1" \
    "poke-env>=0.8.0,<0.9.0" \
    "numpy>=1.24.0" \
    "pydantic>=2.0.0"

ENV SHOWDOWN_DIR=/opt/pokemon-showdown

# Basic healthcheck against the FastAPI app port
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Start Pokemon Showdown on port 8000 (for poke-env) and the OpenEnv FastAPI app on port 8001
CMD bash -lc "cd /opt/pokemon-showdown && node pokemon-showdown start --no-security --port 8000 & uvicorn env.server.app:app --host 0.0.0.0 --port 8001"

