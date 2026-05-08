# Stage 1: Build the React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/rag_agent/frontend
COPY rag_agent/frontend/package*.json ./
RUN npm install --prefer-offline
COPY rag_agent/frontend/ ./
RUN npm run build

# Stage 2: Python application
FROM python:3.11-slim AS app

# System dependencies needed by psycopg and pgvector
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Overwrite the frontend dist with the compiled output from Stage 1
COPY --from=frontend-builder /app/rag_agent/frontend/dist ./rag_agent/frontend/dist

# Create data directory for runtime state
RUN mkdir -p rag_agent/data

# Expose the API port (override with PORT env var at runtime)
EXPOSE 8000

# Run database migrations then start the API server
CMD ["sh", "-c", "python -m alembic upgrade head && python -m rag_agent.api"]
