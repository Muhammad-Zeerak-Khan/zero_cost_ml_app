# STAGE 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Create virtual environment and install CPU-optimized ML dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install packages and aggressively clear pip cache to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# STAGE 2: Production Runner
FROM python:3.10-slim AS runner

WORKDIR /app

# Copy the pre-built virtual environment from Stage 1
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source code
COPY src/ ./src/
COPY app.py .

# Create a non-root user to enforce enterprise security policies
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Boot FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]