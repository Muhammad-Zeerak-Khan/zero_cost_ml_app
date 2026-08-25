# Stage 1: Fast dependency resolution
FROM python:3.10-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN uv pip install -r requirements.txt

# Stage 2: Production Image
FROM python:3.10-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/opt/venv/bin:$PATH" \
    HOME=/home/user \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app
# Copy strictly the virtual environment from Stage 1
COPY --from=builder --chown=user /opt/venv /opt/venv
COPY --chown=user src/ src/
EXPOSE 7860
CMD ["python", "src/app.py"]