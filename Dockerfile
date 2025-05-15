# syntax=docker/dockerfile:1.4

# Use the official Python image as a base image
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.8.2

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
COPY poetry.lock pyproject.toml ./


# Create a non-root user and switch to it
RUN adduser --disabled-password --gecompat -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Install Python dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the application
COPY --chown=appuser:appuser . .

# Install the package in development mode
RUN poetry install --no-interaction --no-ansi

# Command to run the application
CMD ["bash"]

# Development stage
FROM base as development

# Install development dependencies
RUN poetry install --with dev --no-interaction --no-ansi

# Production stage
FROM base as production

# Copy only the necessary files
COPY --from=development /app /app

# Command to run the application
CMD ["python", "-m", "agente_prueba1"]
