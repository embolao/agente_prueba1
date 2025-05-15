# syntax=docker/dockerfile:1.4

# Use the official Python image as a parent image
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies directly with pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN adduser --disabled-password -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy the rest of the application
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Command to run the application
CMD ["bash"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest black flake8 mypy

# Production stage
FROM base as production

# Install the package in development mode
RUN pip install .


# Command to run the application
CMD ["python", "-m", "agente_prueba1"]
