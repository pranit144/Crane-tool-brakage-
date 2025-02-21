# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /code

# Set environment variables
ENV PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /code/uploads /code/logs

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Make directories writable
RUN chmod -R 777 /code/uploads /code/logs

# Command to run the application
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --timeout 120 app:app 