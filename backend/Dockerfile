FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/data/model_cache /app/data/models

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=""
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PYTHONUNBUFFERED=1

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy data files first
COPY ./data/model_cache/transformer_model.pkl /app/data/model_cache/
COPY ./data/model_cache/movie_embeddings.pkl /app/data/model_cache/
COPY ./data/models/advanced_subtitle_embeddings.pkl /app/data/models/
COPY ./data/models/movie_clusters.pkl /app/data/models/

# Copy the rest of the application
COPY . .

# Verify file existence and sizes
RUN ls -l /app/start.sh && \
    for f in \
    /app/data/model_cache/transformer_model.pkl \
    /app/data/model_cache/movie_embeddings.pkl \
    /app/data/models/advanced_subtitle_embeddings.pkl \
    /app/data/models/movie_clusters.pkl; \
    do \
    if [ -f "$f" ]; then \
        echo "File $f exists with size $(ls -lh $f | awk '{print $5}')"; \
    else \
        echo "File $f is missing" && exit 1; \
    fi; \
    done

EXPOSE 5001

CMD ["/app/start.sh"]