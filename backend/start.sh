#!/bin/bash

echo "Verifying model files..."
for f in \
    /app/data/model_cache/transformer_model.pkl \
    /app/data/model_cache/movie_embeddings.pkl \
    /app/data/models/advanced_subtitle_embeddings.pkl \
    /app/data/models/movie_clusters.pkl; do
    if [ ! -f "$f" ] || [ ! -s "$f" ]; then
        echo "Error: $f is missing or empty" && exit 1
    fi
done
echo "All model files verified"
python app.py 