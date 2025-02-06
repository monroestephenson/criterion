#!/usr/bin/env python3
import os
import re
import json
import pickle
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_transformer_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    """
    Load and return a SentenceTransformer model.
    Using a larger model can capture more nuance.
    """
    model = SentenceTransformer(model_name)
    return model

def clean_subtitle_text(text):
    """
    Remove SRT numbering and timestamps.
    """
    lines = text.splitlines()
    cleaned_lines = []
    timestamp_pattern = re.compile(r'\d{2}:\d{2}:\d{2},\d{3}')
    for line in lines:
        line = line.strip()
        # Skip indices and timestamp lines.
        if line.isdigit() or '-->' in line or timestamp_pattern.search(line):
            continue
        if not line:
            continue
        cleaned_lines.append(line)
    return " ".join(cleaned_lines)

def chunk_text(text, max_chunk_size=500):
    """
    Split text into chunks of up to max_chunk_size words.
    You can experiment with chunk sizes to best capture local context.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size):
        chunk = " ".join(words[i:i + max_chunk_size])
        chunks.append(chunk)
    return chunks

def load_subtitle_files(subtitles_dir: str, movies_metadata_path: str):
    """
    Load subtitle files (assumed .srt) and match them to movies using metadata.
    
    Assumes that the filename (without extension) matches the movie title.
    """
    movie_scripts = {}
    
    # Load movie metadata.
    with open(movies_metadata_path, 'r', encoding='utf-8') as f:
        movies_metadata = json.load(f)
    
    # Map normalized titles to metadata.
    metadata_map = {}
    for movie in movies_metadata:
        title_norm = movie['title'].strip().lower()
        metadata_map[title_norm] = movie

    subtitles_path = Path(subtitles_dir)
    subtitle_files = list(subtitles_path.glob("*.srt"))
    logger.info(f"Found {len(subtitle_files)} subtitle files.")
    
    for file_path in subtitle_files:
        try:
            # Assume the file stem is the movie title.
            movie_title = file_path.stem.strip().lower()
            if movie_title not in metadata_map:
                logger.warning(f"Subtitle file '{file_path.name}' not matched to any metadata.")
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            cleaned_text = clean_subtitle_text(content)
            
            # Use film_id as a unique identifier.
            film_id = metadata_map[movie_title]["film_id"]
            movie_scripts[film_id] = {
                "script": cleaned_text,
                "metadata": metadata_map[movie_title]
            }
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
    
    logger.info(f"Loaded scripts for {len(movie_scripts)} movies.")
    return movie_scripts

def build_movie_chunk_embeddings(movie_scripts, model, max_chunk_size=500, batch_size=8):
    """
    For each movie, split the subtitle text into chunks and compute embeddings for each chunk.
    
    Args:
        movie_scripts: Dict with keys as film_ids and values containing 'script' and 'metadata'
        model: A SentenceTransformer model.
        max_chunk_size: Maximum number of words per chunk.
        batch_size: Number of chunks to embed in one batch.
    
    Returns:
        movie_embeddings: Dict mapping film_id to:
            - "chunk_embeddings": A numpy array of shape (num_chunks, embedding_dim)
            - "metadata": Movie metadata.
            - "chunks": (optional) The list of text chunks.
    """
    movie_embeddings = {}
    for film_id, data in tqdm(movie_scripts.items(), desc="Building chunk embeddings"):
        script = data["script"]
        # Split script into smaller chunks.
        chunks = chunk_text(script, max_chunk_size)
        chunk_embeddings = []
        # Process chunks in batches for efficiency.
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            embeddings = model.encode(batch_chunks, convert_to_numpy=True)
            chunk_embeddings.extend(embeddings)
        movie_embeddings[film_id] = {
            "chunk_embeddings": np.array(chunk_embeddings),
            "metadata": data["metadata"],
            "chunks": chunks  # You can remove this if you don't need the raw text.
        }
    return movie_embeddings

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    A small epsilon is added to avoid division by zero.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def compute_movie_similarity(movie_a, movie_b, aggregation="max"):
    """
    Compute a similarity score between two movies using their chunk embeddings.
    """
    embeddings_a = movie_a["chunk_embeddings"]
    embeddings_b = movie_b["chunk_embeddings"]
    
    if len(embeddings_a) == 0 or len(embeddings_b) == 0:
        return 0.0  # Return zero similarity if either movie has no embeddings
    
    sims = []
    for vec_a in embeddings_a:
        # Compute similarities between one chunk of A and all chunks in B.
        sims_b = [cosine_similarity(vec_a, vec_b) for vec_b in embeddings_b]
        if sims_b:  # Only append if we found similarities
            sims.append(max(sims_b))
    
    if not sims:  # If no similarities were found
        return 0.0
        
    if aggregation == "max":
        overall_sim = max(sims)
    elif aggregation == "mean":
        overall_sim = np.mean(sims)
    else:
        overall_sim = np.mean(sims)
    return overall_sim

def find_similar_movies(query_film_id, movie_embeddings, top_k=5, aggregation="max"):
    """
    Given a query movie, find similar movies by comparing their chunk embeddings.
    
    Args:
        query_film_id: The film_id of the query movie.
        movie_embeddings: Dict mapping film_ids to their embeddings and metadata.
        top_k: Number of similar movies to return.
        aggregation: Aggregation method to compute similarity between movies.
    
    Returns:
        A list of tuples: (film_id, similarity_score, metadata)
    """
    if query_film_id not in movie_embeddings:
        logger.error(f"Film id {query_film_id} not found in the embeddings.")
        return []
    
    query_movie = movie_embeddings[query_film_id]
    similarities = []
    for film_id, data in movie_embeddings.items():
        if film_id == query_film_id:
            continue
        sim = compute_movie_similarity(query_movie, data, aggregation)
        similarities.append((film_id, sim, data["metadata"]))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def save_embeddings_checkpoint(embeddings, checkpoint_path: str):
    """
    Save embeddings checkpoint to disk using pickle.
    """
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(embeddings, f)
    logger.info(f"Saved embeddings checkpoint to {checkpoint_path}")

def load_embeddings_checkpoint(checkpoint_path: str):
    """
    Load embeddings checkpoint from disk.
    """
    if not os.path.exists(checkpoint_path):
        logger.info("Embeddings checkpoint not found.")
        return None
    with open(checkpoint_path, 'rb') as f:
        embeddings = pickle.load(f)
    logger.info(f"Loaded embeddings checkpoint from {checkpoint_path}")
    return embeddings

if __name__ == "__main__":
    try:
        # Paths (update these paths as needed)
        subtitles_dir = "/Users/monroestephenson/Downloads/Criterion_Collection_Recomendation/raw_data/subtitles"
        movies_metadata_path = "/Users/monroestephenson/Downloads/Criterion_Collection_Recomendation/criterion_collection/criterion_movies.json"
        checkpoint_path = "/Users/monroestephenson/Downloads/Criterion_Collection_Recomendation/advanced_subtitle_embeddings.pkl"
        
        logger.info("Loading transformer model...")
        model = load_transformer_model()  # using a model that may yield richer embeddings
        
        # Load subtitle texts matched to movie metadata.
        movie_scripts = load_subtitle_files(subtitles_dir, movies_metadata_path)
        
        # Load precomputed embeddings if available.
        movie_embeddings = load_embeddings_checkpoint(checkpoint_path)
        if movie_embeddings is None:
            movie_embeddings = build_movie_chunk_embeddings(movie_scripts, model, max_chunk_size=500, batch_size=8)
            save_embeddings_checkpoint(movie_embeddings, checkpoint_path)
        
        # For demonstration, pick a query movie (using its film_id) and find similar movies.
        query_film_id = next(iter(movie_embeddings))
        query_title = movie_embeddings[query_film_id]["metadata"]["title"]
        logger.info(f"Finding movies similar to '{query_title}' (film_id: {query_film_id})...")
        
        similar_movies = find_similar_movies(query_film_id, movie_embeddings, top_k=5, aggregation="max")
        logger.info("Similar movies based on deep subtitle cuts:")
        for film_id, sim, metadata in similar_movies:
            logger.info(f"- {metadata['title']} (film_id: {film_id}, similarity: {sim:.3f})")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        exit(1)