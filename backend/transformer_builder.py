# transformer_builder.py

import numpy as np
import json
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
import random
import os
from urllib.parse import urlparse
from collections import defaultdict
import pickle
import io
import mmap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_transformer_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load and return a SentenceTransformer model.
    """
    model = SentenceTransformer(model_name)
    return model

def save_embeddings_checkpoint(embeddings, checkpoint_dir: Path, prefix: str):
    """Save embeddings checkpoint using numpy for efficiency"""
    checkpoint_file = checkpoint_dir / f"{prefix}_embeddings.npz"
    
    # Separate embeddings and metadata for efficient storage
    slugs = []
    vectors = []
    metadata = {}
    
    for slug, data in embeddings.items():
        slugs.append(slug)
        vectors.append(data['embedding'])
        metadata[slug] = data['metadata']
    
    # Save embeddings as numpy array for efficiency
    np.savez_compressed(
        checkpoint_file,
        slugs=slugs,
        vectors=np.stack(vectors)
    )
    
    # Save metadata separately as JSON
    with open(checkpoint_dir / f"{prefix}_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved embeddings checkpoint: {checkpoint_file}")

def load_embeddings_checkpoint(checkpoint_dir: Path, prefix: str):
    """Load embeddings from checkpoint"""
    checkpoint_file = checkpoint_dir / f"{prefix}_embeddings.npz"
    metadata_file = checkpoint_dir / f"{prefix}_metadata.json"
    
    if not checkpoint_file.exists() or not metadata_file.exists():
        return None
    
    # Load embeddings
    data = np.load(checkpoint_file)
    slugs = data['slugs']
    vectors = data['vectors']
    
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Reconstruct embeddings dictionary
    embeddings = {}
    for slug, vector in zip(slugs, vectors):
        embeddings[slug] = {
            'embedding': vector,
            'metadata': metadata[slug]
        }
    
    return embeddings

def build_movie_embeddings(processed_movies, model, batch_size=32):
    """Enhanced version with batching and progress bar"""
    movie_embeddings = {}
    movies_to_process = [(slug, data) for slug, data in processed_movies.items() 
                        if data.get("aggregated_text")]
    
    logger.info(f"Found {len(movies_to_process)} movies with review text to process")
    
    for i in tqdm(range(0, len(movies_to_process), batch_size), desc="Computing embeddings"):
        batch = movies_to_process[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} of {len(movies_to_process)//batch_size + 1}")
        logger.info(f"Batch size: {len(batch)} movies")
        
        texts = [data.get("aggregated_text", "") for _, data in batch]
        
        # Compute embeddings for batch
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        # Store results
        for (slug, data), embedding in zip(batch, embeddings):
            movie_embeddings[slug] = {
                "embedding": embedding,
                "metadata": data
            }
    
    return movie_embeddings

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def build_user_profile_embedding(user_movies, movie_embeddings, rating_weight=True):
    """
    Build user profile with weighted ratings and genre consideration
    """
    embeddings = []
    weights = []
    
    for movie in user_movies:
        slug = movie["slug"]
        if slug in movie_embeddings:
            # Normalize rating to 0-1 range
            rating = movie["user_rating"] / 5.0 if rating_weight else 1.0
            
            # Add embedding with rating weight
            embeddings.append(movie_embeddings[slug]["embedding"])
            weights.append(rating)
    
    if not embeddings:
        raise ValueError("No valid movies found for user profile")
    
    # Convert to numpy arrays
    embeddings = np.array(embeddings)
    weights = np.array(weights)
    
    # Compute weighted average
    weighted_profile = np.average(embeddings, axis=0, weights=weights)
    
    # Normalize the profile
    return weighted_profile / np.linalg.norm(weighted_profile)

def recommend_movies(user_profile, movie_embeddings, user_seen_slugs, top_k=10, min_similarity=0.3):
    """Enhanced recommendation with genre awareness and diversity"""
    user_genres = set()
    for slug in user_seen_slugs:
        if slug in movie_embeddings:
            metadata = movie_embeddings[slug]["metadata"]
            if "genres" in metadata:
                user_genres.update(metadata["genres"])
    
    candidates = []
    for slug, data in movie_embeddings.items():
        if slug not in user_seen_slugs:
            # Base similarity
            similarity = cosine_similarity(user_profile, data["embedding"])
            
            # Genre bonus (reduced from 0.1 to 0.05)
            genre_overlap = len(set(data["metadata"].get("genres", [])) & user_genres)
            genre_bonus = 0.05 * genre_overlap
            
            # Adjust similarity
            adjusted_similarity = similarity + genre_bonus
            adjusted_similarity = min(adjusted_similarity, 1.0)
            
            if adjusted_similarity >= min_similarity:
                candidates.append((slug, adjusted_similarity, data["metadata"]))
    
    # Sort by similarity
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Ensure diversity
    selected = []
    seen_genres = defaultdict(int)
    
    for candidate in candidates:
        if len(selected) >= top_k:
            break
        
        candidate_genres = set(candidate[2].get("genres", []))
        
        # Check genre diversity
        genre_count = sum(seen_genres[g] for g in candidate_genres)
        if genre_count <= 2:  # Allow some genre overlap but not too much
            selected.append(candidate)
            for genre in candidate_genres:
                seen_genres[genre] += 1
    
    return selected[:top_k]

def split_user_data(user_movies, split_ratio=0.5):
    """
    Split user's movie data into training and test sets.
    
    Args:
        user_movies: List of dictionaries containing movie data
        split_ratio: Fraction of data to use for training (default: 0.5)
    
    Returns:
        train_movies, test_movies: Two lists of movie dictionaries
    """
    # Create a copy to avoid modifying the original
    movies = user_movies.copy()
    random.shuffle(movies)
    
    split_idx = int(len(movies) * split_ratio)
    train_movies = movies[:split_idx]
    test_movies = movies[split_idx:]
    
    return train_movies, test_movies

def evaluate_recommendations(train_movies, test_movies, movie_embeddings, model):
    """
    Enhanced evaluation with rating-aware metrics
    """
    try:
        user_profile = build_user_profile_embedding(train_movies, movie_embeddings)
    except ValueError as e:
        logger.error(f"Error building user profile: {e}")
        return None
    
    # Get recommendations
    recommendations = recommend_movies(
        user_profile,
        movie_embeddings,
        set(movie["slug"] for movie in train_movies),
        top_k=10,
        min_similarity=0.6
    )
    
    # Calculate metrics
    test_movies_dict = {movie["slug"]: movie for movie in test_movies}
    recommended_slugs = set(slug for slug, _, _ in recommendations)
    test_slugs = set(test_movies_dict.keys())
    
    hits = recommended_slugs.intersection(test_slugs)
    
    # Calculate rating-weighted metrics
    weighted_precision = 0
    if hits:
        for slug in hits:
            rec_sim = next(sim for rec_slug, sim, _ in recommendations if rec_slug == slug)
            actual_rating = test_movies_dict[slug]["user_rating"] / 5.0
            weighted_precision += min(rec_sim, actual_rating) / max(rec_sim, actual_rating)
        weighted_precision /= len(hits)
    
    metrics = {
        "precision": len(hits) / len(recommended_slugs) if recommended_slugs else 0,
        "recall": len(hits) / len(test_slugs) if test_slugs else 0,
        "weighted_precision": weighted_precision,
        "recommendations": recommendations
    }
    
    return metrics

def load_random_user_data(reviews_dir="/Users/monroestephenson/Downloads/Criterion_Collection_Recomendation/raw_data/movie_reviews", 
                         movies_list_path="/Users/monroestephenson/Downloads/Criterion_Collection_Recomendation/movies_list.json",
                         min_reviews=10):
    """Load reviews for a randomly selected user who has at least min_reviews in the Criterion Collection."""
    # First load valid movies list
    with open(movies_list_path, 'r', encoding='utf-8') as f:
        valid_movies = {movie['url']: {
            'title': movie['title'],
            'film_id': movie['film_id']
        } for movie in json.load(f)}
    
    json_files = [f for f in os.listdir(reviews_dir) if f.endswith('.json')]
    user_reviews = defaultdict(dict)  # Changed to dict to prevent duplicates
    
    for json_file in json_files:
        try:
            with open(os.path.join(reviews_dir, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                movie_data = data.get('movie', {})
                movie_url = movie_data.get('url', '')
                
                # Skip if movie is not in our valid movies list
                if movie_url not in valid_movies:
                    continue
                    
                reviews = data.get('reviews', [])
                movie_info = valid_movies[movie_url]
                
                for review in reviews:
                    username = review.get('username')
                    if username and review.get('rating'):
                        rating = float(review.get('rating', '').count('★') + 0.5 * ('½' in review.get('rating', '')))
                        
                        # Use film_id as key to prevent duplicates
                        user_reviews[username][movie_info['film_id']] = {
                            "slug": movie_url.split('/film/')[-1].strip('/'),
                            "user_rating": rating,
                            "review_text": review.get('text', ''),
                            "movie_title": movie_info['title'],
                            "film_id": movie_info['film_id']
                        }
        except Exception as e:
            logger.warning(f"Error processing {json_file}: {e}")
            continue
    
    # Convert dict of reviews to list and filter users
    qualified_users = []
    for username, reviews in user_reviews.items():
        if len(reviews) >= min_reviews:
            qualified_users.append((username, list(reviews.values())))
    
    if not qualified_users:
        raise ValueError(f"No users found with at least {min_reviews} reviews of Criterion Collection films")
    
    selected_user, user_movies = random.choice(qualified_users)
    
    logger.info(f"Selected user '{selected_user}' with {len(user_movies)} unique Criterion Collection reviews")
    return selected_user, user_movies

def save_model_and_embeddings(model, movie_embeddings, cache_dir):
    """Save model and embeddings to cache with device handling and numpy arrays"""
    logger.info("Moving model to CPU before saving...")
    
    # Move model to CPU before saving
    if hasattr(model, 'to'):
        model = model.to('cpu')
    
    model_path = os.path.join(cache_dir, "transformer_model.pkl")
    embeddings_path = os.path.join(cache_dir, "movie_embeddings.npz")
    metadata_path = os.path.join(cache_dir, "metadata.pkl")
    
    # Convert embeddings to numpy arrays
    slugs = []
    vectors = []
    metadata = {}
    
    for slug, data in movie_embeddings.items():
        slugs.append(slug)
        vectors.append(data['embedding'])
        metadata[slug] = data['metadata']
    
    # Save embeddings as numpy array
    logger.info(f"Saving embeddings to {embeddings_path}")
    np.savez_compressed(embeddings_path, 
                       slugs=np.array(slugs), 
                       vectors=np.stack(vectors))
    
    # Save metadata separately
    logger.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save model
    logger.info(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def load_cached_model_and_embeddings(cache_dir):
    """Load cached model and embeddings if they exist using memory mapping"""
    model_path = os.path.join(cache_dir, "transformer_model.pkl")
    embeddings_path = os.path.join(cache_dir, "movie_embeddings.npz")
    metadata_path = os.path.join(cache_dir, "metadata.pkl")
    
    if not all(os.path.exists(p) for p in [model_path, embeddings_path, metadata_path]):
        return None, None
        
    try:
        # Load model
        logger.info("Loading model file...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            if hasattr(model, 'to'):
                model = model.to('cpu')
        logger.info("Successfully loaded model")
        
        # Load embeddings using numpy memory mapping
        logger.info("Memory mapping embeddings file...")
        embeddings_data = np.load(embeddings_path, mmap_mode='r')
        slugs = embeddings_data['slugs']
        vectors = embeddings_data['vectors']
        
        # Load metadata
        logger.info("Loading metadata...")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Reconstruct embeddings dictionary using memory-mapped arrays
        movie_embeddings = {}
        for i, slug in enumerate(slugs):
            movie_embeddings[str(slug)] = {
                'embedding': vectors[i],
                'metadata': metadata[str(slug)]
            }
            
        logger.info("Successfully loaded embeddings")
        
        return model, movie_embeddings
        
    except Exception as e:
        logger.error(f"Error loading cached files: {str(e)}")
        return None, None

if __name__ == "__main__":
    try:
        # Try to load cached model and embeddings first
        model, movie_embeddings = None, None
        
        if model is None or movie_embeddings is None:
            # Load and process everything if cache doesn't exist
            processed_file = "/Users/monroestephenson/Downloads/Criterion_Collection_Recomendation/processed_movies.json"
            logger.info(f"Loading processed movies from {processed_file}")
            with open(processed_file, "r", encoding="utf-8") as f:
                processed_movies = json.load(f)
            
            model = load_transformer_model()
            movie_embeddings = build_movie_embeddings(processed_movies, model)
            
            # Save model and embeddings for future use
            save_model_and_embeddings(model, movie_embeddings)
        
        # Load random user data
        username, user_movies = load_random_user_data(min_reviews=10)
        
        # Split user data
        train_movies, test_movies = split_user_data(user_movies)
        logger.info(f"\nEvaluating recommendations for user: {username}")
        logger.info(f"Training set: {len(train_movies)} movies")
        logger.info(f"Test set: {len(test_movies)} movies")
        
        # Show some of user's highest-rated movies from training set
        sorted_train = sorted(train_movies, key=lambda x: x['user_rating'], reverse=True)
        logger.info("\nUser's top-rated movies (training set):")
        for movie in sorted_train[:5]:
            logger.info(f"- {movie['movie_title']}: {movie['user_rating']} stars")
        
        # Get and evaluate recommendations
        metrics = evaluate_recommendations(train_movies, test_movies, movie_embeddings, model)
        
        if metrics:
            logger.info(f"\nPrecision: {metrics['precision']:.3f}")
            logger.info(f"Recall: {metrics['recall']:.3f}")
            
            logger.info("\nTop Recommendations:")
            for slug, sim, meta in metrics['recommendations']:
                title = meta.get("title", slug)
                logger.info(f"- {title} (similarity: {sim:.3f})")
            
            # Show actual test set movies for comparison
            logger.info("\nActual movies user rated highly (test set):")
            sorted_test = sorted(test_movies, key=lambda x: x['user_rating'], reverse=True)
            for movie in sorted_test[:5]:
                logger.info(f"- {movie['movie_title']}: {movie['user_rating']} stars")
                
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        exit(1)