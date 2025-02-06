import logging
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from web_app.backend.transformer_builder import (
    load_transformer_model,
    build_movie_embeddings,
    build_user_profile_embedding,
    recommend_movies,
    split_user_data,
    load_cached_model_and_embeddings,
    save_model_and_embeddings
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_random_user_data(reviews_dir="/Users/monroestephenson/Downloads/Criterion_Collection_Recomendation/movie_reviews", min_reviews=10):
    """Load all reviews for a randomly selected user who has at least min_reviews."""
    json_files = [f for f in os.listdir(reviews_dir) if f.endswith('.json')]
    user_reviews = defaultdict(list)
    
    for json_file in json_files:
        try:
            with open(os.path.join(reviews_dir, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                movie_data = data.get('movie', {})
                reviews = data.get('reviews', [])
                
                for review in reviews:
                    username = review.get('username')
                    if username and review.get('rating'):  # Only count reviews with ratings
                        user_reviews[username].append({
                            "slug": movie_data.get('url', '').split('/film/')[-1].strip('/'),
                            "user_rating": float(review.get('rating', '').count('★') + 0.5 * ('½' in review.get('rating', ''))),
                            "review_text": review.get('text', ''),
                            "movie_title": movie_data.get('title', '')
                        })
        except Exception as e:
            logger.warning(f"Error processing {json_file}: {e}")
            continue
    
    qualified_users = [user for user, reviews in user_reviews.items() 
                      if len(reviews) >= min_reviews]
    
    if not qualified_users:
        raise ValueError(f"No users found with at least {min_reviews} reviews")
    
    selected_user = random.choice(qualified_users)
    user_movies = user_reviews[selected_user]
    
    logger.info(f"Selected user '{selected_user}' with {len(user_movies)} reviews")
    return selected_user, user_movies

def evaluate_recommendations(train_movies, test_movies, movie_embeddings, model, top_k=10):
    """Evaluate recommendation quality using train/test split."""
    try:
        user_profile = build_user_profile_embedding(train_movies, movie_embeddings)
    except ValueError as e:
        logger.error(f"Error building user profile: {e}")
        return None
    
    user_seen_slugs = set(movie["slug"] for movie in train_movies)
    test_slugs = set(movie["slug"] for movie in test_movies)
    
    recommendations = recommend_movies(
        user_profile,
        movie_embeddings,
        user_seen_slugs,
        top_k=top_k,
        min_similarity=0.3
    )
    
    recommended_slugs = set(slug for slug, _, _ in recommendations)
    hits = recommended_slugs.intersection(test_slugs)
    
    metrics = {
        "recall": len(hits) / len(test_slugs) if test_slugs else 0,
        "precision": len(hits) / len(recommended_slugs) if recommended_slugs else 0,
        "recommendations": recommendations
    }
    
    return metrics

def test_user_recommendations(num_users=5, min_reviews=10):
    """Test recommendations for multiple random users."""
    # Try to load cached model and embeddings
    model, movie_embeddings = load_cached_model_and_embeddings()
    
    if model is None or movie_embeddings is None:
        # Load everything if cache doesn't exist
        processed_file = "/Users/monroestephenson/Downloads/Criterion_Collection_Recomendation/checkpoints_20250203_144151/final_processed_checkpoint.json"
        logger.info(f"Loading processed movies from {processed_file}")
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_movies = json.load(f)
        
        model = load_transformer_model()
        movie_embeddings = build_movie_embeddings(processed_movies, model)
        
        # Save for future use
        save_model_and_embeddings(model, movie_embeddings)
    
    total_precision = 0
    total_recall = 0
    
    for i in range(num_users):
        try:
            logger.info(f"\n=== Testing User {i+1}/{num_users} ===")
            
            # Load random user data
            username, user_movies = load_random_user_data(min_reviews=min_reviews)
            
            # Split user data
            train_movies, test_movies = split_user_data(user_movies)
            logger.info(f"\nEvaluating recommendations for user: {username}")
            logger.info(f"Training set: {len(train_movies)} movies")
            logger.info(f"Test set: {len(test_movies)} movies")
            
            # Show user's highest-rated movies from training set
            sorted_train = sorted(train_movies, key=lambda x: x['user_rating'], reverse=True)
            logger.info("\nUser's top-rated movies (training set):")
            for movie in sorted_train[:5]:
                logger.info(f"- {movie['movie_title']}: {movie['user_rating']} stars")
            
            # Get and evaluate recommendations
            metrics = evaluate_recommendations(train_movies, test_movies, movie_embeddings, model)
            
            if metrics:
                total_precision += metrics['precision']
                total_recall += metrics['recall']
                
                logger.info(f"\nPrecision: {metrics['precision']:.3f}")
                logger.info(f"Recall: {metrics['recall']:.3f}")
                
                logger.info("\nTop Recommendations:")
                for slug, sim, meta in metrics['recommendations']:
                    title = meta.get("title", slug)
                    logger.info(f"- {title} (similarity: {sim:.3f})")
                
                logger.info("\nActual movies user rated highly (test set):")
                sorted_test = sorted(test_movies, key=lambda x: x['user_rating'], reverse=True)
                for movie in sorted_test[:5]:
                    logger.info(f"- {movie['movie_title']}: {movie['user_rating']} stars")
        
        except Exception as e:
            logger.error(f"Error processing user: {e}")
            continue
    
    # Print average metrics
    avg_precision = total_precision / num_users
    avg_recall = total_recall / num_users
    logger.info(f"\n=== Overall Metrics ===")
    logger.info(f"Average Precision: {avg_precision:.3f}")
    logger.info(f"Average Recall: {avg_recall:.3f}")

def test_specific_user(username="mostephen", min_reviews=5):
    """Test recommendations for a specific user."""
    # Load model and embeddings from cache
    model, movie_embeddings = load_cached_model_and_embeddings()
    
    if model is None or movie_embeddings is None:
        # Load everything if cache doesn't exist
        processed_file = "/Users/monroestephenson/Downloads/Criterion_Collection_Recomendation/checkpoints_20250203_144151/final_processed_checkpoint.json"
        logger.info(f"Loading processed movies from {processed_file}")
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_movies = json.load(f)
        
        model = load_transformer_model()
        movie_embeddings = build_movie_embeddings(processed_movies, model)
        save_model_and_embeddings(model, movie_embeddings)
    
    try:
        # Load user's Criterion Collection reviews
        reviews_dir = "/Users/monroestephenson/Downloads/Criterion_Collection_Recomendation/movie_reviews"
        user_movies = []
        
        # Walk through all JSON files to find user's reviews
        for json_file in os.listdir(reviews_dir):
            if not json_file.endswith('.json'):
                continue
                
            with open(os.path.join(reviews_dir, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                movie_data = data.get('movie', {})
                reviews = data.get('reviews', [])
                
                # Look for reviews by our target user
                for review in reviews:
                    if review.get('username', '').lower() == username.lower():
                        movie_url = movie_data.get('url', '')
                        rating_str = review.get('rating', '')
                        if rating_str:  # Only include reviews with ratings
                            user_movies.append({
                                "slug": movie_url.split('/film/')[-1].strip('/'),
                                "user_rating": float(rating_str.count('★') + 0.5 * ('½' in rating_str)),
                                "review_text": review.get('text', ''),
                                "movie_title": movie_data.get('title', '')
                            })
        
        if len(user_movies) < min_reviews:
            logger.error(f"Not enough reviews found. Found {len(user_movies)} Criterion Collection reviews.")
            return
            
        logger.info(f"\nFound {len(user_movies)} Criterion Collection reviews for user: {username}")
        
        # Split user data
        train_movies, test_movies = split_user_data(user_movies)
        logger.info(f"Training set: {len(train_movies)} movies")
        logger.info(f"Test set: {len(test_movies)} movies")
        
        # Show user's highest-rated movies from training set
        sorted_train = sorted(train_movies, key=lambda x: x['user_rating'], reverse=True)
        logger.info("\nUser's top-rated Criterion Collection movies (training set):")
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
            
            logger.info("\nActual highly-rated movies (test set):")
            sorted_test = sorted(test_movies, key=lambda x: x['user_rating'], reverse=True)
            for movie in sorted_test[:5]:
                logger.info(f"- {movie['movie_title']}: {movie['user_rating']} stars")
                
    except Exception as e:
        logger.error(f"Error processing user data: {e}")

if __name__ == "__main__":
    test_user_recommendations(num_users=5, min_reviews=10)