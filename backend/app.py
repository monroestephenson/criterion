from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
import os
import sys
import psutil
import gc
from transformer_builder import (
    load_transformer_model,
    build_movie_embeddings,
    load_cached_model_and_embeddings,
    save_model_and_embeddings,
    split_user_data,
    evaluate_recommendations
    # Remove or comment out test_specific_user if not needed
    # test_specific_user
)
from bs4 import BeautifulSoup
import requests
import time
from collections import defaultdict
import pickle
from transformer_builder_scripts import compute_movie_similarity
from movie_clustering import (
    create_movie_clusters, 
    find_similar_movies, 
    save_clustering_results, 
    load_clustering_results,
    load_or_create_clusters
)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - Memory: %(memory_usage).2f MB'
)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Add memory usage to log record
old_factory = logging.getLogRecordFactory()

def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    record.memory_usage = get_memory_usage()
    return record

logging.setLogRecordFactory(record_factory)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
MODELS = None
clustering_results = None
CRITERION_MOVIES = None
REVIEW_MOVIES = None

MODEL_PATHS = {
    'review_model': "./data/model_cache",
    'subtitle_model': "./data/models/advanced_subtitle_embeddings.pkl",
    'clustering_model': "./data/models/movie_clusters.pkl"
}

def load_models():
    """Load both models with proper error handling"""
    global MODELS
    if MODELS is not None:
        logger.info("Using cached models")
        return MODELS
        
    logger.info("Starting model loading process")
    models = {}
    
    try:
        # Load review model
        logger.info("Loading review model from %s", MODEL_PATHS['review_model'])
        model, movie_embeddings = load_cached_model_and_embeddings(MODEL_PATHS['review_model'])
        
        if model is not None and movie_embeddings is not None:
            logger.info("Review model loaded successfully. Embeddings shape: %s", 
                       len(movie_embeddings) if movie_embeddings else "None")
            models['review'] = (model, movie_embeddings)
        else:
            logger.error("Failed to load review model - received None")
            return None
            
        # Load subtitle model
        logger.info("Loading subtitle model from %s", MODEL_PATHS['subtitle_model'])
        try:
            with open(MODEL_PATHS['subtitle_model'], 'rb') as f:
                subtitle_embeddings = pickle.load(f)
                logger.info("Subtitle embeddings loaded. Size: %s", 
                           len(subtitle_embeddings) if subtitle_embeddings else "None")
                models['subtitle'] = subtitle_embeddings
        except Exception as e:
            logger.error("Error loading subtitle model: %s", str(e))
            return None
            
        logger.info("All models loaded successfully")
        return models
        
    except Exception as e:
        logger.exception("Unexpected error during model loading: %s", str(e))
        return None

def initialize_app():
    """Initialize all application data"""
    global MODELS, clustering_results, CRITERION_MOVIES, REVIEW_MOVIES
    
    try:
        # Load movie lists first (smaller files)
        logger.info("Loading movie lists...")
        with open("./data/criterion_movies.json", 'r') as f:
            CRITERION_MOVIES = {movie['title'].lower(): movie for movie in json.load(f)}
        logger.info(f"Loaded {len(CRITERION_MOVIES)} criterion movies")
        
        with open("./data/movies_list.json", 'r') as f:
            REVIEW_MOVIES = {movie['title'].lower(): movie for movie in json.load(f)}
        logger.info(f"Loaded {len(REVIEW_MOVIES)} review movies")
        
        # Initialize models
        logger.info("Starting model initialization")
        logger.info(f"Initial memory usage: {get_memory_usage():.2f} MB")
        
        MODELS = load_models()
        if MODELS is None:
            raise Exception("Failed to load models")
            
        logger.info(f"Memory after loading models: {get_memory_usage():.2f} MB")
        gc.collect()
        
        # Initialize clustering
        logger.info("Loading clustering results...")
        clustering_results = load_or_create_clusters(MODELS)
        if clustering_results is None:
            raise Exception("Failed to load clustering results")
            
        logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.exception(f"Error during initialization: {str(e)}")
        return False

# Initialize application
logger.info("Starting application initialization")
if not initialize_app():
    logger.error("Failed to initialize application. Exiting...")
    sys.exit(1)

logger.info("Application initialization complete")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        username = data.get("username")
        logger.info(f"Received request for username: {username}")
        
        if not username:
            return jsonify({"error": "Username is required."}), 400

        # Use already loaded subtitle embeddings from MODELS
        subtitle_embeddings = MODELS['subtitle']
        if subtitle_embeddings is None:
            return jsonify({"error": "Subtitle model not properly initialized"}), 500

        # Extract review model and embeddings
        model, movie_embeddings = MODELS['review']
        if model is None or movie_embeddings is None:
            return jsonify({"error": "Review model not properly initialized"}), 500

        # Try Letterboxd first, passing subtitle_embeddings
        user_movies = get_letterboxd_reviews(username, subtitle_embeddings)
        
        # If Letterboxd fails or returns too few reviews, try Criterion
        if len(user_movies) < 5:
            logger.info(f"Not enough Letterboxd reviews ({len(user_movies)}), trying Criterion Collection")
            criterion_movies = get_user_reviews(username)
            if len(criterion_movies) >= 5:
                user_movies = criterion_movies
            elif len(criterion_movies) + len(user_movies) >= 5:
                user_movies.extend(criterion_movies)
        
        logger.info(f"Found {len(user_movies)} total reviews for user")
        
        if len(user_movies) < 5:
            return jsonify({
                "error": f"Not enough reviews found. Found {len(user_movies)} reviews.",
                "reviews_found": len(user_movies)
            }), 400

        # Get recommendations from both models
        review_based_metrics = evaluate_recommendations(user_movies, [], movie_embeddings, model)
        
        # Get recommendations from subtitle model
        subtitle_recommendations = get_subtitle_recommendations(user_movies, subtitle_embeddings)
        if not review_based_metrics:
            return jsonify({"error": "Could not generate recommendations"}), 500

        # Get top rated movies from all reviews
        sorted_movies = sorted(user_movies, key=lambda x: x['user_rating'], reverse=True)
        top_rated = [
            {
                "title": movie['movie_title'],
                "rating": movie['user_rating'],
                "slug": movie['slug']
            }
            for movie in sorted_movies[:5]
        ]

        # Combine and deduplicate recommendations
        combined_recommendations = combine_recommendations(
            review_based_metrics['recommendations'],
            subtitle_recommendations
        )

        # Format response
        response = {
            "user_stats": {
                "total_reviews": len(user_movies),
                "training_reviews": len(user_movies),
                "test_reviews": 0
            },
            "top_rated_movies": top_rated,
            "recommendations": combined_recommendations
        }

        return jsonify(response)

    except Exception as e:
        logger.exception("Error processing recommendation request")
        return jsonify({"error": str(e)}), 500

def get_user_reviews(username):
    """Retrieve user reviews from the Criterion Collection dataset."""
    reviews_dir = "./data/movie_reviews"
    user_movies = []
    
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
    
    return user_movies

def get_letterboxd_reviews(username, subtitle_embeddings=None):
    """Scrape user reviews from Letterboxd."""
    base_url = f"https://letterboxd.com/{username}/films/reviews/page/"
    user_movies = []
    page = 1
    
    # Create a set of lowercase Criterion titles if subtitle_embeddings is provided
    criterion_titles = set()
    if subtitle_embeddings:
        criterion_titles = {
            data['metadata']['title'].lower()
            for data in subtitle_embeddings.values()
        }
    
    logger.info(f"Found {len(criterion_titles)} Criterion Collection titles")
    
    while True:
        try:
            current_url = f"{base_url}{page}/"
            logger.info(f"Fetching page {page}: {current_url}")
            
            response = requests.get(current_url, headers={'User-Agent': 'Mozilla/5.0'})
            
            if response.status_code != 200:
                logger.info(f"Reached end of reviews at page {page}")
                break
                
            soup = BeautifulSoup(response.text, 'html.parser')
            film_entries = soup.find_all('h2', class_='headline-2')
            
            if not film_entries:
                logger.info(f"No reviews found on page {page}")
                break
                
            logger.info(f"Found {len(film_entries)} reviews on page {page}")
            
            for entry in film_entries:
                try:
                    parent_section = entry.find_parent()
                    film_link = entry.find('a')
                    if not film_link:
                        continue
                        
                    movie_title = film_link.text.strip()
                    movie_url = film_link['href']
                    
                    # Skip if movie is not in Criterion Collection
                    if criterion_titles and movie_title.lower() not in criterion_titles:
                        continue
                    
                    rating_elem = parent_section.find('span', class_='rating')
                    if rating_elem:
                        rating_str = rating_elem.text.strip()
                        rating = float(rating_str.count('★') + 0.5 * ('½' in rating_str))
                    else:
                        continue
                    
                    review_text = ''
                    review_elem = parent_section.find_next_sibling()
                    if review_elem:
                        review_text = review_elem.text.strip()
                    
                    user_movies.append({
                        "slug": movie_url.split('/film/')[-1].strip('/'),
                        "user_rating": rating,
                        "review_text": review_text,
                        "movie_title": movie_title
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing review: {str(e)}")
                    continue
            
            page += 1
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error fetching page {page}: {str(e)}")
            break
    
    logger.info(f"Successfully processed {len(user_movies)} Criterion Collection reviews")
    return user_movies

def test_manual_recommendation(username="mostephen"):
    """Test the recommendation pipeline directly without the Flask endpoint"""
    try:
        logger.info(f"\nTesting recommendations for user: {username}")
        
        # Get user reviews
        user_movies = get_user_reviews(username)
        logger.info(f"Found {len(user_movies)} reviews for user")
        
        if len(user_movies) < 5:
            logger.error(f"Not enough reviews found. Found {len(user_movies)} Criterion Collection reviews.")
            return
            
        # Split user data and get recommendations
        train_movies, test_movies = split_user_data(user_movies)
        logger.info(f"Training set: {len(train_movies)} movies")
        logger.info(f"Test set: {len(test_movies)} movies")
        
        # Show user's highest-rated movies
        sorted_train = sorted(train_movies, key=lambda x: x['user_rating'], reverse=True)
        logger.info("\nUser's top-rated movies:")
        for movie in sorted_train[:5]:
            logger.info(f"- {movie['movie_title']}: {movie['user_rating']} stars")
            
        # Get recommendations
        metrics = evaluate_recommendations(train_movies, test_movies, movie_embeddings, model)
        
        if metrics:
            logger.info(f"\nPrecision: {metrics['precision']:.3f}")
            logger.info(f"Recall: {metrics['recall']:.3f}")
            
            logger.info("\nTop Recommendations:")
            for slug, sim, meta in metrics['recommendations']:
                title = meta.get("title", slug)
                logger.info(f"- {title} (similarity: {sim:.3f})")
        else:
            logger.error("Could not generate recommendations")
            
    except Exception as e:
        logger.exception("Error in manual test")

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok"})

def get_subtitle_recommendations(user_movies, subtitle_embeddings, top_k=10):
    """Get recommendations based on subtitle embeddings"""
    try:
        similarities = defaultdict(float)
        counts = defaultdict(int)
        matched_count = 0
        
        # Create a set of valid Criterion titles for faster lookup
        criterion_titles = {title.lower() for title in CRITERION_MOVIES.keys()}
        
        # Filter subtitle embeddings to only include Criterion movies
        valid_subtitle_embeddings = {
            film_id: data
            for film_id, data in subtitle_embeddings.items()
            if data['metadata']['title'].lower() in criterion_titles
        }
        
        logger.info(f"Found {len(valid_subtitle_embeddings)} valid Criterion Collection movies in subtitle data")
        
        # Create efficient title mapping
        title_to_id = {
            data['metadata']['title'].lower(): film_id 
            for film_id, data in valid_subtitle_embeddings.items()
        }
        
        for movie in user_movies:
            movie_title = movie['movie_title'].lower()
            rating_weight = movie['user_rating'] / 5.0
            
            if movie_title in title_to_id:
                matched_count += 1
                film_id = title_to_id[movie_title]
                data = valid_subtitle_embeddings[film_id]
                
                logger.info(f"Found subtitle match for: {movie['movie_title']}")
                
                # Only compare against other valid Criterion movies
                for other_id, other_data in valid_subtitle_embeddings.items():
                    if other_id != film_id:
                        sim = compute_movie_similarity(
                            data, 
                            other_data, 
                            aggregation="max"
                        )
                        similarities[other_id] += sim * rating_weight
                        counts[other_id] += 1
            else:
                logger.info(f"No subtitle match found for: {movie['movie_title']}")
        
        logger.info(f"Matched {matched_count} out of {len(user_movies)} user movies with subtitle data")
        
        if matched_count == 0:
            logger.warning("No movies matched with subtitle data. Cannot generate subtitle-based recommendations.")
            return []
            
        # Calculate average similarities
        recommendations = []
        for film_id in similarities:
            if counts[film_id] > 0:
                avg_sim = similarities[film_id] / counts[film_id]
                metadata = valid_subtitle_embeddings[film_id]['metadata']
                recommendations.append((
                    metadata.get('url', '').split('/film/')[-1].strip('/'),  # slug
                    float(avg_sim),
                    metadata
                ))
        
        # Sort by similarity
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
        
    except Exception as e:
        logger.error(f"Error getting subtitle recommendations: {e}")
        logger.exception("Detailed error:")  # This will print the full stack trace
        return []

def combine_recommendations(review_recs, subtitle_recs, weight_review=0.6):
    """Combine recommendations with proper score normalization"""
    combined_scores = {}
    
    # Get max similarities for normalization
    max_review_sim = max((sim for _, sim, _ in review_recs), default=1.0) if review_recs else 1.0
    max_subtitle_sim = max((sim for _, sim, _ in subtitle_recs), default=1.0) if subtitle_recs else 1.0
    
    logger.info(f"Raw max similarities - Review: {max_review_sim}, Subtitle: {max_subtitle_sim}")
    
    # Process review-based recommendations
    for slug, sim, meta in review_recs:
        title = meta.get('title', '').lower()
        if title in REVIEW_MOVIES:
            # Normalize to 0-1 range
            normalized_sim = (sim / max_review_sim) if max_review_sim > 0 else 0
            # Store the normalized score without weighting
            combined_scores[slug] = {
                'review_score': normalized_sim,
                'subtitle_score': 0,
                'meta': meta,
                'source': 'review'
            }
    
    # Process subtitle-based recommendations
    for slug, sim, meta in subtitle_recs:
        title = meta.get('title', '').lower()
        if title in CRITERION_MOVIES:
            # Normalize to 0-1 range
            normalized_sim = (sim / max_subtitle_sim) if max_subtitle_sim > 0 else 0
            
            if slug in combined_scores:
                # Add subtitle score to existing entry
                combined_scores[slug]['subtitle_score'] = normalized_sim
                combined_scores[slug]['source'] += '+subtitle'
            else:
                combined_scores[slug] = {
                    'review_score': 0,
                    'subtitle_score': normalized_sim,
                    'meta': meta,
                    'source': 'subtitle'
                }
    
    # Create final recommendations with weighted combination
    final_recommendations = []
    for slug, data in combined_scores.items():
        # Calculate weighted average of scores
        weighted_score = (
            (data['review_score'] * weight_review) +
            (data['subtitle_score'] * (1 - weight_review))
        )
        
        # Convert to percentage (0-100 range)
        percentage_score = weighted_score * 100
        
        final_recommendations.append({
            'title': data['meta'].get('title', ''),
            'similarity': round(percentage_score, 1),
            'slug': slug,
            'source': data['source']
        })
    
    # Sort by similarity score
    final_recommendations.sort(key=lambda x: x['similarity'], reverse=True)
    return final_recommendations[:10]

@app.route("/cluster_similar_movies/<movie_id>", methods=["GET"])
def get_cluster_similar_movies(movie_id):
    global MODELS, clustering_results
    
    if MODELS is None or clustering_results is None:
        return jsonify({"error": "Models not initialized"}), 500
        
    try:
        # Find similar movies
        similar_movies = find_similar_movies(movie_id, clustering_results)
        
        # Format response
        recommendations = []
        for similar_id, similarity in similar_movies:
            metadata = MODELS['subtitle'][similar_id]['metadata']
            recommendations.append({
                'title': metadata.get('title', ''),
                'similarity': round(similarity * 100, 1),
                'slug': similar_id,
                'year': metadata.get('year', ''),
                'director': metadata.get('director', ''),
                'genres': metadata.get('genres', [])
            })
            
        return jsonify({
            "recommendations": recommendations
        })
        
    except Exception as e:
        logger.exception("Error finding cluster-based similar movies")
        return jsonify({"error": str(e)}), 500

@app.route("/recommendations/<movie_id>", methods=["GET"])
def get_recommendations(movie_id):
    global MODELS, clustering_results
    if MODELS is None or clustering_results is None:
        return jsonify({"error": "Models not initialized"}), 500
    
    try:
        # Get similar movies using clustering
        similar_movies = find_similar_movies(movie_id, clustering_results)
        
        # Format recommendations
        recommendations = []
        for similar_id, similarity in similar_movies:
            # Get metadata from subtitle embeddings
            metadata = MODELS['subtitle'][similar_id]['metadata']
            recommendations.append({
                'title': metadata.get('title', ''),
                'similarity': round(float(similarity) * 100, 1),  # Ensure similarity is a float
                'slug': similar_id,
                'year': metadata.get('year', ''),  # Add year for poster lookup
                'director': metadata.get('director', ''),
                'genres': metadata.get('genres', [])
            })
        
        logger.info(f"Found {len(recommendations)} recommendations for movie {movie_id}")
        # Add debug logging
        logger.info(f"Sample recommendation: {recommendations[0] if recommendations else 'None'}")
        
        return jsonify({
            "recommendations": recommendations,
            "status": "success"
        })
        
    except Exception as e:
        logger.exception("Error getting recommendations")
        return jsonify({"error": str(e)}), 500

# Optional: Add an endpoint to force cluster recreation
@app.route("/admin/recreate_clusters", methods=["POST"])
def recreate_clusters():
    """Admin endpoint to force cluster recreation"""
    try:
        clustering_results = create_movie_clusters(
            MODELS['subtitle'],
            MODELS['review']
        )
        
        if clustering_results:
            save_clustering_results(clustering_results, MODEL_PATHS['clustering_model'])
            app.clustering_results = clustering_results
            return jsonify({"message": "Clusters recreated successfully"})
        else:
            return jsonify({"error": "Failed to create clusters"}), 500
            
    except Exception as e:
        logger.exception("Error recreating clusters")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Remove or comment out the test line
    # test_manual_recommendation("sorefined")
    # Enable the Flask server
    app.run(debug=True, port=5001, host='0.0.0.0')