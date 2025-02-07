import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_builder import load_transformer_model, build_movie_embeddings, save_model_and_embeddings
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Creating model cache directory...")
        os.makedirs("./data/model_cache", exist_ok=True)

        logger.info("Loading processed movies...")
        with open("./data/processed_movies.json", "r") as f:
            processed_movies = json.load(f)

        logger.info("Loading transformer model...")
        model = load_transformer_model()
        
        logger.info("Building movie embeddings...")
        movie_embeddings = build_movie_embeddings(processed_movies, model)
        
        logger.info("Saving model and embeddings...")
        save_model_and_embeddings(model, movie_embeddings, "./data/model_cache")
        
        logger.info("Model rebuild complete!")
        
    except Exception as e:
        logger.error(f"Error rebuilding models: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 