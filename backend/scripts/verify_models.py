import pickle
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

files = [
    "./data/model_cache/transformer_model.pkl",
    "./data/model_cache/movie_embeddings.pkl",
    "./data/models/advanced_subtitle_embeddings.pkl",
    "./data/models/movie_clusters.pkl"
]

def verify_pickle_file(file_path):
    """Verify a pickle file by reading it in chunks"""
    try:
        if os.path.exists(file_path):
            logger.info(f"Verifying {file_path}...")
            with open(file_path, 'rb') as f:
                # Read in 10MB chunks
                chunk_size = 1024 * 1024 * 10
                data = bytearray()
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    data.extend(chunk)
                pickle.loads(data)
                logger.info(f"✓ {file_path} is valid")
                return True
        else:
            logger.warning(f"✗ {file_path} is missing")
            return False
    except Exception as e:
        logger.error(f"✗ Error with {file_path}: {str(e)}")
        return False

def main():
    all_valid = True
    for f in files:
        if not verify_pickle_file(f):
            all_valid = False
    
    if not all_valid:
        sys.exit(1)

if __name__ == "__main__":
    main() 