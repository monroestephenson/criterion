import os

def setup_test_environment():
    # Get the absolute path to the backend directory
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set environment variables to use local paths
    os.environ['FLASK_ENV'] = 'development'
    os.environ['MODEL_PATH'] = os.path.join(backend_dir, 'data', 'models')
    os.environ['CACHE_PATH'] = os.path.join(backend_dir, 'data', 'model_cache')
    
    # Print for debugging
    print("\nEnvironment variables set to:")
    print(f"FLASK_ENV: {os.getenv('FLASK_ENV')}")
    print(f"MODEL_PATH: {os.getenv('MODEL_PATH')}")
    print(f"CACHE_PATH: {os.getenv('CACHE_PATH')}")
    print(f"Backend directory: {backend_dir}\n")
    
    # Verify directories and files exist
    cache_path = os.getenv('CACHE_PATH')
    model_path = os.getenv('MODEL_PATH')
    
    print("Checking directory structure:")
    print(f"Cache directory exists: {os.path.exists(cache_path)}")
    print(f"Model directory exists: {os.path.exists(model_path)}")
    
    # Check specific files
    expected_files = {
        os.path.join(cache_path, 'transformer_model.pkl'): False,
        os.path.join(cache_path, 'movie_embeddings.pkl'): False,
        os.path.join(model_path, 'advanced_subtitle_embeddings.pkl'): False,
        os.path.join(model_path, 'movie_clusters.pkl'): False
    }
    
    print("\nChecking required files:")
    for file_path in expected_files:
        exists = os.path.isfile(file_path)
        expected_files[file_path] = exists
        print(f"File exists - {os.path.basename(file_path)}: {exists}")
        if exists:
            print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    # Verify all required files exist
    missing_files = [f for f, exists in expected_files.items() if not exists]
    if missing_files:
        raise ValueError(f"Missing required files:\n" + "\n".join(missing_files))
    
    print("\nAll required files found and verified!")
    
    # Print current working directory
    print(f"\nCurrent working directory: {os.getcwd()}")
    
    # List contents of model directories
    print("\nContents of cache directory:")
    if os.path.exists(cache_path):
        print("\n".join(f"- {f}" for f in os.listdir(cache_path)))
    
    print("\nContents of model directory:")
    if os.path.exists(model_path):
        print("\n".join(f"- {f}" for f in os.listdir(model_path))) 