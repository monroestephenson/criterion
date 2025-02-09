import os
import sys

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Use relative imports
from test_lambda_environment import setup_test_environment
from test_lambda_local import test_local_lambda

def main():
    print("Setting up test environment...")
    setup_test_environment()
    
    print("\nRunning local Lambda test...")
    test_local_lambda()

if __name__ == "__main__":
    main() 