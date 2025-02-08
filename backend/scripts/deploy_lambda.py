import boto3
import os
import subprocess
from botocore.config import Config

def deploy_to_lambda():
    # Debug: Print environment variables
    print(f"AWS_PROFILE: {os.getenv('AWS_PROFILE')}")
    print(f"AWS_DEFAULT_REGION: {os.getenv('AWS_DEFAULT_REGION')}")
    
    # Use SSO profile configuration
    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE'))
    
    # Debug: Print session credentials
    credentials = session.get_credentials()
    if credentials is None:
        print("No credentials found in session!")
    else:
        print("Credentials found in session")
    
    # Create clients with session and explicit region
    region = os.getenv('AWS_DEFAULT_REGION', 'eu-west-1')  # Add fallback to eu-west-1
    config = Config(
        region_name=region,
        signature_version='v4',
        retries={
            'max_attempts': 3,
            'mode': 'standard'
        }
    )
    
    ecr_client = session.client('ecr', config=config)
    lambda_client = session.client('lambda', config=config)
    
    # ECR repository name
    repository_name = 'monroes-movie-api'
    
    try:
        # Create ECR repository if it doesn't exist
        try:
            ecr_client.create_repository(repositoryName=repository_name)
        except ecr_client.exceptions.RepositoryAlreadyExistsException:
            pass
        
        # Get ECR repository URI
        response = ecr_client.describe_repositories(repositoryNames=[repository_name])
        repository_uri = response['repositories'][0]['repositoryUri']
        
        # Build Docker image with platform specification and no provenance
        subprocess.run([
            'docker', 'build',
            '--platform', 'linux/amd64',
            '--provenance=false',  # Disable multi-arch manifest
            '--no-cache',  # Force a clean build
            '-f', 'Dockerfile.lambda',
            '-t', repository_name,
            '.'
        ], check=True)
        
        # Tag image for ECR
        subprocess.run([
            'docker', 'tag',
            f'{repository_name}:latest',
            f'{repository_uri}:latest'
        ], check=True)
        
        # Get ECR login token and use it to login
        auth = ecr_client.get_authorization_token()
        token = auth['authorizationData'][0]['authorizationToken']
        endpoint = auth['authorizationData'][0]['proxyEndpoint']
        
        # Use get-login-password instead of direct login
        login_cmd = f"aws ecr get-login-password --region {region} --profile {os.getenv('AWS_PROFILE')} | docker login --username AWS --password-stdin {endpoint}"
        subprocess.run(login_cmd, shell=True, check=True)
        
        # Push image to ECR
        subprocess.run([
            'docker', 'push',
            f'{repository_uri}:latest'
        ], check=True)
        
        # Get the image URI
        image_uri = f'{repository_uri}:latest'
        
        # Try to update the function, create it if it doesn't exist
        try:
            lambda_client.update_function_code(
                FunctionName='movie-recommender',
                ImageUri=image_uri
            )
        except lambda_client.exceptions.ResourceNotFoundException:
            print("Lambda function doesn't exist, creating it...")
            lambda_client.create_function(
                FunctionName='movie-recommender',
                Role='arn:aws:iam::186292285156:role/lambda-movie-recommender-role',
                PackageType='Image',
                Code={'ImageUri': image_uri},
                Timeout=30,
                MemorySize=2048,
                Environment={
                    'Variables': {
                        'FLASK_ENV': 'production',
                        'MODEL_PATH': '/var/task/data/models',
                        'CACHE_PATH': '/var/task/data/model_cache'
                    }
                }
            )
        
        print("Deployment successful!")
        
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    deploy_to_lambda() 