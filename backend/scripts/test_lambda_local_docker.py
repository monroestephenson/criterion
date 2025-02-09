import os
import sys
import json
import subprocess
import time

def test_lambda_in_docker(reuse_image=True):
    if not reuse_image:
        print("Building Lambda Docker image for testing...")
        subprocess.run([
            'docker', 'build',
            '-f', 'Dockerfile.lambda',
            '-t', 'lambda-test',
            '.'
        ], check=True)
    
    print("\nStarting Lambda container...")
    container = subprocess.run([
        'docker', 'run',
        '-d',  # Run in background
        '-p', '9000:8080',  # Lambda runtime API port
        '-e', 'FLASK_ENV=development',
        '-e', 'MODEL_PATH=/var/task/data/models',
        '-e', 'CACHE_PATH=/var/task/data/model_cache',
        'lambda-test'
    ], capture_output=True, text=True, check=True)
    
    container_id = container.stdout.strip()
    
    try:
        # Wait for container to be ready
        time.sleep(5)
        
        # Test the endpoint
        test_endpoint(container_id)
        
    finally:
        print("\nCleaning up...")
        subprocess.run(['docker', 'stop', container_id], check=True)
        subprocess.run(['docker', 'rm', container_id], check=True)

def test_endpoint(container_id):
    # Create API Gateway v2 format event
    event = {
        "version": "2.0",
        "routeKey": "POST /recommend",
        "rawPath": "/recommend",
        "headers": {
            "content-type": "application/json",
            "Host": "lambda-test",
            "User-Agent": "curl/7.64.1"
        },
        "requestContext": {
            "accountId": "123456789012",
            "apiId": "api-id",
            "domainName": "lambda-test",
            "domainPrefix": "test",
            "http": {
                "method": "POST",
                "path": "/recommend",
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",
                "userAgent": "curl/7.64.1"
            },
            "requestId": "request-id",
            "routeKey": "POST /recommend",
            "stage": "$default",
            "time": "03/Feb/2024:19:03:58 +0000",
            "timeEpoch": 1707073438
        },
        "body": json.dumps({
            "username": "mostephen",
            "source": "letterboxd"
        }),
        "isBase64Encoded": False
    }
    
    print("\nSending test request...")
    response = subprocess.run([
        'curl', '-XPOST',
        'http://localhost:9000/2015-03-31/functions/function/invocations',
        '-d', json.dumps(event)
    ], capture_output=True, text=True)
    
    print("\nResponse:")
    print(json.dumps(json.loads(response.stdout), indent=2))

if __name__ == "__main__":
    test_lambda_in_docker(reuse_image=True)
