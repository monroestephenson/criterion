import os
import sys
import json
from test_lambda_environment import setup_test_environment

# Set up environment before importing the handler
setup_test_environment()

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lambda_handler import handler

def test_local_lambda():
    # Create the request body
    request_body = {
        "username": "mostephen",
        "source": "letterboxd"  # Add source parameter to match frontend
    }
    
    # Simulate API Gateway event
    event = {
        "version": "2.0",
        "routeKey": "POST /recommend",
        "rawPath": "/recommend",
        "rawQueryString": "",
        "headers": {
            "Content-Type": "application/json",
            "Host": "api.example.com",
            "User-Agent": "Custom/1.0",
            "Accept": "*/*"
        },
        "requestContext": {
            "accountId": "123456789012",
            "apiId": "api-id",
            "domainName": "api.example.com",
            "domainPrefix": "api",
            "http": {
                "method": "POST",
                "path": "/recommend",
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",
                "userAgent": "Custom/1.0"
            },
            "requestId": "request-id",
            "routeKey": "POST /recommend",
            "stage": "$default",
            "time": "03/Feb/2024:19:03:58 +0000",
            "timeEpoch": 1707073438
        },
        "body": json.dumps(request_body),  # Properly stringify the JSON body
        "isBase64Encoded": False
    }
    
    # Call the handler
    try:
        print("\nSending request with body:", event["body"])
        response = handler(event, context=None)
        print("\nResponse:")
        print(json.dumps(response, indent=2))
        return response
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    test_local_lambda() 