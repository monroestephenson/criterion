from mangum import Mangum
from app import app
import logging
import sys
import json
from asgiref.wsgi import WsgiToAsgi

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Convert WSGI app to ASGI
asgi_app = WsgiToAsgi(app)

# Create the Mangum handler with ASGI app
mangum_handler = Mangum(
    asgi_app,
    lifespan="off"
)

def handler(event, context):
    """Main Lambda handler with debugging"""
    try:
        # Log the incoming event structure
        logger.info("=== Lambda Handler Start ===")
        logger.info("Event structure:")
        logger.info(f"Headers: {event.get('headers', {})}")
        logger.info(f"HTTP Method: {event.get('requestContext', {}).get('http', {}).get('method')}")
        logger.info(f"Path: {event.get('requestContext', {}).get('http', {}).get('path')}")
        
        # Log raw body
        if 'body' in event:
            logger.info("Raw body type: %s", type(event['body']))
            logger.info("Raw body content: %s", event['body'])
        
        # Ensure proper content type
        if 'headers' not in event:
            event['headers'] = {}
        event['headers']['Content-Type'] = 'application/json'
        
        # Handle body
        if 'body' in event:
            try:
                # Parse and re-stringify the body
                if isinstance(event['body'], str):
                    logger.info("Parsing body string to JSON")
                    body_dict = json.loads(event['body'])
                    event['body'] = json.dumps(body_dict)
                    logger.info("Processed body: %s", event['body'])
                    
                    # Add raw body to headers for debugging
                    event['headers']['X-Raw-Body'] = event['body']
                    logger.info("Added raw body to headers")
            except json.JSONDecodeError as e:
                logger.error("JSON decode error: %s", str(e))
                return {
                    "statusCode": 400,
                    "body": json.dumps({"error": f"Invalid JSON in request body: {str(e)}"}),
                    "headers": {
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    }
                }
        
        logger.info("Calling Mangum handler")
        response = mangum_handler(event, context)
        logger.info("Mangum handler response: %s", response)
        logger.info("=== Lambda Handler End ===")
        return response
        
    except Exception as e:
        logger.error("Error in handler: %s", str(e))
        logger.error("Stack trace:", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        } 