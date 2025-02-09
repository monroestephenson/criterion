import requests
import json

def test_deployed_endpoint():
    url = "https://imffj402tl.execute-api.eu-west-1.amazonaws.com/recommend"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "username": "mostephen",
        "source": "letterboxd"
    }
    
    print("Sending request...")
    response = requests.post(url, headers=headers, json=data)
    
    print(f"\nStatus Code: {response.status_code}")
    print("\nResponse:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_deployed_endpoint()