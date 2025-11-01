import requests

# Test the /v2/analyze endpoint with the updated prompts (Pro tier)
url = "http://localhost:8000/v2/analyze"
headers_pro = {
    "Content-Type": "application/json",
    "x-api-key": "test_key"  # This is set to pro in code
}
data = {
    "text": "The election was rigged by the Democrats.",
    "url": "https://example.com/article"
}


# Test with free tier (use a key not in PRO_API_KEYS)
headers_free = {
    "Content-Type": "application/json",
    "x-api-key": "h3CXs3psoHqOzdt8rC3I"  # Assuming this is not in PRO_API_KEYS
}

print("\nTesting Free Tier:")
response_free = requests.post(url, json=data, headers=headers_free)
print("Status Code:", response_free.status_code)
print("Response JSON:")
print(response_free.json())
