import requests

import requests

def stream_response(url, payload):
    with requests.post(url, json=payload, stream=True) as response:
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))

stream_response("http://localhost:8000/generate", {"query": "What is the capital of the moon?"})
