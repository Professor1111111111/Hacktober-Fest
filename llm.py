import requests

OLLAMA_URL = "http://localhost:11434"


def available_models():
    response = requests.get(
        f"{OLLAMA_URL}/api/tags",
        timeout=5
    )
    response.raise_for_status()

    return [
        model["name"]
        for model in response.json().get("models", [])
    ]


def chat(model, system, user):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "stream": False,
            "options": {
                "temperature": 0.1
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()["message"]["content"]