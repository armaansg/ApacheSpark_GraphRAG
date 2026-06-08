import json
import os
import urllib.request
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["COMPANY_API_KEY"]
BASE = os.environ["COMPANY_API_BASE"].rstrip("/")
EMBED_MODEL = os.environ["EMBEDDING_MODEL"]
CHAT_MODEL = os.environ["CHAT_MODEL"]

def post(path, payload):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)
    

emb = post("/embeddings", {
    "model": EMBED_MODEL,
    "input": "SparkSession SQL catalyst",
})
print("embedding OK, dims:", len(emb["data"][0]["embedding"]))

chat = post("/chat/completions", {
    "model": CHAT_MODEL,
    "messages": [{"role": "user", "content": "Reply with exactly: API works"}],
    "max_tokens": 20,
})

print("chat OK:", chat["choices"][0]["message"]["content"])