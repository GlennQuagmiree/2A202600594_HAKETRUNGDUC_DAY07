"""
Script goi API Flask de lay benchmark data voi Local Embedder
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import urllib.request
import urllib.parse
import json

BASE = "http://127.0.0.1:5000"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def api_post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def api_upload(filepath):
    """Multipart upload using urllib"""
    import uuid
    boundary = uuid.uuid4().hex
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        file_data = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
    ).encode("utf-8") + file_data + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="chunk_size"\r\n\r\n300\r\n'
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/api/store/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

# -- Clear store --
print("Clearing store...")
api_post("/api/store/clear", {})

# -- Upload all docs --
files = ["python_intro.txt", "rag_system_design.md", "vector_store_notes.md",
         "vi_retrieval_notes.md", "customer_support_playbook.txt"]

for fname in files:
    fpath = os.path.join(DATA_DIR, fname)
    print(f"Uploading {fname}...")
    result = api_upload(fpath)
    print(f"  -> {result.get('message','')}, store size: {result.get('size',0)}")

# -- Get store size --
req = urllib.request.Request(f"{BASE}/api/store/size")
with urllib.request.urlopen(req) as r:
    size_data = json.loads(r.read())
print(f"\nTotal chunks in store: {size_data['size']}")

# -- Run benchmark queries --
queries = [
    "What is Python and what is it used for?",
    "How does chunking affect retrieval quality in a RAG system?",
    "What is the role of metadata in vector search?",
    "How does cosine similarity work for text embeddings?",
    "What are common failure cases in retrieval systems?",
]

print("\n== BENCHMARK RESULTS (Local Embedder) ==")
for i, q in enumerate(queries, 1):
    result = api_post("/api/store/search", {"query": q, "top_k": 3})
    results = result.get("results", [])
    top = results[0] if results else {}
    snippet = top.get("content", "")[:80].replace("\n", " ")
    score = top.get("score", 0)
    doc_id = top.get("id", "?")
    source = top.get("metadata", {}).get("source", "?")
    print(f"Q{i}: {q}")
    print(f"     Top: [{doc_id}] score={score:.4f}")
    print(f"     Source: {source}")
    print(f"     Preview: {snippet}...")
    print()

print("== DONE ==")
