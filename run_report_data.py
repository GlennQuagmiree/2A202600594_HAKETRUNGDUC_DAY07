"""
Script thu thập dữ liệu thực tế để điền vào REPORT.md
Chạy: python run_report_data.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from src.chunking import (
    FixedSizeChunker, SentenceChunker, RecursiveChunker,
    ChunkingStrategyComparator, compute_similarity
)
from src.embeddings import LocalEmbedder
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.models import Document

print("Loading Local Embedder (all-MiniLM-L6-v2)...")
embedder = LocalEmbedder()
print("Local Embedder ready!\n")

# ── Read documents ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def read_file(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return f.read()

docs = {
    "python_intro.txt": read_file("python_intro.txt"),
    "rag_system_design.md": read_file("rag_system_design.md"),
    "vector_store_notes.md": read_file("vector_store_notes.md"),
    "vi_retrieval_notes.md": read_file("vi_retrieval_notes.md"),
    "customer_support_playbook.txt": read_file("customer_support_playbook.txt"),
}

# ── Section 2 — Document inventory ──────────────────────────────────────
print("\n== SECTION 2: Document Inventory ==")
for name, content in docs.items():
    print(f"  {name}: {len(content)} chars")

# ── Section 3 — Baseline Chunking Comparator ────────────────────────────
print("\n== SECTION 3: ChunkingStrategyComparator ==")
comparator = ChunkingStrategyComparator()
for doc_name in ["python_intro.txt", "rag_system_design.md", "vector_store_notes.md"]:
    result = comparator.compare(docs[doc_name], chunk_size=200)
    print(f"\n  [Document: {doc_name}]")
    for strategy, stats in result.items():
        preserves = "Yes" if stats["avg_length"] > 100 else "Partial"
        print(f"    {strategy}: count={stats['count']}, avg_len={stats['avg_length']:.1f}, preserves_ctx={preserves}")

# ── Custom strategy comparison ──────────────────────────────────────────
print("\n== My Strategy (RecursiveChunker chunk_size=300 vs baseline 200) ==")
doc_text = docs["rag_system_design.md"]
baseline = RecursiveChunker(chunk_size=200)
my_strat = RecursiveChunker(chunk_size=300)
b_chunks = baseline.chunk(doc_text)
m_chunks = my_strat.chunk(doc_text)
print(f"  Baseline recursive(200): count={len(b_chunks)}, avg={sum(len(c) for c in b_chunks)/len(b_chunks):.1f}")
print(f"  My recursive(300):       count={len(m_chunks)}, avg={sum(len(c) for c in m_chunks)/len(m_chunks):.1f}")

# ── Section 5 — Similarity with Local Embedder ─────────────────────────
print("\n== SECTION 5: Similarity Predictions (Local Embedder) ==")
pairs = [
    ("The weather today is very nice and sunny.", "The weather today is very nice and sunny."),
    ("The dog chased the cat up the tree.", "A canine ran after the feline up the oak."),
    ("I love coding in Python and building AI models.", "Software engineering is a great career path."),
    ("We are learning about data foundations today.", "Apples are red and grow on trees."),
    ("The room is extremely hot.", "The room is freezing cold."),
]
for i, (a, b) in enumerate(pairs, 1):
    va = embedder(a); vb = embedder(b)
    score = compute_similarity(va, vb)
    print(f"  Pair {i}: {score:.4f}")

# ── Section 6 — Benchmark queries with Local Embedder ─────────────────
print("\n== SECTION 6: Benchmark Queries (Local Embedder, in-memory) ==")

store = EmbeddingStore(embedding_fn=embedder)
all_docs = []
for name, content in docs.items():
    chunker = RecursiveChunker(chunk_size=300)
    chunks = chunker.chunk(content)
    category = "vietnamese" if "vi_" in name else "english"
    doc_type = "tutorial" if "intro" in name or "notes" in name else "design"
    for j, chunk in enumerate(chunks):
        all_docs.append(Document(
            id=f"{name}_{j}",
            content=chunk,
            metadata={"source": name, "language": category, "type": doc_type}
        ))
print("  Embedding and loading docs into store...")
store.add_documents(all_docs)
print(f"  Total chunks in store: {store.get_collection_size()}")

benchmark_queries = [
    "What is Python and what is it used for?",
    "How does chunking affect retrieval quality in a RAG system?",
    "What is the role of metadata in vector search?",
    "How does cosine similarity work for text embeddings?",
    "What are common failure cases in retrieval systems?",
]

for i, q in enumerate(benchmark_queries, 1):
    results = store.search(q, top_k=3)
    top = results[0] if results else {}
    snippet = top.get("content", "")[:80].replace("\n", " ") if top else ""
    score = top.get("score", 0)
    print(f"  Q{i}: '{q}'")
    print(f"       Top chunk: [{top.get('id','?')}] score={score:.4f}")
    print(f"       Preview: {snippet}...")

print("\n== DONE ==")
