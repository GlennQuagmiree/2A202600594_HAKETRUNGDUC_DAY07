import sys
from src.chunking import compute_similarity
from src.embeddings import _mock_embed, LocalEmbedder

# 5 cặp câu thử nghiệm
pairs = [
    (
        "The weather today is very nice and sunny.",
        "The weather today is very nice and sunny.",
        "Cặp 1: Trùng khớp hoàn toàn"
    ),
    (
        "The dog chased the cat up the tree.",
        "A canine ran after the feline up the oak.",
        "Cặp 2: Đồng nghĩa (khác từ vựng)"
    ),
    (
        "I love coding in Python and building AI models.",
        "Software engineering is a great career path.",
        "Cặp 3: Cùng chủ đề (lập trình/công nghệ)"
    ),
    (
        "We are learning about data foundations today.",
        "Apples are red and grow on trees.",
        "Cặp 4: Hoàn toàn khác biệt"
    ),
    (
        "The room is extremely hot.",
        "The room is freezing cold.",
        "Cặp 5: Ngược nghĩa (nóng vs lạnh)"
    )
]

print("=== CHẠY VỚI MOCK EMBEDDER ===")
for a, b, desc in pairs:
    vec_a = _mock_embed(a)
    vec_b = _mock_embed(b)
    sim = compute_similarity(vec_a, vec_b)
    print(f"{desc}: {sim:.4f}")

print("\n=== CHẠY VỚI LOCAL EMBEDDER ===")
try:
    embedder = LocalEmbedder()
    for a, b, desc in pairs:
        vec_a = embedder(a)
        vec_b = embedder(b)
        sim = compute_similarity(vec_a, vec_b)
        print(f"{desc}: {sim:.4f}")
except Exception as e:
    print("Lưu ý: Không thể tải LocalEmbedder (có thể bạn chưa cài sentence-transformers hoặc lỗi tải model).")
